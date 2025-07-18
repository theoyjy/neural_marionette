import torch
import numpy as np
import sys
import os
import pickle
import open3d as o3d
from torch.distributions.normal import Normal
from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import crop_sequence, episodic_normalization, voxelize
import cv2
import imageio


def load_voxel(file, opt_file, start, scale=1.0, x_trans=0.0, z_trans=0.0):
    x = np.load(file)[..., :3]
    x = crop_sequence(x, start, opt_file.Ttot, opt_file.sample_rate)
    x = episodic_normalization(x, scale, x_trans, z_trans)

    vox_seq = []
    for t in range(len(x)):
        vox_seq.append(voxelize(x[t], (opt_file.grid_size,) * 3, is_binarized=True))
    
    vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float().cuda()

    return vox_seq

def drawPlate(center, orientation, color=[0.6, 0.9, 0.6], radius=0.02, compute_vertex_normals=False):
    plate = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.01, resolution=80)
    plate.translate([0, 0, -0.005])
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = orientation / (np.linalg.norm(orientation) + 1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4:
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    plate.transform(np.concatenate((np.concatenate((R, center[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    plate.paint_uniform_color(color)

    if compute_vertex_normals:
        plate.compute_vertex_normals()
    
    return plate

exp_dir = 'pretrained/aist'
opt_file = os.path.join(exp_dir, 'opt.pickle')
with open(opt_file, 'rb') as f:
    opt = pickle.load(f)

Tcond = 5
Tgen = 25
sample_num = 3

opt.Ttot = Tcond

if __name__ == "__main__":

    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.backends.cudnn.deterministic = True

    resume_file = os.path.join(exp_dir, 'aist_pretrained.pth')
    checkpoint = torch.load(resume_file)
    network = NeuralMarionette(opt).cuda()
    network.load_state_dict(checkpoint)
    network.eval()
    network.anneal(1)  # to enable extracting affinity

    ###################################################################################
    filenames = ['data/demo/source/gHO_sBM_cAll_d20_mHO1_ch05.npy']

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1025, height=958, visible=True)
    
    for filename in filenames:
        motion_name = filename.split('/')[-1].replace('.npy', '')
        cond_voxel = load_voxel(filename, opt, 0)
        
        with torch.no_grad():
            K = opt.nkeypoints
            detector_log = network.kypt_detector(cond_voxel[None])
            keypoints = detector_log['keypoints']
            affinity = detector_log['affinity']
            _ = network.dyna_module.encode(keypoints, affinity)
            A = network.dyna_module.A
            priority = network.dyna_module.priority
            parents = network.dyna_module.parents

            ###########################################################################
            prev_state = network.dyna_module.init_kypt_rnn_state.expand(sample_num, -1)
            offset = network.dyna_module.get_offset(keypoints).expand(sample_num, -1, -1, -1)
            cond_keypoints = []
            gen_keypoints = []

            for t in range(Tcond):
                keypoint = keypoints[:, t].clone()
                keypoint_flat = keypoint.view(1, -1).expand(sample_num, -1)

                params_post = network.dyna_module.extract_post_dist(torch.cat([prev_state, keypoint_flat], dim=-1))
                post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
                post_std = torch.nn.functional.softplus(post_std) + 1e-4
                z_kypt_post_dist = Normal(post_mean, post_std)
                z_kypt_sampled = z_kypt_post_dist.rsample()
                keypoint_sampled_flat, _ = network.dyna_module.extract_kypt_from_latent_and_state(torch.cat([prev_state, z_kypt_sampled], dim=-1), offset)
                keypoint_distance = (keypoint_sampled_flat - keypoint_flat).pow(2).sum(dim=-1)
                min_sampled_idx = keypoint_distance.argmin()
                keypoint_sampled_flat = keypoint_sampled_flat[min_sampled_idx][None].expand(sample_num, -1)
                z_kypt_sampled = z_kypt_sampled[min_sampled_idx][None].expand(sample_num, -1)
                prev_state = prev_state[min_sampled_idx][None].expand(sample_num, -1)
                cond_keypoints.append(keypoint_flat[min_sampled_idx].view(K, 4))

                rnn_input = torch.cat([keypoint_sampled_flat, z_kypt_sampled], dim=-1)
                prev_state = network.dyna_module.kypt_rnn_cell(rnn_input, prev_state)

            for tgen in range(Tgen):
                params_prior = network.dyna_module.extract_prior_dist(prev_state)
                prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
                prior_std = torch.nn.functional.softplus(prior_std) + 1e-4
                z_kypt_prior_dist = Normal(prior_mean, prior_std)
                z_kypt_sampled = z_kypt_prior_dist.rsample()
                keypoint_sampled_flat, _ = network.dyna_module.extract_kypt_from_latent_and_state(torch.cat([prev_state, z_kypt_sampled], dim=-1), offset)
                gen_keypoints.append(keypoint_sampled_flat.view(-1, K, 4))

                rnn_input = torch.cat([keypoint_sampled_flat, z_kypt_sampled], dim=-1)
                prev_state = network.dyna_module.kypt_rnn_cell(rnn_input, prev_state)

            cond_keypoints = torch.stack(cond_keypoints, dim=0)[None]
            gen_keypoints = torch.stack(gen_keypoints, dim=0)[None]
            
            for sample_id in range(sample_num):
                full_keypoints = torch.cat([cond_keypoints, gen_keypoints[:, :, sample_id]], dim=1)
                first_feature = detector_log['first_feature']
                first_frame = cond_voxel[None, 0]
                decode_log = network.kypt_detector.decode_from_dyna(full_keypoints, first_feature, first_frame)
                gen_voxel = decode_log['gen'].squeeze(0)
                gen_voxel[gen_voxel < 0.5] = 0
                gen_voxel[gen_voxel >= 0.5] = 1
                # gen_voxel[:Tcond] = cond_voxel[:Tcond]

                ############################################################################################################
                min_z = 1e4
                max_z = -1

                for t in range(len(gen_voxel)):
                    coords = np.stack(np.where(gen_voxel[t, 0].clone().detach().cpu().numpy()), axis=-1) / ((64 - 1) / 2) - 1
                    if min_z > coords[:, -1].min():
                        min_z = coords[:, -1].min()
                    if max_z < coords[:, -1].max():
                        max_z = coords[:, -1].max()
                
                z_len = (max_z - min_z)
                
                imgs = []

                for t in range(len(gen_voxel)):
                    coords = np.stack(np.where(gen_voxel[t, 0].clone().detach().cpu().numpy()), axis=-1) / ((64 - 1) / 2) - 1
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coords)
                    pcd.estimate_normals()
                    pcd.orient_normals_consistent_tangent_plane(5)
                    pcd_normals = np.asarray(pcd.normals)

                    for i in range(len(coords)):
                        if t < Tcond:
                            color = list(np.array([0.6, 0.6, 1.0]) * ((coords[i, -1] - min_z) / z_len * 0.8 + 0.2))
                        else:
                            color = list(np.array([0.6, 1.0, 0.6]) * ((coords[i, -1] - min_z) / z_len * 0.8 + 0.2))
                        
                        vis.add_geometry(drawPlate(coords[i], pcd_normals[i], color, 0.03))
                
                    # force a draw pass before capture
                    vis.poll_events()
                    vis.update_renderer()
                    ctr = vis.get_view_control()
                    parameters = o3d.io.read_pinhole_camera_parameters('data/demo/source/source.json')
                    # print('Current working directory:', os.getcwd())
                    ctr.convert_from_pinhole_camera_parameters(parameters)
                    img = vis.capture_screen_float_buffer(False)
                    img = np.asarray(img) * 255.
                    vis.clear_geometries()

                    final_img = img.astype(np.uint8)

                    save_dir = 'output/demo/generation/%s' % motion_name
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    if not os.path.exists(os.path.join(save_dir, 'gen_result_imgs_%d' % sample_id)):
                        os.makedirs(os.path.join(save_dir, 'gen_result_imgs_%d' % sample_id))

                    cv2.imwrite(os.path.join(save_dir, 'gen_result_imgs_%d' % sample_id, '%02d.png' % t), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
                    imgs.append(final_img)

                imageio.mimsave(os.path.join(save_dir, 'gen_result_%d.gif' % sample_id), imgs, duration=0.1)
                print('Sample %d generation finished, results saved to %s' % (sample_id, save_dir))
    print('Generation finished, results saved to %s' % save_dir)