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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import glob


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

opt.Ttot = 21
sample_rate = 10
sample_num = 10000

def process_single_mesh(mesh_file, frame_idx, total_frames):
    """
    Process single mesh file (for multi-threading)
    
    Args:
        mesh_file: Mesh file path
        frame_idx: Frame index
        total_frames: Total frame count
        
    Returns:
        dict: Dictionary containing processing results
    """
    try:
        print(f"Processing file {frame_idx+1}/{total_frames}: {os.path.basename(mesh_file)}")
        
        # Load mesh
        if mesh_file.endswith('.obj') or mesh_file.endswith('.ply'):
            mesh = o3d.io.read_triangle_mesh(mesh_file)
        else:
            # Try to load as point cloud
            pcd = o3d.io.read_point_cloud(mesh_file)
            points = np.asarray(pcd.points)
        
        if 'mesh' in locals() and len(mesh.vertices) > 0:
            points = np.asarray(mesh.vertices)
            mesh_data = mesh
        elif 'pcd' in locals() and len(pcd.points) > 0:
            points = np.asarray(pcd.points)
            mesh_data = pcd
        else:
            raise ValueError(f"Cannot load file: {mesh_file}")
        
        # Normalize point cloud (mimicking original code processing)
        points_norm = episodic_normalization(points[None], scale=1.0, x_trans=0.0, z_trans=0.0)[0]
        
        # Voxelization
        try:
            voxel = voxelize(points_norm, (opt.grid_size,) * 3, is_binarized=True)
        except Exception as e:
            print(f"Voxelization failed: {e}")
            raise
        
        return {
            'frame_idx': frame_idx,
            'mesh': mesh_data,
            'points_norm': points_norm,
            'voxel': voxel,
            'success': True
        }
        
    except Exception as e:
        print(f"Failed to process file {mesh_file}: {e}")
        return {
            'frame_idx': frame_idx,
            'mesh': None,
            'points_norm': None,
            'voxel': None,
            'success': False,
            'error': str(e)
        }

def load_mesh_sequence( mesh_folder, file_pattern="*.obj", max_frames=None):
    """
    Load mesh sequence and convert to voxels (multi-threaded version)
    
    Args:
        mesh_folder: Folder path containing mesh files
        file_pattern: File matching pattern, such as "*.obj", "frame_*.ply"
        max_frames: Maximum frame count limit
    
    Returns:
        voxel_sequence: (T, grid_size, grid_size, grid_size)
        mesh_sequence: Original mesh data list
    """
    start_time = time.time()
    
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder, file_pattern)))
    
    # if max_frames:
    #     mesh_files = mesh_files[:max_frames]
    
    if len(mesh_files) == 0:
        raise ValueError(f"No files matching {file_pattern} found in {mesh_folder}")
    
    print(f"Found {len(mesh_files)} mesh files")
    print(f"Starting multi-threaded processing...")
    
    # Multi-threaded processing
    max_workers = min(8, len(mesh_files))  # Limit maximum thread count
    print(f"  - Using {max_workers} threads")
    
    voxel_sequence = []
    mesh_sequence = []
    points_sequence = []
    
    # List to store results (sorted by frame index)
    results = [None] * len(mesh_files)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_mesh, mesh_file, i, len(mesh_files)): i 
            for i, mesh_file in enumerate(mesh_files)
        }
        
        # Collect results
        for future in as_completed(future_to_idx):
            result = future.result()
            if result['success']:
                results[result['frame_idx']] = result
                print(f"Completed file {result['frame_idx']+1}/{len(mesh_files)}")
            else:
                print(f"File {result['frame_idx']+1} processing failed: {result.get('error', 'Unknown error')}")
    
    # Sort results in order
    for i, result in enumerate(results):
        if result is None or not result['success']:
            raise ValueError(f"File {i+1} processing failed")
        
        mesh_sequence.append(result['mesh'])
        points_sequence.append(result['points_norm'])
        voxel_sequence.append(result['voxel'])
    
    # Convert to torch tensor
    voxel_sequence = torch.from_numpy(np.stack(voxel_sequence, axis=0)).float().cuda()
    
    processing_time = time.time() - start_time
    print(f"Multi-threaded processing completed!")
    print(f"  - Voxel sequence shape: {voxel_sequence.shape}")
    print(f"  - Processing time: {processing_time:.2f} seconds")
    print(f"  - Average per frame: {processing_time/len(mesh_files):.3f} seconds")
    
    return voxel_sequence, mesh_sequence, points_sequence



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

    ###########################################################################################
    # Interpolation Parameters
    start_frame = 50
    end_frame = 100
    num_interpolation = 10
    motion_name = "custom_interpolation"
    ###########################################################################################


    # filenames = ['data/demo/source/gHO_sBM_cAll_d20_mHO1_ch05.npy']
    meshfolder = "D:/Code/VVEditor/Rafa_Approves_hd_4k"

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1025, height=958, visible=False)
    target_voxel, _, _ = load_mesh_sequence(meshfolder)

    # for filename in filenames:
        # motion_name = filename.split('/')[-1].replace('.npy', '')
        # target_voxel = load_voxel(filename, opt, 0)
    
    with torch.no_grad():
        T_total = target_voxel.shape[0]
        if end_frame >= T_total:
            sys.exit(f"Error: 'end_frame' ({end_frame}) is out of bounds. Sequence has only {T_total} frames.")

        detector_log = network.kypt_detector(target_voxel[None])
        keypoints = detector_log['keypoints']
        affinity = detector_log['affinity']
        parents = network.dyna_module.parents
        _ = network.dyna_module.encode(keypoints, affinity)

        prev_state = network.dyna_module.init_kypt_rnn_state.expand(sample_num, -1)
        offset = network.dyna_module.get_offset(keypoints).expand(sample_num, -1, -1, -1)

        # --- Warm-up phase to get state at start_frame ---
        print(f"Warming up model to frame {start_frame}...")
        state_at_start = prev_state
        if start_frame > 0:
            for t in range(start_frame):
                keypoint_t = keypoints[:, t].clone().view(1, -1).expand(sample_num, -1)
                
                params_post = network.dyna_module.extract_post_dist(torch.cat([state_at_start, keypoint_t], dim=-1))
                post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
                post_std = torch.nn.functional.softplus(post_std) + 1e-4
                z_kypt_sampled = Normal(post_mean, post_std).rsample()
                
                keypoint_sampled_flat, _ = network.dyna_module.extract_kypt_from_latent_and_state(torch.cat([state_at_start, z_kypt_sampled], dim=-1), offset)
                keypoint_distance = (keypoint_sampled_flat - keypoint_t).pow(2).sum(dim=-1)
                min_sampled_idx = keypoint_distance.argmin()
                
                keypoint_best = keypoint_sampled_flat[min_sampled_idx][None]
                z_best = z_kypt_sampled[min_sampled_idx][None]
                
                rnn_input = torch.cat([keypoint_best, z_best], dim=-1)
                state_at_start = network.dyna_module.kypt_rnn_cell(rnn_input.expand(sample_num, -1), state_at_start)
        print("Warm-up finished.")

        def get_reconstructed_keypoint(t, current_state):
            keypoint_real = keypoints[:, t].clone().view(1, -1).expand(sample_num, -1)
            params_post = network.dyna_module.extract_post_dist(torch.cat([current_state, keypoint_real], dim=-1))
            post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
            post_std = torch.nn.functional.softplus(post_std) + 1e-4
            z_kypt_sampled = Normal(post_mean, post_std).rsample()
            
            keypoint_sampled_flat, _ = network.dyna_module.extract_kypt_from_latent_and_state(torch.cat([current_state, z_kypt_sampled], dim=-1), offset)
            keypoint_distance = (keypoint_sampled_flat - keypoint_real).pow(2).sum(dim=-1)
            min_sampled_idx = keypoint_distance.argmin()

            return keypoint_sampled_flat[min_sampled_idx].view(opt.nkeypoints, 4)

        # --- Get Start and End Keypoints for Interpolation ---
        print("Extracting start and end keypoints...")
        keypoints_start = get_reconstructed_keypoint(start_frame, state_at_start)
        
        # Continue RNN to get state at end_frame
        state_at_end = state_at_start
        for t in range(start_frame, end_frame):
            keypoint_t = keypoints[:, t].clone().view(1, -1).expand(sample_num, -1)
            params_post = network.dyna_module.extract_post_dist(torch.cat([state_at_end, keypoint_t], dim=-1))
            post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
            post_std = torch.nn.functional.softplus(post_std) + 1e-4
            z_kypt_sampled = Normal(post_mean, post_std).rsample()
            
            keypoint_sampled_flat, _ = network.dyna_module.extract_kypt_from_latent_and_state(torch.cat([state_at_end, z_kypt_sampled], dim=-1), offset)
            keypoint_distance = (keypoint_sampled_flat - keypoint_t).pow(2).sum(dim=-1)
            min_sampled_idx = keypoint_distance.argmin()
            
            keypoint_best = keypoint_sampled_flat[min_sampled_idx][None]
            z_best = z_kypt_sampled[min_sampled_idx][None]
            
            rnn_input = torch.cat([keypoint_best, z_best], dim=-1)
            state_at_end = network.dyna_module.kypt_rnn_cell(rnn_input.expand(sample_num, -1), state_at_end)

        keypoints_end = get_reconstructed_keypoint(end_frame, state_at_end)
        print("Keypoints extracted.")

        # --- Interpolate in Keypoint Space ---
        print(f"Interpolating {num_interpolation} frames...")
        interpolated_keypoints = []
        # Add start and end keyframes if you want them in the final video
        # interpolated_keypoints.append(keypoints_start)

        for i in range(num_interpolation):
            alpha = (i + 1.0) / (num_interpolation + 1.0)
            keypoints_interp = (1 - alpha) * keypoints_start + alpha * keypoints_end
            interpolated_keypoints.append(keypoints_interp)

        # interpolated_keypoints.append(keypoints_end)

        selected_keypoints = torch.stack(interpolated_keypoints, dim=0)[None]
        selected_keypoints[0, :, :, -1] = selected_keypoints[0, 0, :, -1].clone() # Use confidence from first frame

        # --- Decode interpolated keypoints to voxels ---
        print("Decoding interpolated keypoints to voxels...")
        first_feature = detector_log['first_feature']
        # Use the frame from the start of interpolation as the reference frame
        first_frame = target_voxel[None, start_frame] 
        decode_log = network.kypt_detector.decode_from_dyna(selected_keypoints, first_feature, first_frame)

        print("Decoding finished.")
        
        predicted_info = network.kypt_detector(decode_log['gen']) 
        predicted_keypoints = predicted_info['keypoints']
        predicted_affinity = predicted_info['affinity']
        dyna_log = network.dyna_module.encode(predicted_keypoints, predicted_affinity)
        R = dyna_log['R'][0]  # (T, K, 3, 3)
        pos = predicted_keypoints[0, :, :, :3][..., None]  # (T, K, 3, 1)
        T4x4 = torch.cat([R, pos], dim=-1)  # (T, K, 3, 4)
        homo = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(R.device)[None, None, None].expand(
            R.size(0), R.size(1), -1, -1)
        T4x4 = torch.cat([T4x4, homo], dim=-2)  # (T, K, 4, 4)

        # 正确获取parents信息，不能直接用上文的parents变量（此时parents未定义或为None）
        # 应该从dyna_module中获取parents，并确保parents为一维numpy数组
        if hasattr(network.dyna_module, 'parents'):
            parents = network.dyna_module.parents
            # 如果parents是tensor，转为numpy
            if hasattr(parents, 'cpu'):
                parents = parents.cpu().numpy()
            # 如果parents是0维或标量，转为一维数组
            parents = np.array(parents).reshape(-1)
        else:
            raise RuntimeError("network.dyna_module没有parents属性，无法保存parents.npy")

        print(parents.shape)

        output_dir = 'output/nm_interp'
        os.makedirs(output_dir, exist_ok=True)
        # 保存预测的关键点、变换矩阵和父子关系
        np.save(os.path.join(output_dir, 'keypoints.npy'), predicted_keypoints[0].cpu().numpy(), allow_pickle=True)
        np.save(os.path.join(output_dir, 'transforms.npy'), T4x4.cpu().numpy(), allow_pickle=True)
        np.save(os.path.join(output_dir, 'parents.npy'), parents, allow_pickle=True)

        from SkelVisualizer import visualize_skeleton
        results = visualize_skeleton(
            data_dir=output_dir,
            output_dir=os.path.join(output_dir, 'visualization'),
            create_sequence=True,    # 创建动画GIF文件
            create_animation=True    # 创建动画数据JSON
        )
        print("创建的文件:")
        for key, value in results.items():
            print(f"{key}: {value}")



        interp_voxel = decode_log['gen'].squeeze(0)
        interp_voxel[interp_voxel < 0.5] = 0
        interp_voxel[interp_voxel >= 0.5] = 1

        ############################################################################################################
        min_z = 1e4
        max_z = -1

        for t in range(len(interp_voxel)):
            coords = np.stack(np.where(interp_voxel[t, 0].clone().detach().cpu().numpy()), axis=-1) / ((64 - 1) / 2) - 1
            if len(coords) == 0: continue
            if min_z > coords[:, -1].min():
                min_z = coords[:, -1].min()
            if max_z < coords[:, -1].max():
                max_z = coords[:, -1].max()
        
        z_len = (max_z - min_z)
        
        imgs = []

        # Adjust sample_rate for visualization coloring
        T = len(interp_voxel)
        sample_rate_vis = T + 1 # Make all interpolated frames have gradient color
        from copy import deepcopy

        for t in range(len(interp_voxel)):
            coords = np.stack(np.where(interp_voxel[t, 0].clone().detach().cpu().numpy()), axis=-1) / ((64 - 1) / 2) - 1
            if len(coords) == 0: continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(5)
            pcd_normals = np.asarray(pcd.normals)

            for i in range(len(coords)):
                if t % sample_rate_vis == 0 or t == T - 1:
                    color = list(np.array([0.6, 0.6, 1.0]) * ((coords[i, -1] - min_z) / z_len * 0.9 + 0.1))
                else:
                    color = list(np.array([0.5 + ((t % sample_rate_vis) / (2 * T)), 0.5 + ((t % sample_rate_vis) / (2 * T)),
                                           0.5 + ((t % sample_rate_vis) / (2 * T))]) * (
                                               (coords[i, -1] - min_z) / z_len * 0.9 + 0.1) + np.array(
                       [(t % sample_rate_vis) / (2 * T), (t % sample_rate_vis) / (2 * T), (t % sample_rate_vis) / (2 * T)]))
                
                vis.add_geometry(drawPlate(coords[i], pcd_normals[i], color, 0.03))
        
            ctr = vis.get_view_control()
            parameters = o3d.io.read_pinhole_camera_parameters('data/source/source.json')
            ctr.convert_from_pinhole_camera_parameters(parameters)
            img = vis.capture_screen_float_buffer(True)
            img = np.asarray(img) * 255.
            vis.clear_geometries()

            final_img = img.astype(np.uint8)

            save_dir = 'output/nm_interp/%s' % motion_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not os.path.exists(os.path.join(save_dir, 'interp_result_imgs')):
                os.makedirs(os.path.join(save_dir, 'interp_result_imgs'))

            cv2.imwrite(os.path.join(save_dir, 'interp_result_imgs', '%02d.png' % t), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
            imgs.append(final_img)

        imageio.mimsave(os.path.join(save_dir, 'interp_result.gif'), imgs, duration=0.3)
