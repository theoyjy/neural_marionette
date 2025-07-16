import argparse
import os
import time
import pickle
import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from scipy.spatial.transform import Rotation, Slerp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import collections

from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import episodic_normalization, voxelize


def extract_skin_weights(A, priority, parents, points, keypoints, HARDNESS=8.0, THRESHOLD=0.2):
    """
    Compute skinning weights for points given bone segments defined by keypoints and parents.
    """
    N, _ = points.shape
    K, _ = keypoints.size()
    points_torch = torch.from_numpy(points).to(A.device)
    # compute weights based on distance to bone segments
    dists = torch.zeros((N, K), device=A.device)
    for k in range(K):
        p = parents[k]
        joint_k = keypoints[k, :3]  # (3,)
        joint_p = keypoints[p, :3]  # (3,)
        seg = joint_p - joint_k  # (3,)
        seg_len2 = seg.pow(2).sum() + 1e-8
        # vector from joint to points: (N,3)
        rel = points_torch - joint_k
        # projection factor t onto infinite line: (N,)
        t = (rel * seg).sum(dim=1) / seg_len2
        proj = joint_k + t.unsqueeze(1) * seg.unsqueeze(0)  # (N,3)
        dists[:, k] = (points_torch - proj).pow(2).sum(dim=1).sqrt()
    # suppress invalid bones
    invalids = torch.where((keypoints[:, -1] < THRESHOLD))[0]
    dists[:, invalids] = float('inf')
    # convert to weights
    weights = torch.exp(-dists * HARDNESS)
    weights[:, invalids] = 0
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    return weights.cpu().numpy()


def draw_skeleton(positions, parents, color=[1,0,0], radius=0.02):
    """Return list of Open3D geometries for skeleton joints and bones."""
    geoms = []
    # spheres at joint positions
    for pt in positions:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius)
        sph.translate(pt)
        sph.paint_uniform_color(color)
        sph.compute_vertex_normals()
        geoms.append(sph)
    # lines for bones
    lines = [[k, int(p)] for k, p in enumerate(parents) if p != k]
    if lines:
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
        geoms.append(ls)
    return geoms


def load_voxel_from_mesh(file, opt, is_bind=False, scale=1.0, x_trans=0.0, z_trans=0.0):
    """
    Read mesh, normalize and voxelize for network input.
    """
    mesh = o3d.io.read_triangle_mesh(file, True)
    points = np.asarray(mesh.vertices)
    if is_bind:
        points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=-1)
    # compute normalization params
    bmin = points.min(axis=0)
    bmax = points.max(axis=0)
    blen = (bmax - bmin).max()
    # normalize points to [-1,1]
    pts_norm = ((points - bmin) * scale / (blen + 1e-5)) * 2 - 1 + np.array([x_trans, 0, z_trans])
    # voxelize
    vox = voxelize(pts_norm, (opt.grid_size,) * 3, is_binarized=True)
    # add batch and time dimensions: (1,1,C,D,H,W)
    vox = torch.from_numpy(vox)[None, None].float().cuda()
    return vox, points, mesh, bmin, blen


def geodesic_weight_correction(mesh, weights, max_geodesic_dist=0.3):
    """
    Use mesh connectivity to correct skinning weights based on geodesic distance.
    """    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    N = len(vertices)
    K = weights.shape[1]
    
    # Build adjacency matrix from mesh connectivity
    edges = set()
    for tri in triangles:
        for i in range(3):
            for j in range(i+1, 3):
                v1, v2 = tri[i], tri[j]
                if v1 != v2:
                    edges.add((min(v1,v2), max(v1,v2)))
    
    # Create sparse adjacency matrix with edge lengths as weights
    rows, cols, data = [], [], []
    for v1, v2 in edges:
        dist = np.linalg.norm(vertices[v1] - vertices[v2])
        rows.extend([v1, v2])
        cols.extend([v2, v1])
        data.extend([dist, dist])
    
    adjacency = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # For each bone, find vertices with significant weights
    corrected_weights = weights.copy()
    for k in range(K):
        # Find seed vertices (high weight for this bone)
        seeds = np.where(weights[:, k] > 0.1)[0]
        if len(seeds) == 0:
            continue
            
        # Compute geodesic distances from all seed vertices
        geodesic_dists = dijkstra(adjacency, indices=seeds, limit=max_geodesic_dist)
        min_geodesic = np.min(geodesic_dists, axis=0)
        
        # Zero out weights for vertices too far geodesically
        far_mask = min_geodesic > max_geodesic_dist
        corrected_weights[far_mask, k] = 0
    
    # Renormalize weights
    row_sums = corrected_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    corrected_weights = corrected_weights / row_sums
    
    return corrected_weights


def main():
    parser = argparse.ArgumentParser(description="Blend skeletons between two meshes using NeuralMarionette.")
    parser.add_argument('mesh1', type=str, help='Path to first mesh file.')
    parser.add_argument('mesh2', type=str, help='Path to second mesh file.')
    parser.add_argument('frames', type=int, help='Number of blending frames to generate.')
    parser.add_argument('--output_dir', type=str, default='output/blend', help='Output directory.')
    parser.add_argument('--is_bind', action='store_true', help='Use bind transform for mesh coords.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = 'pretrained/aist'
    opt_path = os.path.join(exp_dir, 'opt.pickle')
    with open(opt_path, 'rb') as f:
        opt = pickle.load(f)
    opt.Ttot = 1

    # load network
    ckpt_path = os.path.join(exp_dir, 'aist_pretrained.pth')
    checkpoint = torch.load(ckpt_path)
    network = NeuralMarionette(opt).cuda()
    network.load_state_dict(checkpoint)
    network.eval()
    network.anneal(1)
    # initialize non-blocking visualizer for skeletons
    vis_skel = o3d.visualization.Visualizer()
    vis_skel.create_window(window_name='Skeleton Preview', width=600, height=600, visible=True)

    # process first mesh
    t0 = time.time()
    # load and voxelize mesh
    vox1, pts1_raw, mesh1, bmin1, blen1 = load_voxel_from_mesh(args.mesh1, opt, args.is_bind)
    # normalize raw points for skinning
    pts1_norm = ((pts1_raw - bmin1) * 1.0 / (blen1 + 1e-5)) * 2 - 1
    with torch.no_grad():
        log1 = network.kypt_detector(vox1)
        kps1 = log1['keypoints'][0, 0]
        affinity1 = log1['affinity']
        dlog1 = network.dyna_module.encode(log1['keypoints'], affinity1)
        R1 = dlog1['R'][0, 0].cpu().numpy() # (K, 3, 3) local-frame rotations
        pos1 = kps1[:, :3].cpu().numpy() # (K, 3) joint positions
        A = network.dyna_module.A
        priority = network.dyna_module.priority
        parents = network.dyna_module.parents
        K = parents.shape[0]  # number of joints
        skin1 = extract_skin_weights(A, priority, parents, pts1_norm, kps1, HARDNESS=30.0, THRESHOLD=0.0)
        # Apply geodesic correction to prevent unrelated parts from moving
        try:
            skin1 = geodesic_weight_correction(mesh1, skin1, max_geodesic_dist=0.2)
            # 2. do weights sum to 1?
            print(np.abs(skin1.sum(axis=1) - 1).max())   # should be < 1e-6

        except ImportError:
            print("scipy not available, skipping geodesic correction")
    t1 = time.time()
    print(f"Mesh1 processing time: {t1 - t0:.4f}s")
    # non-blocking display of mesh1 skeleton
    for geo in draw_skeleton(pos1, parents, color=[1,0,0]): vis_skel.add_geometry(geo)
    vis_skel.poll_events(); vis_skel.update_renderer(); time.sleep(1)
    vis_skel.clear_geometries()

    # process second mesh
    t2 = time.time()
    vox2, pts2_raw, mesh2, bmin2, blen2 = load_voxel_from_mesh(args.mesh2, opt, args.is_bind)
    pts2_norm = ((pts2_raw - bmin2) * 1.0 / (blen2 + 1e-5)) * 2 - 1
    with torch.no_grad():
        log2 = network.kypt_detector(vox2)
        kps2 = log2['keypoints'][0, 0]
        affinity2 = log2['affinity']
        dlog2 = network.dyna_module.encode(log2['keypoints'], affinity2)
        R2 = dlog2['R'][0, 0].cpu().numpy()
        pos2 = kps2[:, :3].cpu().numpy()
        # Convert pos2 to the real-world coords of mesh-2 …
        pos2_world = ((pos2 + 1) / 2.0) * (blen2 + 1e-5) + bmin2
        # … then re-normalise with mesh-1’s bounding box
        pos2 = ((pos2_world - bmin1) / (blen1 + 1e-5)) * 2.0 - 1
        skin2 = extract_skin_weights(A, priority, parents, pts2_norm, kps2, HARDNESS=30.0, THRESHOLD=0.0)
    t3 = time.time()
    print(f"Mesh2 processing time: {t3 - t2:.4f}s")
    # non-blocking display of mesh2 skeleton
    for geo in draw_skeleton(pos2, parents, color=[0,1,0]): vis_skel.add_geometry(geo)
    vis_skel.poll_events(); vis_skel.update_renderer(); time.sleep(1)
    vis_skel.clear_geometries()

    # prepare per-joint Scipy SLERP between mesh1 and mesh2 skeletons
    slerp_joints = []
    key_times = [0.0, 1.0]
    for k in range(K):
        slerp_joints.append(
            Slerp(key_times, Rotation.from_matrix(np.stack([R1[k], R2[k]], axis=0)))
        )

    # compute bind and inverse bind transforms for mesh1 skeleton
    # Inverse bind will later undo rest-pose offsets during LBS.
    T_bind = np.zeros((K, 4, 4), dtype=np.float32)
    for k in range(K):
        T_bind[k] = np.eye(4, dtype=np.float32)
        T_bind[k, :3, :3] = R1[k]
        T_bind[k, :3, 3] = pos1[k]
    inv_bind = np.linalg.inv(T_bind)

    # a) 权重总和应≈1
    row_sum = skin1.sum(axis=1)
    print("min / max row-sum :", row_sum.min(), row_sum.max())

    # b) 是否存在全 0 行（这些顶点无骨骼约束）
    idx_zero = np.where(row_sum < 1e-6)[0]
    print("# zero-weight vertices :", idx_zero.size)

    # c) 单一骨骼支配过强？
    dominant = (skin1 > 0.95).sum(axis=1)   # True 列表: >95% 全落单骨
    print("# dominated vertices  :", dominant.sum())

    dom_mask = (skin1 > 0.95).sum(1).astype(bool)
    dom_jid  = skin1[dom_mask].argmax(1)    # 这些顶点实际支配骨骼编号

    hist = collections.Counter(dom_jid.tolist())
    print("骨骼编号  : 受影响顶点数")
    for jid, cnt in hist.most_common():
        print(f"{jid:02d} : {cnt}")

    dbg_pc = o3d.geometry.PointCloud()
    dbg_pc.points = o3d.utility.Vector3dVector(pts1_raw[dom_mask])  # 用原始坐标
    dbg_pc.paint_uniform_color([1,0,0])  # 红色
    o3d.visualization.draw_geometries([mesh1, dbg_pc])


    # dominated mask 与 pos1_norm / skin1 已有
    M = 6                     # 给最近 6 根骨骼平均分
    for v in np.where(dom_mask)[0]:
        # 1️⃣ 按欧氏距离选最近骨骼
        near = np.linalg.norm(pos1 - pts1_norm[v], axis=1).argsort()[:M]
        skin1[v] = 0
        skin1[v, near] = 1.0 / M   # 均匀分配

    # 最后再归一
    skin1 /= skin1.sum(1, keepdims=True)


    folder = os.path.abspath(args.output_dir)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(args.frames):
        ti = time.time()

        alpha = (i + 1) / (args.frames + 1)
        
        # interpolate each joint rotation
        R_i = np.zeros_like(R1)
        for k in range(K):
            R_i[k] = slerp_joints[k]([alpha]).as_matrix()[0]

        # interpolate joint positions linearly
        pos_i = pos1 * (1 - alpha) + pos2 * alpha

        # Blocking display of interpolated skeleton using Open3D
        # o3d.visualization.draw_geometries(
        #     draw_skeleton(pos_i, parents, color=[0,0.5,i/args.frames]),
        #     window_name=f"Blend Skeleton Frame {i}",
        #     width=600,
        #     height=600,
        #     point_show_normal=False
        # )
        # continue

        # non-blocking display of interpolated skeleton
        for geo in draw_skeleton(pos_i, parents, color=[0,0.5,i/args.frames]): vis_skel.add_geometry(geo)
        vis_skel.poll_events(); vis_skel.update_renderer(); time.sleep(0.5)
        vis_skel.clear_geometries()
        
        # build transforms (reuse K from SLERP setup)
        T4 = np.zeros((K, 4, 4), dtype=np.float32)
        for k in range(K):
            T4[k, :3, :3] = R_i[k]
            T4[k, :3, 3] = pos_i[k]
            T4[k, 3, 3] = 1.0

        # Build per-joint composite skinning transforms: deform * inv_bind
        T_skin = np.zeros_like(T4)
        for k in range(K):
            T_skin[k] = T4[k] @ inv_bind[k]

        # Linear Blend Skinning with proper rotation handling
        pts_h = np.concatenate([pts1_norm, np.ones((pts1_norm.shape[0], 1))], axis=1)
        new_pts_norm_arr = np.zeros_like(pts1_norm)
        
        # Apply skinning weights to points
        # for n in range(pts_h.shape[0]):
        #     # Accumulate weighted transforms
        #     blended_transform = np.zeros((4, 4), dtype=np.float32)
        #     total_weight = 0
            
        #     for k in range(K):
        #         weight = skin1[n, k]
        #         if weight > 1e-6:  # Only process significant weights
        #             blended_transform += weight * T_skin[k]
        #             total_weight += weight
            
        #     if total_weight > 1e-6:
        #         # Apply the blended transformation
        #         transformed_pt = blended_transform @ pts_h[n]
        #         new_pts_norm_arr[n] = transformed_pt[:3]
        #     else:
        #         # No bone influence, keep original position
        #         new_pts_norm_arr[n] = pts1_norm[n]

        # for n, p in enumerate(pts_h):
        #     # Accumulate weighted transforms
        #     acc = np.zeros(3)
        #     for k, w in enumerate(skin1[n]):
        #         if w > 1e-6:
        #             acc += w * (T_skin[k] @ p)[:3]
        #     new_pts_norm_arr[n] = acc

        # fully vectorised alternative (optional, faster):
        T_skin_stack = np.stack(T_skin, axis=0) # (K,4,4)
        pts_blended = np.einsum('kij,nj->nki', T_skin_stack, pts_h)   # (N,K,4)
        pts_blended = pts_blended[..., :3]                            # (N,K,3)
        new_pts_norm_arr = (skin1[..., None] * pts_blended).sum(axis=1)   # (N,3)

        # unnormalize all at once to original space
        new_pts = ((new_pts_norm_arr + 1) / 2) * (blen1 + 1e-5) + bmin1

        if alpha == 0:
            err = np.linalg.norm(new_pts - pts1_raw, axis=1).mean()
            print('Mean error @alpha=0', err)

         # write mesh frame
        mesh_o = deepcopy(mesh1)
        mesh_o.vertices = o3d.utility.Vector3dVector(new_pts)
        mesh_o.compute_vertex_normals()
        # clear triangle normals to avoid OBJ warning
        mesh_o.triangle_normals = o3d.utility.Vector3dVector([])

        out_path = os.path.join(folder, f"frame_{i:03d}.obj")
        # write mesh frame with vertex normals only (OBJ does not support triangle normals)
        o3d.io.write_triangle_mesh(
            out_path,
            mesh_o,
        )
        ti_end = time.time()
        print(f"Frame {i} processed in {ti_end - ti:.4f}s, saved to {out_path}")
        # display interpolated mesh
        # o3d.visualization.draw_geometries([mesh_o], window_name=f"Blend Frame {i}")

    vis_skel.destroy_window()


if __name__ == '__main__':
    main()
