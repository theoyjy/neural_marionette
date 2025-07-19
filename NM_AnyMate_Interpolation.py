import argparse
import copy
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
from scipy.optimize import linear_sum_assignment
import scipy

from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import episodic_normalization, voxelize
import subprocess, json, tempfile, shutil, sys, pathlib
import trimesh

# Path to Anymate project and its virtual environment
ANYMATE_PROJECT_PATH = pathlib.Path(r"D:/Code/HuggingFace/Anymate")
ANYMATE_PYTHON = ANYMATE_PROJECT_PATH / "env" / "Scripts" / "python.exe"
ANYMATE_SCRIPT = ANYMATE_PROJECT_PATH / "AnymatePredictor.py"

def predict_with_anymate(mesh_path: str, frame_idx: int = 0) -> dict:
    """Run Anymate pipeline and return the descriptor JSON it writes."""
    out_json = mesh_path + ".json"
    if not os.path.exists(out_json):
        cmd = [
            str(ANYMATE_PYTHON), str(ANYMATE_SCRIPT),
            "--mesh_path", mesh_path,
            "--frame_idx", str(frame_idx)
        ]
        subprocess.check_call(cmd)
    with open(out_json) as fp:
        return json.load(fp) 

def to_world(norm_xyz, bmin, blen):
    return ((norm_xyz + 1) * 0.5) * (blen + 1e-5) + bmin

def joint_map_by_greedy(src_xyz, dst_xyz, thresh=0.08):
    """Return list `map_src_to_dst[i] == j` or -1 if no good match."""
    M,N = len(src_xyz), len(dst_xyz)
    unused = set(range(N))
    mapping = [-1]*M
    for i in range(M):
        d = np.linalg.norm(dst_xyz - src_xyz[i], axis=1)
        j = np.argmin(d)
        if d[j] < thresh and j in unused:
            mapping[i] = j
            unused.remove(j)
    return mapping

def to_top4(weights32: np.ndarray):
    """
    Convert (N,32) weight matrix to:
      weights_4  (N,4) float32   – weights for the 4 strongest bones
      joints_4   (N,4) uint32    – column indices of those bones
    Rows already alpha-normalised so we re-normalise the picked 4.
    """
    idx_top4 = np.argsort(weights32, axis=1)[:, -4:]          # (N,4)
    w_top4   = np.take_along_axis(weights32, idx_top4, axis=1)
    w_top4  /= w_top4.sum(axis=1, keepdims=True) + 1e-8
    return w_top4.astype(np.float32), idx_top4.astype(np.uint32)

def topo_sort(parents):
    """Return an index array that orders joints so parent < child."""
    n = len(parents)
    visited, order = [False]*n, []
    def dfs(v):
        if visited[v]: return
        visited[v] = True
        if parents[v] >= 0: dfs(parents[v])
        order.append(v)
    for v in range(n):
        dfs(v)
    return order 

def reorder_hierarchy(joints, parents, order):
    """
    Permute joints & parents by `order`, and remap parent indices
    into the new coordinate system so that parent < child holds.
    """
    joints_sorted   = joints [order]
    parents_sorted  = parents[order]

    # build inverse map old_index → new_index
    inverse = np.zeros_like(order)
    inverse[order] = np.arange(len(order))

    # remap parents; keep -1 for root
    parents_remap = np.where(parents_sorted >= 0,
                             inverse[parents_sorted],
                             -1)
    return joints_sorted, parents_remap

def canonicalise_hierarchy(joints, parents):
    # 0️⃣ convert self-parent to root
    parents = parents.copy()
    parents[parents == np.arange(len(parents))] = -1

    # 1️⃣ ensure single root (optional but typical)
    if (parents < 0).sum() > 1:
        main_root = np.argmin(joints[:,1])  # pelvis
        parents[(parents < 0) & (np.arange(len(parents)) != main_root)] = main_root

    # 2️⃣ topo-sort & remap
    order = topo_sort(parents)
    return reorder_hierarchy(joints, parents, order), order

def load_skinning(path):
    S = np.load(path, allow_pickle=True)
   
    if isinstance(S, np.lib.npyio.NpzFile):
        Sw = S["weights"].astype(np.float32)   # (N,4) 或 (N,32)
        Si = S["joints" ].astype(np.uint32)    # (N,4)
    else:
        # single file .npy，32 bits weights matrix
        weights32 = S.astype(np.float32)        # (N,32)
        idx_top4  = np.argsort(weights32, 1)[:, -4:]
        Si       = idx_top4.astype(np.uint32)   # 4 个骨骼 ID
        Sw       = np.take_along_axis(weights32, idx_top4, 1)
        Sw      /= Sw.sum(1, keepdims=True) + 1e-8


    # Check if weights are already in top-4 format
    if Sw.shape[1] != 4 :
        Sw, Si = to_top4(Sw)
        print("Converted to top-4 format")

    return Sw, Si

def align_vectors(a, b):
    """
    Returns a rotation matrix that aligns vector a to vector b.
    """
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        # 180 degree rotation: pick any orthogonal axis
        axis = np.eye(3)[np.argmin(np.abs(a))]
        R = Rotation.from_rotvec(np.pi * axis).as_matrix()
        return R
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],    0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return R

def rt_to_mat(R, t):
    """
    Construct a 4x4 transformation matrix from a rotation matrix R (3x3) and translation vector t (3,).
    """
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def global_to_local(joints, parents):
    local_T = []
    for j,p in enumerate(parents):
        if p < 0:
            R = np.eye(3); t = joints[j]
        else:
            t = joints[j] - joints[p]
            R = align_vectors((0,1,0), t)      # simple aim-at (or use NM’s R)
        local_T.append(rt_to_mat(R, t))
    return np.stack(local_T)    # (K,4,4)

def quat_to_mat(quat):
    """
    Converts a quaternion (K,4) or (4,) to a 4x4 transformation matrix (K,4,4) or (4,4).
    """
    rot = Rotation.from_quat(quat)
    mat = rot.as_matrix()
    if mat.ndim == 2:
        M = np.eye(4)
        M[:3,:3] = mat
        return M
    else:
        M = np.tile(np.eye(4), (mat.shape[0], 1, 1))
        M[:,:3,:3] = mat
        return M

def safe_rot(mat3):
    """将 3×3 矩阵转成 Rotation，若矩阵非法则返回单位旋转。"""
    if not np.isfinite(mat3).all() or np.linalg.norm(mat3) < 1e-6:
        return Rotation.identity()
    # 简单正交化（SVD）保证 det≈1
    U, _, Vt = np.linalg.svd(mat3)
    mat3_ok = U @ Vt
    if np.linalg.det(mat3_ok) < 0:         # 若翻转，调换符号
        mat3_ok[:, 0] *= -1
    return Rotation.from_matrix(mat3_ok)

def interpolate_chain(T0, TT, alpha):
    K = T0.shape[0]
    out = np.empty_like(T0, dtype=np.float32)

    for k in range(K):
        R0 = safe_rot(T0[k, :3, :3])
        RT = safe_rot(TT[k, :3, :3])

        if np.allclose(R0.as_quat(), RT.as_quat()):      # 完全相同就省掉 Slerp
            Rk = R0.as_matrix()
        else:
            slerp = Slerp([0.0, 1.0], Rotation.concatenate([R0, RT]))
            Rk = slerp([alpha]).as_matrix()[0]

        tk = (1.0 - alpha) * T0[k, :3, 3] + alpha * TT[k, :3, 3]

        out[k] = np.eye(4, dtype=np.float32)
        out[k, :3, :3] = Rk
        out[k, :3,  3] = tk
    return out


def apply_lbs(rest_verts, weights4, joints4, bone_T):
    """
    rest_verts : (N,3)
    weights4   : (N,4)
    joints4    : (N,4)  int / uint   – index of the driving bone for each weight
    bone_T     : (K,4,4)             – world-space bone transforms
    → skinned  : (N,3)
    """

    assert rest_verts.shape[0] == weights4.shape[0] == joints4.shape[0], \
       "Vertex count mismatch!"

    assert weights4.shape[1] == joints4.shape[1] == 4, \
        "Expecting exactly 4 influences per vertex."

    N = rest_verts.shape[0]

    # — homogeneous coordinates
    v_h = np.hstack([rest_verts,
                     np.ones((N,1), dtype=rest_verts.dtype)])      # (N,4)

    # — gather the 4 transforms per vertex
    # joints4.reshape(-1)  →  (N*4,)   →  (N*4,4,4)
    T_flat = bone_T[joints4.reshape(-1)]
    T_v4   = T_flat.reshape(N, 4, 4, 4)                           # (N,4,4,4)

    # — transform each vertex copy (einsum avoids awkward matmul broadcast)
    # nlij * nj  ->  nli
    v_trans = np.einsum('nlij,nj->nli', T_v4, v_h)                # (N,4,4)

    # — linear-blend skinning
    skinned_h = (weights4[..., None] * v_trans).sum(axis=1)       # (N,4)
    return skinned_h[:, :3]                                       # (N,3)           

def global_mats(R_local, pos_local, parents):
    """Return per-joint global 4×4 matrices from local (R,t)."""
    K = len(parents)
    G = np.empty((K,4,4), dtype=np.float32)
    for j in range(K):
        G[j] = rt_to_mat(R_local[j], pos_local[j])
        p = parents[j]
        if p >= 0:
            G[j] = G[p] @ G[j]
    return G

def globals_to_locals(G, parents, eps=1e-8):
        """
        G        : (K,4,4) global transforms
        parents  : (K,)    parent indices  (root = –1)
        returns  : (K,4,4) local transforms  s.t.  G[j] = G[parent]·L[j]
        """
        K = len(parents)
        L = np.empty_like(G)
        for j in range(K):
            p = parents[j]
            if p < 0:
                L[j] = G[j]
            else:
                if abs(np.linalg.det(G[p])) < eps:      # singular → treat as root
                    L[j] = np.eye(4, dtype=np.float32)
                    L[j][:3,3] = G[j][:3,3]      # 仍保留平移
                else:
                    L[j] = np.linalg.inv(G[p]) @ G[j]

        return L

def build_retargeted_transforms(R0_nm, P0_nm,
                                RT_nm, PT_nm,
                                map_nm2any,
                                parents_nm, parents_any, rest_globals_any):
    """
    • R0_nm, RT_nm:  (K_nm,3,3)
    • P0_nm, PT_nm:  (K_nm,3)  world-space joint centres
    • map_nm2any: list len K_nm  (value = index in Anymate or -1)
    • Returns T0_any, TT_any  (K_any,4,4) – *local* to Anymate parents.
    """

    K_nm, K_any = len(parents_nm), len(parents_any)
    rest_L = globals_to_locals(rest_globals_any, parents_any)

    # 1️⃣ global NM matrices
    G0_nm = rest_globals_any.copy()   # (24,4,4)
    GT_nm = rest_globals_any.copy()

    # 2️⃣ start Anymate globals as identity
    G0_any = np.tile(np.eye(4, dtype=np.float32), (K_any,1,1))
    GT_any = G0_any.copy()

    # 3️⃣ copy where mapping exists
    for j_src, j_dst in enumerate(map_nm2any):      # 24 iterations
        if 0 <= j_dst < K_any:
            G0_any[j_dst] = G0_nm[j_src]
            GT_any[j_dst] = GT_nm[j_src]


    # 4️⃣  propagate: if a child is still identity but its parent was filled,
    #     copy the parent’s global so the chain moves coherently
    for j in range(K_any):
        if j in map_nm2any:        # 已映射，跳过
            continue
        p = parents_any[j]
        if p >= 0:
            G0_any[j] = G0_any[p] @ rest_L[j] # 父的当前全局 × bind-local
            GT_any[j] = GT_any[p] @ rest_L[j]


    # 5️⃣ convert to local wrt Anymate parents
    T0 = globals_to_locals(G0_any, parents_any)       # (K_any,4,4)
    TT = globals_to_locals(GT_any, parents_any)
    return T0, TT

def build_bind_globals(joints_world, parents):
    """
    joints_world : (K,3)  第一帧 Anymate 关节的世界坐标
    parents      : (K,)   parent 索引（root = –1）
    返回         : (K,4,4) bind pose 全局矩阵
    旋转设为单位矩阵，平移即关节坐标。
    """
    K = len(parents)
    G = np.tile(np.eye(4, dtype=np.float32), (K, 1, 1))
    G[:, :3, 3] = joints_world               # 只填 t

    # 若想让子骨相对父骨的平移保持一致，可递归累加：
    for j in range(K):
        p = parents[j]
        if p >= 0:
            G[j, :3, 3] = joints_world[j]    # 已经是绝对坐标，可直接用
    return G


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

def draw_mapping(j_any, j_nm, map_any2nm,
                 r_joint=0.01, r_line=0.005, line_res=20):

    geos = []

    # ──准备 3 种模板球──
    ball_any = o3d.geometry.TriangleMesh.create_sphere(r_joint, 4)
    ball_any.paint_uniform_color([0,1,0])        # 绿

    ball_nm  = o3d.geometry.TriangleMesh.create_sphere(r_joint, 4)
    ball_nm.paint_uniform_color([1,0,0])         # 红

    ball_unm = o3d.geometry.TriangleMesh.create_sphere(r_joint, 4)
    ball_unm.paint_uniform_color([0,0,1])        # 蓝

    # ──NM 红球──
    for p in j_nm:
        g = copy.deepcopy(ball_nm)
        g.translate(p, relative=False)
        geos.append(g)

    # ──Anymate──
    for a_id, nm_id in enumerate(map_any2nm):
        pa = j_any[a_id]

        if nm_id >= 0:          # 已匹配：绿球 + 青色圆柱
            pn = j_nm[nm_id]
            g = copy.deepcopy(ball_any); g.translate(pa, relative=False)
            geos.append(g)
            geos.append(make_cylinder(pa, pn, r_line, line_res))
        else:                   # 未匹配：蓝球
            g = copy.deepcopy(ball_unm); g.translate(pa, relative=False)
            geos.append(g)

    # ──显示──
    vis = o3d.visualization.Visualizer()
    vis.create_window('Mapping Debug', 900, 700)
    for g in geos: vis.add_geometry(g)
    vis.run(); vis.destroy_window()


def make_cylinder(p0, p1, radius, res):
    """返回一根 world-space 圆柱；颜色 = 青色"""
    v = p1 - p0; length = np.linalg.norm(v)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, length, res, 1)
    cyl.paint_uniform_color([0,1,1])             # 青

    # 先把圆柱局部 z 轴对齐到 v（先平移再旋转）
    cyl.translate([0,0,length/2])
    if length > 1e-6:
        z = np.array([0,0,1.0])
        axis = np.cross(z, v)
        if np.linalg.norm(axis) > 1e-6:
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.dot(z, v) / length)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis*angle)
            cyl.rotate(R, center=[0,0,0])
    cyl.translate(p0)
    return cyl

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


def hungarian_match(src, dst, w_len=0.4):
    """src, dst: (K,3) xyz; returns len(src) → id(dst) or -1"""
    Ms, Md = len(src), len(dst)
    D = np.linalg.norm(src[:,None,:] - dst[None,:,:], axis=-1)  # (Ms,Md)
    # add bone-length similarity (optional)
    # cost  = α·Euclid + β·|boneLen_src – boneLen_dst|
    row_ind, col_ind = linear_sum_assignment(D)
    mapping = [-1]*Ms
    for i,j in zip(row_ind, col_ind):
        if D[i,j] < 0.15:            # relaxed threshold
            mapping[i] = j
    return mapping


def build_cost(J_any, J_nm, parents_nm, alpha_len=0.2, alpha_bone=1.2):
    """Return an M×K cost matrix."""
    M, K = len(J_any), len(J_nm)

    # Euclidean distance
    D_xyz = np.linalg.norm(J_any[:,None,:] - J_nm[None,:,:], axis=-1)  # (M,K)

    # Bone-length similarity: |‖joint-parent‖_any - ‖joint-parent‖_nm|
    len_any = np.zeros((M,))
    len_nm  = np.zeros((K,))
    for k,p in enumerate(parents_nm):
        if p>=0:
            len_nm[k]  = np.linalg.norm(J_nm[k]-J_nm[p])
    # For Anymate we approximate parent as the nearest neighbour on the MST
    tree_any = scipy.spatial.KDTree(J_any)
    _, nn = tree_any.query(J_any, k=2)
    len_any = np.linalg.norm(J_any - J_any[nn[:,1]], axis=1)      # crude but works

    D_len = np.abs(len_any[:,None] - len_nm[None,:])              # (M,K)

    return alpha_len*D_xyz + alpha_bone* (D_len ** 2)  # (M,K) ←平方拉大差距


def fuse_anymate_into_nm(R_nm, P_nm,   # original NM (K×3×3, K×3)
                         J_any, mapping_any2nm, 
                         parents_nm, freeze_rot=False):
    """Return new R_nm, P_nm where the mapped joints get Anymate translation."""
    R = R_nm.copy()
    P = P_nm.copy()
    for i_any, j_nm in enumerate(mapping_any2nm):
        if j_nm >= 0:
            P[j_nm] = J_any[i_any]
            if not freeze_rot:
                # Face the child direction computed from Anymate
                parent = parents_nm[j_nm]
                if parent >= 0:
                    v = P[j_nm] - P[parent]
                    R[j_nm] = align_vecs_to_zy(v, R[parent])  # user-defined
    return R, P

def align_vecs_to_zy(v_child_world, R_parent):
    """
    Build a 3×3 rotation that:
    • Points the local Y-axis along vector v_child_world (in world space)
    • Keeps the parent’s local Z-axis projected as close as possible
    Returns a numpy (3,3) matrix.
    """
    v = v_child_world
    if np.linalg.norm(v) < 1e-8:      # zero-length → just copy parent
        return R_parent.copy()

    # 1️⃣ desired world Y direction for the child
    y_axis = v / np.linalg.norm(v)

    # 2️⃣ choose a Z axis: take parent’s Z, make it orthogonal to the new Y
    z_parent = R_parent[:, 2]         # third column of parent matrix
    z_axis = z_parent - y_axis * np.dot(z_parent, y_axis)
    if np.linalg.norm(z_axis) < 1e-8:      # parent Z was parallel to new Y
        # fall back to global X
        z_axis = np.array([1.0, 0.0, 0.0])
        z_axis = z_axis - y_axis * np.dot(z_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # 3️⃣ X completes the right-hand frame
    x_axis = np.cross(y_axis, z_axis)

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are axes
    return rot.astype(np.float32)

def locals_to_globals(L, parents):
    """
    L        : (K,4,4)  每关节局部矩阵
    parents  : (K,)     父索引 (root = -1)
    返回     : (K,4,4)  全局矩阵
    """
    K = len(parents)
    G = np.empty_like(L)
    for j in range(K):
        p = parents[j]
        G[j] = L[j] if p < 0 else G[p] @ L[j]
    return G


def main():
    parser = argparse.ArgumentParser(description="Blend skeletons between two meshes using NeuralMarionette.")
    parser.add_argument('--mesh0', type=str, help='Path to first mesh file.')
    parser.add_argument('--meshT', type=str, help='Path to second mesh file.')
    parser.add_argument('--frames', type=int, help='Number of blending frames to generate.')
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
    vox0, pts0_raw, mesh0, bmin0, blen0 = load_voxel_from_mesh(args.mesh0, opt, args.is_bind)
    # normalize raw points for skinning
    # pts0_norm = ((pts0_raw - bmin0) * 1.0 / (blen0 + 1e-5)) * 2 - 1

    with torch.no_grad():
        log0 = network.kypt_detector(vox0)
        pos0_nm = log0['keypoints'][0,0][:,:3].cpu().numpy()
        pos0_world = to_world(pos0_nm, bmin0, blen0)

        kps0 = log0['keypoints'][0, 0]
        affinity0 = log0['affinity']
        dlog0 = network.dyna_module.encode(log0['keypoints'], affinity0)
        R0 = dlog0['R'][0, 0].cpu().numpy() # (K, 3, 3) local-frame rotations
        pos0 = kps0[:, :3].cpu().numpy() # (K, 3) joint positions
        A = network.dyna_module.A
        priority = network.dyna_module.priority
        parents_nm = network.dyna_module.parents
        K = parents_nm.shape[0]  # number of joints
       
    t1 = time.time()
    print(f"Mesh0 processing time: {t1 - t0:.4f}s")
    # non-blocking display of mesh0 skeleton
    for geo in draw_skeleton(pos0, parents_nm, color=[1,0,0]): vis_skel.add_geometry(geo)
    vis_skel.poll_events(); vis_skel.update_renderer(); time.sleep(1)
    vis_skel.clear_geometries()

    # process second mesh
    t2 = time.time()
    voxT, ptsT_raw, meshT, bminT, blenT = load_voxel_from_mesh(args.meshT, opt, args.is_bind)
    # ptsT_norm = ((ptsT_raw - bminT) * 1.0 / (blenT + 1e-5)) * 2 - 1
    with torch.no_grad():
        logT = network.kypt_detector(voxT)
        posT_nm = logT['keypoints'][0,0][:,:3].cpu().numpy()
        posT_world = to_world(posT_nm, bminT, blenT)

        kpsT = logT['keypoints'][0, 0]
        affinityT = logT['affinity']
        dlogT = network.dyna_module.encode(logT['keypoints'], affinityT)
        RT = dlogT['R'][0, 0].cpu().numpy()
        posT = kpsT[:, :3].cpu().numpy()

    t3 = time.time()
    print(f"MeshT processing time: {t3 - t2:.4f}s")
    # non-blocking display of T skeleton
    for geo in draw_skeleton(posT, parents_nm, color=[0,1,0]): vis_skel.add_geometry(geo)
    vis_skel.poll_events(); vis_skel.update_renderer(); time.sleep(1)
    vis_skel.clear_geometries()

    # -- load Anymate data for mesh-0
    dat_any0 = predict_with_anymate(args.mesh0, frame_idx=0)
    j_any_w0  = np.load(dat_any0['joints'])
    parents_any0 = np.load(dat_any0['connectivity'])
    (j_any_w0, parents_any0), order0 = canonicalise_hierarchy(j_any_w0,
                                                            parents_any0)
    Sw_exp0, Si_exp0 = load_skinning(dat_any0['skinning'])

    mesh0 = trimesh.load(dat_any0["normalized_mesh"])
    rest_verts = np.asarray(mesh0.vertices, dtype=np.float32)
    rest_verts = rest_verts / dat_any0["scale"] + dat_any0["center"]

    rest_globals_any = build_bind_globals(j_any_w0, parents_any0)

    # -- load Anymate data for mesh-T
    dat_anyT = predict_with_anymate(args.meshT, frame_idx=0)
    j_any_wT  = np.load(dat_anyT['joints'])
    parents_anyT = np.load(dat_anyT['connectivity'])
    (j_any_wT, parents_anyT), orderT = canonicalise_hierarchy(j_any_wT,
                                                            parents_anyT)
    Sw_expT, Si_expT = load_skinning(dat_anyT['skinning'])

    print("NM joint bbox :", pos0_world.min(0), pos0_world.max(0))
    print("AM joint bbox :", j_any_w0.min(0), j_any_w0.max(0))
    # pause to see the joint positions
    vis_skel.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos0_world)))
    vis_skel.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(j_any_w0)))
    vis_skel.poll_events()

    # -- align Anymate joints to NM
    C0 = build_cost(j_any_w0, pos0_world, parents_nm.cpu().numpy())
    row,col = linear_sum_assignment(C0)
    map_any2nm = [-1]*len(j_any_w0);  # M-length
    for i,j in zip(row,col):
        if C0[i,j] < 0.15: 
            map_any2nm[i] = j

    mapped = [i for i,x in enumerate(map_any2nm) if x >= 0]
    print(f"[map] matched {len(mapped)}/{len(j_any_w0)} Anymate joints")
    print("     worst cost :", C0[mapped, [map_any2nm[i] for i in mapped]].max())
    print("     mean cost  :", C0[mapped, [map_any2nm[i] for i in mapped]].mean())


    print("map_any2nm length :", len(map_any2nm))
    print("unique mapped nm :", np.unique([x for x in map_any2nm if x>=0]).shape[0])
    print("unmatched count  :", map_any2nm.count(-1))
    draw_mapping(j_any_w0, pos0_world, map_any2nm)


    # fuse Anymate into NM
    R0_fused, P0_fused = fuse_anymate_into_nm(R0, pos0_world, j_any_w0, map_any2nm, parents_nm)
    RT_fused, PT_fused = fuse_anymate_into_nm(RT, posT_world, j_any_wT, map_any2nm, parents_nm)
   
    def bone_len(P, parents):
        return np.array([ np.linalg.norm(P[j]-P[p]) if p>=0 else 0
                        for j,p in enumerate(parents) ])
    
    # 匹配关节过少 / 平均 cost >0.3 m → 阈值太严或坐标没对齐
    print(f"[fuse] Δ骨长(max) =", np.abs(bone_len(P0_fused, parents_nm) -
                                        bone_len(pos0_world,    parents_nm)).max())
    # after Δ骨长(max) 计算
    delta = np.abs(bone_len(P0_fused, parents_nm) -
                bone_len(pos0_world,    parents_nm))
    bad_id = delta.argmax()
    print(f"[debug] worst bone = {bad_id:2d}  Δlen = {delta[bad_id]:.3f} m",
        " ← parent", parents_nm[bad_id])

    # … then compute T0 / TT in the Anymate joint order
    map_nm2any = [-1] * len(parents_nm)
    roots_any = np.where(parents_any0 < 0)[0]          # 通常 0 或 0,1
    root_any = roots_any[np.argmin(np.linalg.norm(j_any_w0[roots_any] -
                                                pos0_world[0], axis=1))]
    map_nm2any[0] = root_any          
    for any_id, nm_id in enumerate(map_any2nm):
        if nm_id >= 0:
            map_nm2any[nm_id] = any_id

    assert len(map_nm2any)==len(parents_nm)
    T0, TT = build_retargeted_transforms(R0_fused,P0_fused,
                                         RT_fused,PT_fused,
                                         map_nm2any,
                                         parents_nm.cpu().numpy(), parents_any0,
                                         rest_globals_any)
    
    singular = [k for k in range(T0.shape[0]) if np.linalg.det(T0[k,:3,:3]) < 1e-6]
    print(f"[local] singular rot count: {len(singular)}") # singular rot >0 → G0_any 仍有空洞

    rest_verts = np.asarray(mesh0.vertices)               # N x 3 in world coords
    assert rest_verts.shape[0] == Sw_exp0.shape[0]

    # 用 TT (或 T0) + skinning 重建原始 frame
    verts_recon = apply_lbs(rest_verts, Sw_exp0, Si_exp0, T0)

    # 原 mesh 顶点
    verts_gt = np.asarray(mesh0.vertices)              # 同 rest_verts 一致

    err = np.linalg.norm(verts_recon - verts_gt, axis=1)
    print("[mesh]  L2 mean =", err.mean(), "cm")
    print("        95-perc =", np.percentile(err, 95), "cm") # 均值 <1 cm，95%<2 cm → 骨架+权重足以重建


    # -- generate intermediate frames
    for k,alpha in enumerate(np.linspace(0,1,args.frames+2)[1:-1],1):
        T = interpolate_chain(T0, TT, alpha)

        # display skeleton
        G = locals_to_globals(T, parents_any0)        # (K,4,4) global matrices
        pos_k = G[:, :3, 3]
        for geo in draw_skeleton(pos_k, parents_any0, color=[0,0,1]):
            vis_skel.add_geometry(geo)
        vis_skel.poll_events()
        vis_skel.update_renderer()
        time.sleep(0.1)
        vis_skel.clear_geometries()

        # apply LBS to rest_verts
        verts_k = apply_lbs(rest_verts, Sw_exp0, Si_exp0, T)
        out = trimesh.Trimesh(verts_k, mesh0.faces, process=False)
        out.export(f"{args.output_dir}/interp_{k:04d}.ply")
    print("✓ Interpolation complete.")

    vis_skel.destroy_window()


if __name__ == '__main__':
    main()
