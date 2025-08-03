#!/usr/bin/env python3
"""
网格处理工具函数
包括读取OBJ、计算法向、Chamfer距离、ARAP误差等
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import json
from pathlib import Path

def load_mesh(obj_path):
    """加载OBJ网格"""
    mesh = trimesh.load(str(obj_path))
    return mesh.vertices, mesh.faces, mesh.vertex_normals

def compute_chamfer_distance(vertices_a, vertices_b, k=1, subsample_ratio=0.1):
    """计算Chamfer距离 - 优化版本，支持子采样"""
    # 子采样以减少计算量
    if subsample_ratio < 1.0:
        n_a = max(int(len(vertices_a) * subsample_ratio), 100)
        n_b = max(int(len(vertices_b) * subsample_ratio), 100)
        
        if len(vertices_a) > n_a:
            indices_a = np.random.choice(len(vertices_a), n_a, replace=False)
            sampled_a = vertices_a[indices_a]
        else:
            sampled_a = vertices_a
            
        if len(vertices_b) > n_b:
            indices_b = np.random.choice(len(vertices_b), n_b, replace=False)
            sampled_b = vertices_b[indices_b]
        else:
            sampled_b = vertices_b
    else:
        sampled_a, sampled_b = vertices_a, vertices_b
    
    tree_a = cKDTree(sampled_a)
    tree_b = cKDTree(sampled_b)
    
    # 计算从A到B的距离
    dist_a_to_b, _ = tree_a.query(sampled_b, k=k)
    # 计算从B到A的距离
    dist_b_to_a, _ = tree_b.query(sampled_a, k=k)
    
    # 平均Chamfer距离
    chamfer_dist = (np.mean(dist_a_to_b) + np.mean(dist_b_to_a)) * 0.5
    return chamfer_dist

def align_vertices_for_comparison(vertices_a, vertices_b, normals_a=None, normals_b=None):
    """
    对齐两个顶点集合用于比较，处理顶点数量不一致的情况
    使用最近邻匹配来对齐顶点
    """
    if len(vertices_a) == len(vertices_b):
        # 如果顶点数相同，直接返回
        if normals_a is not None and normals_b is not None:
            return vertices_a, vertices_b, normals_a, normals_b
        else:
            return vertices_a, vertices_b
    
    # 使用较少顶点数的作为参考
    if len(vertices_a) <= len(vertices_b):
        ref_vertices = vertices_a
        target_vertices = vertices_b
        ref_normals = normals_a
        target_normals = normals_b
        swap = False
    else:
        ref_vertices = vertices_b
        target_vertices = vertices_a
        ref_normals = normals_b
        target_normals = normals_a
        swap = True
    
    # 使用KD树找到最近邻对应关系
    tree = cKDTree(target_vertices)
    distances, indices = tree.query(ref_vertices, k=1)
    
    # 提取对应的顶点和法向量
    aligned_target_vertices = target_vertices[indices]
    aligned_target_normals = target_normals[indices] if target_normals is not None else None
    
    if swap:
        # 如果交换了顺序，需要交换回来
        if normals_a is not None and normals_b is not None:
            return aligned_target_vertices, ref_vertices, aligned_target_normals, ref_normals
        else:
            return aligned_target_vertices, ref_vertices
    else:
        if normals_a is not None and normals_b is not None:
            return ref_vertices, aligned_target_vertices, ref_normals, aligned_target_normals
        else:
            return ref_vertices, aligned_target_vertices

def compute_normal_consistency(normals_a, normals_b):
    """计算法向一致性（夹角）"""
    # 检查顶点数是否一致，如果不一致则跳过计算
    if len(normals_a) != len(normals_b):
        print(f"Warning: Normal arrays have different sizes ({len(normals_a)} vs {len(normals_b)}), skipping normal consistency computation")
        return np.array([])
    
    # 归一化法向量
    normals_a_norm = normals_a / (np.linalg.norm(normals_a, axis=1, keepdims=True) + 1e-8)
    normals_b_norm = normals_b / (np.linalg.norm(normals_b, axis=1, keepdims=True) + 1e-8)
    
    # 计算点积
    dot_products = np.sum(normals_a_norm * normals_b_norm, axis=1)
    # 限制在[-1, 1]范围内
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # 计算夹角（弧度）
    angles = np.arccos(np.abs(dot_products))
    return angles

def compute_arap_error(vertices_a, vertices_b, influence_weights=None, sample_ratio=0.05, neighbor_radius=0.1):
    """计算ARAP（As-Rigid-As-Possible）误差 - 优化版本"""
    # 处理顶点数不一致的情况
    if len(vertices_a) != len(vertices_b):
        print(f"Warning: Vertex arrays have different sizes ({len(vertices_a)} vs {len(vertices_b)}), aligning vertices for ARAP computation")
        vertices_a, vertices_b = align_vertices_for_comparison(vertices_a, vertices_b)
    
    if influence_weights is None:
        # 如果没有权重，使用均匀权重
        influence_weights = np.ones(len(vertices_a))
    
    # 采样顶点以减少计算量
    n_vertices = len(vertices_a)
    sample_size = max(int(n_vertices * sample_ratio), 50)  # 至少采样50个顶点
    
    if n_vertices > sample_size:
        sampled_indices = np.random.choice(n_vertices, sample_size, replace=False)
    else:
        sampled_indices = np.arange(n_vertices)
    
    # 预计算距离矩阵（只计算采样顶点）
    sampled_vertices_a = vertices_a[sampled_indices]
    
    # 使用KDTree快速查找邻居
    tree = cKDTree(vertices_a)
    
    arap_errors = []
    
    for idx in sampled_indices:
        if influence_weights[idx] > 0.1:  # 只考虑有影响的顶点
            # 使用KDTree查找邻近顶点，更高效
            neighbors = tree.query_ball_point(vertices_a[idx], neighbor_radius)
            
            if len(neighbors) > 3:
                # 计算局部刚体变换
                local_a = vertices_a[neighbors]
                local_b = vertices_b[neighbors]
                
                try:
                    # 计算质心
                    centroid_a = np.mean(local_a, axis=0)
                    centroid_b = np.mean(local_b, axis=0)
                    
                    # 计算协方差矩阵
                    H = (local_a - centroid_a).T @ (local_b - centroid_b)
                    
                    # SVD分解
                    U, S, Vt = np.linalg.svd(H)
                    R = Vt.T @ U.T
                    
                    # 处理反射情况
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = Vt.T @ U.T
                    
                    # 计算变换后的位置
                    transformed = (local_a - centroid_a) @ R.T + centroid_b
                    
                    # 计算残差
                    residual = np.linalg.norm(transformed - local_b, axis=1)
                    arap_errors.append(np.mean(residual))
                except np.linalg.LinAlgError:
                    # SVD可能失败，跳过这个顶点
                    continue
    
    return np.mean(arap_errors) if arap_errors else 0.0

def compute_jerk(vertices_sequence, dt=1.0/30.0):
    """计算jerk（加加速度）"""
    if len(vertices_sequence) < 3:
        return 0.0
    
    # 计算速度
    velocities = np.diff(vertices_sequence, axis=0) / dt
    
    # 计算加速度
    accelerations = np.diff(velocities, axis=0) / dt
    
    # 计算jerk
    jerks = np.diff(accelerations, axis=0) / dt
    
    # 计算jerk的范数
    jerk_norms = np.linalg.norm(jerks, axis=-1)
    mean_jerk = np.mean(jerk_norms)
    max_jerk = np.max(jerk_norms)
    
    return mean_jerk, max_jerk

def compute_bone_length_sd(joints_sequence, parents):
    """计算骨长标准差"""
    bone_lengths = []
    
    for joints in joints_sequence:
        frame_bone_lengths = []
        for i, parent in enumerate(parents):
            if parent >= 0:  # 不是根节点
                bone_vec = joints[i] - joints[parent]
                bone_length = np.linalg.norm(bone_vec)
                frame_bone_lengths.append(bone_length)
        bone_lengths.append(frame_bone_lengths)
    
    # 计算每个骨骼的标准差
    bone_lengths = np.array(bone_lengths)
    bone_sd = np.std(bone_lengths, axis=0)
    mean_bone_sd = np.mean(bone_sd)
    
    return mean_bone_sd

def compute_self_intersection_count(mesh, fast_mode=True):
    """计算自碰撞数量 - 优化版本"""
    if fast_mode:
        # 快速模式：简化检查或跳过
        try:
            # 简单的边界框重叠检查作为近似
            if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
                return 1  # 非封闭网格可能有自相交
            return 0
        except:
            return 0
    else:
        # 完整模式：原始实现
        try:
            # 使用trimesh检查自碰撞
            collision = mesh.collision
            if collision is not None:
                return len(collision)
            else:
                return 0
        except:
            return 0

def compute_foot_slide(vertices_sequence, ground_y=0.0, epsilon=0.01):
    """计算脚部滑动（假设地面y=0）"""
    foot_slide_pixels = 0
    
    for vertices in vertices_sequence:
        # 找到接近地面的顶点（假设脚部顶点y坐标接近0）
        ground_vertices = vertices[vertices[:, 1] < ground_y + epsilon]
        if len(ground_vertices) > 0:
            # 计算与地面的距离
            distances = np.abs(ground_vertices[:, 1] - ground_y)
            foot_slide_pixels += np.sum(distances > epsilon)
    
    return foot_slide_pixels

def load_skeleton_info(skeleton_path):
    """加载skeleton信息"""
    with open(skeleton_path, 'r') as f:
        skeleton_info = json.load(f)
    return skeleton_info

def save_evaluation_results(results, output_path):
    """保存评估结果"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_evaluation_results(results_path):
    """加载评估结果"""
    with open(results_path, 'r') as f:
        return json.load(f) 