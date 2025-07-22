#!/usr/bin/env python3
"""
完整的体积视频插值管道 - 最终版本
=================================

功能：
1. 加载文件夹中的.obj文件，每帧通过voxel化后交给NeuralMarionette预测skeleton
2. 自动检测最佳rest pose，基于skeleton进行网格拓扑统一
3. 使用 C++ CLI 版本的 DemBones 进行蒙皮权重预测
4. 生成任意两帧之间的指定数量插值

使用方法：
python complete_vv_pipeline.py <folder_path> --start_frame 0 --end_frame 10 --num_interp 5

前置要求：
- 需要 DemBones C++ 可执行文件在 PATH 中或当前目录
- 可从 https://github.com/electronicarts/dem-bones 下载源码编译
"""

import argparse
import os
import glob
import time
import pickle
import numpy as np
import torch
import open3d as o3d
import threading
import queue
from copy import deepcopy
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree

# 导入必要的模块
from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import voxelize
import subprocess
import tempfile

# 导入GenerateSkel的函数
from GenerateSkel import (
    process_single_mesh,
    sanitize_parents,
    draw_skeleton,
    _as_colmajor,
    _as_rowmajor
)

class CompleteVVPipeline:
    """完整的体积视频插值管道"""
    
    def __init__(self, folder_path, output_dir=None):
        self.folder_path = folder_path
        self.output_dir = output_dir or os.path.join(folder_path, 'vv_complete_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载NeuralMarionette
        self._load_neural_marionette()
        
        # 管道数据
        self.all_mesh_data = []
        self.rest_pose_idx = None
        self.unified_vertices = None
        self.skinning_weights = None
        self.bone_transforms = None
        self.parents = None


    def points_to_voxel(self, pts_norm, grid_size):
        occ = voxelize(pts_norm, (grid_size, grid_size, grid_size), is_binarized=True)  # (1,D,H,W) bool
        occ = torch.from_numpy(occ).float() # (1, 64, 64, 64) float32
        return occ

    def read_mesh_verts(self, path: str):
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_vertices():
            raise ValueError(f"Failed to load mesh from {path}")
        return np.asarray(mesh.vertices, np.float32)

    def load_voxel_from_mesh(self, file, opt, is_bind=False,
                         scale=1.0, x_trans=0.0, z_trans=0.0):
        """
        Read mesh, normalize and voxelize for network input.
        Returns: vox (N=1, C=4, D, H, W), raw points, mesh, bmin, blen
        """
        mesh = o3d.io.read_triangle_mesh(file, True)
        points = np.asarray(mesh.vertices)

        if is_bind:
            points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=-1)

        # ---------- 归一化 ----------
        bmin = points.min(axis=0)
        bmax = points.max(axis=0)
        blen = (bmax - bmin).max()

        pts_norm = ((points - bmin) * scale / (blen + 1e-5)) * 2 - 1 \
                + np.array([x_trans, 0, z_trans])
        
        # ---------- 体素化 ----------
        vox = self.points_to_voxel(pts_norm, self.opt.grid_size, is_binarized=True)

        return vox, points, mesh, bmin, blen

        
    def _load_neural_marionette(self):
        """加载预训练的NeuralMarionette网络"""
        exp_dir = 'pretrained/aist'
        opt_path = os.path.join(exp_dir, 'opt.pickle')
        with open(opt_path, 'rb') as f:
            self.opt = pickle.load(f)
        self.opt.Ttot = 1

        ckpt_path = os.path.join(exp_dir, 'aist_pretrained.pth')
        checkpoint = torch.load(ckpt_path)
        self.network = NeuralMarionette(self.opt).cuda()
        self.network.load_state_dict(checkpoint)
        self.network.eval()
        self.network.anneal(1)
        
        print("✓ NeuralMarionette loaded successfully")
    
    def step1_process_frames(self, start_frame=0, end_frame=None):
        """
        步骤1：多帧一起处理，提取一致的skeleton数据
        参考Neural Marionette的多帧处理逻辑
        """
        print(f"\n=== 步骤1：多帧联合处理 ({start_frame} 到 {end_frame}) ===")
        
        obj_files = sorted(glob.glob(os.path.join(self.folder_path, "*.obj")))
        if not obj_files:
            raise ValueError(f"在 {self.folder_path} 中未找到.obj文件")
        
        # 确定处理范围
        assert 0 <= start_frame < len(obj_files)
        assert 0 <= end_frame < len(obj_files)

        self.all_mesh_data = []

        selected_files = obj_files[start_frame:end_frame+1]
        print(f"处理 {len(selected_files)} 个文件")
        
        # 第一步：加载所有帧的voxel数据
        all_pts = []
        all_voxels = []
        

        for i, obj_file in enumerate(selected_files):
            print(f"加载 {i+1}/{len(selected_files)}: {os.path.basename(obj_file)}")

            # mesh_info_path = os.path.join(mesh_info_folder, f"mesh_info_{i:03d}.pkl")
            # if os.path.exists(mesh_info_path):
            #     print(f"  ✓ 已存在mesh信息: {mesh_info_path}")
            #     with open(mesh_info_path, 'rb') as f:
            #         mesh_info = pickle.load(f)
            #     all_pts.append(mesh_info['pts_raw'])
            #     # print(f"  vox形状: {tuple(vox.shape)}")
            #     # all_voxels.append(vox[0])
            #     continue

            try:
                # 加载并体素化mesh
                mesh = self.read_mesh_verts(obj_file)  # 确保mesh加载成功
                pts_raw = np.asarray(mesh.vertices)
                all_pts.append(pts_raw)
                # all_voxels.append(vox[0])

                # with open(mesh_info_path, 'wb') as f:
                #     pickle.dump(mesh_info, f)
                # print(f"  ✓ 成功加载并保存mesh信息: {mesh_info_path}")
                
            except Exception as e:
                print(f"  ❌ 加载 {obj_file} 失败: {e}")
                continue
        
        all_pts = np.vstack(all_pts)  # (N,3) all vertices from all meshes
        print(f"shape: {all_pts.shape}")

        bmin = all_pts.min(axis=0)  # (3,) min corner
        blen = (all_pts.max(0) - bmin).max()

        for i, obj_file in enumerate(selected_files):
            print(f"处理 {i+1}/{len(selected_files)}: {os.path.basename(obj_file)}")
            pts_norm = (all_pts[i] - bmin) / (blen + 1e-5) * 2 - 1 + np.array([0, 0, 0])
            vox = self.points_to_voxel(pts_norm, self.opt.grid_size, is_binarized=True)
            all_voxels.append(vox)

        # 第二步：多帧联合处理获得骨骼
        print(f"\n多帧联合骨骼检测...")
        device = next(self.network.parameters()).device
        print(f"使用设备: {device}")
        
        # 将所有voxel堆叠成序列
        print(f"堆叠所有voxel帧...")
        voxel_seq = torch.stack(all_voxels, dim=0).to(device) 
        voxel_seq = voxel_seq.unsqueeze(0)  # (1, N, 1, 64, 64, 64)

        frame_data_folder = os.path.join(self.output_dir, 'frame_data')
        os.makedirs(frame_data_folder, exist_ok=True)

        # clear memory
        all_pts.clear()
        all_voxels.clear()

        with torch.no_grad():
            # 使用所有帧进行骨骼检测，保证时间一致性
            detector_log = self.network.kypt_detector(voxel_seq)  # (1, T, D, H, W)
            keypoints = detector_log['keypoints']    # (1, T, K, 4)
            first_feature = detector_log['first_feature']  # (1, C, G, G, G)
            
            B, T, K, _ = keypoints.shape
            print(f"检测到关键点: Batch={B}, Time={T}, Joints={K}")
            
            # 获取父节点关系
            parents = self.network.dyna_module.parents.cpu().numpy()
            
            # 对每一帧进行解码以获得详细的骨骼信息
            for t in range(T):
                frame_data_path = os.path.join(frame_data_folder, f"frame_{t:03d}.pkl")

                if os.path.exists(frame_data_path):
                    print(f"  ✓ 已存在帧数据: {frame_data_path}")
                    with open(frame_data_path, 'rb') as f:
                        frame_data = pickle.load(f)
                        self.all_mesh_data.append(frame_data)
                    continue


                # 构造当前帧的keypoints
                selected = torch.zeros(B, 2, K, 4, device=device, dtype=keypoints.dtype)
                selected[:, 0] = keypoints[:, t]     # 当前帧
                selected[:, 1] = keypoints[:, min(t+1, T-1)]  # 下一帧或最后一帧
                selected[:, :, :, 3] = keypoints[:, t, :, 3].unsqueeze(1)  # 保持置信度
                
                first_frame = voxel_seq[t:t+1].unsqueeze(0)  # (1, 1, D, H, W)
                
                # 解码获得更详细的骨骼信息
                decode_log = self.network.kypt_detector.decode_from_dyna(selected, first_feature, first_frame)
                
                # 提取当前帧的骨骼信息
                frame_kps = keypoints[0, t].cpu().numpy()  # (K, 4)
                joints = frame_kps[:, :3]  # (K, 3) joint positions
                
                # 获取旋转信息
                affinity = detector_log['affinity'] if 'affinity' in detector_log else None
                if affinity is not None:
                    R = decode_log['R'][0, 0].cpu().numpy()  # (K, 3, 3)
                else:
                    # 如果没有affinity，使用单位旋转矩阵
                    R = np.tile(np.eye(3), (K, 1, 1))
                
                # 归一化顶点坐标
                mesh_info = all_mesh_info[t]
                pts_norm = ((mesh_info['pts_raw'] - mesh_info['bmin']) / 
                           (mesh_info['blen'] + 1e-5)) * 2 - 1
                
                # 保存帧数据
                frame_data = {
                    'base_name': mesh_info['base_name'],
                    'parents': parents,
                    'kps': frame_kps,
                    'joints': joints,
                    'R': R,
                    'bmin': mesh_info['bmin'],
                    'blen': mesh_info['blen'],
                    'pts_raw': mesh_info['pts_raw'],
                    'pts_norm': pts_norm,
                    'mesh_vertices': mesh_info['mesh_vertices'],
                    'mesh_triangles': mesh_info['mesh_triangles'],
                }
                
                with open(frame_data_path, 'wb') as f:
                    pickle.dump(frame_data, f)
                    frame_visualization = draw_skeleton(joints, parents)
                    o3d.visualization.draw_geometries([frame_visualization])
                    print(f"  ✓ 保存帧数据: {frame_data_path}")

                self.all_mesh_data.append(frame_data)
                
                if t < 3:  # 只显示前3个的详细信息
                    joints_world = (joints + 1) * 0.5 * mesh_info['blen'] + mesh_info['bmin']
                    print(f"  ✓ 帧{t}: {len(mesh_info['pts_raw'])} 顶点, {K} 关节")
        
        if not self.all_mesh_data:
            raise RuntimeError("没有成功处理任何mesh")
        
        print(f"✓ 多帧联合处理完成: {len(self.all_mesh_data)} 帧")
        print(f"✓ 骨骼结构一致性: {K} 个关节，相同的父节点关系")
        return len(self.all_mesh_data)
    
    def step2_detect_rest_pose(self):
        """
        步骤2：自动检测最佳rest pose
        基于网格和骨骼的几何特性，不依赖语义关节
        """
        print("\n=== 步骤2：检测最佳Rest Pose (基于几何特性) ===")
        
        scores = []
        
        for i, data in enumerate(self.all_mesh_data):
            joints = data['joints']  # 归一化坐标
            mesh_vertices = data['pts_norm']
            parents = sanitize_parents(data['parents'])
            R_rotations = data['R']  # 旋转矩阵
            
            # 指标1：骨骼结构稳定性（骨长变化小）
            bone_lengths = []
            for j in range(len(joints)):
                if parents[j] >= 0:
                    bone_length = np.linalg.norm(joints[j] - joints[parents[j]])
                    bone_lengths.append(bone_length)
            
            if bone_lengths:
                bone_length_std = np.std(bone_lengths)
                bone_length_mean = np.mean(bone_lengths)
                bone_stability = 1.0 / (bone_length_std / (bone_length_mean + 1e-6) + 0.01)  # 变异系数倒数
            else:
                bone_stability = 0
            
            # 指标2：关节分布均匀性（关节在空间中分布均匀）
            if len(joints) > 1:
                joint_center = joints.mean(axis=0)
                joint_distances = np.linalg.norm(joints - joint_center, axis=1)
                joint_spread = np.std(joint_distances)  # 分布标准差
                joint_coverage = np.ptp(joints, axis=0).mean()  # 覆盖范围
                distribution_score = joint_spread * joint_coverage  # 既要分散又要覆盖范围大
            else:
                distribution_score = 0
            
            # 指标3：旋转矩阵接近单位矩阵（minimal rotation）
            rotation_deviations = []
            for R in R_rotations:
                deviation = np.linalg.norm(R - np.eye(3), 'fro')  # Frobenius范数
                rotation_deviations.append(deviation)
            
            avg_rotation_deviation = np.mean(rotation_deviations)
            rotation_minimality = 1.0 / (avg_rotation_deviation + 0.1)  # 偏差越小越好
            
            # 指标4：网格紧凑性和中心性
            mesh_center = mesh_vertices.mean(axis=0)
            mesh_bounds = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
            mesh_compactness = 1.0 / (np.prod(mesh_bounds) + 1e-6)  # 体积越小越紧凑
            
            # 关节-网格中心对齐
            if len(joints) > 0:
                joint_center = joints.mean(axis=0)
                center_alignment = 1.0 / (np.linalg.norm(joint_center - mesh_center) + 0.1)
            else:
                center_alignment = 0
            
            # 指标5：骨骼层次结构质量
            # 检查parent关系的合理性
            hierarchy_score = 0
            for j in range(len(joints)):
                if parents[j] >= 0:
                    parent_pos = joints[parents[j]]
                    child_pos = joints[j]
                    # 好的层次结构中，子关节不应该离根部太远
                    distance_to_parent = np.linalg.norm(child_pos - parent_pos)
                    hierarchy_score += 1.0 / (distance_to_parent + 0.1)
            
            hierarchy_score = hierarchy_score / max(1, len(joints))
            
            # 组合加权得分 - 更注重几何特性而不是语义
            score = (
                3.0 * bone_stability +          # 骨骼稳定性最重要
                2.0 * rotation_minimality +     # 最小旋转很重要
                1.5 * distribution_score +      # 关节分布
                1.0 * center_alignment +        # 中心对齐
                1.0 * hierarchy_score +         # 层次结构
                0.5 * mesh_compactness          # 网格紧凑性
            )
            
            scores.append(score)
            print(f"  帧 {i:2d}: 骨长稳定={bone_stability:.3f}, 旋转最小={rotation_minimality:.3f}, "
                  f"关节分布={distribution_score:.3f}, 中心对齐={center_alignment:.3f}, "
                  f"层次={hierarchy_score:.3f}, 总分={score:.3f}")
        
        # 选择得分最高的帧
        self.rest_pose_idx = np.argmax(scores)
        rest_data = self.all_mesh_data[self.rest_pose_idx]
        
        print(f"✓ 基于几何特性选择帧 {self.rest_pose_idx} 作为rest pose: {rest_data['base_name']}")
        print(f"  含有 {len(rest_data['pts_raw'])} 顶点, {len(rest_data['joints'])} 关节")
        print(f"  几何得分: {scores[self.rest_pose_idx]:.3f}")
        
        return self.rest_pose_idx
    
    def step3_unify_mesh_topology(self):
        """
        步骤3：基于骨骼结构的网格拓扑统一
        从rest pose开始向两边进行remap，保证更好的时间一致性
        """
        print("\n=== 步骤3：基于骨骼的网格拓扑统一 (从rest pose向两边) ===")
        
        if self.rest_pose_idx is None:
            raise RuntimeError("必须先检测rest pose")
        
        template_data = self.all_mesh_data[self.rest_pose_idx]
        template_pts = template_data['pts_norm']
        template_joints = template_data['joints']
        target_vertex_count = len(template_pts)
        
        print(f"使用帧{self.rest_pose_idx}的 {target_vertex_count} 顶点作为模板")
        print(f"从rest pose向前后两个方向进行拓扑统一")
        
        unified_frames = [None] * len(self.all_mesh_data)
        
        # 首先设置rest pose
        unified_frames[self.rest_pose_idx] = template_pts
        print(f"  帧 {self.rest_pose_idx}: Rest pose设置完成")
        
        # 向后处理 (rest_pose_idx + 1 到 end)
        print(f"向后处理: 帧 {self.rest_pose_idx+1} 到 {len(self.all_mesh_data)-1}")
        prev_pts = template_pts
        prev_joints = template_joints
        
        for i in range(self.rest_pose_idx + 1, len(self.all_mesh_data)):
            current_data = self.all_mesh_data[i]
            current_pts = current_data['pts_norm']
            current_joints = current_data['joints']
            
            # 使用前一帧作为参考进行映射
            if len(current_pts) == target_vertex_count:
                correspondence_quality = self._check_correspondence_quality(
                    prev_pts, current_pts, prev_joints, current_joints
                )
                
                if correspondence_quality > 0.8:
                    unified_frames[i] = current_pts
                    print(f"  帧 {i}: 良好对应关系 (质量={correspondence_quality:.3f})")
                else:
                    remapped_pts = self._bone_guided_remapping(
                        prev_pts, current_pts, prev_joints, current_joints
                    )
                    unified_frames[i] = remapped_pts
                    print(f"  帧 {i}: 骨骼引导重映射 (质量={correspondence_quality:.3f})")
            else:
                remapped_pts = self._bone_guided_remapping(
                    prev_pts, current_pts, prev_joints, current_joints
                )
                unified_frames[i] = remapped_pts
                print(f"  帧 {i}: 顶点数映射 {len(current_pts)} → {target_vertex_count}")
            
            # 更新参考帧
            prev_pts = unified_frames[i]
            prev_joints = current_joints
        
        # 向前处理 (rest_pose_idx - 1 到 0)
        print(f"向前处理: 帧 {self.rest_pose_idx-1} 到 0")
        prev_pts = template_pts
        prev_joints = template_joints
        
        for i in range(self.rest_pose_idx - 1, -1, -1):
            current_data = self.all_mesh_data[i]
            current_pts = current_data['pts_norm']
            current_joints = current_data['joints']
            
            # 使用后一帧作为参考进行映射
            if len(current_pts) == target_vertex_count:
                correspondence_quality = self._check_correspondence_quality(
                    prev_pts, current_pts, prev_joints, current_joints
                )
                
                if correspondence_quality > 0.8:
                    unified_frames[i] = current_pts
                    print(f"  帧 {i}: 良好对应关系 (质量={correspondence_quality:.3f})")
                else:
                    remapped_pts = self._bone_guided_remapping(
                        prev_pts, current_pts, prev_joints, current_joints
                    )
                    unified_frames[i] = remapped_pts
                    print(f"  帧 {i}: 骨骼引导重映射 (质量={correspondence_quality:.3f})")
            else:
                remapped_pts = self._bone_guided_remapping(
                    prev_pts, current_pts, prev_joints, current_joints
                )
                unified_frames[i] = remapped_pts
                print(f"  帧 {i}: 顶点数映射 {len(current_pts)} → {target_vertex_count}")
            
            # 更新参考帧
            prev_pts = unified_frames[i]
            prev_joints = current_joints
        
        self.unified_vertices = np.stack(unified_frames, axis=0)
        print(f"✓ 双向网格拓扑统一完成: {self.unified_vertices.shape}")
        print(f"  时间一致性应该更好，因为相邻帧之间的映射更稳定")
        
        return self.unified_vertices.shape
    
    def _check_correspondence_quality(self, template_pts, current_pts, template_joints, current_joints):
        """检查顶点对应关系质量"""
        if len(template_pts) != len(current_pts):
            return 0.0
        
        # 关节对齐检查
        joint_distances = np.linalg.norm(template_joints - current_joints, axis=1)
        joint_alignment = np.exp(-joint_distances.mean())
        
        # 网格中心对齐检查
        template_center = template_pts.mean(axis=0)
        current_center = current_pts.mean(axis=0)
        center_distance = np.linalg.norm(template_center - current_center)
        center_alignment = np.exp(-center_distance * 5)
        
        # 组合质量得分
        quality = 0.7 * joint_alignment + 0.3 * center_alignment
        return quality
    
    def _bone_guided_remapping(self, template_pts, current_pts, template_joints, current_joints):
        """基于骨骼引导的顶点重映射"""
        n_template = len(template_pts)
        n_current = len(current_pts)
        
        # 为大型网格使用简化重映射
        if n_template > 8000:
            print(f"    大型网格 ({n_template} 顶点), 使用简化重映射")
            tree = cKDTree(current_pts)
            distances, indices = tree.query(template_pts, k=1)
            return current_pts[indices]
        
        # 创建骨骼影响坐标系
        template_coords = self._compute_bone_coordinates(template_pts, template_joints)
        current_coords = self._compute_bone_coordinates(current_pts, current_joints)
        
        # 使用k-d树进行高效最近邻搜索
        tree = cKDTree(current_coords)
        k = min(3, n_current)
        distances, indices = tree.query(template_coords, k=k)
        
        remapped_pts = np.zeros_like(template_pts)
        
        if k == 1:
            remapped_pts = current_pts[indices.flatten()]
        else:
            # 加权平均
            safe_distances = distances + 1e-8
            weights = 1.0 / safe_distances
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            for i in range(n_template):
                vertex_indices = indices[i]
                vertex_weights = weights[i]
                remapped_pts[i] = np.sum(
                    vertex_weights.reshape(-1, 1) * current_pts[vertex_indices], axis=0
                )
        
        return remapped_pts
    
    def _compute_bone_coordinates(self, vertices, joints):
        """计算骨骼影响坐标系 - 优化版本"""
        n_vertices = len(vertices)
        n_joints = len(joints)
        
        # 预计算网格尺度
        mesh_bounds = vertices.max(axis=0) - vertices.min(axis=0)
        mesh_scale = np.linalg.norm(mesh_bounds)
        
        # 对大型网格限制关节影响特征
        if n_vertices > 5000:
            # 只使用前8个最重要的关节
            joint_subset = joints[:min(8, n_joints)]
            bone_coords = np.zeros((n_vertices, 3 + len(joint_subset)))
        else:
            joint_subset = joints
            bone_coords = np.zeros((n_vertices, 3 + n_joints))
        
        bone_coords[:, :3] = vertices
        
        # 向量化距离计算
        vertices_expanded = vertices[:, np.newaxis, :]  # (N, 1, 3)
        joints_expanded = joint_subset[np.newaxis, :, :]  # (1, K, 3)
        
        distances = np.linalg.norm(vertices_expanded - joints_expanded, axis=2)
        normalized_distances = distances / (mesh_scale + 1e-8)
        influences = np.exp(-normalized_distances * 3.0)
        
        bone_coords[:, 3:] = influences
        
        return bone_coords
    
    def step4_compute_skinning(self):
        """
        步骤4：使用 C++ CLI 版本的 DemBones 计算蒙皮权重
        """
        print("\n=== 步骤4：DemBones蒙皮权重计算 (C++ CLI版本) ===")
        
        if self.unified_vertices is None:
            raise RuntimeError("必须先统一网格拓扑")
        
        # 获取骨骼父节点数据
        rest_data = self.all_mesh_data[self.rest_pose_idx]
        self.parents = sanitize_parents(rest_data['parents'])
        
        F, N, _ = self.unified_vertices.shape
        K = len(self.parents)
        
        print(f"骨骼: {K} 个关节")
        print(f"动画: {F} 帧, {N} 个顶点")
        
        # 智能数据子采样以确保DemBones能成功运行
        processed_vertices, vertex_indices, frame_indices = self._prepare_demBones_data()
        
        F_sub, N_sub, _ = processed_vertices.shape
        print(f"DemBones输入: {F_sub} 帧, {N_sub} 顶点 (子采样), {frame_indices} frame_indices")
        
        try:
            # 使用修复后的DemBones API
            rest_pose_sub, weights_sub, transforms_sub = self._compute_skinning_weights_fixed(
                processed_vertices, self.parents
            )
            
            print(f"✓ DemBones子采样数据成功:")
            print(f"  Rest pose: {rest_pose_sub.shape}")
            print(f"  蒙皮权重: {weights_sub.shape}")
            print(f"  骨骼变换: {transforms_sub.shape}")
            
            # 扩展结果到完整分辨率
            self.skinning_weights, self.bone_transforms = self._expand_demBones_results(
                weights_sub, transforms_sub, vertex_indices, frame_indices, N, F
            )
            
            print(f"✓ 扩展到完整分辨率:")
            print(f"  完整蒙皮权重: {self.skinning_weights.shape}")
            print(f"  完整骨骼变换: {self.bone_transforms.shape}")
            print(f"  权重范围: [{self.skinning_weights.min():.4f}, {self.skinning_weights.max():.4f}]")
                
        except Exception as e:
            print(f"❌ DemBones失败: {e}")
            raise RuntimeError(f"DemBones计算失败: {e}")
        
        # 保存结果
        skinning_results = {
            'rest_pose': self.unified_vertices[self.rest_pose_idx],
            'skinning_weights': self.skinning_weights,
            'bone_transforms': self.bone_transforms,
            'parents': self.parents,
            'rest_pose_idx': self.rest_pose_idx,
            'unified_vertices': self.unified_vertices
        }
        
        results_path = os.path.join(self.output_dir, 'skinning_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(skinning_results, f)
        
        print(f"✓ 蒙皮结果已保存到 {results_path}")
    
    def _prepare_demBones_data(self):
        """智能子采样数据以确保DemBones成功运行"""
        F, N, _ = self.unified_vertices.shape
        
        # 目标限制值 - 基于性能测试：50个顶点很快，100个顶点较慢
        MAX_VERTICES = 80  # 在50-100之间选择一个安全值
        MAX_FRAMES = 10
        
        # 顶点子采样
        if N > MAX_VERTICES:
            vertex_ratio = MAX_VERTICES / N
            vertex_indices = self._sample_vertices_intelligently(vertex_ratio)
            print(f"  顶点子采样: {N} → {len(vertex_indices)} ({vertex_ratio:.2%})")
        else:
            vertex_indices = np.arange(N)
            print(f"  无需顶点子采样: {N} 顶点")
        
        # 帧子采样
        if F > MAX_FRAMES:
            frame_indices = np.linspace(0, F-1, MAX_FRAMES, dtype=int)
            if self.rest_pose_idx not in frame_indices:
                frame_indices[0] = self.rest_pose_idx  # 确保包含rest pose
            frame_indices = np.sort(frame_indices)
            print(f"  帧子采样: {F} → {len(frame_indices)} 帧")
        else:
            frame_indices = np.arange(F)
            print(f"  无需帧子采样: {F} 帧")
        
        # 创建子采样数据
        processed_vertices = self.unified_vertices[frame_indices][:, vertex_indices]
        
        return processed_vertices, vertex_indices, frame_indices
    
    def _sample_vertices_intelligently(self, ratio):
        """智能顶点采样以保持网格结构 - 优化版本避免卡死"""
        N = self.unified_vertices.shape[1]
        n_samples = int(N * ratio)
        
        print(f"    智能采样: {N} → {n_samples} 顶点")
        
        # 使用rest pose作为参考
        rest_vertices = self.unified_vertices[self.rest_pose_idx]
        
        # 对于大型网格，使用更高效的采样策略
        if N > 20000:
            print(f"    大型网格检测，使用快速均匀采样")
            # 简单均匀采样避免复杂计算
            step = N // n_samples
            indices = np.arange(0, N, step)[:n_samples]
            return np.sort(indices)
        
        # 中等大小网格：使用改进的最远点采样
        if N > 5000:
            print(f"    中等网格，使用批量最远点采样")
            # 批量处理，每次处理一批候选点
            indices = [0]
            remaining = np.arange(1, N)
            batch_size = min(1000, len(remaining))
            
            while len(indices) < n_samples and len(remaining) > 0:
                if len(remaining) <= batch_size:
                    # 处理剩余所有点
                    candidates = remaining
                else:
                    # 随机选择一批候选点
                    candidates = np.random.choice(remaining, batch_size, replace=False)
                
                # 在候选点中找最远的
                max_dist = -1
                best_idx = None
                
                for idx in candidates:
                    min_dist = min(np.linalg.norm(rest_vertices[idx] - rest_vertices[selected]) 
                                 for selected in indices[-min(10, len(indices)):])  # 只比较最近的10个点
                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_idx = idx
                
                if best_idx is not None:
                    indices.append(best_idx)
                    remaining = remaining[remaining != best_idx]
                else:
                    break
            
            # 如果还不够，随机补充
            if len(indices) < n_samples and len(remaining) > 0:
                needed = n_samples - len(indices)
                additional = np.random.choice(remaining, min(needed, len(remaining)), replace=False)
                indices.extend(additional)
            
            return np.sort(indices[:n_samples])
        
        # 小型网格：使用完整最远点采样
        else:
            print(f"    小型网格，使用完整最远点采样")
            indices = [0]
            remaining = set(range(1, N))
            
            for i in range(n_samples - 1):
                if not remaining:
                    break
                    
                max_dist = -1
                best_idx = None
                
                # 随机采样一部分候选点以加速
                candidates = list(remaining)
                if len(candidates) > 500:
                    candidates = np.random.choice(candidates, 500, replace=False)
                
                for idx in candidates:
                    min_dist = min(np.linalg.norm(rest_vertices[idx] - rest_vertices[selected]) 
                                 for selected in indices[-min(5, len(indices)):])  # 只比较最近的5个点
                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_idx = idx
                
                if best_idx is not None:
                    indices.append(best_idx)
                    remaining.remove(best_idx)
                
                # 每100个点显示进度
                if i % 100 == 0:
                    print(f"      采样进度: {i+1}/{n_samples-1}")
            
            return np.sort(indices[:n_samples])
    
    def _compute_skinning_weights_fixed(self, frames_vertices, parents):
        """使用修复后的DemBones API计算蒙皮权重 - 带超时和回退机制"""
        print("使用修复后的DemBones API...")
        
        F, N, _ = frames_vertices.shape
        K = len(parents)
        
        # 检查数据合理性
        if F < 2:
            print("⚠️ 帧数不足，使用简化蒙皮权重")
            return self._create_simple_skinning_weights(frames_vertices[0], K)
        
        if N > 8000:
            print("⚠️ 顶点数过多，进一步子采样")
            # 进一步减少顶点数
            subsample_ratio = 6000 / N
            subsample_indices = np.linspace(0, N-1, 6000, dtype=int)
            frames_vertices = frames_vertices[:, subsample_indices]
            N = 6000
            print(f"    进一步子采样到 {N} 顶点")
        
        # 尝试多种配置的DemBones
        configs = [
            # 配置1：最简单设置
            {
                'nIters': 10, 'nInitIters': 2, 'nTransIters': 1, 
                'nWeightsIters': 1, 'nnz': 4, 'weightsSmooth': 1e-3,
                'timeout': 60, 'name': '最简配置'
            },
            # 配置2：中等设置
            {
                'nIters': 15, 'nInitIters': 3, 'nTransIters': 2, 
                'nWeightsIters': 1, 'nnz': 6, 'weightsSmooth': 1e-4,
                'timeout': 120, 'name': '中等配置'
            },
            # 配置3：原始设置（但超时时间更短）
            {
                'nIters': 20, 'nInitIters': 5, 'nTransIters': 3, 
                'nWeightsIters': 2, 'nnz': 8, 'weightsSmooth': 1e-4,
                'timeout': 180, 'name': '完整配置'
            }
        ]
        
        for i, config in enumerate(configs):
            print(f"\n尝试DemBones {config['name']} (配置 {i+1}/{len(configs)})")
            
            try:
                result = self._try_demBones_with_timeout(frames_vertices, parents, config)
                if result is not None:
                    print(f"✓ {config['name']} 成功！")
                    return result
                else:
                    print(f"❌ {config['name']} 超时或失败")
                    
            except Exception as e:
                print(f"❌ {config['name']} 异常: {e}")
                continue
        
        # 所有DemBones配置都失败，使用回退方案
        print("⚠️ 所有DemBones配置都失败，使用简化蒙皮权重")
        return self._create_simple_skinning_weights(frames_vertices[0], K)
    
    def _try_demBones_with_timeout(self, frames_vertices, parents, config):
        """使用 C++ CLI 版本运行 DemBones"""
        F, N, _ = frames_vertices.shape
        K = len(parents)
        
        print(f"    Rest pose: {frames_vertices[0].shape}, Animated: {frames_vertices[1:].shape}")
        print(f"    参数: iters={config['nIters']}, nnz={config['nnz']}")
        
        # 查找 DemBones 可执行文件
        demBones_exe = self._find_demBones_executable()
        if demBones_exe is None:
            print("    ⚠️ 未找到 DemBones 可执行文件，使用简化蒙皮权重")
            return None
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 准备数据文件
                input_file = os.path.join(temp_dir, 'input.dembones')
                output_file = os.path.join(temp_dir, 'output.dembones')
                
                # 写入输入数据
                self._write_demBones_input(input_file, frames_vertices, parents, config)
                
                # 运行 DemBones CLI
                print(f"    运行命令: {demBones_exe}")
                start_time = time.time()
                
                # DemBones CLI 命令格式: demBones input.dembones output.dembones
                result = subprocess.run(
                    [demBones_exe, input_file, output_file],
                    capture_output=True,
                    text=True,
                    timeout=config.get('timeout', 300)
                )
                
                elapsed_time = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"    ✅ DemBones CLI 执行成功！耗时 {elapsed_time:.2f} 秒")
                    
                    # 读取结果
                    if os.path.exists(output_file):
                        return self._read_demBones_output(output_file, frames_vertices[0], K, F)
                    else:
                        print("    ❌ 输出文件未生成")
                        return None
                else:
                    print(f"    ❌ DemBones CLI 执行失败 (返回码: {result.returncode})")
                    if result.stdout:
                        print(f"    标准输出: {result.stdout[:500]}...")
                    if result.stderr:
                        print(f"    标准错误: {result.stderr[:500]}...")
                    return None
                
            except subprocess.TimeoutExpired:
                print(f"    ❌ DemBones CLI 超时 ({config.get('timeout', 300)} 秒)")
                return None
            except Exception as e:
                print(f"    ❌ DemBones CLI 异常: {e}")
                return None
    
    def _find_demBones_executable(self):
        """查找 DemBones 可执行文件"""
        # 常见的可能位置
        possible_paths = [
            'demBones.exe',  # 在 PATH 中
            'DemBones.exe',
            'demBones.bat',  # Windows 批处理文件
            './demBones.exe',  # 当前目录
            './DemBones.exe',
            './demBones.bat',
            './bin/demBones.exe',  # bin 目录
            './bin/DemBones.exe',
            './bin/demBones.bat',
            'C:/Program Files/DemBones/demBones.exe',  # 系统安装
            'C:/Program Files/DemBones/DemBones.exe',
            '../demBones/bin/demBones.exe',  # 相邻目录
            '../demBones/bin/DemBones.exe',
            # Linux 路径
            'demBones',
            './demBones',
            './bin/demBones',
            '/usr/local/bin/demBones',
            '/usr/bin/demBones',
        ]
        
        for path in possible_paths:
            try:
                # 测试是否可以执行
                result = subprocess.run([path, '--help'], 
                                      capture_output=True, timeout=5, text=True)
                if result.returncode == 0 or 'dembones' in result.stdout.lower() or 'usage' in result.stdout.lower():
                    print(f"    ✓ 找到 DemBones: {path}")
                    return path
            except:
                continue
        
        print("    ⚠️ 未找到 DemBones 可执行文件")
        print("    📥 建议下载并编译: https://github.com/electronicarts/dem-bones")
        print("    🔄 将使用简化蒙皮权重算法")
        return None
    
    def _write_demBones_input(self, input_file, frames_vertices, parents, config):
        """写入 DemBones 输入文件（基于官方格式）"""
        F, N, _ = frames_vertices.shape
        K = len(parents)
        
        with open(input_file, 'w') as f:
            # 写入基本信息
            f.write(f"# DemBones Input File\n")
            f.write(f"nV {N}\n")  # 顶点数
            f.write(f"nB {K}\n")  # 骨骼数
            f.write(f"nF {F-1}\n")  # 动画帧数 (不包括rest pose)
            f.write(f"nS 1\n")  # subject数量
            
            # 写入算法参数
            f.write(f"nIters {config['nIters']}\n")
            f.write(f"nInitIters {config['nInitIters']}\n") 
            f.write(f"nTransIters {config['nTransIters']}\n")
            f.write(f"nWeightsIters {config['nWeightsIters']}\n")
            f.write(f"nnz {config['nnz']}\n")
            f.write(f"weightsSmooth {config['weightsSmooth']}\n")
            
            # 写入骨骼层次结构
            f.write("# Bone hierarchy (parents)\n")
            for i, parent in enumerate(parents):
                f.write(f"parent {i} {parent}\n")
            
            # 写入frame起始索引和subject ID
            f.write("fStart 0\n")
            f.write("subjectID 0\n")
            
            # 写入rest pose顶点
            f.write("# Rest pose vertices\n")
            rest_pose = frames_vertices[0]
            for i in range(N):
                f.write(f"v {rest_pose[i, 0]:.6f} {rest_pose[i, 1]:.6f} {rest_pose[i, 2]:.6f}\n")
            
            # 写入动画顶点
            f.write("# Animated vertices\n")
            for frame_idx in range(1, F):
                frame_vertices = frames_vertices[frame_idx]
                for i in range(N):
                    f.write(f"a {frame_vertices[i, 0]:.6f} {frame_vertices[i, 1]:.6f} {frame_vertices[i, 2]:.6f}\n")
    
    def _read_demBones_output(self, output_file, rest_pose, K, F):
        """读取 DemBones 输出文件"""
        try:
            weights = None
            N = len(rest_pose)
            
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # 解析权重矩阵
            # DemBones 通常输出格式为每行一个顶点的权重
            weights = np.zeros((N, K), dtype=np.float32)
            
            reading_weights = False
            vertex_idx = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 检查是否开始读取权重
                if 'weight' in line.lower() or reading_weights:
                    reading_weights = True
                    
                    # 尝试解析权重数据
                    try:
                        if line.startswith('w ') or ' ' in line:
                            values = line.replace('w ', '').split()
                            if len(values) >= K and vertex_idx < N:
                                for k in range(K):
                                    weights[vertex_idx, k] = float(values[k])
                                vertex_idx += 1
                    except:
                        continue
            
            # 如果没有读取到权重，创建简单权重
            if vertex_idx == 0:
                print("    ⚠️ 无法解析DemBones输出，创建简化权重")
                return self._create_simple_skinning_weights(rest_pose, K)
            
            # 归一化权重
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums < 1e-8] = 1.0
            weights = weights / row_sums
            
            # 创建单位变换矩阵（简化版本）
            T_all = np.zeros((F, K, 4, 4), dtype=np.float32)
            for f in range(F):
                for b in range(K):
                    T_all[f, b] = np.eye(4)
            
            print(f"    ✓ 读取权重矩阵: {weights.shape}")
            print(f"    权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            
            return (rest_pose, weights, T_all)
                
        except Exception as e:
            print(f"    ❌ 读取输出文件失败: {e}")
            return None
    
    def _create_simple_skinning_weights(self, rest_pose, K):
        """创建简化的蒙皮权重（当DemBones失败时的回退方案）"""
        print("创建简化蒙皮权重...")
        N = len(rest_pose)
        
        # 创建基于距离的蒙皮权重
        rest_data = self.all_mesh_data[self.rest_pose_idx]
        joints = rest_data['joints']
        
        # 将joints从归一化坐标转换为与rest_pose相同的坐标系
        joints_scaled = joints  # 假设已经在正确坐标系中
        
        weights = np.zeros((N, K), dtype=np.float32)
        
        # 为每个顶点计算到每个关节的距离
        for v in range(N):
            vertex = rest_pose[v]
            distances = np.array([np.linalg.norm(vertex - joints_scaled[j]) for j in range(min(K, len(joints_scaled)))])
            
            # 转换距离为权重（距离越近权重越大）
            inv_distances = 1.0 / (distances + 1e-3)
            
            # 只保留最近的几个关节
            top_k = min(4, len(inv_distances))
            top_indices = np.argsort(inv_distances)[-top_k:]
            
            for idx in top_indices:
                if idx < K:
                    weights[v, idx] = inv_distances[idx]
            
            # 归一化
            weight_sum = weights[v].sum()
            if weight_sum > 1e-8:
                weights[v] /= weight_sum
            else:
                # 如果所有权重都是0，给第一个关节设置权重
                weights[v, 0] = 1.0
        
        # 创建单位变换矩阵
        F = len(self.all_mesh_data)
        T_all = np.zeros((F, K, 4, 4), dtype=np.float32)
        for f in range(F):
            for b in range(K):
                T_all[f, b] = np.eye(4)
        
        print(f"✓ 简化蒙皮权重创建完成: {weights.shape}")
        print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        
        return rest_pose, weights, T_all
    
    def _expand_demBones_results(self, weights_sub, transforms_sub, vertex_indices, frame_indices, N_full, F_full):
        """将DemBones结果从子采样数据扩展到完整分辨率"""
        K = len(self.parents)
        
        # 扩展蒙皮权重
        full_weights = np.zeros((N_full, K), dtype=np.float32)
        
        # 直接分配子采样权重
        full_weights[vertex_indices] = weights_sub
        
        # 对未采样顶点进行插值
        rest_vertices_full = self.unified_vertices[self.rest_pose_idx]
        rest_vertices_sub = rest_vertices_full[vertex_indices]
        
        tree = cKDTree(rest_vertices_sub)
        unsampled_indices = np.setdiff1d(np.arange(N_full), vertex_indices)
        
        if len(unsampled_indices) > 0:
            distances, nearest_idx = tree.query(rest_vertices_full[unsampled_indices], k=3)
            
            for i, orig_idx in enumerate(unsampled_indices):
                dists = distances[i]
                neighs = nearest_idx[i]
                
                weights_dists = 1.0 / (dists + 1e-8)
                weights_dists /= weights_dists.sum()
                
                for j in range(len(neighs)):
                    full_weights[orig_idx] += weights_dists[j] * weights_sub[neighs[j]]
        
        # 扩展骨骼变换（简化版本）
        full_transforms = np.zeros((F_full, K, 4, 4), dtype=np.float32)
        for f in range(F_full):
            for b in range(K):
                full_transforms[f, b] = np.eye(4)
        
        return full_weights, full_transforms
    
    def step5_generate_interpolation(self, start_frame_idx, end_frame_idx, num_interp):
        """
        步骤5：在任意两帧之间生成指定数量的插值
        """
        print(f"\n=== 步骤5：生成插值 (帧{start_frame_idx} → 帧{end_frame_idx}, {num_interp}个插值) ===")
        
        if self.bone_transforms is None:
            raise RuntimeError("必须先计算蒙皮权重")
        
        if start_frame_idx >= len(self.all_mesh_data) or end_frame_idx >= len(self.all_mesh_data):
            raise ValueError(f"帧索引超出范围 (最大: {len(self.all_mesh_data)-1})")
        
        # 获取起始和结束帧的骨骼变换
        T_from = self.bone_transforms[start_frame_idx]  # (K, 4, 4)
        T_to = self.bone_transforms[end_frame_idx]      # (K, 4, 4)
        
        # 创建插值权重
        alphas = np.linspace(0, 1, num_interp + 2)[1:-1]  # 排除端点
        
        interpolated_meshes = []
        rest_data = self.all_mesh_data[self.rest_pose_idx]
        
        for i, alpha in enumerate(alphas):
            print(f"  生成插值帧 {i+1}/{num_interp} (alpha={alpha:.3f})")
            
            # 插值骨骼变换
            T_interp = self._interpolate_bone_transforms(T_from, T_to, alpha)
            
            # 应用线性混合蒙皮
            vertices_interp = self._apply_linear_blend_skinning(
                self.unified_vertices[self.rest_pose_idx],  # Rest pose顶点
                T_interp
            )
            
            # 转换回世界坐标
            vertices_world = (vertices_interp + 1) * 0.5 * rest_data['blen'] + rest_data['bmin']
            
            # 创建网格
            mesh_interp = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vertices_world),
                triangles=o3d.utility.Vector3iVector(rest_data['mesh_triangles'])
            )
            mesh_interp.compute_vertex_normals()
            
            # 保存网格
            output_path = os.path.join(self.output_dir, f'interpolated_{start_frame_idx:03d}_{end_frame_idx:03d}_{i:03d}.obj')
            o3d.io.write_triangle_mesh(output_path, mesh_interp)
            
            interpolated_meshes.append(vertices_world)
        
        print(f"✓ 生成了 {len(interpolated_meshes)} 个插值帧")
        print(f"✓ 保存到: {self.output_dir}/interpolated_{start_frame_idx:03d}_{end_frame_idx:03d}_*.obj")
        
        return interpolated_meshes
    
    def _interpolate_bone_transforms(self, T_from, T_to, alpha):
        """使用SLERP插值骨骼变换"""
        T_interp = np.zeros_like(T_from)
        
        for k in range(len(T_from)):
            # 提取旋转和平移
            R_from = T_from[k, :3, :3]
            R_to = T_to[k, :3, :3]
            t_from = T_from[k, :3, 3]
            t_to = T_to[k, :3, 3]
            
            # 旋转的SLERP插值
            try:
                rot_from = Rotation.from_matrix(R_from)
                rot_to = Rotation.from_matrix(R_to)
                slerp = Slerp([0, 1], Rotation.concatenate([rot_from, rot_to]))
                R_interp = slerp([alpha]).as_matrix()[0]
            except:
                # 回退到线性插值
                R_interp = (1 - alpha) * R_from + alpha * R_to
            
            # 平移的线性插值
            t_interp = (1 - alpha) * t_from + alpha * t_to
            
            # 重构变换矩阵
            T_interp[k, :3, :3] = R_interp
            T_interp[k, :3, 3] = t_interp
            T_interp[k, 3, 3] = 1.0
        
        return T_interp
    
    def _apply_linear_blend_skinning(self, rest_vertices, bone_transforms):
        """应用线性混合蒙皮变形rest pose顶点"""
        N = len(rest_vertices)
        K = len(bone_transforms)
        
        # 转换为齐次坐标
        rest_h = np.hstack([rest_vertices, np.ones((N, 1))])
        
        # 应用蒙皮
        deformed_vertices = np.zeros((N, 3))
        
        for v in range(N):
            blended_transform = np.zeros((4, 4))
            
            for k in range(K):
                weight = self.skinning_weights[v, k]
                if weight > 1e-6:
                    blended_transform += weight * bone_transforms[k]
            
            # 应用混合变换
            deformed_h = blended_transform @ rest_h[v]
            deformed_vertices[v] = deformed_h[:3]
        
        return deformed_vertices


def main():
    parser = argparse.ArgumentParser(description="完整的体积视频插值管道")
    parser.add_argument('folder_path', type=str, help='包含.obj文件的文件夹路径')
    parser.add_argument('--start_frame', type=int, default=0, help='处理起始帧索引')
    parser.add_argument('--end_frame', type=int, default=None, help='处理结束帧索引')
    parser.add_argument('--interp_from', type=int, default=None, help='插值起始帧索引')
    parser.add_argument('--interp_to', type=int, default=None, help='插值结束帧索引')
    parser.add_argument('--num_interp', type=int, default=5, help='插值帧数')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder_path):
        print(f"❌ 错误: 文件夹 {args.folder_path} 不存在")
        return
    
    # 初始化管道
    pipeline = CompleteVVPipeline(args.folder_path, args.output_dir)
    
    try:
        # 步骤1：处理帧数据
        pipeline.step1_process_frames(args.start_frame, args.end_frame)
        
        # 步骤2：检测rest pose
        # pipeline.step2_detect_rest_pose()
        
        # # 步骤3：统一网格拓扑
        # pipeline.step3_unify_mesh_topology()
        
        # # 步骤4：计算蒙皮权重
        # pipeline.step4_compute_skinning()
        
        # # 步骤5：生成插值
        # if args.interp_from is not None and args.interp_to is not None:
        #     pipeline.step5_generate_interpolation(args.start_frame, args.end_frame, args.num_interp)
        # else:
        #     # 默认在前两帧之间插值
        #     pipeline.step5_generate_interpolation(0, min(1, len(pipeline.all_mesh_data)-1), args.num_interp)
        
        print("\n🎉 完整管道执行成功！")
        print(f"📁 结果保存在: {pipeline.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 管道执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
