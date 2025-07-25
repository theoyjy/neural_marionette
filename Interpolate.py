import numpy as np
import torch
import os
import pickle
import open3d as o3d
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import json
import glob
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time
from functools import partial

class VolumetricInterpolator:
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None):
        """
        初始化体素视频插值器
        
        Args:
            skeleton_data_dir: 骨骼数据目录路径
            mesh_folder_path: 网格文件目录路径
            weights_path: 预计算的蒙皮权重路径（可选）
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.mesh_folder_path = Path(mesh_folder_path)
        self.weights_path = weights_path
        
        # 加载骨骼数据
        self.load_skeleton_data()
        
        # 加载网格序列
        self.load_mesh_sequence()
        
        # 初始化蒙皮器
        self.skinner = None
        self.skinning_weights = None
        self.reference_frame_idx = None
        
        if weights_path and os.path.exists(weights_path):
            self.load_skinning_weights(weights_path)
        
        # 插值相关参数
        self.interpolation_cache = {}
        
    def load_skeleton_data(self):
        """加载骨骼预测数据"""
        try:
            # 加载关键点数据 [num_frames, num_joints, 4] (x, y, z, confidence)
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # 加载变换矩阵 [num_frames, num_joints, 4, 4]
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # 加载父节点关系 [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            # 加载旋转矩阵 [num_frames, num_joints, 3, 3]
            if (self.skeleton_data_dir / 'rotations.npy').exists():
                self.rotations = np.load(self.skeleton_data_dir / 'rotations.npy')
            else:
                self.rotations = None
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"✅ 成功加载骨骼数据:")
            print(f"  - 帧数: {self.num_frames}")
            print(f"  - 关节数: {self.num_joints}")
            print(f"  - 关键点形状: {self.keypoints.shape}")
            print(f"  - 变换矩阵形状: {self.transforms.shape}")
            
        except Exception as e:
            raise ValueError(f"无法加载骨骼数据: {e}")
    
    def load_mesh_sequence(self):
        """加载网格序列"""
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        
        if len(self.mesh_files) == 0:
            raise ValueError(f"在 {self.mesh_folder_path} 中未找到obj文件")
        
        print(f"✅ 成功加载网格序列:")
        print(f"  - 网格文件数: {len(self.mesh_files)}")
        print(f"  - 骨骼帧数: {self.num_frames}")
        
        if len(self.mesh_files) != self.num_frames:
            print(f"⚠️  警告: 网格文件数 ({len(self.mesh_files)}) 与骨骼帧数 ({self.num_frames}) 不匹配")
    
    def load_skinning_weights(self, weights_path):
        """加载预计算的蒙皮权重"""
        try:
            data = np.load(weights_path)
            self.skinning_weights = data['weights']
            self.rest_pose_vertices = data['rest_vertices']
            self.rest_pose_transforms = data['rest_transforms']
            self.reference_frame_idx = data['reference_frame_idx'].item()
            
            print(f"✅ 成功加载蒙皮权重:")
            print(f"  - 权重矩阵形状: {self.skinning_weights.shape}")
            print(f"  - 参考帧: {self.reference_frame_idx}")
            
            return True
        except Exception as e:
            print(f"⚠️  加载蒙皮权重失败: {e}")
            return False
    
    def check_and_optimize_weights(self, frame_start, frame_end, num_interpolate):
        """
        检查权重文件的reference_frame_idx是否等于start_idx，如果不是则调用Skinning.py重新优化权重
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_interpolate: 插值帧数
            
        Returns:
            bool: 是否需要重新优化权重
        """
        if self.skinning_weights is None:
            print("🔧 没有预加载权重，需要重新优化")
            return True
        
        if self.reference_frame_idx != frame_start:
            print(f"🔧 权重文件的参考帧 ({self.reference_frame_idx}) 与起始帧 ({frame_start}) 不匹配")
            print("   将调用Skinning.py重新优化权重...")
            return True
        
        print(f"✅ 权重文件参考帧 ({self.reference_frame_idx}) 与起始帧 ({frame_start}) 匹配")
        return False
    
    def optimize_weights_using_skinning(self, frame_start, frame_end, max_optimize_frames=5):
        """
        直接使用Skinning.py的成熟方法优化权重
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            max_optimize_frames: 最大优化帧数（默认5）
            
        Returns:
            bool: 是否成功优化权重
        """
        print(f"🔧 使用Skinning.py优化权重: {frame_start} -> {frame_end}")
        
        try:
            # 导入Skinning模块
            from Skinning import AutoSkinning
            
            # 创建临时权重文件路径
            import tempfile
            import os
            temp_weights_path = os.path.join(tempfile.gettempdir(), f"interpolation_weights_{frame_start}.npz")
            
            # 初始化Skinning对象
            skinner = AutoSkinning(
                skeleton_data_dir=str(self.skeleton_data_dir),
                reference_frame_idx=frame_start  # 使用起始帧作为参考帧
            )
            
            # 加载网格序列
            skinner.load_mesh_sequence(str(self.mesh_folder_path))
            
            # 确保归一化参数已初始化
            if skinner.reference_frame_idx not in skinner.frame_normalization_params:
                print("   确保归一化参数已初始化...")
                skinner.frame_normalization_params[skinner.reference_frame_idx] = skinner.compute_mesh_normalization_params(skinner.reference_mesh)
            
            # 选择优化帧（限制数量）
            total_frames = frame_end - frame_start + 1
            if total_frames <= max_optimize_frames:
                # 如果总帧数不多，使用所有帧
                optimize_frames = list(range(frame_start, frame_end + 1))
            else:
                # 均匀采样优化帧
                step = max(1, total_frames // max_optimize_frames)
                optimize_frames = list(range(frame_start, frame_end + 1, step))[:max_optimize_frames]
                # 确保包含起始和结束帧
                if frame_start not in optimize_frames:
                    optimize_frames.insert(0, frame_start)
                if frame_end not in optimize_frames:
                    optimize_frames.append(frame_end)
                # 限制数量
                optimize_frames = optimize_frames[:max_optimize_frames]
            
            print(f"   优化帧: {optimize_frames}")
            
            # 直接使用Skinning的优化方法
            print(f"   调用Skinning.py的optimize_reference_frame_skinning...")
            skinner.skinning_weights = skinner.optimize_reference_frame_skinning(
                optimization_frames=optimize_frames,
                regularization_lambda=0.01,
                max_iter=200  # 适中的迭代次数
            )
            
            if skinner.skinning_weights is not None:
                print(f"✅ 权重优化完成")
                print(f"   权重矩阵形状: {skinner.skinning_weights.shape}")
                
                # 保存权重
                skinner.save_skinning_weights(temp_weights_path)
                
                # 加载优化后的权重到插值器
                self.load_skinning_weights(temp_weights_path)
                print(f"✅ 权重已加载到插值器")
                return True
            else:
                print("❌ 权重优化失败")
                return False
                
        except Exception as e:
            print(f"❌ 调用Skinning.py优化权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compute_mesh_normalization_params(self, mesh):
        """计算网格归一化参数"""
        vertices = np.asarray(mesh.vertices)
        
        bmax = np.amax(vertices, axis=0)
        bmin = np.amin(vertices, axis=0)
        blen = (bmax - bmin).max()
        
        params = {
            'bmin': bmin,
            'bmax': bmax,
            'blen': blen,
            'scale': 1.0,
            'x_trans': 0.0,
            'z_trans': 0.0
        }
        
        return params
    
    def normalize_mesh_vertices(self, vertices, normalization_params):
        """归一化网格顶点"""
        params = normalization_params
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        normalized = ((vertices - params['bmin']) * params['scale'] / (params['blen'] + 1e-5)) * 2 - 1 + trans_offset
        return normalized
    
    def apply_lbs_transform(self, rest_vertices, weights, transforms):
        """应用改进的Linear Blend Skinning变换，保持网格体积和骨骼对齐"""
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        rest_vertices_homo = np.hstack([rest_vertices, np.ones((num_vertices, 1))])
        transformed_vertices = np.zeros((num_vertices, 3))
        
        # 改进的权重处理：确保权重和为1且非负
        weights = np.maximum(weights, 0)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-8)
        
        # 计算每个关节的变换贡献
        joint_contributions = []
        for j in range(num_joints):
            joint_transform = transforms[j]
            transformed_homo = (joint_transform @ rest_vertices_homo.T).T
            transformed_xyz = transformed_homo[:, :3]
            joint_weights = weights[:, j:j+1]
            joint_contributions.append(joint_weights * transformed_xyz)
        
        # 应用权重混合
        for contribution in joint_contributions:
            transformed_vertices += contribution
        
        # 体积保持：计算原始网格的体积特征
        if num_vertices > 3:
            # 计算原始网格的边界框
            bbox_min = np.min(rest_vertices, axis=0)
            bbox_max = np.max(rest_vertices, axis=0)
            original_volume = np.prod(bbox_max - bbox_min)
            
            # 计算变换后网格的边界框
            bbox_min_transformed = np.min(transformed_vertices, axis=0)
            bbox_max_transformed = np.max(transformed_vertices, axis=0)
            transformed_volume = np.prod(bbox_max_transformed - bbox_min_transformed)
            
            # 如果体积变化过大，进行缩放调整
            volume_ratio = transformed_volume / (original_volume + 1e-8)
            if volume_ratio < 0.5 or volume_ratio > 2.0:
                # 计算缩放因子
                scale_factor = np.power(volume_ratio, 1.0/3.0)  # 立方根
                # 计算网格中心
                center = np.mean(transformed_vertices, axis=0)
                # 应用缩放
                transformed_vertices = center + scale_factor * (transformed_vertices - center)
        
        return transformed_vertices
    
    def align_mesh_with_skeleton(self, mesh_vertices, skeleton_transforms):
        """
        将网格顶点与骨骼对齐
        
        Args:
            mesh_vertices: 网格顶点 [N, 3]
            skeleton_transforms: 骨骼变换矩阵 [K, 4, 4]
            
        Returns:
            aligned_vertices: 对齐后的顶点
        """
        # 计算网格中心
        mesh_center = np.mean(mesh_vertices, axis=0)
        
        # 计算骨骼中心（使用所有关节的平均位置）
        joint_positions = skeleton_transforms[:, :3, 3]  # [K, 3]
        skeleton_center = np.mean(joint_positions, axis=0)
        
        # 计算偏移量
        offset = skeleton_center - mesh_center
        
        # 应用偏移
        aligned_vertices = mesh_vertices + offset
        
        return aligned_vertices
    
    def compute_lbs_loss(self, weights_flat, rest_vertices, target_vertices, transforms, 
                        regularization_lambda=0.01):
        """计算LBS损失函数"""
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        weights = weights_flat.reshape(num_vertices, num_joints)
        weights = np.maximum(weights, 0)
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        predicted_vertices = self.apply_lbs_transform(rest_vertices, weights, transforms)
        reconstruction_loss = np.mean(np.sum((predicted_vertices - target_vertices)**2, axis=1))
        sparsity_loss = np.mean(np.sum(weights**2, axis=1))
        
        total_loss = reconstruction_loss + regularization_lambda * sparsity_loss
        return total_loss
    
    def interpolate_skeleton_transforms(self, frame_start, frame_end, t):
        """
        使用相对变换插值骨骼变换（与Skinning.py保持一致）
        
        关键修复：
        1. 使用相对变换而不是绝对变换
        2. 保持与Skinning.py相同的坐标系处理
        3. 确保骨骼长度和姿态正确
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            t: 插值参数 [0, 1]
            
        Returns:
            interpolated_transforms: 插值后的变换矩阵 [num_joints, 4, 4]
        """
        # 获取参考帧（使用起始帧作为参考）
        reference_frame = frame_start
        
        # 获取变换矩阵
        transforms_start = self.transforms[frame_start]  # [num_joints, 4, 4]
        transforms_end = self.transforms[frame_end]      # [num_joints, 4, 4]
        transforms_ref = self.transforms[reference_frame] # [num_joints, 4, 4]
        
        # 计算相对变换（与Skinning.py保持一致）
        relative_transforms_start = np.zeros_like(transforms_start)
        relative_transforms_end = np.zeros_like(transforms_end)
        
        for j in range(self.num_joints):
            # 计算从参考帧到起始帧的相对变换
            if np.linalg.det(transforms_ref[j][:3, :3]) > 1e-6:
                ref_inv = np.linalg.inv(transforms_ref[j])
                relative_transforms_start[j] = transforms_start[j] @ ref_inv
            else:
                relative_transforms_start[j] = np.eye(4)
            
            # 计算从参考帧到结束帧的相对变换
            if np.linalg.det(transforms_ref[j][:3, :3]) > 1e-6:
                ref_inv = np.linalg.inv(transforms_ref[j])
                relative_transforms_end[j] = transforms_end[j] @ ref_inv
            else:
                relative_transforms_end[j] = np.eye(4)
        
        # 插值相对变换
        interpolated_relative_transforms = np.zeros_like(transforms_start)
        
        for j in range(self.num_joints):
            # 提取旋转部分 (3x3)
            R_start = relative_transforms_start[j][:3, :3]
            R_end = relative_transforms_end[j][:3, :3]
            
            # 提取平移部分
            pos_start = relative_transforms_start[j][:3, 3]
            pos_end = relative_transforms_end[j][:3, 3]
            
            # SLERP插值旋转
            quat_start = R.from_matrix(R_start).as_quat()
            quat_end = R.from_matrix(R_end).as_quat()
            
            # 确保四元数在同一半球
            if np.dot(quat_start, quat_end) < 0:
                quat_end = -quat_end
            
            # SLERP插值
            quat_interp = (1-t) * quat_start + t * quat_end
            quat_interp = quat_interp / np.linalg.norm(quat_interp)
            R_interp = R.from_quat(quat_interp).as_matrix()
            
            # 线性插值平移
            pos_interp = (1-t) * pos_start + t * pos_end
            
            # 构建相对变换矩阵
            relative_transform_interp = np.eye(4)
            relative_transform_interp[:3, :3] = R_interp
            relative_transform_interp[:3, 3] = pos_interp
            interpolated_relative_transforms[j] = relative_transform_interp
        
        # 将相对变换转换回绝对变换
        interpolated_transforms = np.zeros_like(transforms_start)
        
        for j in range(self.num_joints):
            # 从参考帧变换到插值帧
            interpolated_transforms[j] = interpolated_relative_transforms[j] @ transforms_ref[j]
        
        return interpolated_transforms
    
    def interpolate_keypoints(self, frame_start, frame_end, t):
        """
        插值关键点位置
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            t: 插值参数 [0, 1]
            
        Returns:
            interpolated_keypoints: 插值后的关键点 [num_joints, 4]
        """
        keypoints_start = self.keypoints[frame_start]  # [num_joints, 4]
        keypoints_end = self.keypoints[frame_end]      # [num_joints, 4]
        
        # 线性插值位置和置信度
        positions_start = keypoints_start[:, :3]
        positions_end = keypoints_end[:, :3]
        positions_interp = (1-t) * positions_start + t * positions_end
        
        # 置信度取最小值（保守策略）
        confidences_start = keypoints_start[:, 3]
        confidences_end = keypoints_end[:, 3]
        confidences_interp = np.minimum(confidences_start, confidences_end)
        
        interpolated_keypoints = np.column_stack([positions_interp, confidences_interp])
        
        return interpolated_keypoints
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, max_optimize_frames = 10,
                                   optimize_weights=True, output_dir=None, debug_frames=None):
        """
        生成插值帧
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_interpolate: 插值帧数
            optimize_weights: 是否优化权重
            output_dir: 输出目录
            debug_frames: 需要调试的帧索引列表（可选）
        """
        print(f"🎬 开始生成插值帧: {frame_start} -> {frame_end} (插值 {num_interpolate} 帧)")
        
        # 验证帧索引
        if frame_start >= self.num_frames or frame_end >= self.num_frames:
            raise ValueError(f"帧索引超出范围: [{0}, {self.num_frames-1}]")
        
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"网格帧索引超出范围: [{0}, {len(self.mesh_files)-1}]")
        
        # 创建输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载参考网格（起始帧）
        reference_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
        reference_vertices = np.asarray(reference_mesh.vertices)
        reference_faces = np.asarray(reference_mesh.triangles) if len(reference_mesh.triangles) > 0 else None
        
        # 改进的归一化策略：计算整体归一化参数
        all_meshes = []
        all_vertices = []
        
        # 收集所有相关帧的网格信息
        frame_indices = [frame_start, frame_end]
        for frame_idx in frame_indices:
            mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
            vertices = np.asarray(mesh.vertices)
            all_meshes.append(mesh)
            all_vertices.append(vertices)
        
        # 计算全局归一化参数
        all_vertices_flat = np.vstack(all_vertices)
        global_bmax = np.amax(all_vertices_flat, axis=0)
        global_bmin = np.amin(all_vertices_flat, axis=0)
        global_blen = (global_bmax - global_bmin).max()
        
        global_normalization_params = {
            'bmin': global_bmin,
            'bmax': global_bmax,
            'blen': global_blen,
            'scale': 1.0,
            'x_trans': 0.0,
            'z_trans': 0.0
        }
        
        # 使用全局参数归一化参考网格
        reference_vertices_norm = self.normalize_mesh_vertices(reference_vertices, global_normalization_params)
        
        # 检查并优化蒙皮权重
        need_optimize = self.check_and_optimize_weights(frame_start, frame_end, num_interpolate)
        
        if need_optimize or optimize_weights:
            print(f"🔧 优化插值蒙皮权重...")
            success = self.optimize_weights_using_skinning(frame_start, frame_end, max_optimize_frames)
            if not success:
                print("⚠️  权重优化失败，将使用简单插值")
                self.skinning_weights = None
        
        # 生成插值帧
        interpolated_frames = []
        interpolation_steps = np.linspace(0, 1, num_interpolate + 2)[1:-1]  # 排除起始和结束帧
        
        print(f"🎨 生成 {len(interpolation_steps)} 个插值帧...")
        
        for i, t in enumerate(tqdm(interpolation_steps, desc="生成插值帧")):
            # 插值骨骼变换
            interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
            
            # 插值关键点
            interpolated_keypoints = self.interpolate_keypoints(frame_start, frame_end, t)
            
            # 应用LBS变换生成网格
            if self.skinning_weights is not None:
                # 确保权重矩阵与顶点数量匹配
                if self.skinning_weights.shape[0] != len(reference_vertices_norm):
                    print(f"⚠️  权重矩阵顶点数 ({self.skinning_weights.shape[0]}) 与参考网格顶点数 ({len(reference_vertices_norm)}) 不匹配")
                    # 调整权重矩阵大小
                    if self.skinning_weights.shape[0] > len(reference_vertices_norm):
                        self.skinning_weights = self.skinning_weights[:len(reference_vertices_norm)]
                    else:
                        # 扩展权重矩阵
                        extended_weights = np.zeros((len(reference_vertices_norm), self.skinning_weights.shape[1]))
                        extended_weights[:self.skinning_weights.shape[0]] = self.skinning_weights
                        # 对新增顶点使用距离初始化
                        keypoints = self.keypoints[frame_start, :, :3]
                        remaining_vertices = reference_vertices_norm[self.skinning_weights.shape[0]:]
                        if len(remaining_vertices) > 0:
                            distances = cdist(remaining_vertices, keypoints)
                            remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                            remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                            extended_weights[self.skinning_weights.shape[0]:] = remaining_weights
                        self.skinning_weights = extended_weights
                
                # 使用与Skinning.py相同的相对变换处理
                print(f"🔧 使用相对变换进行LBS...")
                
                # 获取参考帧变换（使用起始帧作为参考）
                reference_transforms = self.transforms[frame_start]
                
                # 计算从参考帧到插值帧的相对变换
                relative_transforms = np.zeros_like(interpolated_transforms)
                for j in range(self.num_joints):
                    if np.linalg.det(reference_transforms[j][:3, :3]) > 1e-6:
                        ref_inv = np.linalg.inv(reference_transforms[j])
                        relative_transforms[j] = interpolated_transforms[j] @ ref_inv
                    else:
                        relative_transforms[j] = np.eye(4)
                
                # 应用LBS变换（使用相对变换）
                transformed_vertices_norm = self.apply_lbs_transform(
                    reference_vertices_norm, self.skinning_weights, relative_transforms
                )
                
                # 使用全局参数反归一化
                transformed_vertices = self.denormalize_mesh_vertices(
                    transformed_vertices_norm, global_normalization_params
                )
            else:
                # 如果没有权重，使用改进的顶点插值
                mesh_start = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
                mesh_end = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_end]))
                
                vertices_start = np.asarray(mesh_start.vertices)
                vertices_end = np.asarray(mesh_end.vertices)
                
                min_vertices = min(len(vertices_start), len(vertices_end))
                
                # 归一化两个网格
                vertices_start_norm = self.normalize_mesh_vertices(vertices_start[:min_vertices], global_normalization_params)
                vertices_end_norm = self.normalize_mesh_vertices(vertices_end[:min_vertices], global_normalization_params)
                
                # 对齐网格和骨骼
                print(f"🔧 对齐网格和骨骼（无权重模式）...")
                vertices_start_aligned = self.align_mesh_with_skeleton(vertices_start_norm, interpolated_transforms)
                vertices_end_aligned = self.align_mesh_with_skeleton(vertices_end_norm, interpolated_transforms)
                
                # 在归一化空间中进行插值
                interpolated_vertices_norm = (1-t) * vertices_start_aligned + t * vertices_end_aligned
                
                # 反归一化
                transformed_vertices = self.denormalize_mesh_vertices(interpolated_vertices_norm, global_normalization_params)
            
            # 创建插值网格
            interpolated_mesh = o3d.geometry.TriangleMesh()
            interpolated_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            if reference_faces is not None:
                interpolated_mesh.triangles = o3d.utility.Vector3iVector(reference_faces)
            
            # 保存插值帧数据
            frame_data = {
                'frame_idx': i,
                'interpolation_t': t,
                'mesh': interpolated_mesh,
                'transforms': interpolated_transforms,
                'keypoints': interpolated_keypoints,
                'vertices': transformed_vertices
            }
            
            interpolated_frames.append(frame_data)
            
            # 调试特定帧
            if debug_frames is not None and i in debug_frames:
                print(f"\n🔍 调试插值帧 {i} (t={t:.3f})...")
                debug_info = self.debug_interpolation_frame(frame_data, i, output_dir if output_dir else "output/debug")
                
                # 额外检查
                print(f"  额外检查:")
                print(f"    - 网格是否有效: {len(transformed_vertices) > 0}")
                print(f"    - 顶点范围: {np.min(transformed_vertices, axis=0)} -> {np.max(transformed_vertices, axis=0)}")
                print(f"    - 是否有NaN: {np.any(np.isnan(transformed_vertices))}")
                print(f"    - 是否有Inf: {np.any(np.isinf(transformed_vertices))}")
            
            # 保存到文件（如果需要）
            if output_dir:
                mesh_output_path = output_path / f"interpolated_frame_{i:04d}.obj"
                o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
                
                # 保存变换数据
                transform_output_path = output_path / f"interpolated_frame_{i:04d}_transforms.npy"
                np.save(transform_output_path, interpolated_transforms)
                
                keypoints_output_path = output_path / f"interpolated_frame_{i:04d}_keypoints.npy"
                np.save(keypoints_output_path, interpolated_keypoints)
        
        print(f"✅ 插值完成！生成了 {len(interpolated_frames)} 个插值帧")
        
        return interpolated_frames
    
    def denormalize_mesh_vertices(self, normalized_vertices, normalization_params):
        """改进的反归一化网格顶点到原始空间"""
        params = normalization_params
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        
        # 改进的反归一化变换
        # 首先移除偏移
        vertices_no_offset = normalized_vertices - trans_offset
        
        # 从[-1,1]范围转换到[0,1]范围
        vertices_01 = (vertices_no_offset + 1) / 2
        
        # 缩放到原始空间
        denormalized = vertices_01 * (params['blen'] + 1e-8) / params['scale'] + params['bmin']
        
        return denormalized
    
    def visualize_interpolation(self, frame_start, frame_end, num_interpolate, 
                              output_dir=None, save_animation=True, max_optimize_frames = 10,
                              interpolated_frames=None):
        """
        可视化插值结果
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_interpolate: 插值帧数
            output_dir: 输出目录
            save_animation: 是否保存动画
            max_optimize_frames: 最大优化帧数
            interpolated_frames: 已生成的插值帧列表（如果为None则重新生成）
        """
        print(f"🎨 可视化插值结果...")
        
        # 如果没有提供插值帧，则重新生成
        if interpolated_frames is None:
            interpolated_frames = self.generate_interpolated_frames(
                frame_start, frame_end, num_interpolate, 
                max_optimize_frames=max_optimize_frames,
                optimize_weights=True, output_dir=output_dir
            )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return
        
        # 创建可视化
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1200, height=800, visible=not save_animation)
        
        # 加载原始帧进行对比
        original_start_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
        original_end_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_end]))
        
        # 设置颜色
        original_start_mesh.paint_uniform_color([1, 0, 0])  # 红色 - 起始帧
        original_end_mesh.paint_uniform_color([0, 0, 1])    # 蓝色 - 结束帧
        
        # 添加原始帧
        vis.add_geometry(original_start_mesh)
        vis.add_geometry(original_end_mesh)
        
        # 为插值帧设置颜色
        for i, frame_data in enumerate(interpolated_frames):
            mesh = frame_data['mesh']
            # 使用绿色到黄色的渐变
            color_ratio = i / len(interpolated_frames)
            color = [color_ratio, 1 - color_ratio * 0.5, 0]
            mesh.paint_uniform_color(color)
            vis.add_geometry(mesh)
        
        # 设置视角
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 2.0
        
        if save_animation and output_dir:
            output_path = Path(output_dir)
            frames_dir = output_path / "animation_frames"
            frames_dir.mkdir(exist_ok=True)
            
            # 保存动画帧
            for i in range(len(interpolated_frames) + 2):  # +2 for start and end frames
                vis.poll_events()
                vis.update_renderer()
                
                # 保存截图
                img = vis.capture_screen_float_buffer(True)
                img = (np.asarray(img) * 255).astype(np.uint8)
                o3d.io.write_image(str(frames_dir / f"frame_{i:04d}.png"), 
                                 o3d.geometry.Image(img))
            
            print(f"📹 动画帧已保存到: {frames_dir}")
        else:
            # 交互式显示
            vis.run()
        
        vis.destroy_window()
        
        print(f"✅ 可视化完成")
    
    def export_interpolation_sequence(self, frame_start, frame_end, num_interpolate, 
                                    output_dir, format='obj', max_optimize_frames = 10,
                                    interpolated_frames=None):
        """
        导出插值序列
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_interpolate: 插值帧数
            output_dir: 输出目录
            format: 输出格式 ('obj', 'ply', 'stl')
            max_optimize_frames: 最大优化帧数
            interpolated_frames: 已生成的插值帧列表（如果为None则重新生成）
        """
        print(f"📦 导出插值序列...")
        
        # 如果没有提供插值帧，则重新生成
        if interpolated_frames is None:
            interpolated_frames = self.generate_interpolated_frames(
                frame_start, frame_end, num_interpolate, 
                max_optimize_frames=max_optimize_frames,
                optimize_weights=True, output_dir=None
            )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 导出序列
        total_frames = len(interpolated_frames) + 2  # 包括起始和结束帧
        
        # 导出起始帧
        start_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
        start_output_path = output_path / f"frame_{0:06d}.{format}"
        if format == 'obj':
            o3d.io.write_triangle_mesh(str(start_output_path), start_mesh)
        elif format == 'ply':
            o3d.io.write_triangle_mesh(str(start_output_path), start_mesh, write_ascii=False)
        elif format == 'stl':
            o3d.io.write_triangle_mesh(str(start_output_path), start_mesh)
        
        # 导出插值帧
        for i, frame_data in enumerate(interpolated_frames):
            mesh = frame_data['mesh']
            frame_idx = i + 1
            output_file = output_path / f"frame_{frame_idx:06d}.{format}"
            
            if format == 'obj':
                o3d.io.write_triangle_mesh(str(output_file), mesh)
            elif format == 'ply':
                o3d.io.write_triangle_mesh(str(output_file), mesh, write_ascii=False)
            elif format == 'stl':
                o3d.io.write_triangle_mesh(str(output_file), mesh)
        
        # 导出结束帧
        end_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_end]))
        end_output_path = output_path / f"frame_{total_frames-1:06d}.{format}"
        if format == 'obj':
            o3d.io.write_triangle_mesh(str(end_output_path), end_mesh)
        elif format == 'ply':
            o3d.io.write_triangle_mesh(str(end_output_path), end_mesh, write_ascii=False)
        elif format == 'stl':
            o3d.io.write_triangle_mesh(str(end_output_path), end_mesh)
        
        # 保存元数据
        metadata = {
            'frame_start': frame_start,
            'frame_end': frame_end,
            'num_interpolate': num_interpolate,
            'total_frames': total_frames,
            'format': format,
            'skeleton_data_dir': str(self.skeleton_data_dir),
            'mesh_folder_path': str(self.mesh_folder_path),
            'interpolation_method': 'skeleton_slerp_lbs',
            'optimization_frames': list(range(frame_start, frame_end + 1))
        }
        
        metadata_path = output_path / "interpolation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 插值序列导出完成:")
        print(f"  - 输出目录: {output_path}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 格式: {format}")
        print(f"  - 元数据: {metadata_path}")

    def validate_interpolation_quality(self, frame_start, frame_end, interpolated_frames):
        """
        验证插值质量
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            interpolated_frames: 插值帧列表
            
        Returns:
            quality_metrics: 质量指标字典
        """
        print(f"🔍 验证插值质量...")
        
        quality_metrics = {}
        
        # 加载原始帧
        original_start_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
        original_end_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_end]))
        
        original_start_vertices = np.asarray(original_start_mesh.vertices)
        original_end_vertices = np.asarray(original_end_mesh.vertices)
        
        # 计算原始帧的体积
        start_bbox_min = np.min(original_start_vertices, axis=0)
        start_bbox_max = np.max(original_start_vertices, axis=0)
        start_volume = np.prod(start_bbox_max - start_bbox_min)
        
        end_bbox_min = np.min(original_end_vertices, axis=0)
        end_bbox_max = np.max(original_end_vertices, axis=0)
        end_volume = np.prod(end_bbox_max - end_bbox_min)
        
        quality_metrics['original_volumes'] = {
            'start_frame': start_volume,
            'end_frame': end_volume,
            'volume_ratio': end_volume / (start_volume + 1e-8)
        }
        
        # 检查插值帧的体积变化
        interpolated_volumes = []
        for i, frame_data in enumerate(interpolated_frames):
            vertices = frame_data['vertices']
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            volume = np.prod(bbox_max - bbox_min)
            interpolated_volumes.append(volume)
        
        quality_metrics['interpolated_volumes'] = interpolated_volumes
        quality_metrics['volume_stability'] = {
            'min_volume': min(interpolated_volumes),
            'max_volume': max(interpolated_volumes),
            'volume_variance': np.var(interpolated_volumes)
        }
        
        # 检查网格连续性
        continuity_scores = []
        for i in range(len(interpolated_frames) - 1):
            vertices_curr = interpolated_frames[i]['vertices']
            vertices_next = interpolated_frames[i + 1]['vertices']
            
            # 计算相邻帧之间的平均顶点位移
            min_vertices = min(len(vertices_curr), len(vertices_next))
            displacement = np.mean(np.linalg.norm(vertices_curr[:min_vertices] - vertices_next[:min_vertices], axis=1))
            continuity_scores.append(displacement)
        
        quality_metrics['continuity'] = {
            'mean_displacement': np.mean(continuity_scores),
            'max_displacement': np.max(continuity_scores),
            'displacement_variance': np.var(continuity_scores)
        }
        
        # 检查骨骼姿态的自然性
        pose_scores = []
        for frame_data in interpolated_frames:
            transforms = frame_data['transforms']
            
            # 检查骨骼长度的一致性
            bone_lengths = []
            for j in range(1, self.num_joints):  # 跳过根节点
                parent_idx = self.parents[j]
                bone_length = np.linalg.norm(transforms[j][:3, 3] - transforms[parent_idx][:3, 3])
                bone_lengths.append(bone_length)
            
            # 计算骨骼长度的方差（越小越自然）
            bone_length_variance = np.var(bone_lengths)
            pose_scores.append(bone_length_variance)
        
        quality_metrics['pose_naturality'] = {
            'mean_bone_length_variance': np.mean(pose_scores),
            'max_bone_length_variance': np.max(pose_scores)
        }
        
        # 打印质量报告
        print(f"📊 插值质量报告:")
        print(f"  - 原始体积比: {quality_metrics['original_volumes']['volume_ratio']:.3f}")
        print(f"  - 插值体积稳定性: {quality_metrics['volume_stability']['volume_variance']:.6f}")
        print(f"  - 平均顶点位移: {quality_metrics['continuity']['mean_displacement']:.6f}")
        print(f"  - 姿态自然性: {quality_metrics['pose_naturality']['mean_bone_length_variance']:.6f}")
        
        return quality_metrics

    def visualize_skeleton_with_mesh(self, frame_data, output_path=None, frame_idx=None):
        """
        可视化单个插值帧的骨骼和网格
        
        Args:
            frame_data: 插值帧数据
            output_path: 输出路径（可选）
            frame_idx: 帧索引（用于文件名）
        """
        try:
            import open3d as o3d
            
            # 创建可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1200, height=800, visible=False)
            
            # 添加网格
            mesh = frame_data['mesh']
            mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
            vis.add_geometry(mesh)
            
            # 添加骨骼
            transforms = frame_data['transforms']
            keypoints = frame_data['keypoints']
            
            # 绘制关节球体
            for j in range(self.num_joints):
                joint_pos = transforms[j][:3, 3]  # 关节位置
                confidence = keypoints[j, 3]  # 置信度
                
                if confidence > 0.2:  # 只显示高置信度的关节
                    # 创建关节球体
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.translate(joint_pos)
                    sphere.paint_uniform_color([1, 0, 0])  # 红色关节
                    vis.add_geometry(sphere)
                    
                    # 绘制到父关节的连接线
                    if j > 0:  # 非根节点
                        parent_idx = self.parents[j]
                        parent_confidence = keypoints[parent_idx, 3]
                        
                        if parent_confidence > 0.2:
                            parent_pos = transforms[parent_idx][:3, 3]
                            
                            # 创建连接线
                            line_points = [parent_pos, joint_pos]
                            lines = [[0, 1]]
                            line_set = o3d.geometry.LineSet()
                            line_set.points = o3d.utility.Vector3dVector(line_points)
                            line_set.lines = o3d.utility.Vector2iVector(lines)
                            line_set.paint_uniform_color([0, 1, 0])  # 绿色骨骼
                            vis.add_geometry(line_set)
            
            # 设置视角
            vis.get_render_option().point_size = 2.0
            vis.get_render_option().line_width = 3.0
            
            if output_path:
                # 保存图像
                vis.poll_events()
                vis.update_renderer()
                img = vis.capture_screen_float_buffer(True)
                img = (np.asarray(img) * 255).astype(np.uint8)
                o3d.io.write_image(str(output_path), o3d.geometry.Image(img))
                print(f"✅ 骨骼+网格可视化已保存: {output_path}")
            else:
                # 交互式显示
                vis.run()
            
            vis.destroy_window()
            
        except Exception as e:
            print(f"❌ 骨骼可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def debug_interpolation_frame(self, frame_data, frame_idx, output_dir):
        """
        调试单个插值帧
        
        Args:
            frame_data: 插值帧数据
            frame_idx: 帧索引
            output_dir: 输出目录
        """
        print(f"\n🔍 调试插值帧 {frame_idx}...")
        
        # 分析网格
        mesh = frame_data['mesh']
        vertices = np.asarray(mesh.vertices)
        
        print(f"  网格统计:")
        print(f"    - 顶点数: {len(vertices)}")
        print(f"    - 边界框: {np.min(vertices, axis=0)} -> {np.max(vertices, axis=0)}")
        print(f"    - 体积: {np.prod(np.max(vertices, axis=0) - np.min(vertices, axis=0)):.6f}")
        
        # 分析骨骼
        transforms = frame_data['transforms']
        keypoints = frame_data['keypoints']
        
        print(f"  骨骼统计:")
        for j in range(min(5, self.num_joints)):  # 只显示前5个关节
            joint_pos = transforms[j][:3, 3]
            confidence = keypoints[j, 3]
            print(f"    - 关节 {j}: 位置={joint_pos}, 置信度={confidence:.3f}")
        
        # 检查骨骼长度
        print(f"  骨骼长度检查:")
        for j in range(1, min(5, self.num_joints)):
            parent_idx = self.parents[j]
            if parent_idx >= 0:
                bone_length = np.linalg.norm(
                    transforms[j][:3, 3] - transforms[parent_idx][:3, 3]
                )
                print(f"    - 骨骼 {parent_idx}->{j}: 长度={bone_length:.6f}")
        
        # 可视化
        output_path = Path(output_dir) / f"debug_frame_{frame_idx:04d}.png"
        self.visualize_skeleton_with_mesh(frame_data, str(output_path), frame_idx)
        
        return {
            'frame_idx': frame_idx,
            'mesh_vertices': len(vertices),
            'mesh_bbox': (np.min(vertices, axis=0), np.max(vertices, axis=0)),
            'mesh_volume': np.prod(np.max(vertices, axis=0) - np.min(vertices, axis=0)),
            'joint_positions': transforms[:, :3, 3],
            'joint_confidences': keypoints[:, 3]
        }

def main():
    """
    主函数 - 演示插值功能
    """
    print("🎬 体素视频插值系统")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_path = "output/skinning_weights_auto.npz"
    output_dir = "output/interpolation_results"
    
    # 检查数据是否存在
    if not os.path.exists(skeleton_data_dir):
        print(f"❌ 骨骼数据目录不存在: {skeleton_data_dir}")
        print("请先运行 SkelSequencePrediction.py 生成骨骼数据")
        return
    
    if not os.path.exists(mesh_folder_path):
        print(f"❌ 网格文件夹不存在: {mesh_folder_path}")
        return
    
    # 初始化插值器
    print("🔧 初始化插值器...")
    interpolator = VolumetricInterpolator(
        skeleton_data_dir=skeleton_data_dir,
        mesh_folder_path=mesh_folder_path,
        weights_path=weights_path if os.path.exists(weights_path) else None
    )
    
    # 设置插值参数
    frame_start = 10
    frame_end = 20
    num_interpolate = 10
    max_optimize_frames = 10
    
    print(f"📋 插值参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    
    # 执行插值
    try:
        # 生成插值帧（只调用一次）
        print("\n🎬 开始生成插值帧...")
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=max_optimize_frames,
            optimize_weights=True, 
            output_dir=output_dir
        )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return
        
        # 验证插值质量
        print("\n🔍 验证插值质量...")
        quality_metrics = interpolator.validate_interpolation_quality(
            frame_start=frame_start,
            frame_end=frame_end,
            interpolated_frames=interpolated_frames
        )
        
        # 导出插值序列（使用已生成的帧）
        print("\n📦 导出插值序列...")
        interpolator.export_interpolation_sequence(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=max_optimize_frames,
            output_dir=output_dir,
            format='obj',
            interpolated_frames=interpolated_frames
        )
        
        # 可视化插值结果（使用已生成的帧）
        print("\n🎨 可视化插值结果...")
        interpolator.visualize_interpolation(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=max_optimize_frames,
            output_dir=output_dir,
            save_animation=True,
            interpolated_frames=interpolated_frames
        )
        
        print(f"\n🎉 插值完成！")
        print(f"📁 结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"❌ 插值过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
