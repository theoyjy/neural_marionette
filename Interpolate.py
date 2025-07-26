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
        """加载蒙皮权重"""
        try:
            data = np.load(weights_path)
            self.skinning_weights = data['weights']
            print(f"✅ 成功加载蒙皮权重:")
            print(f"  - 权重矩阵形状: {self.skinning_weights.shape}")
            return True
        except Exception as e:
            print(f"❌ 加载蒙皮权重失败: {e}")
            return False

    def optimize_weights_using_skinning(self, frame_start, frame_end, max_optimize_frames=5):
        """
        使用Skinning.py优化权重
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            max_optimize_frames: 最大优化帧数
            
        Returns:
            success: 是否成功
        """
        start_time = time.time()
        
        try:
            from Skinning import AutoSkinning
            
            print(f"🔧 调用Skinning.py进行权重优化...")
            print(f"  - 参考帧: {frame_start}")
            print(f"  - 优化帧范围: {frame_start}-{frame_end}")
            print(f"  - 最大优化帧数: {max_optimize_frames}")
            
            # 生成权重文件路径 - 修复路径问题
            weights_filename = f"skinning_weights_ref{frame_start}_opt{frame_start}-{frame_end}_step1.npz"
            
            # 确保使用正确的输出目录
            if hasattr(self, 'output_dir') and self.output_dir:
                # 如果插值器有output_dir，使用它
                weights_path = Path(self.output_dir) / "skinning_weights" / weights_filename
            else:
                # 否则使用默认路径
                weights_path = Path("output") / "skinning_weights" / weights_filename
            
            print(f"  - 权重文件路径: {weights_path}")
            
            # 检查是否已存在权重文件
            if weights_path.exists():
                print(f"✅ 发现已存在的权重文件: {weights_path}")
                self.load_skinning_weights(str(weights_path))
                return True
            
            # 创建输出目录
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 初始化Skinning系统
            skinner = AutoSkinning(
                skeleton_data_dir=self.skeleton_data_dir,
                reference_frame_idx=frame_start
            )
            
            # 加载网格序列
            skinner.load_mesh_sequence(self.mesh_folder_path)
            
            # 选择优化帧
            optimize_frames = []
            for i in range(frame_start, min(frame_end + 1, frame_start + max_optimize_frames)):
                if i < len(skinner.mesh_files):
                    optimize_frames.append(i)
            
            if not optimize_frames:
                print("⚠️  没有需要优化的帧")
                return False
            
            print(f"  - 优化帧: {optimize_frames}")
            
            # 直接使用Skinning的优化方法
            print(f"   调用Skinning.py的optimize_reference_frame_skinning...")
            optimization_start = time.time()
            
            skinner.skinning_weights = skinner.optimize_reference_frame_skinning(
                optimization_frames=optimize_frames,
                regularization_lambda=0.01,
                max_iter=200  # 适中的迭代次数
            )
            
            optimization_time = time.time() - optimization_start
            
            if skinner.skinning_weights is not None:
                print(f"✅ 权重优化完成")
                print(f"  - 权重矩阵形状: {skinner.skinning_weights.shape}")
                print(f"  - 优化耗时: {optimization_time:.2f}秒")
                
                # 保存权重
                skinner.save_skinning_weights(str(weights_path))
                print(f"  - 权重已保存到: {weights_path}")
                
                # 加载优化后的权重到插值器
                self.load_skinning_weights(str(weights_path))
                print(f"✅ 权重已加载到插值器")
                
                total_time = time.time() - start_time
                print(f"⏱️  总耗时: {total_time:.2f}秒")
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
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None):
        """
        生成插值帧
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_interpolate: 插值帧数
            max_optimize_frames: 最大优化帧数
            optimize_weights: 是否优化权重
            output_dir: 输出目录
            debug_frames: 调试帧列表
            
        Returns:
            interpolated_frames: 插值帧列表
        """
        total_start_time = time.time()
        
        print(f"🎬 开始生成插值帧...")
        print(f"  - 起始帧: {frame_start}")
        print(f"  - 结束帧: {frame_end}")
        print(f"  - 插值帧数: {num_interpolate}")
        print(f"  - 输出目录: {output_dir}")
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 检查帧索引范围
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"帧索引超出范围: {len(self.mesh_files)}")
        
        if frame_start >= frame_end:
            raise ValueError(f"起始帧必须小于结束帧: {frame_start} >= {frame_end}")
        
        # 生成插值参数
        t_values = np.linspace(0, 1, num_interpolate + 2)[1:-1]  # 排除起始和结束帧
        
        interpolated_frames = []
        
        # 权重优化
        if optimize_weights and self.skinning_weights is None:
            print(f"\n🔧 开始权重优化...")
            optimization_start = time.time()
            
            if not self.optimize_weights_using_skinning(frame_start, frame_end, max_optimize_frames):
                print("⚠️  权重优化失败，将使用简单插值")
            
            optimization_time = time.time() - optimization_start
            print(f"⏱️  权重优化总耗时: {optimization_time:.2f}秒")
        
        # 生成插值帧
        print(f"\n🎬 开始生成 {len(t_values)} 个插值帧...")
        frame_generation_start = time.time()
        
        for i, t in enumerate(t_values):
            frame_start_time = time.time()
            print(f"  🔄 生成插值帧 {i+1}/{len(t_values)} (t={t:.3f})...")
            
            try:
                # 插值骨骼变换
                interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
                
                # 生成插值帧数据
                frame_data = self.generate_single_interpolated_frame(
                    frame_start, frame_end, t, interpolated_transforms, output_dir, i
                )
                
                if frame_data:
                    interpolated_frames.append(frame_data)
                    
                    # 调试特定帧
                    if debug_frames and i in debug_frames:
                        self.debug_interpolation_frame(frame_data, i, output_dir)
                    
                    frame_time = time.time() - frame_start_time
                    print(f"    ✅ 完成 (耗时: {frame_time:.2f}秒)")
                else:
                    print(f"    ❌ 生成失败")
                    
            except Exception as e:
                print(f"    ❌ 生成插值帧失败: {e}")
                import traceback
                traceback.print_exc()
        
        frame_generation_time = time.time() - frame_generation_start
        total_time = time.time() - total_start_time
        
        print(f"\n✅ 插值帧生成完成！")
        print(f"  - 生成帧数: {len(interpolated_frames)}")
        print(f"  - 帧生成耗时: {frame_generation_time:.2f}秒")
        print(f"  - 平均每帧: {frame_generation_time/len(t_values):.3f}秒")
        print(f"  - 总耗时: {total_time:.2f}秒")
        
        return interpolated_frames
    
    def generate_single_interpolated_frame(self, frame_start, frame_end, t, interpolated_transforms, output_dir, frame_idx):
        """
        生成单个插值帧
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            t: 插值参数 [0, 1]
            interpolated_transforms: 插值后的变换矩阵
            output_dir: 输出目录
            frame_idx: 帧索引
            
        Returns:
            frame_data: 插值帧数据字典
        """
        # 加载参考网格（起始帧）
        reference_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
        reference_vertices = np.asarray(reference_mesh.vertices)
        reference_faces = np.asarray(reference_mesh.triangles) if len(reference_mesh.triangles) > 0 else None
        
        # 改进的归一化策略：计算整体归一化参数
        all_meshes = []
        all_vertices = []
        
        # 收集所有相关帧的网格信息
        frame_indices = [frame_start, frame_end]
        for idx in frame_indices:  # 修复：使用idx而不是frame_idx避免变量名冲突
            mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[idx]))
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
            print(f"    🔧 使用相对变换进行LBS...")
            
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
            
            # 修复坐标系问题：将骨骼变换到网格坐标系
            print(f"    🔧 修复坐标系对齐...")
            
            # 计算网格中心
            mesh_center = np.mean(transformed_vertices, axis=0)
            
            # 计算骨骼中心（使用插值后的绝对变换）
            joint_positions = interpolated_transforms[:, :3, 3]
            joint_center = np.mean(joint_positions, axis=0)
            
            # 计算偏移量
            offset = mesh_center - joint_center
            
            # 调整骨骼位置到网格坐标系
            adjusted_transforms = interpolated_transforms.copy()
            for j in range(self.num_joints):
                adjusted_transforms[j][:3, 3] += offset
            
            # 更新插值后的变换
            interpolated_transforms = adjusted_transforms
            
            print(f"      - 网格中心: {mesh_center}")
            print(f"      - 调整前骨骼中心: {joint_center}")
            print(f"      - 调整后骨骼中心: {np.mean(adjusted_transforms[:, :3, 3], axis=0)}")
            print(f"      - 偏移量: {offset}")
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
            print(f"    🔧 对齐网格和骨骼（无权重模式）...")
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
        
        # 插值关键点
        interpolated_keypoints = self.interpolate_keypoints(frame_start, frame_end, t)
        
        # 保存插值帧数据
        frame_data = {
            'frame_idx': frame_idx,
            'interpolation_t': t,
            'mesh': interpolated_mesh,
            'transforms': interpolated_transforms,
            'keypoints': interpolated_keypoints,
            'vertices': transformed_vertices
        }
        
        # 保存到文件（如果需要）
        if output_dir:
            mesh_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}.obj"
            o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
            
            # 保存变换数据
            transform_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_transforms.npy"
            np.save(transform_output_path, interpolated_transforms)
            
            keypoints_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_keypoints.npy"
            np.save(keypoints_output_path, interpolated_keypoints)
        
        return frame_data
    
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
    
    def visualize_skeleton_with_mesh(self, frame_data, output_path=None, frame_idx=None):
        """
        可视化单个插值帧的骨骼和网格
        
        修复：确保骨骼和网格在同一个坐标系中
        
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
            
            # 获取网格顶点以确定坐标系
            mesh_vertices = np.asarray(mesh.vertices)
            mesh_center = np.mean(mesh_vertices, axis=0)
            mesh_scale = np.max(mesh_vertices, axis=0) - np.min(mesh_vertices, axis=0)
            
            # 添加骨骼
            transforms = frame_data['transforms']
            keypoints = frame_data['keypoints']
            
            # 检查骨骼是否在正确的坐标系中
            joint_positions = transforms[:, :3, 3]
            joint_center = np.mean(joint_positions, axis=0)
            
            # 如果骨骼和网格中心差距太大，说明坐标系不匹配
            center_distance = np.linalg.norm(joint_center - mesh_center)
            print(f"🔍 坐标系检查:")
            print(f"  - 网格中心: {mesh_center}")
            print(f"  - 骨骼中心: {joint_center}")
            print(f"  - 中心距离: {center_distance:.6f}")
            
            # 如果距离太大，将骨骼变换到网格坐标系
            if center_distance > 1.0:  # 阈值可调整
                print(f"⚠️  检测到坐标系不匹配，调整骨骼位置...")
                
                # 计算偏移量
                offset = mesh_center - joint_center
                
                # 调整所有关节位置
                adjusted_transforms = transforms.copy()
                for j in range(self.num_joints):
                    adjusted_transforms[j][:3, 3] += offset
                
                transforms = adjusted_transforms
                print(f"✅ 骨骼已调整，新中心: {np.mean(transforms[:, :3, 3], axis=0)}")
            
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
            
            print(f"✅ 可视化完成")
            
        except Exception as e:
            print(f"❌ 骨骼可视化失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数 - 用于测试"""
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    # 初始化插值器
    interpolator = VolumetricInterpolator(
        skeleton_data_dir=skeleton_data_dir,
        mesh_folder_path=mesh_folder_path,
        weights_path=None
    )
    
    # 测试参数
    frame_start = 10
    frame_end = 20
    num_interpolate = 5
    
    print(f"🧪 测试插值功能...")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    
    # 生成插值帧
    interpolated_frames = interpolator.generate_interpolated_frames(
        frame_start=frame_start,
        frame_end=frame_end,
        num_interpolate=num_interpolate,
        max_optimize_frames=5,
        optimize_weights=True,
        output_dir="output/test_interpolation"
    )
    
    if interpolated_frames:
        print(f"✅ 插值测试成功！生成了 {len(interpolated_frames)} 个插值帧")
    else:
        print(f"❌ 插值测试失败！")

if __name__ == "__main__":
    main()
