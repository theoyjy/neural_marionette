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
from copy import deepcopy

class VolumetricInterpolator:
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None):
        """
        Initialize volumetric video interpolator
        
        Args:
            skeleton_data_dir: skeleton data directory path
            mesh_folder_path: mesh file directory path
            weights_path: precomputed skinning weights path (optional)
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
            # 对于Neural Marionette，跳过骨骼数据加载
            if hasattr(self, 'use_network_keypoints') and self.use_network_keypoints:
                print(f"Neural Marionette mode - skip loading skeleton data")
                print(f"  - Use network generated keypoints")
                # 设置默认值
                self.num_frames = 40  # demo数据的帧数
                self.num_joints = 24  # default number of joints
                return True
            
            # Load keypoints data [num_frames, num_joints, 4] (x, y, z, confidence)
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # Load transformation matrix [num_frames, num_joints, 4, 4]
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # Load parent node relationship [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            # Load rotation matrix [num_frames, num_joints, 3, 3]
            if (self.skeleton_data_dir / 'rotations.npy').exists():
                self.rotations = np.load(self.skeleton_data_dir / 'rotations.npy')
            else:
                self.rotations = None
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"Successfully loaded skeleton data:")
            print(f"  - Frames: {self.num_frames}")
            print(f"  - Joints: {self.num_joints}")
            print(f"  - Keypoints shape: {self.keypoints.shape}")
            print(f"  - Transforms shape: {self.transforms.shape}")
            
        except Exception as e:
            raise ValueError(f"Cannot load skeleton data: {e}")
    
    def load_mesh_sequence(self):
        """Load mesh sequence"""
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        
        if len(self.mesh_files) == 0:
            raise ValueError(f"No obj files found in {self.mesh_folder_path}")
        
        print(f"Successfully loaded mesh sequence:")
        print(f"  - Mesh files: {len(self.mesh_files)}")
        print(f"  - Skeleton frames: {self.num_frames}")
        
        if len(self.mesh_files) != self.num_frames:
            print(f"Warning: Mesh file count ({len(self.mesh_files)}) doesn't match skeleton frame count ({self.num_frames})")
    
    def load_skinning_weights(self, weights_path):
        """Load skinning weights"""
        try:
            data = np.load(weights_path)
            self.skinning_weights = data['weights']
            
            # 🔧 修复：确保reference_frame_idx被正确设置
            if 'reference_frame_idx' in data:
                self.reference_frame_idx = data['reference_frame_idx'].item()
            else:
                # 如果权重文件中没有reference_frame_idx，尝试从文件名推断
                import re
                match = re.search(r'ref(\d+)_', str(weights_path))
                if match:
                    self.reference_frame_idx = int(match.group(1))
                    print(f"⚠️  从文件名推断参考帧: {self.reference_frame_idx}")
                else:
                    self.reference_frame_idx = 0  # 默认使用0
                    print(f"⚠️  使用默认参考帧: {self.reference_frame_idx}")
            
            print(f"Successfully loaded skinning weights:")
            print(f"  - Weight matrix shape: {self.skinning_weights.shape}")
            print(f"  - Reference frame: {self.reference_frame_idx}")
            
            return True
        except Exception as e:
            print(f"Failed to load skinning weights: {e}")
            import traceback
            traceback.print_exc()
            return False

    def optimize_weights_using_skinning(self, frame_start, frame_end, max_optimize_frames=5):
        """
        ✅ 恢复Work版本的简单权重优化逻辑
        
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
            
            # 生成权重文件路径 - 与Work版本一致
            weights_filename = f"skinning_weights_ref{frame_start}_opt{frame_start}-{frame_end}_step1.npz"
            
                        # 🔧 优先使用work版本权重文件（动作效果更好）
            work_weights_filename = "skinning_weights_ref0_work.npz"
            sparse_weights_filename = "skinning_weights_ref0_sparse.npz"

            # 确保使用正确的输出目录
            if hasattr(self, 'output_dir') and self.output_dir:
                # 如果插值器有output_dir，使用它
                weights_path = Path(self.output_dir) / "skinning_weights" / weights_filename
                work_weights_path = Path(self.output_dir) / "skinning_weights" / work_weights_filename
                sparse_weights_path = Path(self.output_dir) / "skinning_weights" / sparse_weights_filename
            else:
                # 否则使用默认路径
                weights_path = Path("output") / "skinning_weights" / weights_filename
                work_weights_path = Path("output") / "skinning_weights" / work_weights_filename
                sparse_weights_path = Path("output") / "skinning_weights" / sparse_weights_filename

            # 优先顺序：work版本 > 稀疏版本 > 原始版本
            if work_weights_path.exists():
                print(f"🎯 使用work版本权重文件（动作效果更好）: {work_weights_path}")
                weights_path = work_weights_path
            elif sparse_weights_path.exists():
                print(f"🎯 使用稀疏权重文件: {sparse_weights_path}")
                weights_path = sparse_weights_path
            
            print(f"  - 权重文件路径: {weights_path}")
            
            # 检查是否已存在权重文件
            if weights_path.exists():
                print(f"✅ 发现已存在的权重文件: {weights_path}")
                print(f"🔍 DEBUG: 权重文件信息:")
                print(f"  - 文件大小: {weights_path.stat().st_size} bytes")
                success = self.load_skinning_weights(str(weights_path))
                if success:
                    print(f"✅ 权重文件加载成功")
                    print(f"  - 参考帧: {getattr(self, 'reference_frame_idx', 'Unknown')}")
                    print(f"  - 权重形状: {self.skinning_weights.shape if self.skinning_weights is not None else 'None'}")
            else:
                print(f"❌ 权重文件加载失败，将重新优化")
                # 如果加载失败，删除损坏的文件并重新优化
                weights_path.unlink(missing_ok=True)
                return False
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
        """Compute mesh normalization parameters"""
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
        """Normalize mesh vertices"""
        params = normalization_params
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        normalized = ((vertices - params['bmin']) * params['scale'] / (params['blen'] + 1e-5)) * 2 - 1 + trans_offset
        return normalized
    
    def apply_lbs_transform(self, rest_vertices, weights, transforms):
        """Apply improved Linear Blend Skinning transformation, keep mesh volume and skeleton aligned"""
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        rest_vertices_homo = np.hstack([rest_vertices, np.ones((num_vertices, 1))])
        transformed_vertices = np.zeros((num_vertices, 3))
        
        # Improved weight processing: ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-8)
        
        # Compute contribution of each joint's transformation
        joint_contributions = []
        for j in range(num_joints):
            joint_transform = transforms[j]
            transformed_homo = (joint_transform @ rest_vertices_homo.T).T
            transformed_xyz = transformed_homo[:, :3]
            joint_weights = weights[:, j:j+1]
            joint_contributions.append(joint_weights * transformed_xyz)
        
        # Apply weight mixing
        for contribution in joint_contributions:
            transformed_vertices += contribution
        
        # Volume preservation: compute volume features of the original mesh
        if num_vertices > 3:
            # Compute bounding box of the original mesh
            bbox_min = np.min(rest_vertices, axis=0)
            bbox_max = np.max(rest_vertices, axis=0)
            original_volume = np.prod(bbox_max - bbox_min)
            
            # Compute bounding box of the transformed mesh
            bbox_min_transformed = np.min(transformed_vertices, axis=0)
            bbox_max_transformed = np.max(transformed_vertices, axis=0)
            transformed_volume = np.prod(bbox_max_transformed - bbox_min_transformed)
            
            # If volume change is too large, perform scaling adjustment
            volume_ratio = transformed_volume / (original_volume + 1e-8)
            if volume_ratio < 0.5 or volume_ratio > 2.0:
                # Compute scaling factor
                scale_factor = np.power(volume_ratio, 1.0/3.0)  # 立方根
                # Compute mesh center
                center = np.mean(transformed_vertices, axis=0)
                # Apply scaling
                transformed_vertices = center + scale_factor * (transformed_vertices - center)
        
        return transformed_vertices
    
    def align_mesh_with_skeleton(self, mesh_vertices, skeleton_transforms):
        """
        Align mesh vertices with skeleton
        
        Args:
            mesh_vertices: mesh vertices [N, 3]
            skeleton_transforms: skeleton transformation matrix [K, 4, 4]
            
        Returns:
            aligned_vertices: aligned vertices
        """
        # Compute mesh center
        mesh_center = np.mean(mesh_vertices, axis=0)
        
        # Compute skeleton center (using average position of all joints)
        joint_positions = skeleton_transforms[:, :3, 3]  # [K, 3]
        skeleton_center = np.mean(joint_positions, axis=0)
        
        # Compute offset
        offset = skeleton_center - mesh_center
        
        # Apply offset
        aligned_vertices = mesh_vertices + offset
        
        return aligned_vertices
    
    def interpolate_skeleton_transforms(self, frame_start, frame_end, t):
        """
        Interpolate skeleton transforms using relative transformation (consistent with Skinning.py)
        
        Key fixes:
        1. Use relative transformation instead of absolute transformation
        2. Keep consistent coordinate system processing with Skinning.py
        3. Ensure correct skeleton length and pose
        
        Args:
            frame_start: start frame index
            frame_end: end frame index
            t: interpolation parameter [0, 1]
            
        Returns:
            interpolated_transforms: interpolated transformation matrix [num_joints, 4, 4]
        """
        # Get reference frame (use start frame as reference)
        reference_frame = frame_start
        
        # Get transformation matrix
        transforms_start = self.transforms[frame_start]  # [num_joints, 4, 4]
        transforms_end = self.transforms[frame_end]      # [num_joints, 4, 4]
        transforms_ref = self.transforms[reference_frame] # [num_joints, 4, 4]
        
        # Compute relative transformation (consistent with Skinning.py)
        relative_transforms_start = np.zeros_like(transforms_start)
        relative_transforms_end = np.zeros_like(transforms_end)
        
        for j in range(self.num_joints):
            # Compute relative transformation from reference frame to start frame
            if np.linalg.det(transforms_ref[j][:3, :3]) > 1e-6:
                ref_inv = np.linalg.inv(transforms_ref[j])
                relative_transforms_start[j] = transforms_start[j] @ ref_inv
            else:
                relative_transforms_start[j] = np.eye(4)
            
            # Compute relative transformation from reference frame to end frame
            if np.linalg.det(transforms_ref[j][:3, :3]) > 1e-6:
                ref_inv = np.linalg.inv(transforms_ref[j])
                relative_transforms_end[j] = transforms_end[j] @ ref_inv
            else:
                relative_transforms_end[j] = np.eye(4)
        
        # Interpolate relative transformation
        interpolated_relative_transforms = np.zeros_like(transforms_start)
        
        for j in range(self.num_joints):
            # Extract rotation part (3x3)
            R_start = relative_transforms_start[j][:3, :3]
            R_end = relative_transforms_end[j][:3, :3]
            
            # Extract translation part
            pos_start = relative_transforms_start[j][:3, 3]
            pos_end = relative_transforms_end[j][:3, 3]
            
            # SLERP interpolation of rotation
            quat_start = R.from_matrix(R_start).as_quat()
            quat_end = R.from_matrix(R_end).as_quat()
            
            # Ensure quaternions are in the same hemisphere
            if np.dot(quat_start, quat_end) < 0:
                quat_end = -quat_end
            
            # SLERP interpolation
            quat_interp = (1-t) * quat_start + t * quat_end
            quat_interp = quat_interp / np.linalg.norm(quat_interp)
            R_interp = R.from_quat(quat_interp).as_matrix()
            
            # Linear interpolation of translation
            pos_interp = (1-t) * pos_start + t * pos_end
            
            # Build relative transformation matrix
            relative_transform_interp = np.eye(4)
            relative_transform_interp[:3, :3] = R_interp
            relative_transform_interp[:3, 3] = pos_interp
            interpolated_relative_transforms[j] = relative_transform_interp
        
        # Convert relative transformation back to absolute transformation
        interpolated_transforms = np.zeros_like(transforms_start)
        
        for j in range(self.num_joints):
            # Transform from reference frame to interpolated frame
            interpolated_transforms[j] = interpolated_relative_transforms[j] @ transforms_ref[j]
        
        return interpolated_transforms
    
    def interpolate_skeleton_transforms_with_reference(self, frame_start, frame_end, t, reference_frame):
        """
        Interpolate skeleton transforms using specified reference frame
        
        Args:
            frame_start: start frame index
            frame_end: end frame index
            t: interpolation parameter [0, 1]
            reference_frame: reference frame index
            
        Returns:
            interpolated_transforms: interpolated transformation matrix [num_joints, 4, 4]
        """
        # Get transformation matrix
        transforms_start = self.transforms[frame_start]  # [num_joints, 4, 4]
        transforms_end = self.transforms[frame_end]      # [num_joints, 4, 4]
        transforms_ref = self.transforms[reference_frame] # [num_joints, 4, 4]
        
        # Compute relative transformation (using specified reference frame)
        relative_transforms_start = np.zeros_like(transforms_start)
        relative_transforms_end = np.zeros_like(transforms_end)
        
        for j in range(self.num_joints):
            # Compute relative transformation from reference frame to start frame
            if np.linalg.det(transforms_ref[j][:3, :3]) > 1e-6:
                ref_inv = np.linalg.inv(transforms_ref[j])
                relative_transforms_start[j] = transforms_start[j] @ ref_inv
            else:
                relative_transforms_start[j] = np.eye(4)
            
            # Compute relative transformation from reference frame to end frame
            if np.linalg.det(transforms_ref[j][:3, :3]) > 1e-6:
                ref_inv = np.linalg.inv(transforms_ref[j])
                relative_transforms_end[j] = transforms_end[j] @ ref_inv
            else:
                relative_transforms_end[j] = np.eye(4)
        
        # Interpolate relative transformation
        interpolated_relative_transforms = np.zeros_like(transforms_start)
        
        for j in range(self.num_joints):
            # Extract rotation part (3x3)
            R_start = relative_transforms_start[j][:3, :3]
            R_end = relative_transforms_end[j][:3, :3]
            
            # Extract translation part
            pos_start = relative_transforms_start[j][:3, 3]
            pos_end = relative_transforms_end[j][:3, 3]
            
            # SLERP interpolation of rotation
            quat_start = R.from_matrix(R_start).as_quat()
            quat_end = R.from_matrix(R_end).as_quat()
            
            # Ensure quaternions are in the same hemisphere
            if np.dot(quat_start, quat_end) < 0:
                quat_end = -quat_end
            
            # SLERP interpolation
            quat_interp = (1-t) * quat_start + t * quat_end
            quat_interp = quat_interp / np.linalg.norm(quat_interp)
            R_interp = R.from_quat(quat_interp).as_matrix()
            
            # Linear interpolation of translation
            pos_interp = (1-t) * pos_start + t * pos_end
            
            # Build relative transformation matrix
            relative_transform_interp = np.eye(4)
            relative_transform_interp[:3, :3] = R_interp
            relative_transform_interp[:3, 3] = pos_interp
            interpolated_relative_transforms[j] = relative_transform_interp
        
        # Convert relative transformation back to absolute transformation
        interpolated_transforms = np.zeros_like(transforms_start)
        
        for j in range(self.num_joints):
            # Transform from reference frame to interpolated frame
            interpolated_transforms[j] = interpolated_relative_transforms[j] @ transforms_ref[j]
        
        return interpolated_transforms
    
    def interpolate_keypoints(self, frame_start, frame_end, t):
        """
        Interpolate keypoints positions
        
        Args:
            frame_start: start frame index
            frame_end: end frame index
            t: interpolation parameter [0, 1]
            
        Returns:
            interpolated_keypoints: interpolated keypoints [num_joints, 4]
        """
        keypoints_start = self.keypoints[frame_start]  # [num_joints, 4]
        keypoints_end = self.keypoints[frame_end]      # [num_joints, 4]
        
        # Linear interpolation of positions and confidences
        positions_start = keypoints_start[:, :3]
        positions_end = keypoints_end[:, :3]
        positions_interp = (1-t) * positions_start + t * positions_end
        
        # Take minimum confidence (conservative strategy)
        confidences_start = keypoints_start[:, 3]
        confidences_end = keypoints_end[:, 3]
        confidences_interp = np.minimum(confidences_start, confidences_end)
        
        interpolated_keypoints = np.column_stack([positions_interp, confidences_interp])
        
        return interpolated_keypoints
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None, smooth_mesh=False, subdivide_iter=3,
                                   use_vertex_colors=False, save_npy_files=False, save_standard_obj=True):
        """
        Generate interpolated frames
        
        Args:
            frame_start: start frame index
            frame_end: end frame index
            num_interpolate: number of interpolation frames
            max_optimize_frames: maximum number of optimization frames
            optimize_weights: whether to optimize weights
            output_dir: output directory
            debug_frames: debug frames list
            smooth_mesh: whether to smooth the mesh
            subdivide_iter: number of subdivision iterations
            use_vertex_colors: whether to add vertex colors
            save_npy_files: whether to save npy files (usually not needed)
            save_standard_obj: whether to save standard obj files (avoid duplicates)
            
        Returns:
            interpolated_frames: interpolated frames list
        """
        total_start_time = time.time()
        
        print(f"Start generating interpolation frames...")
        print(f"  - Start frame: {frame_start}")
        print(f"  - End frame: {frame_end}")
        print(f"  - Interpolation frames: {num_interpolate}")
        print(f"  - Output directory: {output_dir}")
        
        # 设置输出目录
        if output_dir:
            # 保存插值输出目录
            self.interpolation_output_dir = output_dir
            # 确保权重文件保存在统一的skinning_weights目录中
            if hasattr(self, 'output_dir') and self.output_dir:
                # 使用已设置的基础输出目录
                pass
            else:
                # 如果没有设置基础输出目录，从插值目录推断
                interpolation_path = Path(output_dir)
                self.output_dir = str(interpolation_path.parent.parent)
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 检查帧索引范围（frame_start和frame_end是排序后文件列表的索引）
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"Frame index out of range: start_frame={frame_start}, end_frame={frame_end}, available frames={len(self.mesh_files)}")
        
        # 检查帧索引是否相等（不允许相等）
        if frame_start == frame_end:
            raise ValueError(f"Start frame cannot be equal to end frame: {frame_start} == {frame_end}")
        
        # 确定实际的起始和结束帧（支持反向插值）
        actual_start = min(frame_start, frame_end)
        actual_end = max(frame_start, frame_end)
        is_reverse = frame_start > frame_end
        
        # 打印实际使用的文件信息
        start_file = self.mesh_files[actual_start].name
        end_file = self.mesh_files[actual_end].name
        print(f"  - Using files: {start_file} (index {actual_start}) -> {end_file} (index {actual_end})")
        
        # 生成插值参数
        t_values = np.linspace(0, 1, num_interpolate + 2)[1:-1]  # 排除起始和结束帧
        
        # 如果是反向插值，反转t值
        if is_reverse:
            t_values = 1.0 - t_values
            print(f"  - Reverse interpolation detected: {frame_start} -> {frame_end}")
            print(f"  - Using actual frame range: {actual_start} -> {actual_end}")
        
        interpolated_frames = []
        
        # ✅ 修复：权重优化也要使用真正的起始帧作为参考！
        print(f"\n🔍 DEBUG: 权重优化检查:")
        print(f"  - optimize_weights: {optimize_weights}")
        print(f"  - self.skinning_weights is None: {self.skinning_weights is None}")
        print(f"  - 将调用权重优化: {optimize_weights and self.skinning_weights is None}")
        
        if optimize_weights and self.skinning_weights is None:
            print(f"\n✅ 开始权重优化（使用真正的起始帧 {frame_start} 作为参考）...")
            optimization_start = time.time()
            
            if not self.optimize_weights_using_skinning(frame_start, frame_end, max_optimize_frames):
                print("⚠️  权重优化失败，将使用简单插值")
            
            optimization_time = time.time() - optimization_start
            print(f"⏱️  权重优化总耗时: {optimization_time:.2f}秒")
        elif self.skinning_weights is not None:
            print(f"\n✅ 使用已有权重矩阵")
            print(f"  - 权重矩阵形状: {self.skinning_weights.shape}")
            print(f"  - 参考帧索引: {getattr(self, 'reference_frame_idx', 'Unknown')}")
        else:
            print(f"\n⚠️  跳过权重优化（optimize_weights=False）")
        
        # 生成插值帧
        print(f"\nStart generating {len(t_values)} interpolation frames...")
        frame_generation_start = time.time()
        
        for i, t in enumerate(t_values):
            frame_start_time = time.time()
            print(f"  Generate interpolation frame {i+1}/{len(t_values)} (t={t:.3f})...")
            
            try:
                # ✅ 修复：恢复Work版本的参数传递方式
                # 使用原始的frame_start, frame_end而不是actual_start, actual_end
                # 这样确保真正的起始帧被用作参考
                interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
                
                # 生成插值帧数据 - 使用原始帧索引
                frame_data = self.generate_single_interpolated_frame(
                    frame_start, frame_end, t, interpolated_transforms, output_dir, i,
                    smooth_mesh, subdivide_iter, save_npy_files, save_standard_obj
                )
                
                if frame_data:
                    interpolated_frames.append(frame_data)
                    
                    # 调试特定帧
                    if debug_frames and i in debug_frames:
                        self.debug_interpolation_frame(frame_data, i, output_dir)
                    
                    frame_time = time.time() - frame_start_time
                    print(f"    Completed (time: {frame_time:.2f} seconds)")
                else:
                    print(f"    Generation failed")
                    
            except Exception as e:
                print(f"    Generate interpolation frame failed: {e}")
                import traceback
                traceback.print_exc()
        
        frame_generation_time = time.time() - frame_generation_start
        total_time = time.time() - total_start_time
        
        print(f"\nInterpolation frame generation completed!")
        print(f"  - Generated frames: {len(interpolated_frames)}")
        print(f"  - Frame generation time: {frame_generation_time:.2f} seconds")
        print(f"  - Average per frame: {frame_generation_time/len(t_values):.3f} seconds")
        print(f"  - Total time: {total_time:.2f} seconds")
        
        return interpolated_frames
    
    def generate_single_interpolated_frame(self, frame_start, frame_end, t, interpolated_transforms, output_dir, frame_idx, smooth_mesh=False, subdivide_iter=3, save_npy_files=False, save_standard_obj=True):
        """
        生成单个插值帧 - 恢复到Work版本的简单正确逻辑
        
        ✅ 正确的Volumetric Interpolation流程:
        1. Learn start frame weights  
        2. Start mesh with texture/colors
        3. Interpolate skeleton
        4. Apply relative transform via LBS
        5. Result: deformed start mesh with colors
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            t: 插值参数 [0, 1]
            interpolated_transforms: 插值后的变换矩阵
            output_dir: 输出目录
            frame_idx: 帧索引
        """
        # ✅ 步骤1: 加载参考网格（固定使用起始帧 - 这是关键！）
        reference_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
        reference_vertices = np.asarray(reference_mesh.vertices)
        reference_faces = np.asarray(reference_mesh.triangles) if len(reference_mesh.triangles) > 0 else None
        
        # 计算全局归一化参数（用于一致的空间处理）
        all_meshes = []
        all_vertices = []
        
        # 收集相关帧的网格信息
        frame_indices = [frame_start, frame_end]
        for idx in frame_indices:
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
        
        # ✅ 步骤2: 应用LBS变换生成网格
        print(f"🔍 DEBUG: skinning_weights状态检查:")
        print(f"  - self.skinning_weights is not None: {self.skinning_weights is not None}")
        if self.skinning_weights is not None:
            print(f"  - 权重矩阵形状: {self.skinning_weights.shape}")
            print(f"  - 权重矩阵范围: [{self.skinning_weights.min():.6f}, {self.skinning_weights.max():.6f}]")
        else:
            print(f"  ❌ 没有权重矩阵！将使用fallback插值")
        
        if self.skinning_weights is not None:
            # 确保权重矩阵与顶点数量匹配（简单处理）
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
            
            # ✅ 步骤3: 使用相对变换进行LBS（关键逻辑！）
            print(f"✅ 使用相对变换进行LBS...")
            print(f"🔍 DEBUG: 变换信息:")
            print(f"  - frame_start: {frame_start}, frame_end: {frame_end}, t: {t}")
            
            # 获取参考帧变换（固定使用起始帧作为参考）
            reference_transforms = self.transforms[frame_start]
            print(f"  - reference_transforms形状: {reference_transforms.shape}")
            print(f"  - interpolated_transforms形状: {interpolated_transforms.shape}")
            
            # 检查插值变换是否真的在变化
            transforms_diff = np.linalg.norm(interpolated_transforms - reference_transforms)
            print(f"  - 插值变换与参考变换的差异magnitude: {transforms_diff:.6f}")
            if transforms_diff < 1e-6:
                print(f"  ⚠️  警告: 插值变换几乎没有变化！")
            
            # 计算从参考帧到插值帧的相对变换
            relative_transforms = np.zeros_like(interpolated_transforms)
            for j in range(self.num_joints):
                if np.linalg.det(reference_transforms[j][:3, :3]) > 1e-6:
                    ref_inv = np.linalg.inv(reference_transforms[j])
                    relative_transforms[j] = interpolated_transforms[j] @ ref_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # 检查相对变换
            relative_magnitude = np.linalg.norm(relative_transforms - np.eye(4))
            print(f"  - 相对变换magnitude: {relative_magnitude:.6f}")
            if relative_magnitude < 1e-6:
                print(f"  ⚠️  警告: 相对变换几乎是单位矩阵！")
            
            # 应用LBS变换（使用相对变换）
            print(f"🔍 DEBUG: LBS变换前顶点统计:")
            print(f"  - reference_vertices_norm 形状: {reference_vertices_norm.shape}")
            print(f"  - reference_vertices_norm 范围: [{reference_vertices_norm.min():.3f}, {reference_vertices_norm.max():.3f}]")
            
            transformed_vertices_norm = self.apply_lbs_transform(
                reference_vertices_norm, self.skinning_weights, relative_transforms
            )
            
            print(f"🔍 DEBUG: LBS变换后顶点统计:")
            print(f"  - transformed_vertices_norm 形状: {transformed_vertices_norm.shape}")
            print(f"  - transformed_vertices_norm 范围: [{transformed_vertices_norm.min():.3f}, {transformed_vertices_norm.max():.3f}]")
            
            # 检查变换是否真的发生了
            vertices_diff = np.linalg.norm(transformed_vertices_norm - reference_vertices_norm)
            print(f"  - 顶点变化magnitude: {vertices_diff:.6f}")
            if vertices_diff < 1e-6:
                print(f"  ⚠️  警告: 顶点几乎没有变化！LBS可能没有生效")
            
            # 使用全局参数反归一化
            transformed_vertices = self.denormalize_mesh_vertices(
                transformed_vertices_norm, global_normalization_params
            )
            
            # ✅ 步骤4: 修复坐标系对齐
            print(f"✅ 修复坐标系对齐...")
            
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
            
            print(f"  - 网格中心: {mesh_center}")
            print(f"  - 调整前骨骼中心: {joint_center}")
            print(f"  - 调整后骨骼中心: {np.mean(adjusted_transforms[:, :3, 3], axis=0)}")
            print(f"  - 偏移量: {offset}")
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
            print(f"对齐网格和骨骼（无权重模式）...")
            vertices_start_aligned = self.align_mesh_with_skeleton(vertices_start_norm, interpolated_transforms)
            vertices_end_aligned = self.align_mesh_with_skeleton(vertices_end_norm, interpolated_transforms)
            
            # 在归一化空间中进行插值
            interpolated_vertices_norm = (1-t) * vertices_start_aligned + t * vertices_end_aligned
            
            # 反归一化
            transformed_vertices = self.denormalize_mesh_vertices(interpolated_vertices_norm, global_normalization_params)
        
        # ✅ 步骤5: 创建插值网格（简单直接）
        interpolated_mesh = o3d.geometry.TriangleMesh()
        interpolated_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        if reference_faces is not None:
            interpolated_mesh.triangles = o3d.utility.Vector3iVector(reference_faces)
        
        # 插值关键点
        interpolated_keypoints = self.interpolate_keypoints(frame_start, frame_end, t)
        
        # ✅ 步骤6: 保存插值帧数据
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
            
            print(f"🔍 DEBUG: 保存插值文件:")
            print(f"  - 输出路径: {mesh_output_path}")
            print(f"  - 网格顶点数: {len(transformed_vertices)}")
            print(f"  - 网格面数: {len(reference_faces) if reference_faces is not None else 0}")
            
            try:
                # 确保输出目录存在
                mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存mesh
                success = o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
                if success:
                    print(f"  ✅ mesh文件保存成功: {mesh_output_path}")
                else:
                    print(f"  ❌ mesh文件保存失败: {mesh_output_path}")
                
                # 保存变换数据
                transform_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_transforms.npy"
                np.save(transform_output_path, interpolated_transforms)
                print(f"  ✅ 变换数据保存: {transform_output_path}")
                
                keypoints_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_keypoints.npy"
                np.save(keypoints_output_path, interpolated_keypoints)
                print(f"  ✅ 关键点数据保存: {keypoints_output_path}")
                
            except Exception as e:
                print(f"  ❌ 保存文件时出错: {e}")
                import traceback
                traceback.print_exc()
        
        return frame_data
    
    def _generate_frame_core(self, frame_start, frame_end, t, interpolated_transforms, 
                           output_dir, frame_idx, smooth_mesh=False, subdivide_iter=3, 
                           save_npy_files=False, save_standard_obj=True,
                           reference_frame=None, skinning_weights=None):
        """
        核心插值帧生成逻辑（可被不同插值方法复用）
        
        Args:
            reference_frame: 参考帧索引，如果为None则使用frame_start
            skinning_weights: 蒙皮权重矩阵，如果为None则使用self.skinning_weights
        """
        if reference_frame is None:
            reference_frame = frame_start
            
        # 准备帧数据
        frame_data = self._prepare_frame_data(frame_start, frame_end, reference_frame)
        reference_vertices = frame_data['reference_vertices']
        reference_faces = frame_data['reference_faces']
        reference_uvs = frame_data['reference_uvs']
        reference_vertex_colors = frame_data['reference_vertex_colors']
        global_normalization_params = frame_data['global_normalization_params']
        
        # 处理蒙皮权重
        processed_weights = self._process_skinning_weights(
            reference_vertices, reference_frame, global_normalization_params, skinning_weights
        )
        
        # 应用LBS变换生成网格
        if processed_weights is not None:
            # 应用LBS变形
            transformed_vertices = self._apply_lbs_deformation(
                reference_vertices, processed_weights, interpolated_transforms, 
                reference_frame, global_normalization_params
            )
            
            # 应用坐标系对齐
            transformed_vertices, interpolated_transforms = self._apply_coordinate_alignment(
                transformed_vertices, interpolated_transforms
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
            print(f"Align mesh and skeleton (no weight mode)...")
            vertices_start_aligned = self.align_mesh_with_skeleton(vertices_start_norm, interpolated_transforms)
            vertices_end_aligned = self.align_mesh_with_skeleton(vertices_end_norm, interpolated_transforms)
            
            # 在归一化空间中进行插值
            interpolated_vertices_norm = (1-t) * vertices_start_aligned + t * vertices_end_aligned
            
            # 反归一化
            transformed_vertices = self.denormalize_mesh_vertices(interpolated_vertices_norm, global_normalization_params)
        
        # 创建输出网格
        interpolated_mesh = self._create_output_mesh(
            transformed_vertices, reference_faces, reference_uvs, reference_vertex_colors,
            smooth_mesh, subdivide_iter
        )
        
        # 插值关键点
        interpolated_keypoints = self.interpolate_keypoints(frame_start, frame_end, t)
        
        # 保存插值帧数据
        frame_result = {
            'frame_idx': frame_idx,
            'interpolation_t': t,
            'mesh': interpolated_mesh,
            'transforms': interpolated_transforms,
            'keypoints': interpolated_keypoints,
            'vertices': transformed_vertices
        }
        
        # 保存到文件（如果需要）
        if output_dir:
            # 只在需要时保存标准obj文件
            if save_standard_obj:
                mesh_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}.obj"
                o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
            
            # 只在需要时保存变换数据
            if save_npy_files:
                transform_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_transforms.npy"
                np.save(transform_output_path, interpolated_transforms)
                
                keypoints_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_keypoints.npy"
                np.save(keypoints_output_path, interpolated_keypoints)
        
        return frame_result
    
    def denormalize_mesh_vertices(self, normalized_vertices, normalization_params):
        """Improved denormalization of mesh vertices to original space"""
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
            print(f"Coordinate system check:")
            print(f"  - Mesh center: {mesh_center}")
            print(f"  - Skeleton center: {joint_center}")
            print(f"  - Center distance: {center_distance:.6f}")
            
            # 如果距离太大，将骨骼变换到网格坐标系
            if center_distance > 1.0:  # 阈值可调整
                print(f"Detected coordinate system mismatch, adjusting skeleton position...")
                
                # 计算偏移量
                offset = mesh_center - joint_center
                
                # 调整所有关节位置
                adjusted_transforms = transforms.copy()
                for j in range(self.num_joints):
                    adjusted_transforms[j][:3, 3] += offset
                
                transforms = adjusted_transforms
                print(f"Skeleton adjusted, new center: {np.mean(transforms[:, :3, 3], axis=0)}")
            
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
                print(f"Skeleton+mesh visualization saved: {output_path}")
            else:
                # 交互式显示
                vis.run()
            
            vis.destroy_window()
            
            print(f"Visualization completed")
            
        except Exception as e:
            print(f"Skeleton visualization failed: {e}")
            import traceback
            traceback.print_exc()

    def smooth_mesh(self, mesh, subdivide_iter=3):
        """
        对网格进行细分平滑处理（类似nmario的实现）
        
        Args:
            mesh: Open3D网格对象
            subdivide_iter: 细分迭代次数，默认为3
            
        Returns:
            smoothed_mesh: 平滑后的网格
        """
        try:
            # 创建临时网格副本
            temp_mesh = o3d.geometry.TriangleMesh()
            temp_mesh.vertices = deepcopy(mesh.vertices)
            temp_mesh.triangles = deepcopy(mesh.triangles)
            
            # 如果有法向量，也复制
            if len(mesh.vertex_normals) > 0:
                temp_mesh.vertex_normals = deepcopy(mesh.vertex_normals)
            
            # 执行Loop细分
            print(f"Execute mesh subdivision (iterations: {subdivide_iter})...")
            print(f"    - Before subdivision: {len(temp_mesh.vertices)} vertices, {len(temp_mesh.triangles)} faces")
            
            temp_mesh = temp_mesh.subdivide_loop(subdivide_iter)
            temp_mesh.compute_vertex_normals()
            
            print(f"    - After subdivision: {len(temp_mesh.vertices)} vertices, {len(temp_mesh.triangles)} faces")
            
            return temp_mesh
            
        except Exception as e:
            print(f"Mesh subdivision failed: {e}")
            return mesh  # Return original mesh

    def _prepare_frame_data(self, frame_start, frame_end, reference_frame=None):
        """
        准备插值帧数据（全局归一化参数、参考网格等）
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引  
            reference_frame: 参考帧索引，如果为None则使用frame_start
            
        Returns:
            字典包含：reference_mesh, reference_vertices, reference_faces, 
                   reference_uvs, reference_vertex_colors, global_normalization_params
        """
        if reference_frame is None:
            reference_frame = frame_start
            
        # 加载参考网格
        reference_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[reference_frame]))
        reference_vertices = np.asarray(reference_mesh.vertices)
        reference_faces = np.asarray(reference_mesh.triangles) if len(reference_mesh.triangles) > 0 else None
        
        # 保留原始mesh的纹理坐标和顶点颜色
        reference_uvs = None
        reference_vertex_colors = None
        if hasattr(reference_mesh, 'triangle_uvs') and len(reference_mesh.triangle_uvs) > 0:
            reference_uvs = np.asarray(reference_mesh.triangle_uvs)
            print(f"Keep original texture coordinates: {len(reference_uvs)}")
        
        if hasattr(reference_mesh, 'vertex_colors') and len(reference_mesh.vertex_colors) > 0:
            reference_vertex_colors = np.asarray(reference_mesh.vertex_colors)
            print(f"Keep original vertex colors: {len(reference_vertex_colors)}")
        
        # 计算全局归一化参数
        all_meshes = []
        all_vertices = []
        
        frame_indices = [frame_start, frame_end]
        for idx in frame_indices:
            mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[idx]))
            vertices = np.asarray(mesh.vertices)
            all_meshes.append(mesh)
            all_vertices.append(vertices)
        
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
        
        return {
            'reference_mesh': reference_mesh,
            'reference_vertices': reference_vertices,
            'reference_faces': reference_faces,
            'reference_uvs': reference_uvs,
            'reference_vertex_colors': reference_vertex_colors,
            'global_normalization_params': global_normalization_params
        }
    
    def _process_skinning_weights(self, reference_vertices, frame_start, global_normalization_params, skinning_weights=None):
        """
        处理蒙皮权重（权重矩阵大小匹配、扩展等）
        
        Args:
            reference_vertices: 参考网格顶点
            frame_start: 起始帧索引
            global_normalization_params: 全局归一化参数
            skinning_weights: 蒙皮权重矩阵，如果为None则使用self.skinning_weights
            
        Returns:
            处理后的权重矩阵
        """
        if skinning_weights is None:
            skinning_weights = self.skinning_weights
            
        if skinning_weights is None:
            return None
            
        reference_vertices_norm = self.normalize_mesh_vertices(reference_vertices, global_normalization_params)
        
        # 确保权重矩阵与顶点数量匹配
        if skinning_weights.shape[0] != len(reference_vertices_norm):
            print(f"Weight matrix vertex number ({skinning_weights.shape[0]}) does not match reference mesh vertex number ({len(reference_vertices_norm)})")
            # 调整权重矩阵大小
            if skinning_weights.shape[0] > len(reference_vertices_norm):
                skinning_weights = skinning_weights[:len(reference_vertices_norm)]
            else:
                # 扩展权重矩阵
                extended_weights = np.zeros((len(reference_vertices_norm), skinning_weights.shape[1]))
                extended_weights[:skinning_weights.shape[0]] = skinning_weights
                # 对新增顶点使用距离初始化
                keypoints = self.keypoints[frame_start, :, :3]
                remaining_vertices = reference_vertices_norm[skinning_weights.shape[0]:]
                if len(remaining_vertices) > 0:
                    from scipy.spatial.distance import cdist
                    distances = cdist(remaining_vertices, keypoints)
                    remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                    remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                    extended_weights[skinning_weights.shape[0]:] = remaining_weights
                skinning_weights = extended_weights
                
        return skinning_weights
    
    def _apply_lbs_deformation(self, reference_vertices, skinning_weights, interpolated_transforms, reference_frame, global_normalization_params):
        """
        应用LBS变形
        
        Args:
            reference_vertices: 参考网格顶点
            skinning_weights: 蒙皮权重矩阵
            interpolated_transforms: 插值后的变换矩阵
            reference_frame: 参考帧索引
            global_normalization_params: 全局归一化参数
            
        Returns:
            变形后的顶点坐标
        """
        # 使用全局参数归一化参考网格
        reference_vertices_norm = self.normalize_mesh_vertices(reference_vertices, global_normalization_params)
        
        # 使用与Skinning.py相同的相对变换处理
        print(f"Use relative transformation for LBS...")
        
        # 获取参考帧变换
        reference_transforms = self.transforms[reference_frame]
        
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
            reference_vertices_norm, skinning_weights, relative_transforms
        )
        
        # 使用全局参数反归一化
        transformed_vertices = self.denormalize_mesh_vertices(
            transformed_vertices_norm, global_normalization_params
        )
        
        return transformed_vertices
    
    def _apply_coordinate_alignment(self, transformed_vertices, interpolated_transforms):
        """
        应用坐标系对齐
        
        Args:
            transformed_vertices: 变形后的顶点
            interpolated_transforms: 插值后的变换矩阵
            
        Returns:
            对齐后的顶点坐标和调整后的变换矩阵
        """
        # 修复坐标系问题：将骨骼变换到网格坐标系
        print(f"Fix coordinate system alignment...")
        
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
        
        print(f"      - Mesh center: {mesh_center}")
        print(f"      - Before adjustment skeleton center: {joint_center}")
        print(f"      - After adjustment skeleton center: {np.mean(adjusted_transforms[:, :3, 3], axis=0)}")
        print(f"      - Offset: {offset}")
        
        return transformed_vertices, adjusted_transforms
    
    def _create_output_mesh(self, transformed_vertices, reference_faces, reference_uvs, reference_vertex_colors, smooth_mesh=False, subdivide_iter=3):
        """
        创建输出网格
        
        Args:
            transformed_vertices: 变形后的顶点
            reference_faces: 参考面片
            reference_uvs: 参考UV坐标
            reference_vertex_colors: 参考顶点颜色
            smooth_mesh: 是否平滑网格
            subdivide_iter: 细分迭代次数
            
        Returns:
            输出网格
        """
        # 创建插值网格
        interpolated_mesh = o3d.geometry.TriangleMesh()
        interpolated_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        if reference_faces is not None:
            interpolated_mesh.triangles = o3d.utility.Vector3iVector(reference_faces)
        
        # 保留原始纹理坐标
        if reference_uvs is not None:
            interpolated_mesh.triangle_uvs = o3d.utility.Vector2dVector(reference_uvs)
            print(f"Keep original texture coordinates to interpolated mesh")
        
        # 保留原始顶点颜色
        if reference_vertex_colors is not None:
            interpolated_mesh.vertex_colors = o3d.utility.Vector3dVector(reference_vertex_colors)
            print(f"Keep original vertex colors to interpolated mesh")
        
        # 确保有法线
        if not interpolated_mesh.has_vertex_normals():
            interpolated_mesh.compute_vertex_normals()
            print(f"Compute vertex normals of interpolated mesh")
        
        # 可选的网格平滑处理
        if smooth_mesh:
            interpolated_mesh = self.smooth_mesh(interpolated_mesh, subdivide_iter)
        
        return interpolated_mesh


class DualReferenceInterpolator(VolumetricInterpolator):
    """
    Dual reference frame interpolator
    
    Use start frame and end frame as reference, respectively optimize skinning weights:
    - First half uses start frame skeleton and skinning
    - Second half uses end frame skeleton and skinning
    """
    
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None):
        super().__init__(skeleton_data_dir, mesh_folder_path, weights_path)
        self.start_frame_weights = None
        self.end_frame_weights = None
        self.start_frame_skinner = None
        self.end_frame_skinner = None
    
    def optimize_dual_reference_weights(self, frame_start, frame_end, max_optimize_frames=5):
        """
        Optimize skinning weights for start frame and end frame
        
        Args:
            frame_start: start frame index
            frame_end: end frame index
            max_optimize_frames: maximum number of optimization frames
            
        Returns:
            success: whether successful
        """
        print(f"Start dual reference frame weight optimization...")
        print(f"  - Start reference frame: {frame_start}")
        print(f"  - End reference frame: {frame_end}")
        
        # 为起始帧优化权重
        print(f"\nOptimize start frame {frame_start} weights...")
        self.start_frame_skinner = self._create_skinner_for_frame(frame_start)
        start_weights = self._optimize_frame_weights(
            self.start_frame_skinner, frame_start, frame_end, max_optimize_frames, "start"
        )
        
        if start_weights is None:
            print("Start frame weight optimization failed")
            return False
        
        # 为结束帧优化权重
        print(f"\nOptimize end frame {frame_end} weights...")
        self.end_frame_skinner = self._create_skinner_for_frame(frame_end)
        end_weights = self._optimize_frame_weights(
            self.end_frame_skinner, frame_start, frame_end, max_optimize_frames, "end"
        )
        
        if end_weights is None:
            print("End frame weight optimization failed")
            return False
        
        self.start_frame_weights = start_weights
        self.end_frame_weights = end_weights
        
        print(f"Dual reference frame weight optimization completed")
        print(f"  - Start frame weights shape: {start_weights.shape}")
        print(f"  - End frame weights shape: {end_weights.shape}")
        
        return True
    
    def _create_skinner_for_frame(self, reference_frame_idx):
        """为指定帧创建蒙皮器"""
        from Skinning import AutoSkinning
        
        skinner = AutoSkinning(
            skeleton_data_dir=self.skeleton_data_dir,
            reference_frame_idx=reference_frame_idx
        )
        skinner.load_mesh_sequence(self.mesh_folder_path)
        return skinner
    
    def _optimize_frame_weights(self, skinner, frame_start, frame_end, max_optimize_frames, frame_type):
        """为指定帧优化权重"""
        try:
            # 选择reference frame附近的帧进行优化（-5到+5范围）
            if frame_type == "start":
                reference_frame = frame_start
            else:  # end
                reference_frame = frame_end
            
            # 选择reference frame附近的帧，范围[-5, +5]
            available_frames = []
            for i in range(max(0, reference_frame - 5), min(self.num_frames, reference_frame + 6)):
                available_frames.append(i)
            
            if not available_frames:
                print(f"No frames to optimize for {frame_type} frame")
                return None
            
            # 限制最多使用max_optimize_frames个帧
            if len(available_frames) > max_optimize_frames:
                # 优先选择reference frame附近的帧
                center_idx = available_frames.index(reference_frame)
                half_range = max_optimize_frames // 2
                
                # 从中心向两边扩展选择帧
                start_idx = max(0, center_idx - half_range)
                end_idx = min(len(available_frames), center_idx + half_range + 1)
                optimize_frames = available_frames[start_idx:end_idx]
                
                # 如果还不够max_optimize_frames个，从两边补充
                while len(optimize_frames) < max_optimize_frames and (start_idx > 0 or end_idx < len(available_frames)):
                    if start_idx > 0:
                        start_idx -= 1
                        optimize_frames.insert(0, available_frames[start_idx])
                    if len(optimize_frames) < max_optimize_frames and end_idx < len(available_frames):
                        optimize_frames.append(available_frames[end_idx])
                        end_idx += 1
                
                print(f"  - Available frames around reference {reference_frame}: {len(available_frames)}, selected {len(optimize_frames)} frames")
            else:
                optimize_frames = available_frames
            
            print(f"  - Optimization frames: {optimize_frames}")
            
            # 统一权重文件命名格式
            reference_frame = frame_start if frame_type == "start" else frame_end
            weights_filename = f"ref{reference_frame}_opt{optimize_frames[0]}-{optimize_frames[-1]}_num{len(optimize_frames)}.npz"
            weights_path = Path(self.weights_path) / weights_filename
            
            if weights_path.exists():
                print(f"  - Found existing weights file: {weights_path}")
                data = np.load(str(weights_path))
                return data['weights']
            
            # 创建输出目录
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 优化权重
            weights = skinner.optimize_reference_frame_skinning(
                optimization_frames=optimize_frames,
                regularization_lambda=0.01,
                max_iter=200
            )
            
            if weights is not None:
                # 保存权重文件
                np.savez_compressed(str(weights_path), weights=weights)
                print(f"  - Weights saved: {weights_path}")
            
            return weights
            
        except Exception as e:
            print(f"{frame_type} frame weight optimization failed: {e}")
            return None
    
    def _get_smoothed_reference_weights(self, t, actual_start, actual_end):
        """
        获取平滑过渡的权重和参考帧 - 总是进行权重混合以确保平滑性
        
        Args:
            t: 插值参数 [0, 1]
            actual_start: 实际起始帧索引
            actual_end: 实际结束帧索引
            
        Returns:
            tuple: (reference_frame, reference_weights, transition_type)
        """
        # 使用平滑权重插值系数，避免线性插值的突兀
        # 使用余弦函数创建S曲线，使过渡更平滑
        smooth_t = 0.5 * (1 - np.cos(np.pi * t))  # S型曲线插值
        
        # 关键修复：总是使用起始帧作为参考帧，避免参考帧跳跃导致骨骼变换不连续
        # 这是jerk值爆炸的真正原因！
        reference_frame = actual_start  # 固定使用起始帧，消除硬切换
        
        print(f"    Smooth weight blending: t={t:.3f}, smooth_t={smooth_t:.3f}, reference_frame={reference_frame}")
        
        try:
            # 总是进行权重混合，使用平滑插值系数
            reference_weights = self._blend_weights_safe(
                self.start_frame_weights, 
                self.end_frame_weights, 
                smooth_t,  # 使用平滑插值系数
                actual_start,  # 主要参考帧（用于确定网格结构）
                actual_end     # 次要参考帧
            )
            transition_type = "always_blended"
            print(f"    ✅ Always blended weights with smooth coefficient: {reference_weights.shape}")
            
        except Exception as e:
            # 如果权重混合失败，退回到简单策略
            print(f"    ⚠️ Weight blending failed ({e}), using fallback strategy")
            if t < 0.5:
                reference_frame = actual_start
                reference_weights = self.start_frame_weights
                print(f"    Using start frame {actual_start} as fallback")
            else:
                reference_frame = actual_end
                reference_weights = self.end_frame_weights
                print(f"    Using end frame {actual_end} as fallback")
            transition_type = "fallback"
        
        return reference_frame, reference_weights, transition_type
    
    def _blend_weights_safe(self, primary_weights, secondary_weights, alpha, primary_frame, secondary_frame):
        """
        安全地混合两个权重矩阵，处理顶点数量不一致问题
        
        Args:
            primary_weights: 主要权重矩阵 [V1, J]
            secondary_weights: 次要权重矩阵 [V2, J]  
            alpha: 混合系数 [0, 1]，0=完全使用primary，1=完全使用secondary
            primary_frame: 主要参考帧索引
            secondary_frame: 次要参考帧索引
            
        Returns:
            blended_weights: 混合后的权重矩阵 [V1, J]
        """
        if primary_weights is None or secondary_weights is None:
            raise ValueError("One of the weight matrices is None")
        
        # 获取主要参考帧的网格信息（决定目标顶点数）
        primary_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[primary_frame]))
        primary_vertices = np.asarray(primary_mesh.vertices)
        target_vertex_count = len(primary_vertices)
        
        print(f"    Blending weights: target_vertices={target_vertex_count}, alpha={alpha:.3f}")
        print(f"    Primary weights shape: {primary_weights.shape}")
        print(f"    Secondary weights shape: {secondary_weights.shape}")
        
        # 确保primary_weights与目标顶点数匹配
        if primary_weights.shape[0] != target_vertex_count:
            print(f"    Adjusting primary weights: {primary_weights.shape[0]} -> {target_vertex_count}")
            if primary_weights.shape[0] > target_vertex_count:
                primary_weights = primary_weights[:target_vertex_count]
            else:
                # 扩展权重矩阵
                extended_primary = np.zeros((target_vertex_count, primary_weights.shape[1]))
                extended_primary[:primary_weights.shape[0]] = primary_weights
                # 对新增顶点使用距离初始化
                if primary_weights.shape[0] < target_vertex_count:
                    remaining_vertices = primary_vertices[primary_weights.shape[0]:]
                    if len(remaining_vertices) > 0:
                        keypoints = self.keypoints[primary_frame, :, :3]
                        from scipy.spatial.distance import cdist
                        distances = cdist(remaining_vertices, keypoints)
                        remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                        remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                        extended_primary[primary_weights.shape[0]:] = remaining_weights
                primary_weights = extended_primary
        
        # 调整secondary_weights到相同的顶点数
        if secondary_weights.shape[0] != target_vertex_count:
            print(f"    Adjusting secondary weights: {secondary_weights.shape[0]} -> {target_vertex_count}")
            
            if secondary_weights.shape[0] > target_vertex_count:
                # 截断多余的顶点
                adjusted_secondary = secondary_weights[:target_vertex_count]
            else:
                # 扩展权重矩阵
                adjusted_secondary = np.zeros((target_vertex_count, secondary_weights.shape[1]))
                adjusted_secondary[:secondary_weights.shape[0]] = secondary_weights
                
                # 对新增顶点使用距离初始化权重
                if secondary_weights.shape[0] < target_vertex_count:
                    remaining_vertices = primary_vertices[secondary_weights.shape[0]:]
                    if len(remaining_vertices) > 0:
                        # 使用主要参考帧的关键点
                        keypoints = self.keypoints[primary_frame, :, :3]
                        from scipy.spatial.distance import cdist
                        distances = cdist(remaining_vertices, keypoints)
                        remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                        remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                        adjusted_secondary[secondary_weights.shape[0]:] = remaining_weights
        else:
            adjusted_secondary = secondary_weights
        
        # 执行权重插值
        blended_weights = (1 - alpha) * primary_weights + alpha * adjusted_secondary
        
        # 确保权重和为1（归一化）
        weight_sums = np.sum(blended_weights, axis=1, keepdims=True)
        weight_sums[weight_sums < 1e-8] = 1e-8  # 避免除零
        blended_weights = blended_weights / weight_sums
        
        print(f"    ✅ Blended weights shape: {blended_weights.shape}")
        
        return blended_weights
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None, smooth_mesh=False, subdivide_iter=3,
                                   use_vertex_colors=False, save_npy_files=False, save_standard_obj=True):
        """
        使用双参考帧方法生成插值帧
        """
        total_start_time = time.time()
        
        print(f"Start dual reference frame interpolation generation...")
        print(f"  - Start frame: {frame_start}")
        print(f"  - End frame: {frame_end}")
        print(f"  - Interpolation frames: {num_interpolate}")
        
        # 检查帧索引范围（frame_start和frame_end是排序后文件列表的索引）
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"Frame index out of range: start_frame={frame_start}, end_frame={frame_end}, available frames={len(self.mesh_files)}")
        
        # 检查帧索引是否相等（不允许相等）
        if frame_start == frame_end:
            raise ValueError(f"Start frame cannot be equal to end frame: {frame_start} == {frame_end}")
        
        # 确定实际的起始和结束帧（支持反向插值）
        actual_start = min(frame_start, frame_end)
        actual_end = max(frame_start, frame_end)
        is_reverse = frame_start > frame_end
        
        # 打印实际使用的文件信息
        start_file = self.mesh_files[actual_start].name
        end_file = self.mesh_files[actual_end].name
        print(f"  - Using files: {start_file} (index {actual_start}) -> {end_file} (index {actual_end})")
        
        # 设置输出目录
        if output_dir:
            # 保存插值输出目录
            self.interpolation_output_dir = output_dir
            # 确保权重文件保存在统一的skinning_weights目录中
            if hasattr(self, 'output_dir') and self.output_dir:
                # 使用已设置的基础输出目录
                pass
            else:
                # 如果没有设置基础输出目录，从插值目录推断
                interpolation_path = Path(output_dir)
                self.output_dir = str(interpolation_path.parent.parent)
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 权重优化
        if optimize_weights:
            print(f"\nStart dual reference frame weight optimization...")
            optimization_start = time.time()
            
            if not self.optimize_dual_reference_weights(actual_start, actual_end, max_optimize_frames):
                print("Dual reference frame weight optimization failed, will use simple interpolation")
                return super().generate_interpolated_frames(
                    frame_start, frame_end, num_interpolate, 
                    max_optimize_frames, False, output_dir, debug_frames, smooth_mesh, subdivide_iter,
                    use_vertex_colors, save_npy_files, save_standard_obj
                )
            
            optimization_time = time.time() - optimization_start
            print(f"Dual reference frame weight optimization time: {optimization_time:.2f} seconds")
        
        # 生成插值参数
        t_values = np.linspace(0, 1, num_interpolate + 2)[1:-1]
        
        # 如果是反向插值，反转t值
        if is_reverse:
            t_values = 1.0 - t_values
            print(f"  - Reverse interpolation detected: {frame_start} -> {frame_end}")
            print(f"  - Using actual frame range: {actual_start} -> {actual_end}")
        
        interpolated_frames = []
        
        print(f"\nStart generating {len(t_values)} dual reference frame interpolation frames...")
        frame_generation_start = time.time()
        
        for i, t in enumerate(t_values):
            frame_start_time = time.time()
            print(f"  Generate interpolation frame {i+1}/{len(t_values)} (t={t:.3f})...")
            
            try:
                # 使用平滑过渡权重选择，避免硬切换
                reference_frame, reference_weights, transition_type = self._get_smoothed_reference_weights(
                    t, actual_start, actual_end
                )
                
                # 选择对应的skinner（只是为了兼容性，实际在_generate_dual_reference_frame中不会用到）
                reference_skinner = self.start_frame_skinner if reference_frame == actual_start else self.end_frame_skinner
                
                # 插值骨骼变换 - 使用对应参考帧的pose
                interpolated_transforms = self.interpolate_skeleton_transforms_with_reference(
                    actual_start, actual_end, t, reference_frame
                )
                
                # 生成插值帧数据
                frame_data = self._generate_dual_reference_frame(
                    actual_start, actual_end, t, interpolated_transforms, 
                    reference_frame, reference_weights, reference_skinner,
                    output_dir, i, smooth_mesh, subdivide_iter
                )
                
                if frame_data:
                    interpolated_frames.append(frame_data)
                    frame_time = time.time() - frame_start_time
                    print(f"    Completed (time: {frame_time:.2f} seconds)")
                else:
                    print(f"    Generation failed")
                    
            except Exception as e:
                print(f"    Generate interpolation frame failed: {e}")
                import traceback
                traceback.print_exc()
        
        frame_generation_time = time.time() - frame_generation_start
        total_time = time.time() - total_start_time
        
        print(f"\nDual reference frame interpolation generation completed!")
        print(f"  - Generated frames: {len(interpolated_frames)}")
        print(f"  - Frame generation time: {frame_generation_time:.2f} seconds")
        print(f"  - Total time: {total_time:.2f} seconds")
        
        return interpolated_frames
    
    def _generate_dual_reference_frame(self, frame_start, frame_end, t, interpolated_transforms, 
                                     reference_frame, reference_weights, reference_skinner,
                                     output_dir, frame_idx, smooth_mesh=False, subdivide_iter=3):
        """生成双参考帧插值帧 - 完全复用baseline逻辑"""
        try:
            print(f"    Generating dual reference frame using reference frame {reference_frame}")
            
            # 使用核心插值逻辑，但指定不同的参考帧和权重
            frame_result = self._generate_frame_core(
                frame_start, frame_end, t, interpolated_transforms,
                output_dir, frame_idx, smooth_mesh, subdivide_iter,
                save_npy_files=False, save_standard_obj=True,
                reference_frame=reference_frame, skinning_weights=reference_weights
            )
            
            if frame_result:
                print(f"    Generated output mesh: {len(frame_result['vertices'])} vertices")
                
                # 返回与原始格式兼容的结果
                return {
                        'mesh': frame_result['mesh'],
                        'transforms': frame_result['transforms'],
                        'keypoints': frame_result['keypoints'],
                    't': t,
                    'reference_frame': reference_frame
                }
            else:
                print(f"    Generate dual reference frame failed: _generate_frame_core returned None")
                return None
            
        except Exception as e:
            print(f"    Generate dual reference frame failed: {e}")
            import traceback
            traceback.print_exc()
            return None


class AdaptiveSimilarityInterpolator(VolumetricInterpolator):
    """
    相似帧自适应插值器
    
    在每个插值区间找到中间帧，从原始数据中找到最相似的骨骼，
    进行自动蒙皮，然后基于相似帧进行插值
    """
    
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None):
        super().__init__(skeleton_data_dir, mesh_folder_path, weights_path)
        self.similarity_cache = {}
        self.adaptive_skinners = {}
    
    def find_most_similar_skeleton(self, target_transforms, search_range=None):
        """
        找到最相似的骨骼
        
        Args:
            target_transforms: 目标变换矩阵 [num_joints, 4, 4]
            search_range: 搜索范围 (start, end)，None表示搜索所有帧
            
        Returns:
            most_similar_frame: 最相似帧的索引
            similarity_score: 相似度分数
        """
        if search_range is None:
            search_range = (0, self.num_frames)
        
        start_frame, end_frame = search_range
        min_distance = float('inf')
        most_similar_frame = start_frame
        
        # 计算目标姿态的关节位置
        target_positions = target_transforms[:, :3, 3]  # [num_joints, 3]
        
        for frame_idx in range(start_frame, end_frame):
            # 获取当前帧的变换矩阵
            current_transforms = self.transforms[frame_idx]  # [num_joints, 4, 4]
            current_positions = current_transforms[:, :3, 3]  # [num_joints, 3]
            
            # 计算欧几里得距离
            distance = np.mean(np.linalg.norm(target_positions - current_positions, axis=1))
            
            if distance < min_distance:
                min_distance = distance
                most_similar_frame = frame_idx
        
        similarity_score = 1.0 / (1.0 + min_distance)  # 转换为相似度分数
        
        return most_similar_frame, similarity_score
    
    def create_adaptive_skinner(self, reference_frame_idx):
        """为指定帧创建自适应蒙皮器"""
        from Skinning import AutoSkinning
        
        skinner = AutoSkinning(
            skeleton_data_dir=self.skeleton_data_dir,
            reference_frame_idx=reference_frame_idx
        )
        skinner.load_mesh_sequence(self.mesh_folder_path)
        return skinner
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None, smooth_mesh=False, subdivide_iter=3,
                                   use_vertex_colors=False, save_npy_files=False, save_standard_obj=True):
        """
        使用相似帧自适应方法生成插值帧
        
        分段逻辑：
        - 将插值区间分成多个段
        - 每段使用一个参考帧
        - 每个参考帧基于其邻近帧进行优化
        """
        total_start_time = time.time()
        
        print(f"Start adaptive similarity interpolation generation...")
        print(f"  - Start frame: {frame_start}")
        print(f"  - End frame: {frame_end}")
        print(f"  - Interpolation frames: {num_interpolate}")
        
        # 检查帧索引范围（frame_start和frame_end是排序后文件列表的索引）
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"Frame index out of range: start_frame={frame_start}, end_frame={frame_end}, available frames={len(self.mesh_files)}")
        
        # 检查帧索引是否相等（不允许相等）
        if frame_start == frame_end:
            raise ValueError(f"Start frame cannot be equal to end frame: {frame_start} == {frame_end}")
        
        # 确定实际的起始和结束帧（支持反向插值）
        actual_start = min(frame_start, frame_end)
        actual_end = max(frame_start, frame_end)
        is_reverse = frame_start > frame_end
        
        # 打印实际使用的文件信息
        start_file = self.mesh_files[actual_start].name
        end_file = self.mesh_files[actual_end].name
        print(f"  - Using files: {start_file} (index {actual_start}) -> {end_file} (index {actual_end})")
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成插值参数
        t_values = np.linspace(0, 1, num_interpolate + 2)[1:-1]
        
        # 如果是反向插值，反转t值
        if is_reverse:
            t_values = 1.0 - t_values
            print(f"  - Reverse interpolation detected: {frame_start} -> {frame_end}")
            print(f"  - Using actual frame range: {actual_start} -> {actual_end}")
        
        interpolated_frames = []
        
        # 分段逻辑：将插值区间分成多个段
        num_segments = min(3, num_interpolate)  # 最多3段
        segment_size = len(t_values) // num_segments
        
        print(f"\nStart generating {len(t_values)} adaptive interpolation frames in {num_segments} segments...")
        frame_generation_start = time.time()
        
        for i, t in enumerate(t_values):
            frame_start_time = time.time()
            print(f"  Generate interpolation frame {i+1}/{len(t_values)} (t={t:.3f})...")
            
            try:
                # 确定当前帧属于哪个段
                segment_idx = min(i // segment_size, num_segments - 1)
                segment_start_t = segment_idx / num_segments
                segment_end_t = (segment_idx + 1) / num_segments
                
                # 计算段内的插值参数
                segment_t = (t - segment_start_t) / (segment_end_t - segment_start_t)
                segment_t = np.clip(segment_t, 0, 1)
                
                print(f"    Segment {segment_idx + 1}/{num_segments} (t={segment_t:.3f})")
                
                # 插值骨骼变换 - 使用实际帧范围
                interpolated_transforms = self.interpolate_skeleton_transforms(actual_start, actual_end, t)
                
                # 找到最相似的骨骼（在原始数据中）
                print(f"    Find most similar skeleton...")
                most_similar_frame, similarity_score = self.find_most_similar_skeleton(
                    interpolated_transforms, search_range=(actual_start, actual_end + 1)
                )
                print(f"    Most similar frame: {most_similar_frame} (similarity: {similarity_score:.3f})")
                
                # 为相似帧创建或获取蒙皮器
                if most_similar_frame not in self.adaptive_skinners:
                    print(f"    Create skinner for similar frame {most_similar_frame}...")
                    skinner = self.create_adaptive_skinner(most_similar_frame)
                    
                    # 优化权重
                    if optimize_weights:
                        print(f"    Optimize weights for similar frame {most_similar_frame}...")
                        # 使用相似帧附近的帧进行优化（-5到+5范围）
                        available_frames = []
                        for i in range(max(0, most_similar_frame - 5), min(self.num_frames, most_similar_frame + 6)):
                            available_frames.append(i)
                        
                        # 限制最多使用max_optimize_frames个帧
                        if len(available_frames) > max_optimize_frames:
                            # 优先选择相似帧附近的帧
                            center_idx = available_frames.index(most_similar_frame)
                            half_range = max_optimize_frames // 2
                            
                            # 从中心向两边扩展选择帧
                            start_idx = max(0, center_idx - half_range)
                            end_idx = min(len(available_frames), center_idx + half_range + 1)
                            optimize_frames = available_frames[start_idx:end_idx]
                            
                            # 如果还不够max_optimize_frames个，从两边补充
                            while len(optimize_frames) < max_optimize_frames and (start_idx > 0 or end_idx < len(available_frames)):
                                if start_idx > 0:
                                    start_idx -= 1
                                    optimize_frames.insert(0, available_frames[start_idx])
                                if len(optimize_frames) < max_optimize_frames and end_idx < len(available_frames):
                                    optimize_frames.append(available_frames[end_idx])
                                    end_idx += 1
                            
                            print(f"    Available frames around similar frame {most_similar_frame}: {len(available_frames)}, selected {len(optimize_frames)} frames")
                        else:
                            optimize_frames = available_frames
                        
                        # 统一权重文件命名格式
                        weights_filename = f"ref{most_similar_frame}_opt{optimize_frames[0]}-{optimize_frames[-1]}_num{len(optimize_frames)}.npz"
                        weights_path = Path(self.weights_path) / weights_filename
                        
                        if weights_path.exists():
                            print(f"    Found existing weights file: {weights_path}")
                            data = np.load(str(weights_path))
                            weights = data['weights']
                        else:
                            # 创建输出目录
                            weights_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            weights = skinner.optimize_reference_frame_skinning(
                                optimization_frames=optimize_frames,
                                regularization_lambda=0.01,
                                max_iter=200
                            )
                            
                            if weights is not None:
                                # 保存权重文件
                                np.savez_compressed(str(weights_path), weights=weights)
                                print(f"    Weights saved: {weights_path}")
                        
                        if weights is not None:
                            skinner.skinning_weights = weights
                            print(f"    Similar frame weight optimization completed")
                        else:
                            print(f"    Similar frame weight optimization failed, use distance initialization")
                    else:
                        print(f"    Skip weight optimization")
                    
                    self.adaptive_skinners[most_similar_frame] = skinner
                else:
                    skinner = self.adaptive_skinners[most_similar_frame]
                    print(f"    Use existing skinner")
                
                # 生成插值帧数据
                frame_data = self._generate_adaptive_frame(
                    frame_start, frame_end, t, interpolated_transforms, 
                    most_similar_frame, skinner, output_dir, i, smooth_mesh, subdivide_iter
                )
                
                if frame_data:
                    interpolated_frames.append(frame_data)
                    frame_time = time.time() - frame_start_time
                    print(f"    Completed (time: {frame_time:.2f} seconds)")
                else:
                    print(f"    Generation failed")
                    
            except Exception as e:
                print(f"    Generate adaptive interpolation frame failed: {e}")
                import traceback
                traceback.print_exc()
        
        frame_generation_time = time.time() - frame_generation_start
        total_time = time.time() - total_start_time
        
        print(f"\nAdaptive similarity interpolation generation completed!")
        print(f"  - Generated frames: {len(interpolated_frames)}")
        print(f"  - Frame generation time: {frame_generation_time:.2f} seconds")
        print(f"  - Total time: {total_time:.2f} seconds")
        
        return interpolated_frames
    
    def _generate_adaptive_frame(self, frame_start, frame_end, t, interpolated_transforms, 
                               similar_frame, skinner, output_dir, frame_idx, smooth_mesh=False, subdivide_iter=3):
        """生成自适应插值帧"""
        try:
            # 直接加载原始相似帧网格，而不是使用skinner中的归一化网格
            similar_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[similar_frame]))
            similar_vertices = np.asarray(similar_mesh.vertices)
            similar_faces = np.asarray(similar_mesh.triangles) if len(similar_mesh.triangles) > 0 else None
            
            print(f"    Loaded similar mesh: {len(similar_vertices)} vertices, {len(similar_faces) if similar_faces is not None else 0} faces")
            
            # 关键修复：计算从相似帧到插值帧的相对变换
            similar_transforms = self.transforms[similar_frame]
            relative_transforms = np.zeros_like(interpolated_transforms)
            
            for j in range(self.num_joints):
                if np.linalg.det(similar_transforms[j][:3, :3]) > 1e-6:
                    similar_inv = np.linalg.inv(similar_transforms[j])
                    relative_transforms[j] = interpolated_transforms[j] @ similar_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            print(f"    Using relative transforms from similar frame {similar_frame}")
            
            # 使用相似帧的权重进行LBS变换（使用相对变换）
            deformed_vertices = skinner.apply_lbs_transform(
                similar_vertices, skinner.skinning_weights, relative_transforms
            )
            
            # 创建输出网格
            output_mesh = o3d.geometry.TriangleMesh()
            output_mesh.vertices = o3d.utility.Vector3dVector(deformed_vertices)
            if similar_faces is not None:
                output_mesh.triangles = o3d.utility.Vector3iVector(similar_faces)
            
            # 可选的网格平滑处理
            if smooth_mesh:
                output_mesh = self.smooth_mesh(output_mesh, subdivide_iter)
            
            print(f"    Generated output mesh: {len(deformed_vertices)} vertices")
            
            # 生成插值关键点数据
            interpolated_keypoints = self.interpolate_keypoints(frame_start, frame_end, t)
            
            # 保存结果
            if output_dir:
                output_path = Path(output_dir) / f"adaptive_frame_{frame_idx:04d}.obj"
                o3d.io.write_triangle_mesh(str(output_path), output_mesh)
                
                # 可视化
                if hasattr(self, 'visualize_skeleton_with_mesh'):
                    viz_path = Path(output_dir) / f"adaptive_frame_{frame_idx:04d}.png"
                    self.visualize_skeleton_with_mesh(
                        {
                            'mesh': output_mesh, 
                            'transforms': interpolated_transforms,
                            'keypoints': interpolated_keypoints
                        },
                        str(viz_path), frame_idx
                    )
            
            return {
                'mesh': output_mesh,
                'transforms': interpolated_transforms,
                'keypoints': interpolated_keypoints,
                't': t,
                'similar_frame': similar_frame
            }
            
        except Exception as e:
            print(f"    Generate adaptive frame failed: {e}")
            import traceback
            traceback.print_exc()
            return None


class NeuralMarionetteInterpolator(VolumetricInterpolator):
    """
    基于Neural Marionette的插值器
    
    使用nmario源码中的VAE-based插值方法：
    - 使用RNN状态和潜在变量进行插值
    - 基于变分自编码器的生成模型
    - 支持时序一致性
    """
    
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None):
        # 对于Neural Marionette，我们不需要预处理的骨骼数据
        # 网络会直接从体素数据中检测关键点
        self.use_network_keypoints = True
        
        super().__init__(skeleton_data_dir, mesh_folder_path, weights_path)
        self.network = None
        self.opt = None
        self.sample_num = 10000  # 使用源码的采样数量
        self.sample_rate = 10
        
        # 初始化Neural Marionette网络
        self._init_neural_marionette()
    
    def _init_neural_marionette(self):
        """初始化Neural Marionette网络"""
        try:
            import pickle
            from model.neural_marionette import NeuralMarionette
            
            # 加载配置
            exp_dir = 'pretrained/aist'
            opt_file = os.path.join(exp_dir, 'opt.pickle')
            
            if not os.path.exists(opt_file):
                print(f"Neural Marionette configuration file does not exist: {opt_file}")
                print("Please ensure the pre-trained model has been downloaded")
                return False
            
            with open(opt_file, 'rb') as f:
                self.opt = pickle.load(f)
            
            # 加载预训练模型
            resume_file = os.path.join(exp_dir, 'aist_pretrained.pth')
            if not os.path.exists(resume_file):
                print(f"Neural Marionette pre-trained model does not exist: {resume_file}")
                print("Please ensure the pre-trained model has been downloaded")
                return False
            
            checkpoint = torch.load(resume_file)
            self.network = NeuralMarionette(self.opt).cuda()
            self.network.load_state_dict(checkpoint)
            self.network.eval()
            self.network.anneal(1)  # Enable affinity extraction
            
            print(f"Neural Marionette network initialized successfully")
            return True
            
        except Exception as e:
            print(f"Neural Marionette network initialization failed: {e}")
            return False
    
    def _load_voxel_sequence(self, frame_start, frame_end):
        """Load voxel sequence - prioritize using specified frame data"""
        try:
            from utils.dataset_utils import crop_sequence, episodic_normalization, voxelize
            
            # 优先使用指定的网格数据（与用户输入的帧相关）
            if hasattr(self, 'mesh_files') and len(self.mesh_files) > 0:
                print(f"Use specified frame mesh data...")
                print(f"    - Start frame: {frame_start}")
                print(f"    - End frame: {frame_end}")
                return self._load_mesh_voxel_sequence(frame_start, frame_end)
            
            # 如果没有网格数据，使用demo数据作为fallback
            demo_source_file = 'data/demo/source/gHO_sBM_cAll_d20_mHO1_ch05.npy'
            if os.path.exists(demo_source_file):
                print(f"Use demo data as fallback: {demo_source_file}")
                return self._load_demo_voxel_sequence(demo_source_file, frame_start, frame_end)
            
            print(f"No available data source found")
            return None, None
            
        except Exception as e:
            print(f"Voxel sequence loading failed: {e}")
            return None, None
    
    def _load_mesh_voxel_sequence(self, frame_start, frame_end):
        """Load voxel sequence from mesh data - use specified frame range"""
        try:
            from utils.dataset_utils import episodic_normalization, voxelize
            
            voxel_sequence = []
            points_sequence = []
            
            print(f"    - Load frame range: {frame_start} to {frame_end}")
            print(f"    - Available mesh files: {len(self.mesh_files)}")
            
            for frame_idx in range(frame_start, frame_end + 1):
                if frame_idx >= len(self.mesh_files):
                    print(f"Frame {frame_idx} out of range, skip")
                    break
                
                print(f"    - Load frame {frame_idx}: {self.mesh_files[frame_idx]}")
                
                # Load mesh
                mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
                vertices = np.asarray(mesh.vertices)
                
                # 归一化处理
                vertices_norm = episodic_normalization(vertices[None], scale=1.0, x_trans=0.0, z_trans=0.0)
                
                # 体素化 - 保持与网络兼容的分辨率
                grid_size = self.opt.grid_size
                voxel = voxelize(vertices_norm[0], (grid_size,) * 3, is_binarized=True)
                voxel_sequence.append(voxel)
                points_sequence.append(vertices_norm[0])
            
            if len(voxel_sequence) == 0:
                print(f"No frames successfully loaded")
                return None, None
            
            # 转换为tensor
            voxel_tensor = torch.from_numpy(np.stack(voxel_sequence, axis=0)).float().cuda()
            print(f"    - Mesh voxelized shape: {voxel_tensor.shape}")
            
            return voxel_tensor, points_sequence
            
        except Exception as e:
            print(f"Mesh voxel sequence loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _load_demo_voxel_sequence(self, demo_file, frame_start, frame_end):
        """Load demo point cloud data and convert to voxel - exactly same as source code"""
        try:
            from utils.dataset_utils import crop_sequence, episodic_normalization, voxelize
            
            # 加载点云数据（与源码完全一致）
            x = np.load(demo_file)[..., :3]  # (T, N, 3)
            print(f"    - Original point cloud data shape: {x.shape}")
            
            # 使用源码的数据处理流程
            x = crop_sequence(x, frame_start, self.opt.Ttot, self.opt.sample_rate)
            print(f"    - Cropped data shape: {x.shape}")
            
            # 归一化处理（与源码一致）
            x = episodic_normalization(x, scale=1.0, x_trans=0.0, z_trans=0.0)
            
            # 体素化（与源码完全一致）
            vox_seq = []
            for t in range(len(x)):
                try:
                    vox_seq.append(voxelize(x[t], (self.opt.grid_size,) * 3, is_binarized=True))
                except Exception as e:
                    print(f"Voxelization failed for frame {t}: {e}")
                    # 创建空体素
                    empty_voxel = np.zeros((self.opt.grid_size,) * 3)
                    vox_seq.append(empty_voxel)
            
            # 转换为tensor（与源码一致）
            vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float().cuda()
            print(f"    - Voxelized shape: {vox_seq.shape}")
            
            return vox_seq, x
            
        except Exception as e:
            print(f"Demo voxel sequence loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _interpolate_with_neural_marionette(self, voxel_sequence, frame_start, frame_end, num_interpolate):
        """Use Neural Marionette for interpolation"""
        try:
            from torch.distributions.normal import Normal
            
            T = voxel_sequence.shape[0]
            K = self.opt.nkeypoints
            
            # 使用源码的参数设置
            sample_num = self.sample_num  # 使用完整采样数量
            print(f"    - Sample number: {sample_num}")
            
            # Set opt.Ttot (if not set)
            if not hasattr(self.opt, 'Ttot'):
                self.opt.Ttot = 21
            
            with torch.no_grad():
                # 检测关键点
                print(f"    - Interpolated voxel shape: {voxel_sequence.shape}")
                detector_log = self.network.kypt_detector(voxel_sequence[None])
                keypoints = detector_log['keypoints']
                affinity = detector_log['affinity']
                _ = self.network.dyna_module.encode(keypoints, affinity)
                
                # Get network parameters
                A = self.network.dyna_module.A
                priority = self.network.dyna_module.priority
                parents = self.network.dyna_module.parents
                
                # 初始化RNN状态
                prev_state = self.network.dyna_module.init_kypt_rnn_state.expand(sample_num, -1)
                offset = self.network.dyna_module.get_offset(keypoints).expand(sample_num, -1, -1, -1)
                
                selected_keypoints = []
                sampled_keypoints = []
                
                # 时序插值
                for t in range(T):
                    keypoint = keypoints[:, t].clone()
                    keypoint_flat = keypoint.view(1, -1).expand(sample_num, -1)
                    
                    if t % self.sample_rate == 0 or t == T - 1:
                        # 使用后验分布
                        params_post = self.network.dyna_module.extract_post_dist(
                            torch.cat([prev_state, keypoint_flat], dim=-1)
                        )
                        params_prior = self.network.dyna_module.extract_prior_dist(prev_state)
                        
                        post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
                        post_std = torch.nn.functional.softplus(post_std) + 1e-4
                        prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
                        prior_std = torch.nn.functional.softplus(prior_std) + 1e-4
                        
                        z_kypt_post_dist = Normal(post_mean, post_std)
                        z_kypt_sampled = z_kypt_post_dist.rsample()
                        z_kypt_prior_dist = Normal(prior_mean, prior_std)
                        z_kypt_sampled_for_choosing = z_kypt_prior_dist.rsample()
                        
                        keypoint_sampled_flat, _ = self.network.dyna_module.extract_kypt_from_latent_and_state(
                            torch.cat([prev_state, z_kypt_sampled], dim=-1), offset
                        )
                        keypoint_sampled_flat_for_choosing, _ = self.network.dyna_module.extract_kypt_from_latent_and_state(
                            torch.cat([prev_state, z_kypt_sampled_for_choosing], dim=-1), offset
                        )
                        
                        # 选择最佳样本
                        keypoint_distance = (keypoint_sampled_flat - keypoint_flat).pow(2).sum(dim=-1)
                        min_sampled_idx = keypoint_distance.argmin()
                        keypoint_sampled_flat = keypoint_sampled_flat[min_sampled_idx][None].expand(sample_num, -1)
                        z_kypt_sampled = z_kypt_sampled[min_sampled_idx][None].expand(sample_num, -1)
                        prev_state = prev_state[min_sampled_idx][None].expand(sample_num, -1)
                        
                        keypoint_distance_for_choosing = (keypoint_sampled_flat_for_choosing - keypoint_sampled_flat).pow(2).sum(dim=-1)
                        min_sampled_idx = keypoint_distance_for_choosing.argmin()
                        
                        sampled_keypoints.append(keypoint_flat)
                        for sampled in sampled_keypoints:
                            selected_keypoints.append(sampled[min_sampled_idx].view(K, 4))
                        sampled_keypoints = []
                    else:
                        # 使用先验分布
                        params_prior = self.network.dyna_module.extract_prior_dist(prev_state)
                        prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
                        prior_std = torch.nn.functional.softplus(prior_std) + 1e-4
                        z_kypt_prior_dist = Normal(prior_mean, prior_std)
                        z_kypt_sampled = z_kypt_prior_dist.rsample()
                        keypoint_sampled_flat, _ = self.network.dyna_module.extract_kypt_from_latent_and_state(
                            torch.cat([prev_state, z_kypt_sampled], dim=-1), offset
                        )
                        sampled_keypoints.append(keypoint_sampled_flat)
                    
                    # 更新RNN状态
                    rnn_input = torch.cat([keypoint_sampled_flat, z_kypt_sampled], dim=-1)
                    prev_state = self.network.dyna_module.kypt_rnn_cell(rnn_input, prev_state)
                
                # 生成插值关键点
                selected_keypoints = torch.stack(selected_keypoints, dim=0)[None]
                selected_keypoints[0, :, :, -1] = selected_keypoints[0, 0, :, -1]
                
                            # 解码生成体素
            first_feature = detector_log['first_feature']
            first_frame = voxel_sequence[None, 0]
            decode_log = self.network.kypt_detector.decode_from_dyna(selected_keypoints, first_feature, first_frame)
            interp_voxel = decode_log['gen'].squeeze(0)
            interp_voxel[interp_voxel < 0.5] = 0
            interp_voxel[interp_voxel >= 0.5] = 1
            
            # 生成指定数量的插值帧
            if num_interpolate > 0:
                # 创建插值时间点
                t_values = np.linspace(0, 1, num_interpolate)
                interpolated_voxels = []
                interpolated_keypoints = []
                
                for t in t_values:
                    # 简单的线性插值（这里可以改进为更复杂的插值方法）
                    frame_idx = int(t * (len(interp_voxel) - 1))
                    frame_idx = min(frame_idx, len(interp_voxel) - 1)
                    
                    interpolated_voxels.append(interp_voxel[frame_idx])
                    if selected_keypoints is not None and frame_idx < selected_keypoints.shape[1]:
                        interpolated_keypoints.append(selected_keypoints[0, frame_idx])
                    else:
                        interpolated_keypoints.append(selected_keypoints[0, 0])  # 使用第一帧作为fallback
                
                interp_voxel = torch.stack(interpolated_voxels, dim=0)
                selected_keypoints = torch.stack(interpolated_keypoints, dim=1)[None]
            
            return interp_voxel, selected_keypoints, parents
                
        except Exception as e:
            print(f"Neural Marionette interpolation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _voxel_to_mesh(self, voxel_sequence):
        """Convert voxel sequence to mesh sequence - use source code rendering"""
        try:
            mesh_sequence = []
            
            for t in range(len(voxel_sequence)):
                # Convert voxel to point cloud coordinates (exactly same as source code)
                coords = np.stack(np.where(voxel_sequence[t, 0].clone().detach().cpu().numpy()), axis=-1) / ((64 - 1) / 2) - 1
                
                if len(coords) == 0:
                    # 如果没有点，创建空网格
                    mesh = o3d.geometry.TriangleMesh()
                    mesh_sequence.append(mesh)
                    continue
                
                # 创建点云（与源码一致）
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords)
                pcd.estimate_normals()
                pcd.orient_normals_consistent_tangent_plane(5)
                pcd_normals = np.asarray(pcd.normals)
                
                # 使用源码的渲染方式：将点云转换为小圆柱体
                mesh = o3d.geometry.TriangleMesh()
                
                # 限制点的数量以避免内存问题
                max_points = min(len(coords), 3000)  # 减少点数以提高性能
                if len(coords) > max_points:
                    # 随机采样点
                    indices = np.random.choice(len(coords), max_points, replace=False)
                    coords = coords[indices]
                    pcd_normals = pcd_normals[indices]
                
                for i in range(len(coords)):
                    # 为每个点创建一个小圆柱体（模拟源码的drawPlate）
                    center = coords[i]
                    normal = pcd_normals[i]
                    
                    # 创建小圆柱体（减少复杂度以提高性能）
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=0.015, height=0.008, resolution=4
                    )
                    cylinder.translate([0, 0, -0.004])
                    
                    # 旋转圆柱体以对齐法向量
                    if np.linalg.norm(normal) > 1e-6:
                        # 计算旋转矩阵
                        z_axis = np.array([0, 0, 1])
                        normal_normalized = normal / np.linalg.norm(normal)
                        
                        # 计算旋转轴和角度
                        rotation_axis = np.cross(z_axis, normal_normalized)
                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            cos_angle = np.dot(z_axis, normal_normalized)
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            
                            # 应用旋转
                            R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(
                                rotation_axis * angle
                            )
                            cylinder.rotate(R)
                    
                    # 移动到正确位置
                    cylinder.translate(center)
                    
                    # 合并到主网格
                    mesh += cylinder
                
                # 如果点太少，添加一个球体作为基础
                if len(coords) < 100:
                    center = np.mean(coords, axis=0) if len(coords) > 0 else np.array([0, 0, 0])
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                    sphere.translate(center)
                    mesh += sphere
                
                # 清理网格
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                mesh.compute_vertex_normals()
                
                mesh_sequence.append(mesh)
            
            print(f"    - Number of generated meshes: {len(mesh_sequence)}")
            return mesh_sequence
            
        except Exception as e:
            print(f"Voxel to mesh conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None, smooth_mesh=False, subdivide_iter=3,
                                   use_vertex_colors=False, save_npy_files=False, save_standard_obj=True):
        """
        Use Neural Marionette method to generate interpolated frames
        """
        total_start_time = time.time()
        
        print(f"Start Neural Marionette interpolation generation...")
        print(f"  - Start frame: {frame_start}")
        print(f"  - End frame: {frame_end}")
        print(f"  - Interpolation frames: {num_interpolate}")
        
        # 检查帧索引范围（frame_start和frame_end是排序后文件列表的索引）
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"Frame index out of range: start_frame={frame_start}, end_frame={frame_end}, available frames={len(self.mesh_files)}")
        
        # 检查帧索引是否相等（不允许相等）
        if frame_start == frame_end:
            raise ValueError(f"Start frame cannot be equal to end frame: {frame_start} == {frame_end}")
        
        # 确定实际的起始和结束帧（支持反向插值）
        actual_start = min(frame_start, frame_end)
        actual_end = max(frame_start, frame_end)
        is_reverse = frame_start > frame_end
        
        # 打印实际使用的文件信息
        start_file = self.mesh_files[actual_start].name
        end_file = self.mesh_files[actual_end].name
        print(f"  - Using files: {start_file} (index {actual_start}) -> {end_file} (index {actual_end})")
        
        # 检查网络是否初始化成功
        if self.network is None:
            print("Neural Marionette network not initialized, cannot perform interpolation")
            return []
        
        # 设置输出目录
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # 加载体素序列 - 使用实际帧范围
            print(f"Load voxel sequence...")
            voxel_sequence, points_sequence = self._load_voxel_sequence(actual_start, actual_end)
            
            if voxel_sequence is None:
                print("Voxel sequence loading failed")
                return []
            
            print(f"    - Voxel sequence shape: {voxel_sequence.shape}")
            
            # 使用Neural Marionette进行插值 - 使用实际帧范围
            print(f"Execute Neural Marionette interpolation...")
            interp_voxel, selected_keypoints, parents = self._interpolate_with_neural_marionette(
                voxel_sequence, actual_start, actual_end, num_interpolate
            )
            
            if interp_voxel is None:
                print("Neural Marionette interpolation failed")
                return []
            
            print(f"    - Interpolated voxel shape: {interp_voxel.shape}")
            
            # 转换为网格序列
            print(f"Convert voxel to mesh...")
            mesh_sequence = self._voxel_to_mesh(interp_voxel)
            
            if len(mesh_sequence) == 0:
                print("Mesh conversion failed")
                return []
            
            print(f"    - Number of generated meshes: {len(mesh_sequence)}")
            
            # 生成插值帧数据
            interpolated_frames = []
            
            for i, mesh in enumerate(mesh_sequence):
                frame_start_time = time.time()
                print(f"  Generate Neural Marionette frame {i+1}/{len(mesh_sequence)}...")
                
                try:
                    # 可选的网格平滑处理
                    if smooth_mesh:
                        mesh = self.smooth_mesh(mesh, subdivide_iter)
                    
                    # 生成关键点数据（从selected_keypoints中提取）
                    if selected_keypoints is not None and i < selected_keypoints.shape[1]:
                        keypoints = selected_keypoints[0, i].detach().cpu().numpy()
                    else:
                        # 使用简单的插值关键点
                        if hasattr(self, 'use_network_keypoints') and self.use_network_keypoints:
                            # 对于Neural Marionette，使用网络生成的关键点
                            keypoints = np.zeros((self.opt.nkeypoints, 4))  # 4维包含置信度
                        else:
                            # 使用实际帧范围进行关键点插值
                            t_value = i / len(mesh_sequence)
                            if is_reverse:
                                t_value = 1.0 - t_value
                            keypoints = self.interpolate_keypoints(actual_start, actual_end, t_value)
                    
                    # 生成变换矩阵（简化版本）
                    transforms = np.eye(4)[None].repeat(self.num_joints, axis=0)
                    for j in range(min(len(keypoints), self.num_joints)):
                        transforms[j][:3, 3] = keypoints[j][:3]
                    
                    # 创建帧数据
                    frame_data = {
                        'frame_idx': i,
                        'interpolation_t': i / len(mesh_sequence),
                        'mesh': mesh,
                        'transforms': transforms,
                        'keypoints': keypoints,
                        'vertices': np.asarray(mesh.vertices) if len(mesh.vertices) > 0 else np.array([])
                    }
                    
                    # 保存到文件
                    if output_dir:
                        mesh_output_path = Path(output_dir) / f"neural_marionette_frame_{i:04d}.obj"
                        o3d.io.write_triangle_mesh(str(mesh_output_path), mesh)
                        
                        # 只在需要时保存变换数据
                        if save_npy_files:
                            transform_output_path = Path(output_dir) / f"neural_marionette_frame_{i:04d}_transforms.npy"
                            np.save(transform_output_path, transforms)
                            
                            keypoints_output_path = Path(output_dir) / f"neural_marionette_frame_{i:04d}_keypoints.npy"
                            np.save(keypoints_output_path, keypoints)
                    
                    interpolated_frames.append(frame_data)
                    
                    frame_time = time.time() - frame_start_time
                    print(f"    Completed (time: {frame_time:.2f} seconds)")
                    
                except Exception as e:
                    print(f"    Generate Neural Marionette frame failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            total_time = time.time() - total_start_time
            
            print(f"\nNeural Marionette interpolation generation completed!")
            print(f"  - Generated frames: {len(interpolated_frames)}")
            print(f"  - Total time: {total_time:.2f} seconds")
            
            return interpolated_frames
            
        except Exception as e:
            print(f"Neural Marionette interpolation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []


def main():
    """Main function - for testing"""
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
    
    print(f"Test interpolation function...")
    print(f"  - Start frame: {frame_start}")
    print(f"  - End frame: {frame_end}")
    print(f"  - Number of interpolated frames: {num_interpolate}")
    
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
        print(f"Interpolation test successful! Generated {len(interpolated_frames)} interpolated frames")
    else:
        print(f"Interpolation test failed!")

if __name__ == "__main__":
    main()
