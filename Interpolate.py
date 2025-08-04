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
        self.weights_path = Path(weights_path) if weights_path else None
        
        # 加载骨骼数据
        self.load_skeleton_data()
        
        # 加载网格序列
        self.load_mesh_sequence()
        
        # 初始化蒙皮器
        self.skinner = None
        self.skinning_weights = None
        self.reference_frame_idx = None
        
        # 如果weights_path是文件，直接加载；如果是目录，则用作权重目录
        if self.weights_path:
            if self.weights_path.is_file():
                # 直接加载权重文件
                self.load_skinning_weights(str(self.weights_path))
            elif self.weights_path.is_dir():
                # 权重目录，在需要时会生成权重文件
                print(f"Weights directory set: {self.weights_path}")
            else:
                print(f"Warning: weights_path does not exist: {self.weights_path}")
        
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
                    print(f"  - Reference frame: {self.reference_frame_idx}")
                else:
                    self.reference_frame_idx = 0  # 默认使用0
                    print(f"  - Use default reference frame: {self.reference_frame_idx}")
            
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
        Simple weight optimization logic
        
        Args:
            frame_start: Start frame index
            frame_end: End frame index
            max_optimize_frames: Maximum number of optimization frames
        """
        start_time = time.time()
        
        try:
            from Skinning import AutoSkinning
            
            print(f"Call Skinning.py for weight optimization...")
            print(f"  - Reference frame: {frame_start}")
            print(f"  - Optimization frame range: {frame_start}-{frame_end}")
            print(f"  - Maximum optimization frames: {max_optimize_frames}")
            
            # 确保权重目录存在
            weights_dir = Path(self.weights_path) if self.weights_path else Path("output/skinning_weights")
            weights_dir.mkdir(parents=True, exist_ok=True)
            print(f"  - Weights directory: {weights_dir}")
            
            
            # Initialize Skinning system
            skinner = AutoSkinning(
                skeleton_data_dir=self.skeleton_data_dir,
                reference_frame_idx=frame_start
            )
            
            # Load mesh sequence
            skinner.load_mesh_sequence(self.mesh_folder_path)
            
            # Select optimization frames
            optimize_frames = []
            num_meshes = len(skinner.mesh_files)
            half = max_optimize_frames // 2

            # 计算起止索引，优先保证参考帧在中间，且不越界
            start_idx = max(0, frame_start - half)
            end_idx = start_idx + max_optimize_frames
            if end_idx > num_meshes:
                end_idx = num_meshes
                start_idx = max(0, end_idx - max_optimize_frames)

            for i in range(start_idx, end_idx):
                optimize_frames.append(i)

            
            if not optimize_frames:
                print("  - No frames to optimize")
                return False
            
            print(f"  - Optimize frames: {optimize_frames}, start_idx: {start_idx}, end_idx: {end_idx}, max_optimize_frames: {max_optimize_frames}")

            weights_path = weights_dir / f"ref{frame_start}_opt{optimize_frames[0]}-{optimize_frames[-1]}_num{len(optimize_frames)}.npz"
            # 检查是否已存在权重文件
            if weights_path.exists():
                print(f"  - Found existing weights file: {weights_path}")
                print(f"  - DEBUG: Weights file information:")
                print(f"  - File size: {weights_path.stat().st_size} bytes")
                success = self.load_skinning_weights(str(weights_path))
                if success:
                    print(f"  - Weights file loaded successfully")
                    print(f"  - Reference frame: {getattr(self, 'reference_frame_idx', 'Unknown')}")
                    print(f"  - Weights shape: {self.skinning_weights.shape if self.skinning_weights is not None else 'None'}")
                    return True
            
            # 直接使用Skinning的优化方法
            print(f"  - Call Skinning.py's optimize_reference_frame_skinning...")
            optimization_start = time.time()
            
            skinner.skinning_weights = skinner.optimize_reference_frame_skinning(
                optimization_frames=optimize_frames,
                regularization_lambda=0.01,
                max_iter=200  # 适中的迭代次数
            )
            
            optimization_time = time.time() - optimization_start
            
            if skinner.skinning_weights is not None:
                print(f"  - Weights optimization completed")
                print(f"  - Weights matrix shape: {skinner.skinning_weights.shape}")
                print(f"  - Optimization time: {optimization_time:.2f} seconds")
                
                # 保存权重
                skinner.save_skinning_weights(str(weights_path))
                print(f"  - Weights saved to: {weights_path}")
                
                # 加载优化后的权重到插值器
                self.load_skinning_weights(str(weights_path))
                print(f"  - Weights loaded to interpolator")
                
                total_time = time.time() - start_time
                print(f"  - Total time: {total_time:.2f} seconds")
                return True
            else:
                print("  - Weights optimization failed")
                return False
                
        except Exception as e:
            # 安全处理可能包含Unicode字符的异常信息
            try:
                error_msg = str(e)
            except UnicodeEncodeError:
                error_msg = repr(e)
            print(f"  - Call Skinning.py for weight optimization failed: {error_msg}")
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
        # Bounds checking for frame indices
        max_skeleton_frame = self.transforms.shape[0] - 1
        if (frame_start > max_skeleton_frame or frame_end > max_skeleton_frame or 
            reference_frame > max_skeleton_frame):
            print(f"WARNING: Frame index out of skeleton bounds!")
            print(f"  Requested: start={frame_start}, end={frame_end}, ref={reference_frame}")
            print(f"  Available skeleton frames: 0-{max_skeleton_frame}")
            
            # Clamp frame indices to valid range
            frame_start = min(frame_start, max_skeleton_frame)
            frame_end = min(frame_end, max_skeleton_frame)
            reference_frame = min(reference_frame, max_skeleton_frame)
            print(f"  Using clamped indices: start={frame_start}, end={frame_end}, ref={reference_frame}")
        
        # Get transformation matrix with bounds checking
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
        
        print(f"\nDEBUG: Weight optimization check:")
        print(f"  - optimize_weights: {optimize_weights}")
        print(f"  - self.skinning_weights is None: {self.skinning_weights is None}")
        print(f"  - Will call weight optimization: {optimize_weights and self.skinning_weights is None}")
        
        if optimize_weights and self.skinning_weights is None:
            print(f"\n  - Start weight optimization (use actual start frame {frame_start} as reference)...")
            optimization_start = time.time()
            
            if not self.optimize_weights_using_skinning(frame_start, frame_end, max_optimize_frames):
                print("  - Weight optimization failed, will use simple interpolation")
            
            optimization_time = time.time() - optimization_start
            print(f"  - Weight optimization total time: {optimization_time:.2f} seconds")
        elif self.skinning_weights is not None:
            print(f"\n  - Use existing weights matrix")
            print(f"  - Weights matrix shape: {self.skinning_weights.shape}")
            print(f"  - Reference frame index: {getattr(self, 'reference_frame_idx', 'Unknown')}")
        else:
            print(f"\n  - Skip weight optimization (optimize_weights=False)")
        
        # Generate interpolation frames
        print(f"\nStart generating {len(t_values)} interpolation frames...")
        frame_generation_start = time.time()
        
        for i, t in enumerate(t_values):
            frame_start_time = time.time()
            print(f"  Generate interpolation frame {i+1}/{len(t_values)} (t={t:.3f})...")
            
            try:
                # Use original frame_start, frame_end instead of actual_start, actual_end
                # This ensures the actual start frame is used as reference
                interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
                
                # Generate interpolation frame data - use original frame indices
                frame_data = self.generate_single_interpolated_frame(
                    frame_start, frame_end, t, interpolated_transforms, output_dir, i,
                    smooth_mesh, subdivide_iter, save_npy_files, save_standard_obj
                )
                
                if frame_data:
                    interpolated_frames.append(frame_data)
                    
                    # Debug specific frames
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
        Generate a single interpolated frame - restore to the simple correct logic of the Work version
        
        Correct Volumetric Interpolation process:
        1. Learn start frame weights  
        2. Start mesh with texture/colors
        3. Interpolate skeleton
        4. Apply relative transform via LBS
        5. Result: deformed start mesh with colors
        
        Args:
            frame_start: Start frame index
            frame_end: End frame index
            t: Interpolation parameter [0, 1]
            interpolated_transforms: Interpolated transformation matrix
            output_dir: Output directory
            frame_idx: Frame index
        """
        # Step 1: Load reference mesh (use start frame as reference - this is critical!)
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
        
        # Use global parameters to normalize the reference mesh
        reference_vertices_norm = self.normalize_mesh_vertices(reference_vertices, global_normalization_params)
        
        # Step 2: Apply LBS transformation to generate mesh
        print(f"DEBUG: skinning_weights status check:")
        print(f"  - self.skinning_weights is not None: {self.skinning_weights is not None}")
        if self.skinning_weights is not None:
            print(f"  - Weights matrix shape: {self.skinning_weights.shape}")
            print(f"  - Weights matrix range: [{self.skinning_weights.min():.6f}, {self.skinning_weights.max():.6f}]")
        else:
            print(f"  - No weights matrix! Will use fallback interpolation")
        
        if self.skinning_weights is not None:
            # Ensure the weight matrix matches the number of vertices (simple handling)
            if self.skinning_weights.shape[0] != len(reference_vertices_norm):
                print(f"  - Weights matrix vertices ({self.skinning_weights.shape[0]}) does not match reference mesh vertices ({len(reference_vertices_norm)})")
                # Adjust the weight matrix size
                if self.skinning_weights.shape[0] > len(reference_vertices_norm):
                    self.skinning_weights = self.skinning_weights[:len(reference_vertices_norm)]
                else:
                    # Extend the weight matrix
                    extended_weights = np.zeros((len(reference_vertices_norm), self.skinning_weights.shape[1]))
                    extended_weights[:self.skinning_weights.shape[0]] = self.skinning_weights
                    # Use distance initialization for new vertices
                    keypoints = self.keypoints[frame_start, :, :3]
                    remaining_vertices = reference_vertices_norm[self.skinning_weights.shape[0]:]
                    if len(remaining_vertices) > 0:
                        distances = cdist(remaining_vertices, keypoints)
                        remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                        remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                        extended_weights[self.skinning_weights.shape[0]:] = remaining_weights
                    self.skinning_weights = extended_weights
            
            # Step 3: Use relative transformation for LBS (critical logic!)
            print(f"  - Use relative transformation for LBS...")
            print(f"DEBUG: transformation information:")
            print(f"  - frame_start: {frame_start}, frame_end: {frame_end}, t: {t}")
            
            # Get reference frame transformation (use start frame as reference)
            reference_transforms = self.transforms[frame_start]
            print(f"  - reference_transforms shape: {reference_transforms.shape}")
            print(f"  - interpolated_transforms shape: {interpolated_transforms.shape}")
            
            # Check if the interpolated transform is really changing
            # transforms_diff = np.linalg.norm(interpolated_transforms - reference_transforms)
            # print(f"  - Difference between interpolated and reference transforms magnitude: {transforms_diff:.6f}")
            # if transforms_diff < 1e-6:
            #     print(f"  - Warning: interpolated transform almost unchanged!")
            
            # Calculate the relative transformation from the reference frame to the interpolated frame
            relative_transforms = np.zeros_like(interpolated_transforms)
            for j in range(self.num_joints):
                if np.linalg.det(reference_transforms[j][:3, :3]) > 1e-6:
                    ref_inv = np.linalg.inv(reference_transforms[j])
                    relative_transforms[j] = interpolated_transforms[j] @ ref_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # Check the relative transformation
            # relative_magnitude = np.linalg.norm(relative_transforms - np.eye(4))
            # print(f"  - Relative transform magnitude: {relative_magnitude:.6f}")
            # if relative_magnitude < 1e-6:
            #     print(f"  - Warning: relative transform is almost identity matrix!")
            
            # Apply LBS transformation (using relative transformation)
            print(f"DEBUG: LBS transformation before vertices statistics:")
            print(f"  - reference_vertices_norm shape: {reference_vertices_norm.shape}")
            print(f"  - reference_vertices_norm range: [{reference_vertices_norm.min():.3f}, {reference_vertices_norm.max():.3f}]")
            
            transformed_vertices_norm = self.apply_lbs_transform(
                reference_vertices_norm, self.skinning_weights, relative_transforms
            )
            
            print(f"DEBUG: LBS transformation after vertices statistics:")
            print(f"  - transformed_vertices_norm shape: {transformed_vertices_norm.shape}")
            print(f"  - transformed_vertices_norm range: [{transformed_vertices_norm.min():.3f}, {transformed_vertices_norm.max():.3f}]")
            
            # Check if the transformation really happened
            # vertices_diff = np.linalg.norm(transformed_vertices_norm - reference_vertices_norm)
            # print(f"  - Vertices change magnitude: {vertices_diff:.6f}")
            # if vertices_diff < 1e-6:
            #     print(f"  - Warning: vertices almost unchanged! LBS may not be effective")
            
            # Use global parameters to denormalize
            transformed_vertices = self.denormalize_mesh_vertices(
                transformed_vertices_norm, global_normalization_params
            )
            
            # Step 4: Fix coordinate alignment
            print(f"  - Fix coordinate alignment...")
            
            # Calculate the mesh center
            mesh_center = np.mean(transformed_vertices, axis=0)
            
            # Calculate the bone center (using the interpolated absolute transformation)
            joint_positions = interpolated_transforms[:, :3, 3]
            joint_center = np.mean(joint_positions, axis=0)
            
            # Calculate the offset
            offset = mesh_center - joint_center
            
            # Adjust the bone position to the mesh coordinate system
            adjusted_transforms = interpolated_transforms.copy()
            for j in range(self.num_joints):
                adjusted_transforms[j][:3, 3] += offset
            
            # Update the interpolated transformation
            interpolated_transforms = adjusted_transforms
            
            print(f"  - Mesh center: {mesh_center}")
            print(f"  - Before adjustment joint center: {joint_center}")
            print(f"  - After adjustment joint center: {np.mean(adjusted_transforms[:, :3, 3], axis=0)}")
            print(f"  - Offset: {offset}")
        else:
            # If there are no weights, use improved vertex interpolation
            mesh_start = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
            mesh_end = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_end]))
            
            vertices_start = np.asarray(mesh_start.vertices)
            vertices_end = np.asarray(mesh_end.vertices)
            
            min_vertices = min(len(vertices_start), len(vertices_end))
            
            # Normalize two meshes
            vertices_start_norm = self.normalize_mesh_vertices(vertices_start[:min_vertices], global_normalization_params)
            vertices_end_norm = self.normalize_mesh_vertices(vertices_end[:min_vertices], global_normalization_params)
            
            # Align mesh and skeleton
            print(f"Align mesh and skeleton (no weights mode)...")
            vertices_start_aligned = self.align_mesh_with_skeleton(vertices_start_norm, interpolated_transforms)
            vertices_end_aligned = self.align_mesh_with_skeleton(vertices_end_norm, interpolated_transforms)
            
            # Interpolate in the normalized space
            interpolated_vertices_norm = (1-t) * vertices_start_aligned + t * vertices_end_aligned
            
            # Denormalize
            transformed_vertices = self.denormalize_mesh_vertices(interpolated_vertices_norm, global_normalization_params)
        
        # 第5步：创建插值网格（与dual方式一致，直接用起始帧的faces/uvs/colors）
        origin_frame_data = self._prepare_frame_data(frame_start, frame_end, frame_start)
        interpolated_mesh = self._create_output_mesh(
            transformed_vertices, origin_frame_data['reference_faces'], origin_frame_data['reference_uvs'], origin_frame_data['reference_vertex_colors'],
            smooth_mesh, subdivide_iter
        )

        # Interpolate keypoints
        interpolated_keypoints = self.interpolate_keypoints(frame_start, frame_end, t)
        
        # Step 6: Save interpolated frame data
        frame_data = {
            'frame_idx': frame_idx,
            'interpolation_t': t,
            'mesh': interpolated_mesh,
            'transforms': interpolated_transforms,
            'keypoints': interpolated_keypoints,
            'vertices': transformed_vertices
        }
        
        # Save to file (if needed)
        if output_dir:
            mesh_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}.obj"
            
            print(f"  - Output path: {mesh_output_path}")
            print(f"  - Mesh vertices: {len(transformed_vertices)}")
            print(f"  - Mesh faces: {len(reference_faces) if reference_faces is not None else 0}")
            
            try:
                # Ensure the output directory exists
                mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save mesh
                success = o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
                if success:
                    print(f"  - Mesh file saved successfully: {mesh_output_path}")
                else:
                    print(f"  - Mesh file saved failed: {mesh_output_path}")
                
                # Save transform data
                # transform_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_transforms.npy"
                # np.save(transform_output_path, interpolated_transforms)
                # print(f"  - Transform data saved: {transform_output_path}")
                
                # keypoints_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_keypoints.npy"
                # np.save(keypoints_output_path, interpolated_keypoints)
                # print(f"  - Keypoints data saved: {keypoints_output_path}")
                
            except Exception as e:
                print(f"  - Error saving file: {e}")
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
            # Save standard obj file (if needed)
            if save_standard_obj:
                mesh_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}.obj"
                o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
            
            # # Save transform data (if needed)
            # if save_npy_files:
            #     transform_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_transforms.npy"
            #     np.save(transform_output_path, interpolated_transforms)
                
            #     keypoints_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_keypoints.npy"
            #     np.save(keypoints_output_path, interpolated_keypoints)
        
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
    Core logic:
    - t < 0.5: Use the start frame as the reference frame and apply the skinning weights of the start frame
    - t >= 0.5: Use the end frame as the reference frame and apply the skinning weights of the end frame
    - Do not blend weights, because the vertices of different frames are inherently different
    - Simple piecewise strategy to avoid complex weight blending
    """
    
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None):
        """
        Initialize the dual reference interpolator
        
        Args:
            skeleton_data_dir: Skeleton data directory path
            mesh_folder_path: Mesh file directory path  
            weights_path: Pre-calculated skinning weights path (optional)
        """
        super().__init__(skeleton_data_dir, mesh_folder_path, weights_path)
        
        # Dual reference interpolator specific states
        self.start_skinning_weights = None
        self.end_skinning_weights = None
        self.start_reference_frame = None
        self.end_reference_frame = None
        
        print(f"Dual reference interpolator initialized")
        print(f"  - Core characteristics: t < 0.5 use start weights, t >= 0.5 use end weights")
        print(f"  - Avoid weight blending, maintain vertex independence")
    
    def optimize_dual_reference_weights(self, frame_start, frame_end, max_optimize_frames=5):
        """
        Optimize the skinning weights for the start and end frames
        
        Args:
            frame_start: Start frame index
            frame_end: End frame index
            max_optimize_frames: Maximum number of optimization frames
            
        Returns:
            success: Whether successful
        """
        print(f"Start dual reference weight optimization...")
        print(f"  - Start frame: {frame_start}")
        print(f"  - End frame: {frame_end}")
        print(f"  - Max optimize frames: {max_optimize_frames}")
        
        # 记录参考帧
        self.start_reference_frame = frame_start
        self.end_reference_frame = frame_end
        
        start_time = time.time()
        
        try:
            from Skinning import AutoSkinning
            
            # 确保使用正确的输出目录
            if hasattr(self, 'output_dir') and self.output_dir:
                base_output_dir = Path(self.output_dir)
            else:
                base_output_dir = Path("output")
                self.output_dir = str(base_output_dir)
            
            # 为起始帧和结束帧分别创建权重文件路径
            weights_dir = Path(self.weights_path) if self.weights_path else Path("output/skinning_weights")
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            
            # 优化起始帧权重
            print(f"\n Optimize start frame weights (frame {frame_start})...")
            success_start, start_weights_path = self._optimize_frame_weights(
                frame_start, max_optimize_frames, weights_dir, "start"
            )
            
            if success_start:
                print(f"Start frame weights optimization successful")
                # 加载起始帧权重
                data = np.load(start_weights_path)
                self.start_skinning_weights = data['weights']
                print(f"Start frame weights shape: {self.start_skinning_weights.shape}")
            else:
                print(f"Start frame weights optimization failed")
                return False
            
            # 优化结束帧权重
            print(f"\n Optimize end frame weights (frame {frame_end})...")
            success_end, end_weights_path = self._optimize_frame_weights(
                frame_end, max_optimize_frames, weights_dir, "end"
            )
            
            if success_end:
                print(f"End frame weights optimization successful")
                # 加载结束帧权重
                data = np.load(end_weights_path)
                self.end_skinning_weights = data['weights']
                print(f"End frame weights shape: {self.end_skinning_weights.shape}")
            else:
                print(f"End frame weights optimization failed")
                return False
            
            total_time = time.time() - start_time
            print(f"\nDual reference weight optimization completed")
            print(f"  - Total time: {total_time:.2f} seconds")
            print(f"  - Start frame weights: {self.start_skinning_weights.shape}")
            print(f"  - End frame weights: {self.end_skinning_weights.shape}")
            
            return True
            
        except Exception as e:
            print(f"Dual reference weight optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _optimize_frame_weights(self, reference_frame, max_optimize_frames, weights_dir, frame_type):
        """
        Optimize the weights for a single reference frame
        
        Args:
            reference_frame: Reference frame index
            max_optimize_frames: Maximum number of optimization frames  
            weights_path: Path to save the weights file
            frame_type: Frame type identifier ("start" or "end")
            
        Returns:
            success: Whether successful
        """
        try:
            # 检查是否已存在权重文件
            
            
            from Skinning import AutoSkinning
            
            # 初始化Skinning系统
            skinner = AutoSkinning(
                skeleton_data_dir=self.skeleton_data_dir,
                reference_frame_idx=reference_frame
            )
            
            # 加载网格序列
            skinner.load_mesh_sequence(self.mesh_folder_path)
            
            # 选择优化帧（以参考帧为中心，确保不越界，且总数不超过max_optimize_frames）
            optimize_frames = []
            num_meshes = len(skinner.mesh_files)
            half = max_optimize_frames // 2

            # 计算起止索引，优先保证参考帧在中间，且不越界
            start_idx = max(0, reference_frame - half)
            end_idx = start_idx + max_optimize_frames
            if end_idx > num_meshes:
                end_idx = num_meshes
                start_idx = max(0, end_idx - max_optimize_frames)

            for i in range(start_idx, end_idx):
                optimize_frames.append(i)
            
            
            if not optimize_frames:
                print(f"No frames to optimize")
                return False
            
            weights_path = weights_dir / f"ref{reference_frame}_opt{optimize_frames[0]}-{optimize_frames[-1]}_num{len(optimize_frames)}.npz"
            if weights_path.exists():
                print(f"Found existing {frame_type} weights file: {weights_path}")
                return True, weights_path
            
            print(f"  - {frame_type} reference frame: {reference_frame}")
            print(f"  - {frame_type} optimization frames: {optimize_frames}")
            
            # 权重优化
            print(f"  - Call Skinning.py to optimize {frame_type} weights...")
            optimization_start = time.time()
            
            skinner.skinning_weights = skinner.optimize_reference_frame_skinning(
                optimization_frames=optimize_frames,
                regularization_lambda=0.01,
                max_iter=200
            )
            
            optimization_time = time.time() - optimization_start
            
            if skinner.skinning_weights is not None:
                print(f"{frame_type} weights optimization completed")
                print(f"    - Weights matrix shape: {skinner.skinning_weights.shape}")
                print(f"    - Optimization time: {optimization_time:.2f} seconds")
                
                # 保存权重
                skinner.save_skinning_weights(str(weights_path))
                print(f"    - Weights saved to: {weights_path}")
                
                return True, weights_path
            else:
                print(f"{frame_type} weights optimization failed")
                return False, None
                
        except Exception as e:
            # 安全处理可能包含Unicode字符的异常信息
            try:
                error_msg = str(e)
            except UnicodeEncodeError:
                error_msg = repr(e)
            print(f"{frame_type} weights optimization error: {error_msg}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None, smooth_mesh=False, subdivide_iter=3,
                                   use_vertex_colors=False, save_npy_files=False, save_standard_obj=True):
        """
        Generate clean dual reference interpolated frames
        
        Args:
            frame_start: Start frame index
            frame_end: End frame index
            num_interpolate: Number of interpolation frames
            max_optimize_frames: Maximum number of optimization frames
            optimize_weights: Whether to optimize weights
            output_dir: Output directory
            debug_frames: Debug frames list
            smooth_mesh: Whether to smooth the mesh
            subdivide_iter: Subdivide iteration times
            use_vertex_colors: Whether to use vertex colors
            save_npy_files: Whether to save npy files
            save_standard_obj: Whether to save standard obj files
            
        Returns:
            interpolated_frames: Interpolated frames list
        """
        total_start_time = time.time()
        
        print(f"Start generating clean dual reference interpolated frames...")
        print(f"  - Start frame: {frame_start}")
        print(f"  - End frame: {frame_end}")
        print(f"  - Interpolation frames: {num_interpolate}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Piecewise logic: t < 0.5 use start weights, t >= 0.5 use end weights")
        
        # 设置输出目录
        if output_dir:
            self.interpolation_output_dir = output_dir
            if hasattr(self, 'output_dir') and self.output_dir:
                pass
            else:
                interpolation_path = Path(output_dir)
                self.output_dir = str(interpolation_path.parent.parent)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 检查帧索引范围
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"Frame index out of range: start_frame={frame_start}, end_frame={frame_end}, available frames={len(self.mesh_files)}")
        
        if frame_start == frame_end:
            raise ValueError(f"Start frame cannot be equal to end frame: {frame_start} == {frame_end}")
        
        # 确定实际的起始和结束帧
        actual_start = min(frame_start, frame_end)
        actual_end = max(frame_start, frame_end)
        is_reverse = frame_start > frame_end
        
        start_file = self.mesh_files[actual_start].name
        end_file = self.mesh_files[actual_end].name
        print(f"  - Using files: {start_file} (index {actual_start}) -> {end_file} (index {actual_end})")
        
        # 生成插值参数
        t_values = np.linspace(0, 1, num_interpolate + 2)[1:-1]  # 排除起始和结束帧
        
        if is_reverse:
            t_values = 1.0 - t_values
            print(f"  - Reverse interpolation detected: {frame_start} -> {frame_end}")
        
        interpolated_frames = []
        
        # 双参考权重优化
        if optimize_weights and (self.start_skinning_weights is None or self.end_skinning_weights is None):
            print(f"\nStart dual reference weight optimization...")
            optimization_start = time.time()
            
            if not self.optimize_dual_reference_weights(frame_start, frame_end, max_optimize_frames):
                print("Dual reference weight optimization failed, using simple interpolation")
            
            optimization_time = time.time() - optimization_start
            print(f"Dual reference weight optimization total time: {optimization_time:.2f} seconds")
        elif self.start_skinning_weights is not None and self.end_skinning_weights is not None:
            print(f"\nUse existing dual reference weights")
            print(f"  - Start frame weights shape: {self.start_skinning_weights.shape}")
            print(f"  - End frame weights shape: {self.end_skinning_weights.shape}")
        else:
            print(f"\nSkip weight optimization (optimize_weights=False)")
        
        # 生成插值帧
        print(f"\nStart generating {len(t_values)} interpolated frames...")
        frame_generation_start = time.time()
        
        for i, t in enumerate(t_values):
            frame_start_time = time.time()
            print(f"Generate interpolated frame {i+1}/{len(t_values)} (t={t:.3f})...")
            
            try:
                # 插值骨骼变换
                interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
                
                # 生成干净的双参考插值帧
                frame_data = self._generate_clean_dual_reference_frame(
                    frame_start, frame_end, t, interpolated_transforms, output_dir, i,
                    smooth_mesh, subdivide_iter, save_npy_files, save_standard_obj
                )
                
                if frame_data:
                    interpolated_frames.append(frame_data)
                    
                    if debug_frames and i in debug_frames:
                        self.debug_interpolation_frame(frame_data, i, output_dir)
                    
                    frame_time = time.time() - frame_start_time
                    print(f"    Completed (time: {frame_time:.2f} seconds)")
                else:
                    print(f"    Failed")
                    
            except Exception as e:
                print(f"    Failed to generate interpolated frame: {e}")
                import traceback
                traceback.print_exc()
        
        frame_generation_time = time.time() - frame_generation_start
        total_time = time.time() - total_start_time
        
        print(f"\nClean dual reference interpolated frames generated!")
        print(f"  - Number of generated frames: {len(interpolated_frames)}")
        print(f"  - Frame generation time: {frame_generation_time:.2f} seconds")
        print(f"  - Average per frame: {frame_generation_time/len(t_values):.3f} seconds")
        print(f"  - Total time: {total_time:.2f} seconds")
        
        return interpolated_frames
    
    def _generate_clean_dual_reference_frame(self, frame_start, frame_end, t, interpolated_transforms, 
                                           output_dir, frame_idx, smooth_mesh=False, subdivide_iter=3, 
                                           save_npy_files=False, save_standard_obj=True):
        """
        Generate clean dual reference interpolated frames
        
        Core logic:
        - t < 0.5: Use start frame as reference, apply start frame's skinning weights
        - t >= 0.5: Use end frame as reference, apply end frame's skinning weights
        - Do not blend weights, maintain vertex independence
        
        Args:
            frame_start: Start frame index
            frame_end: End frame index
            t: Interpolation parameter [0, 1]
            interpolated_transforms: Interpolated transformation matrix
            output_dir: Output directory
            frame_idx: Frame index
            smooth_mesh: Whether to smooth the mesh
            subdivide_iter: Subdivide iteration times
            save_npy_files: Whether to save npy files
            save_standard_obj: Whether to save standard obj files
            
        Returns:
            frame_data: Interpolated frame data
        """
        # Core piecewise logic: decide which reference frame and weights to use
        if t < 0.5:
            # Use start frame as reference
            reference_frame = frame_start
            reference_weights = self.start_skinning_weights
            reference_label = "start"
            print(f" t={t:.3f} < 0.5, use start frame {frame_start} as reference")
        else:
            # Use end frame as reference
            reference_frame = frame_end
            reference_weights = self.end_skinning_weights
            reference_label = "end"
            print(f" t={t:.3f} >= 0.5, use end frame {frame_end} as reference")
        
        # Prepare frame data
        frame_data = self._prepare_frame_data(frame_start, frame_end, reference_frame)
        reference_vertices = frame_data['reference_vertices']
        reference_faces = frame_data['reference_faces']
        reference_uvs = frame_data['reference_uvs']
        reference_vertex_colors = frame_data['reference_vertex_colors']
        global_normalization_params = frame_data['global_normalization_params']
        
        # Process skinning weights
        processed_weights = self._process_skinning_weights(
            reference_vertices, reference_frame, global_normalization_params, reference_weights
        )
        
        # Apply LBS transformation to generate mesh
        if processed_weights is not None:
            print(f" Use {reference_label} frame weights for LBS deformation...")
            print(f"      - Weights matrix shape: {processed_weights.shape}")
            
            # Apply LBS deformation
            transformed_vertices = self._apply_lbs_deformation(
                reference_vertices, processed_weights, interpolated_transforms, 
                reference_frame, global_normalization_params
            )
            
            # 应用坐标系对齐
            transformed_vertices, interpolated_transforms = self._apply_coordinate_alignment(
                transformed_vertices, interpolated_transforms
            )
        else:
            # 如果没有权重，使用简单的顶点插值
            print(f" No weights, using simple vertex interpolation...")
            mesh_start = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_start]))
            mesh_end = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_end]))
            
            vertices_start = np.asarray(mesh_start.vertices)
            vertices_end = np.asarray(mesh_end.vertices)
            
            min_vertices = min(len(vertices_start), len(vertices_end))
            
            # 归一化两个网格
            vertices_start_norm = self.normalize_mesh_vertices(vertices_start[:min_vertices], global_normalization_params)
            vertices_end_norm = self.normalize_mesh_vertices(vertices_end[:min_vertices], global_normalization_params)
            
            # 对齐网格和骨骼
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
        
        # 准备帧数据
        frame_result = {
            'frame_idx': frame_idx,
            'interpolation_t': t,
            'reference_frame': reference_frame,
            'reference_label': reference_label,
            'mesh': interpolated_mesh,
            'transforms': interpolated_transforms,
            'keypoints': interpolated_keypoints,
            'vertices': transformed_vertices
        }
        
        # 保存到文件
        if output_dir:
            try:
                # 确保输出目录存在
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # 保存mesh文件（带参考标识）
                if save_standard_obj:
                    mesh_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_ref_{reference_label}.obj"
                    success = o3d.io.write_triangle_mesh(str(mesh_output_path), interpolated_mesh)
                    if success:
                        print(f"      Mesh file saved successfully: {mesh_output_path}")
                    else:
                        print(f"      Mesh file saved failed: {mesh_output_path}")
                
                # 保存额外数据
                if save_npy_files:
                    transform_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_transforms.npy"
                    np.save(transform_output_path, interpolated_transforms)
                    
                    keypoints_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_keypoints.npy"
                    np.save(keypoints_output_path, interpolated_keypoints)
                    
                    # 保存插值元数据
                    metadata = {
                        'frame_idx': frame_idx,
                        'interpolation_t': t,
                        'reference_frame': reference_frame,
                        'reference_label': reference_label,
                        'frame_start': frame_start,
                        'frame_end': frame_end
                    }
                    metadata_output_path = Path(output_dir) / f"interpolated_frame_{frame_idx:04d}_metadata.json"
                    with open(metadata_output_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
            except Exception as e:
                print(f"      Error saving file: {e}")
                import traceback
                traceback.print_exc()
        
        return frame_result



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
