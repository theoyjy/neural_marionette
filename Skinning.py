import json
import numpy as np
import trimesh
import os
import sys
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.neighbors import NearestNeighbors
import pickle
from tqdm import tqdm
import open3d as o3d
from scipy.sparse import csr_matrix

class AutoSkinning:
    def __init__(self, skeleton_data_dir, reference_frame_idx=0):
        """
        Initialize reverse mesh unifier
        
        Args:
            skeleton_data_dir: Directory path containing skeleton data npy files
            reference_frame_idx: Reference frame index (used as unification target)
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.reference_frame_idx = reference_frame_idx
        
        # 加载骨骼数据
        self.load_skeleton_data()
        
        self.reference_mesh = None
        self.canonicalization_maps = {}
        
        # Normalization parameters for each frame, for handling independent normalization of each mesh
        self.frame_normalization_params = {}
        
        # LBS related attributes
        self.skinning_weights = None  # [V, J] Vertex to joint weight matrix
        self.rest_pose_vertices = None  # Rest pose vertex coordinates
        self.rest_pose_transforms = None  # Rest pose transformation matrices
        
    def load_skeleton_data(self):
        """Load Skeleton Data in Numpy Format"""
        try:
            # Load Keypoints Data [num_frames, num_joints, 4] (x, y, z, confidence)
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # Load Transforms Matrix [num_frames, num_joints, 4, 4]
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # Load Parent Node Relationship [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"Successfully Loaded Skeleton Data:")
            print(f"  - Number of Frames: {self.num_frames}")
            print(f"  - Number of Joints: {self.num_joints}")
            print(f"  - Keypoints Shape: {self.keypoints.shape} (includes confidence)")
            print(f"  - Transforms Matrix Shape: {self.transforms.shape}")
            print(f"  - Parent Node Relationship Shape: {self.parents.shape}")
            
        except Exception as e:
            raise ValueError(f"Failed to Load Skeleton Data: {e}")
            
        # Try to Load Other Optional Data
        try:
            if (self.skeleton_data_dir / 'affinity.npy').exists():
                self.affinity = np.load(self.skeleton_data_dir / 'affinity.npy')
                print(f"  - Affinity Matrix Shape: {self.affinity.shape}")
            else:
                self.affinity = None
                
            if (self.skeleton_data_dir / 'priority.npy').exists():
                self.priority = np.load(self.skeleton_data_dir / 'priority.npy')
                print(f"  - Priority Shape: {self.priority.shape}")
            else:
                self.priority = None
                
            if (self.skeleton_data_dir / 'A.npy').exists():
                self.A = np.load(self.skeleton_data_dir / 'A.npy')
                print(f"  - A Matrix Shape: {self.A.shape}")
            else:
                self.A = None
                
            if (self.skeleton_data_dir / 'rotations.npy').exists():
                self.rotations = np.load(self.skeleton_data_dir / 'rotations.npy')
                print(f"  - Rotation Matrix Shape: {self.rotations.shape}")
            else:
                self.rotations = None
        except Exception as e:
            print(f"Warning: Failed to Load Optional Data: {e}")

    def compute_mesh_normalization_params(self, mesh):
        """
        Compute normalization parameters for a single mesh (simulating episodic_normalization process)
        
        Args:
            mesh: Open3D mesh object
            
        Returns:
            normalization_params: Dictionary of normalization parameters
        """
        vertices = np.asarray(mesh.vertices)
        
        # Compute bounding box (same logic as episodic_normalization)
        bmax = np.amax(vertices, axis=0)
        bmin = np.amin(vertices, axis=0)
        blen = (bmax - bmin).max()
        
        # Default normalization parameters (consistent with episodic_normalization defaults)
        scale = 1.0
        x_trans = 0.0
        z_trans = 0.0
        
        params = {
            'bmin': bmin,
            'bmax': bmax,
            'blen': blen,
            'scale': scale,
            'x_trans': x_trans,
            'z_trans': z_trans
        }
        
        return params
    
    def normalize_mesh_vertices(self, vertices, normalization_params):
        """
        Normalize mesh vertices using given normalization parameters
        
        Args:
            vertices: Original vertex coordinates
            normalization_params: Normalization parameters
            
        Returns:
            normalized_vertices: Normalized vertex coordinates
        """
        params = normalization_params
        
        # Apply the same transformation as episodic_normalization
        # Formula: ((seq - bmin) * scale / (blen + 1e-5)) * 2 - 1 + [x_trans, 0, z_trans]
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        normalized = ((vertices - params['bmin']) * params['scale'] / (params['blen'] + 1e-5)) * 2 - 1 + trans_offset
        
        return normalized

    def load_reference_vertices(self, reference_frame_idx):
        """
        Load reference vertices
        """
        self.reference_frame_idx = reference_frame_idx
        self.reference_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[self.reference_frame_idx]))
        print(f"Number of Reference Vertices: {len(self.reference_mesh.vertices)}")
        
        # pre-compute reference mesh normalization params
        self.frame_normalization_params[self.reference_frame_idx] = self.compute_mesh_normalization_params(self.reference_mesh)


    def load_mesh_sequence(self, mesh_folder_path):
        """
        Prepare mesh sequence
        """
        self.mesh_folder_path = Path(mesh_folder_path)
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        
        if len(self.mesh_files) != self.num_frames:
            print(f"Warning: ({len(self.mesh_files)}) and ({self.num_frames}) do not match")

        self.load_reference_vertices(self.reference_frame_idx)
        
        
    def apply_lbs_transform(self, rest_vertices, weights, transforms):
        """
        Apply Linear Blend Skinning transformation
        
        Args:
            rest_vertices: Rest pose vertices [V, 3]
            weights: Skinning weights [V, J]
            transforms: Joint transformation matrices [J, 4, 4]
            
        Returns:
            transformed_vertices: Transformed vertices [V, 3]
        """
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        # Convert vertices to homogeneous coordinates
        rest_vertices_homo = np.hstack([rest_vertices, np.ones((num_vertices, 1))])  # [V, 4]
        
        # Initialize output vertices
        transformed_vertices = np.zeros((num_vertices, 3))
        
        # Apply transformation and blend for each joint
        for j in range(num_joints):
            # Get current joint transformation matrix [4, 4]
            joint_transform = transforms[j]
            
            # Transform all vertices
            transformed_homo = (joint_transform @ rest_vertices_homo.T).T  # [V, 4]
            transformed_xyz = transformed_homo[:, :3]  # [V, 3]
            
            # Blend according to weights
            joint_weights = weights[:, j:j+1]  # [V, 1]
            transformed_vertices += joint_weights * transformed_xyz
        
        return transformed_vertices
    
    def compute_lbs_loss(self, weights_flat, rest_vertices, target_vertices, transforms, 
                        regularization_lambda=0.01):
        """
        Compute LBS loss function
        
        Args:
            weights_flat: Flattened weight vector [V*J]
            rest_vertices: Rest pose vertices [V, 3]
            target_vertices: Target vertices [V, 3]
            transforms: Joint transformation matrices [J, 4, 4]
            regularization_lambda: Regularization coefficient
            
        Returns:
            loss: Scalar loss value
        """
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        # Reshape weight matrix
        weights = weights_flat.reshape(num_vertices, num_joints)
        
        # Ensure weights are non-negative and normalized
        weights = np.maximum(weights, 0)
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        # Apply LBS transformation
        predicted_vertices = self.apply_lbs_transform(rest_vertices, weights, transforms)
        
        # Compute reconstruction loss
        reconstruction_loss = np.mean(np.sum((predicted_vertices - target_vertices)**2, axis=1))
        
        # Add sparsity regularization (encourage each vertex to be influenced by few joints)
        sparsity_loss = np.mean(np.sum(weights**2, axis=1))
        
        # Add smoothness regularization (optional, requires mesh connectivity information)
        smoothness_loss = 0.0
        
        total_loss = reconstruction_loss + regularization_lambda * sparsity_loss + smoothness_loss
        
        return total_loss
    
    def optimize_skinning_weights_for_frame(self, target_frame_idx, max_iter=1000, 
                                          init_method='distance_based', regularization_lambda=0.01):
        """
        Optimize skinning weights for specific frame
        
        Args:
            target_frame_idx: Target frame index
            max_iter: Maximum number of iterations
            init_method: Initialization method ('distance_based', 'uniform', 'random')
            
        Returns:
            optimized_weights: Optimized weight matrix [V, J]
            loss_history: Loss history
        """
        # Get data
        rest_vertices = self.rest_pose_vertices  # Use reference frame as rest pose
        target_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[target_frame_idx]))
        target_vertices = np.asarray(target_mesh.vertices)
        
        # Normalization processing (keep same space as keypoints)
        if target_frame_idx not in self.frame_normalization_params:
            self.frame_normalization_params[target_frame_idx] = self.compute_mesh_normalization_params(target_mesh)
        
        target_vertices_norm = self.normalize_mesh_vertices(target_vertices, self.frame_normalization_params[target_frame_idx])
        rest_vertices_norm = self.normalize_mesh_vertices(rest_vertices, self.frame_normalization_params[self.reference_frame_idx])
        
        # Ensure rest and target vertex counts match
        original_rest_vertices = len(rest_vertices_norm)  # Save original rest vertex count
        if len(rest_vertices_norm) != len(target_vertices_norm):
            print(f"Warning: rest vertex count ({len(rest_vertices_norm)}) does not match target vertex count ({len(target_vertices_norm)})")
            # Use smaller count for optimization
            min_vertices = min(len(rest_vertices_norm), len(target_vertices_norm))
            rest_vertices_norm_used = rest_vertices_norm[:min_vertices]
            target_vertices_norm_used = target_vertices_norm[:min_vertices]
            print(f"Adjusted to use first {min_vertices} vertices for optimization")
        else:
            rest_vertices_norm_used = rest_vertices_norm
            target_vertices_norm_used = target_vertices_norm
        
        num_vertices = len(rest_vertices_norm_used)
        
        # Get transformation matrices
        target_transforms = self.transforms[target_frame_idx]  # [J, 4, 4]
        rest_transforms = self.transforms[self.reference_frame_idx]  # [J, 4, 4]
        
        # Compute relative transformation (from rest pose to target pose)
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(self.num_joints):
            if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:  # Check if invertible
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            else:
                relative_transforms[j] = np.eye(4)
        
        num_joints = self.num_joints
        
        # Initialize weights
        if init_method == 'distance_based':
            # Distance-based initialization
            keypoints = self.keypoints[self.reference_frame_idx, :, :3]
            distances = cdist(rest_vertices_norm_used, keypoints)
            weights_init = np.exp(-distances**2 / (2 * 0.1**2))
            weights_init = weights_init / (np.sum(weights_init, axis=1, keepdims=True) + 1e-8)
        elif init_method == 'uniform':
            # Uniform initialization
            weights_init = np.ones((num_vertices, num_joints)) / num_joints
        else:
            # Random initialization
            weights_init = np.random.rand(num_vertices, num_joints)
            weights_init = weights_init / (np.sum(weights_init, axis=1, keepdims=True) + 1e-8)
        
        # Flatten weights for optimization
        weights_flat_init = weights_init.flatten()
        
        # Define objective function
        def objective(weights_flat):
            return self.compute_lbs_loss(weights_flat, rest_vertices_norm_used, target_vertices_norm_used, 
                                       relative_transforms, regularization_lambda)
        
        # Use efficient optimization method: large block parallel optimization
        print(f"Using efficient optimization method...")
        print(f"Vertex count: {num_vertices}, Joint count: {num_joints}")
        
        # Use multi-threaded optimization method for all mesh sizes
        if num_vertices > 5000:
            # For large meshes, use sampling strategy
            # Ensure sample size does not exceed available vertices
            sample_size = min(3000, num_vertices, len(target_vertices_norm_used))
            sample_indices = np.random.choice(min(num_vertices, len(target_vertices_norm_used)), sample_size, replace=False)
            print(f"Large mesh detected, sampling {sample_size} vertices for optimization (rest: {num_vertices}, target: {len(target_vertices_norm_used)})")
            
            # Sample vertices and targets
            sampled_rest = rest_vertices_norm_used[sample_indices]
            sampled_target = target_vertices_norm_used[sample_indices]
            sampled_weights_init = weights_init[sample_indices]
            
            # Optimize sampled weights
            optimized_sampled_weights, _ = self.optimize_sampled_weights(
                sampled_rest, sampled_target, sampled_weights_init, 
                relative_transforms, regularization_lambda, max_iter // 5
            )
            
            # Interpolate optimization results to all vertices
            optimized_weights = weights_init.copy()
            optimized_weights[sample_indices] = optimized_sampled_weights
            
            # Use nearest neighbor interpolation for unsampled vertices
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(sampled_rest)
            distances, indices = nbrs.kneighbors(rest_vertices_norm_used)
            
            for i in range(num_vertices):
                if i not in sample_indices:
                    # Use distance-weighted averaging
                    weights_sum = np.sum(1.0 / (distances[i] + 1e-6))
                    weighted_weights = np.zeros(num_joints)
                    for j, neighbor_idx in enumerate(indices[i]):
                        weight = (1.0 / (distances[i][j] + 1e-6)) / weights_sum
                        weighted_weights += weight * optimized_sampled_weights[neighbor_idx]
                    optimized_weights[i] = weighted_weights
                    # Re-normalize
                    optimized_weights[i] = optimized_weights[i] / (np.sum(optimized_weights[i]) + 1e-8)
        else:
            # For small meshes, use direct multi-threaded optimization (no sampling)
            print(f"Small mesh detected, directly optimizing all {num_vertices} vertices")
            optimized_weights, _ = self.optimize_sampled_weights(
                rest_vertices_norm_used, target_vertices_norm_used, weights_init,
                relative_transforms, regularization_lambda, max_iter // 3
            )
        
        # If optimization used fewer vertices than original rest vertices, need to expand to original size
        if num_vertices < original_rest_vertices:
            print(f"Expanding weight matrix: {optimized_weights.shape} -> ({original_rest_vertices}, {num_joints})")
            # Create full-size weight matrix
            full_optimized_weights = np.zeros((original_rest_vertices, num_joints))
            # Copy optimized weights
            full_optimized_weights[:num_vertices] = optimized_weights
            # Use distance-weighted initialization for remaining vertices
            if original_rest_vertices > num_vertices:
                keypoints = self.keypoints[self.reference_frame_idx, :, :3]
                remaining_vertices = rest_vertices_norm[num_vertices:original_rest_vertices]
                distances = cdist(remaining_vertices, keypoints)
                remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                full_optimized_weights[num_vertices:] = remaining_weights
            
            optimized_weights = full_optimized_weights
        
        # Compute final loss (using vertices from optimization)
        final_loss = self.compute_lbs_loss(optimized_weights[:num_vertices].flatten(), rest_vertices_norm_used, 
                                         target_vertices_norm_used, relative_transforms, regularization_lambda)
        
        print(f"Optimization completed, final loss: {final_loss:.6f}")
        print(f"Returned weight matrix shape: {optimized_weights.shape}")
        
        return optimized_weights, [final_loss]
    
    def optimize_sampled_weights(self, rest_vertices, target_vertices, weights_init, 
                               relative_transforms, regularization_lambda, max_iter):
        """
        优化采样的权重（高效版本 - 多线程 + 向量化）
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        num_vertices, num_joints = weights_init.shape
        optimized_weights = weights_init.copy()
        
        print(f"Efficient optimization of sampled weights: {num_vertices} vertices")
        
        # 优化参数
        chunk_size = 1000  # 更大的块以提高并行效率
        learning_rate = 0.03  # 更大的学习率
        num_threads = min(8, (num_vertices + chunk_size - 1) // chunk_size)  # 动态线程数
        
        print(f"Using {num_threads} threads, chunk size: {chunk_size}")
        
        # 预计算变换矩阵的转置，避免重复计算
        transforms_t = relative_transforms.transpose(0, 2, 1)  # [J, 4, 4] -> [J, 4, 4]
        
        def optimize_chunk(chunk_data):
            """Optimize a single data chunk"""
            chunk_indices, chunk_rest, chunk_target, chunk_weights = chunk_data
            
            # 向量化的LBS变换计算
            def fast_apply_lbs(vertices, weights, transforms):
                """Fast LBS transformation (vectorized version)"""
                num_verts = vertices.shape[0]
                vertices_homo = np.hstack([vertices, np.ones((num_verts, 1))])  # [N, 4]
                
                # 预计算所有关节的变换结果
                transformed_vertices = np.zeros((num_verts, 3))
                
                # 向量化计算
                for j in range(num_joints):
                    # 使用预计算的变换矩阵
                    joint_transform = transforms[j]  # [4, 4]
                    transformed_homo = (joint_transform @ vertices_homo.T).T  # [N, 4]
                    transformed_xyz = transformed_homo[:, :3]  # [N, 3]
                    
                    # 权重应用
                    joint_weights = weights[:, j:j+1]  # [N, 1]
                    transformed_vertices += joint_weights * transformed_xyz
                
                return transformed_vertices
            
            # 快速梯度计算
            def compute_gradient_fast(weights, vertices, target):
                """Fast gradient computation (vectorized)"""
                predicted = fast_apply_lbs(vertices, weights, relative_transforms)
                error = predicted - target
                
                # 计算每个顶点的主要关节
                top_k = min(3, num_joints)  # 只优化前3个关节
                top_joints = np.argsort(weights, axis=1)[:, -top_k:]  # [N, top_k]
                
                gradient = np.zeros_like(weights)
                eps = 1e-5
                
                # 批量计算梯度
                for k in range(top_k):
                    joint_idx = top_joints[:, k]  # [N]
                    
                    # 创建扰动权重
                    weights_plus = weights.copy()
                    for i in range(len(weights)):
                        weights_plus[i, joint_idx[i]] += eps
                    
                    # 归一化
                    weights_plus = weights_plus / (np.sum(weights_plus, axis=1, keepdims=True) + 1e-8)
                    
                    # 计算扰动后的预测
                    predicted_plus = fast_apply_lbs(vertices, weights_plus, relative_transforms)
                    error_plus = predicted_plus - target
                    
                    # 计算梯度
                    loss = np.mean(np.sum(error**2, axis=1))
                    loss_plus = np.mean(np.sum(error_plus**2, axis=1))
                    
                    # 批量更新梯度
                    for i in range(len(weights)):
                        gradient[i, joint_idx[i]] = (loss_plus - loss) / eps
                
                return gradient
            
            # 主优化循环
            for sub_iter in range(3):  # Increase inner iteration count
                # 计算当前预测
                predicted = fast_apply_lbs(chunk_rest, chunk_weights, relative_transforms)
                error = predicted - chunk_target
                
                # 计算梯度
                gradient = compute_gradient_fast(chunk_weights, chunk_rest, chunk_target)
                
                # 更新权重
                chunk_weights -= learning_rate * gradient
                chunk_weights = np.maximum(chunk_weights, 0)
                chunk_weights = chunk_weights / (np.sum(chunk_weights, axis=1, keepdims=True) + 1e-8)
            
            # 计算最终损失
            final_predicted = fast_apply_lbs(chunk_rest, chunk_weights, relative_transforms)
            chunk_loss = np.mean(np.sum((final_predicted - chunk_target)**2, axis=1))
            
            return chunk_indices, chunk_weights, chunk_loss
        
        # Main optimization loop
        start_time = time.time()
        for iteration in range(max_iter):
            total_loss = 0.0
            
            # Randomly shuffle vertex order
            perm = np.random.permutation(num_vertices)
            
            # Prepare data chunks
            chunk_data_list = []
            for start_idx in range(0, num_vertices, chunk_size):
                end_idx = min(start_idx + chunk_size, num_vertices)
                chunk_indices = perm[start_idx:end_idx]
                
                chunk_rest = rest_vertices[chunk_indices]
                chunk_target = target_vertices[chunk_indices]
                chunk_weights = optimized_weights[chunk_indices].copy()
                
                chunk_data_list.append((chunk_indices, chunk_rest, chunk_target, chunk_weights))
            
            # Multi-threaded parallel optimization
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(optimize_chunk, chunk_data): chunk_data 
                                 for chunk_data in chunk_data_list}
                
                # Collect results
                for future in as_completed(future_to_chunk):
                    chunk_indices, chunk_weights, chunk_loss = future.result()
                    optimized_weights[chunk_indices] = chunk_weights
                    total_loss += chunk_loss * len(chunk_weights) / num_vertices
            
            # Progress report
            if iteration % 5 == 0:  # More frequent progress report
                elapsed = time.time() - start_time
                print(f"Iteration {iteration}: loss = {total_loss:.6f}, time = {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Optimization completed, total time: {total_time:.2f}s")
        
        # Compute final loss
        final_predicted = self.apply_lbs_transform(rest_vertices, optimized_weights, relative_transforms)
        final_loss = np.mean(np.sum((final_predicted - target_vertices)**2, axis=1))
        
        return optimized_weights, [final_loss]
    
    def optimize_standard_weights(self, rest_vertices, target_vertices, weights_init,
                                relative_transforms, regularization_lambda, max_iter):
        """
        标准权重优化（中等规模）
        """
        num_vertices, num_joints = weights_init.shape
        optimized_weights = weights_init.copy()
        
        print(f"Standard optimization: {num_vertices} vertices")
        
        chunk_size = 200  # Medium-sized chunk size
        learning_rate = 0.01
        
        for iteration in range(max_iter):
            total_loss = 0.0
            
            for start_idx in range(0, num_vertices, chunk_size):
                end_idx = min(start_idx + chunk_size, num_vertices)
                
                chunk_rest = rest_vertices[start_idx:end_idx]
                chunk_target = target_vertices[start_idx:end_idx]
                chunk_weights = optimized_weights[start_idx:end_idx].copy()
                
                # 简化的梯度下降
                for sub_iter in range(2):  # 只做2次内层迭代
                    predicted = self.apply_lbs_transform(chunk_rest, chunk_weights, relative_transforms)
                    error = predicted - chunk_target
                    
                    # 计算梯度（只对部分关节）
                    gradient = np.zeros_like(chunk_weights)
                    eps = 1e-5
                    
                    for i in range(min(len(chunk_weights), 50)):  # 只优化前50个顶点
                        top_joints = np.argsort(chunk_weights[i])[-3:]  # 只优化前3个关节
                        
                        for j in top_joints:
                            chunk_weights_plus = chunk_weights.copy()
                            chunk_weights_plus[i, j] += eps
                            chunk_weights_plus[i] = chunk_weights_plus[i] / (np.sum(chunk_weights_plus[i]) + 1e-8)
                            
                            predicted_plus = self.apply_lbs_transform(chunk_rest, chunk_weights_plus, relative_transforms)
                            error_plus = predicted_plus - chunk_target
                            
                            loss = np.mean(np.sum(error**2, axis=1))
                            loss_plus = np.mean(np.sum(error_plus**2, axis=1))
                            
                            gradient[i, j] = (loss_plus - loss) / eps
                    
                    chunk_weights -= learning_rate * gradient
                    chunk_weights = np.maximum(chunk_weights, 0)
                    chunk_weights = chunk_weights / (np.sum(chunk_weights, axis=1, keepdims=True) + 1e-8)
                
                optimized_weights[start_idx:end_idx] = chunk_weights
                
                predicted = self.apply_lbs_transform(chunk_rest, chunk_weights, relative_transforms)
                chunk_loss = np.mean(np.sum((predicted - chunk_target)**2, axis=1))
                total_loss += chunk_loss * len(chunk_weights) / num_vertices
            
            if iteration % 5 == 0:
                print(f"Standard optimization iteration {iteration}: loss = {total_loss:.6f}")
        
        return optimized_weights

    def calc_optimize_frames(self, start_frame_idx, end_frame_idx, step):
        """
        Calculate optimized frames
        """
        total_frames = len(self.mesh_files)
        if start_frame_idx is None:
            start_frame_idx = 0
        if end_frame_idx is None:
            end_frame_idx = total_frames
        if step is None:
            step = 2
        optimization_frames = list(range(start_frame_idx, end_frame_idx, step))
        
        # Remove reference frame
        if self.reference_frame_idx in optimization_frames:
            optimization_frames.remove(self.reference_frame_idx)

        return optimization_frames
    
    def _initialize_weights_by_distance(self):
        """
        Distance-based weight initialization backup solution
        """
        try:
            num_vertices = len(self.rest_pose_vertices)
            num_joints = self.keypoints.shape[1]
            
            # Use reference frame keypoints
            keypoints = self.keypoints[self.reference_frame_idx, :, :3]
            
            # Calculate distance from vertices to keypoints
            from scipy.spatial.distance import cdist
            distances = cdist(self.rest_pose_vertices, keypoints)
            
            # Use Gaussian weights
            weights = np.exp(-distances**2 / (2 * 0.1**2))
            weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
            
            print(f"Using distance initialization for backup weights: {weights.shape}")
            return weights
            
        except Exception as e:
            print(f"Distance initialization failed: {e}")
            # Last backup solution: uniform weights
            num_vertices = len(self.rest_pose_vertices) if hasattr(self, 'rest_pose_vertices') else 1000
            num_joints = self.keypoints.shape[1] if hasattr(self, 'keypoints') else 24
            weights = np.ones((num_vertices, num_joints)) / num_joints
            print(f"Using uniform weights as last backup solution: {weights.shape}")
            return weights

    def optimize_reference_frame_skinning(self, optimization_frames=None, regularization_lambda=0.01, max_iter=1000):
        """
        Optimize skinning weights for reference frame
        
        Args:
            regularization_lambda: regularization coefficient
            max_iter: maximum number of iterations
            
        Returns:
            skinning_weights: 优化后的权重矩阵 [V, J]
        """
        # 设置rest pose为reference frame
        self.rest_pose_vertices = np.asarray(self.reference_mesh.vertices)
        self.rest_pose_transforms = self.transforms[self.reference_frame_idx]
        
        print(f"Start optimizing skinning weights for reference frame (frame {self.reference_frame_idx})...")
        
        # Optimize all other frames
        all_weights = []
        all_losses = []
        
        # 选择几个代表性帧进行优化
        if optimization_frames is None:
            optimization_frames = self.calc_optimize_frames(None, None, None)

        
        print(f"Using {len(optimization_frames)} frames for weight optimization: {optimization_frames}")
        
        # 多线程并行优化各帧权重
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # 确定线程数 - 限制为最多4个线程以平衡性能和资源使用
        num_threads = min(4, len(optimization_frames))
        print(f"Using {num_threads} threads for parallel weight optimization...")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_frame = {
                executor.submit(self.optimize_skinning_weights_for_frame, 
                               frame_idx, max_iter, regularization_lambda): frame_idx 
                for frame_idx in optimization_frames
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_frame), 
                             total=len(optimization_frames), 
                             desc="Multi-threaded optimization of each frame weights"):
                frame_idx = future_to_frame[future]
                try:
                    weights, loss_history = future.result()
                    all_weights.append(weights)
                    all_losses.extend(loss_history)
                    print(f"Frame {frame_idx} optimization completed")
                except Exception as e:
                    print(f"Frame {frame_idx} optimization failed: {e}")
                    # Use distance initialization as backup solution
                    backup_weights = self._initialize_weights_by_distance()
                    all_weights.append(backup_weights)
                    print(f"Using backup weight matrix")
        
        print(f"Multi-threaded optimization completed, processed {len(all_weights)} frames")
        
        # 验证所有权重矩阵形状一致
        if all_weights:
            # 检查所有权重矩阵的形状
            shapes = [w.shape for w in all_weights]
            print(f"Collected weight matrix shapes: {shapes}")
            
            # 确保所有形状相同
            if len(set(shapes)) > 1:
                print("Warning: Detected different shape weight matrices, unifying shapes...")
                # 找到最大的形状
                max_vertices = max(shape[0] for shape in shapes)
                max_joints = max(shape[1] for shape in shapes)
                target_shape = (max_vertices, max_joints)
                print(f"Target shape: {target_shape}")
                
                # 统一所有权重矩阵的形状
                unified_weights = []
                for i, weights in enumerate(all_weights):
                    if weights.shape != target_shape:
                        print(f"Adjusting weight matrix {i}: {weights.shape} -> {target_shape}")
                        unified = np.zeros(target_shape)
                        # Copy existing weights
                        unified[:weights.shape[0], :weights.shape[1]] = weights
                        # Use distance initialization for new vertices
                        if weights.shape[0] < target_shape[0]:
                            keypoints = self.keypoints[self.reference_frame_idx, :, :3]
                            remaining_vertices = self.rest_pose_vertices[weights.shape[0]:target_shape[0]]
                            if len(remaining_vertices) > 0:
                                # Normalize remaining vertices
                                remaining_norm = self.normalize_mesh_vertices(
                                    remaining_vertices, 
                                    self.frame_normalization_params[self.reference_frame_idx]
                                )
                                distances = cdist(remaining_norm, keypoints)
                                remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                                remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                                unified[weights.shape[0]:target_shape[0], :remaining_weights.shape[1]] = remaining_weights
                        unified_weights.append(unified)
                    else:
                        unified_weights.append(weights)
                all_weights = unified_weights
                print(f"Shape unification completed, all weight matrix shapes: {[w.shape for w in all_weights]}")
            
            # Average all frame weights as final result
            self.skinning_weights = np.mean(all_weights, axis=0)
            print(f"Weight optimization completed, using average weights of {len(all_weights)} frames")
            print(f"Final weight matrix shape: {self.skinning_weights.shape}")
            
            # Compute weight statistics
            weights_per_vertex = np.sum(self.skinning_weights > 0.01, axis=1)  # Number of joints affecting each vertex
            print(f"Average number of joints affecting each vertex: {np.mean(weights_per_vertex):.2f}")
            print(f"Weight sparsity: {np.mean(self.skinning_weights > 0.01):.3f}")
        else:
            print("Warning: No successful weight optimization for any frame")
            return None
        
        return self.skinning_weights
    
    def validate_skinning_weights(self, test_frames=None):
        """
        Validate skinning weights effect
        
        Args:
            test_frames: 测试帧列表，None表示测试所有帧
            
        Returns:
            validation_results: 验证结果字典
        """
        if self.skinning_weights is None:
            print("Error: Skinning weights not calculated, please call optimize_reference_frame_skinning first")
            return None
        
        if test_frames is None:
            test_frames = list(range(min(len(self.mesh_files), 20)))  # Limit test frames
        
        results = {
            'frame_errors': {},
            'average_error': 0.0,
            'max_error': 0.0,
            'min_error': float('inf')
        }
        
        print("Validate skinning weights effect on frames:", test_frames)
        
        rest_vertices_norm = self.normalize_mesh_vertices(
            self.rest_pose_vertices, 
            self.frame_normalization_params[self.reference_frame_idx]
        )
        
        total_error = 0.0
        valid_frames = 0
        
        for frame_idx in tqdm(test_frames, desc="Validate frames"):
            if frame_idx >= len(self.mesh_files):
                continue
            
            # Load target mesh
            target_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
            target_vertices = np.asarray(target_mesh.vertices)
            
            # Normalize
            if frame_idx not in self.frame_normalization_params:
                self.frame_normalization_params[frame_idx] = self.compute_mesh_normalization_params(target_mesh)
            
            target_vertices_norm = self.normalize_mesh_vertices(
                target_vertices, 
                self.frame_normalization_params[frame_idx]
            )
            
            # Compute relative transformation
            target_transforms = self.transforms[frame_idx]
            rest_transforms = self.transforms[self.reference_frame_idx]
            
            relative_transforms = np.zeros_like(target_transforms)
            for j in range(self.num_joints):
                if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                    rest_inv = np.linalg.inv(rest_transforms[j])
                    relative_transforms[j] = target_transforms[j] @ rest_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # Use LBS to predict vertex positions
            predicted_vertices = self.apply_lbs_transform(
                rest_vertices_norm, self.skinning_weights, relative_transforms
            )
            
            # Handle vertex number mismatch
            if predicted_vertices.shape[0] != target_vertices_norm.shape[0]:
                print(f"Frame {frame_idx}: vertex number mismatch (predicted: {predicted_vertices.shape[0]}, target: {target_vertices_norm.shape[0]})")
                # Use smaller number for comparison
                min_vertices = min(predicted_vertices.shape[0], target_vertices_norm.shape[0])
                predicted_vertices_used = predicted_vertices[:min_vertices]
                target_vertices_used = target_vertices_norm[:min_vertices]
                print(f"Using first {min_vertices} vertices for error calculation")
            else:
                predicted_vertices_used = predicted_vertices
                target_vertices_used = target_vertices_norm
            
            # Compute error
            vertex_errors = np.linalg.norm(predicted_vertices_used - target_vertices_used, axis=1)
            frame_error = np.mean(vertex_errors)
            
            results['frame_errors'][frame_idx] = {
                'mean_error': frame_error,
                'max_error': np.max(vertex_errors),
                'min_error': np.min(vertex_errors),
                'std_error': np.std(vertex_errors)
            }
            
            total_error += frame_error
            valid_frames += 1
            
            results['max_error'] = max(results['max_error'], frame_error)
            results['min_error'] = min(results['min_error'], frame_error)
        
        if valid_frames > 0:
            results['average_error'] = total_error / valid_frames
            
            print(f"Validation completed!")
            print(f"Average reconstruction error: {results['average_error']:.6f}")
            print(f"Maximum error: {results['max_error']:.6f}")
            print(f"Minimum error: {results['min_error']:.6f}")
        
        return results
    
    def test_lbs_reconstruction_quality(self, test_frames=None, save_meshes=False, output_dir="output/lbs_test"):
        """
        Test LBS reconstruction quality
        
        Args:
            test_frames: 测试帧列表，None表示自动选择
            save_meshes: 是否保存重建的网格
            output_dir: 输出目录
            
        Returns:
            detailed_results: 详细测试结果
        """
        import time
        import matplotlib.pyplot as plt
        
        if self.skinning_weights is None:
            print("Error: Need to load or calculate skinning weights first")
            return None
        
        # Automatically select test frames
        if test_frames is None:
            total_frames = len(self.mesh_files)
            if total_frames <= 20:
                test_frames = list(range(total_frames))
            else:
                # Select representative frames: start, middle, end, and some random frames
                test_frames = []
                test_frames.extend([0, 1, 2])  # Start frames
                test_frames.extend([total_frames//4, total_frames//2, 3*total_frames//4])  # Middle frames
                test_frames.extend([total_frames-3, total_frames-2, total_frames-1])  # End frames
                # Add some random frames
                import random
                random_frames = random.sample(range(3, total_frames-3), min(6, total_frames-9))
                test_frames.extend(random_frames)
                test_frames = sorted(list(set(test_frames)))  # Remove duplicates and sort
        
        print(f"🔍 Test LBS reconstruction quality")
        print(f"Test frames: {test_frames}")
        print(f"Reference frame: {self.reference_frame_idx}")
        
        # Create output directory
        if save_meshes:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare rest pose data
        rest_vertices_norm = self.normalize_mesh_vertices(
            self.rest_pose_vertices, 
            self.frame_normalization_params[self.reference_frame_idx]
        )
        
        # Test results
        detailed_results = {
            'test_config': {
                'test_frames': test_frames,
                'reference_frame': self.reference_frame_idx,
                'num_vertices': len(rest_vertices_norm),
                'num_joints': self.num_joints,
                'save_meshes': save_meshes,
                'output_dir': str(output_dir) if save_meshes else None
            },
            'frame_results': {},
            'summary_stats': {},
            'performance_stats': {}
        }
        
        all_errors = []
        all_times = []
        distance_errors = []  # Error vs. distance from reference frame
        
        print(f"\nStart testing {len(test_frames)} frames...")
        
        for i, frame_idx in enumerate(tqdm(test_frames, desc="Test reconstruction quality")):
            if frame_idx >= len(self.mesh_files):
                continue
            
            # Load target mesh
            target_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
            target_vertices = np.asarray(target_mesh.vertices)
            
            # Normalize
            if frame_idx not in self.frame_normalization_params:
                self.frame_normalization_params[frame_idx] = self.compute_mesh_normalization_params(target_mesh)
            
            target_vertices_norm = self.normalize_mesh_vertices(
                target_vertices, 
                self.frame_normalization_params[frame_idx]
            )
            
            # Compute relative transformation
            target_transforms = self.transforms[frame_idx]
            rest_transforms = self.transforms[self.reference_frame_idx]
            
            relative_transforms = np.zeros_like(target_transforms)
            for j in range(self.num_joints):
                if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                    rest_inv = np.linalg.inv(rest_transforms[j])
                    relative_transforms[j] = target_transforms[j] @ rest_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # LBS reconstruction test
            start_time = time.time()
            predicted_vertices = self.apply_lbs_transform(
                rest_vertices_norm, self.skinning_weights, relative_transforms
            )
            lbs_time = time.time() - start_time
            
            # Compute detailed error metrics
            vertex_errors = np.linalg.norm(predicted_vertices - target_vertices_norm, axis=1)
            
            frame_result = {
                'frame_idx': frame_idx,
                'distance_from_ref': abs(frame_idx - self.reference_frame_idx),
                'mean_error': float(np.mean(vertex_errors)),
                'median_error': float(np.median(vertex_errors)),
                'std_error': float(np.std(vertex_errors)),
                'min_error': float(np.min(vertex_errors)),
                'max_error': float(np.max(vertex_errors)),
                'rmse': float(np.sqrt(np.mean(vertex_errors**2))),
                'p90_error': float(np.percentile(vertex_errors, 90)),
                'p95_error': float(np.percentile(vertex_errors, 95)),
                'p99_error': float(np.percentile(vertex_errors, 99)),
                'lbs_time': lbs_time,
                'vertices_with_large_error': int(np.sum(vertex_errors > 0.05)),  # Number of vertices with large error
                'error_ratio_large': float(np.sum(vertex_errors > 0.05) / len(vertex_errors))  # Large error ratio
            }
            
            detailed_results['frame_results'][frame_idx] = frame_result
            
            # Collect statistics
            all_errors.extend(vertex_errors)
            all_times.append(lbs_time)
            distance_errors.append((frame_result['distance_from_ref'], frame_result['mean_error']))
            
            # Save mesh (if needed)
            if save_meshes:
                # Save reconstructed mesh
                reconstructed_mesh = o3d.geometry.TriangleMesh()
                reconstructed_mesh.vertices = o3d.utility.Vector3dVector(predicted_vertices)
                if hasattr(target_mesh, 'triangles') and len(target_mesh.triangles) > 0:
                    reconstructed_mesh.triangles = target_mesh.triangles
                
                mesh_output_path = output_path / f"frame_{frame_idx:06d}_reconstructed.obj"
                o3d.io.write_triangle_mesh(str(mesh_output_path), reconstructed_mesh)
                
                # Save error visualization mesh
                normalized_errors = vertex_errors / np.max(vertex_errors)
                error_colors = plt.cm.plasma(normalized_errors)[:, :3]  # Use plasma color mapping
                
                error_mesh = o3d.geometry.TriangleMesh()
                error_mesh.vertices = target_mesh.vertices
                error_mesh.triangles = target_mesh.triangles
                error_mesh.vertex_colors = o3d.utility.Vector3dVector(error_colors)
                
                error_output_path = output_path / f"frame_{frame_idx:06d}_error_colored.obj"
                o3d.io.write_triangle_mesh(str(error_output_path), error_mesh)
        
        # Compute summary statistics
        if all_errors:
            all_errors = np.array(all_errors)
            detailed_results['summary_stats'] = {
                'total_tested_frames': len(test_frames),
                'total_vertices': len(all_errors),
                'overall_mean_error': float(np.mean(all_errors)),
                'overall_median_error': float(np.median(all_errors)),
                'overall_std_error': float(np.std(all_errors)),
                'overall_min_error': float(np.min(all_errors)),
                'overall_max_error': float(np.max(all_errors)),
                'overall_rmse': float(np.sqrt(np.mean(all_errors**2))),
                'overall_p90': float(np.percentile(all_errors, 90)),
                'overall_p95': float(np.percentile(all_errors, 95)),
                'overall_p99': float(np.percentile(all_errors, 99)),
                'vertices_with_large_error_total': int(np.sum(all_errors > 0.05)),
                'large_error_ratio': float(np.sum(all_errors > 0.05) / len(all_errors))
            }
        
        # Performance statistics
        if all_times:
            detailed_results['performance_stats'] = {
                'mean_lbs_time': float(np.mean(all_times)),
                'total_lbs_time': float(np.sum(all_times)),
                'min_lbs_time': float(np.min(all_times)),
                'max_lbs_time': float(np.max(all_times)),
                'fps_estimate': float(len(all_times) / np.sum(all_times)) if np.sum(all_times) > 0 else 0
            }
        
        # Analyze error vs. distance
        if distance_errors:
            distances, errors = zip(*distance_errors)
            if len(set(distances)) > 1:  # Different distance data points
                correlation = np.corrcoef(distances, errors)[0, 1]
                detailed_results['distance_analysis'] = {
                    'correlation_with_distance': float(correlation),
                    'distance_error_pairs': distance_errors
                }
        
        # Output summary
        print(f"\nTest completed summary:")
        if 'summary_stats' in detailed_results:
            stats = detailed_results['summary_stats']
            print(f"Overall average error: {stats['overall_mean_error']:.6f}")
            print(f"Overall RMSE: {stats['overall_rmse']:.6f}")
            print(f"Error range: [{stats['overall_min_error']:.6f}, {stats['overall_max_error']:.6f}]")
            print(f"Large error vertex ratio: {stats['large_error_ratio']*100:.2f}%")
        
        if 'performance_stats' in detailed_results:
            perf = detailed_results['performance_stats']
            print(f"Average LBS time: {perf['mean_lbs_time']:.3f}s")
            print(f"Estimated FPS: {perf['fps_estimate']:.1f} FPS")
        
        if 'distance_analysis' in detailed_results:
            dist_analysis = detailed_results['distance_analysis']
            print(f"Error vs. distance from reference frame correlation: {dist_analysis['correlation_with_distance']:.3f}")
        
        # Save detailed results
        if save_meshes:
            import json
            results_path = output_path / "test_results.json"
            
            # 准备可序列化的结果
            serializable_results = detailed_results.copy()
            # 移除不可序列化的部分
            if 'distance_analysis' in serializable_results:
                serializable_results['distance_analysis'] = {
                    'correlation_with_distance': detailed_results['distance_analysis']['correlation_with_distance']
                }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nDetailed results saved:")
            print(f"    Test results: {results_path}")
            print(f"    Reconstructed mesh: {output_path}/*_reconstructed.obj")
            print(f"    Error visualization: {output_path}/*_error_colored.obj")
        
        return detailed_results
    
    def save_skinning_weights(self, output_path):
        """
        Save self's skinning weights
        """
        if self.skinning_weights is None:
            print("Error: No skinning weights to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save weights and related information
        skinning_data = {
            'weights': self.skinning_weights,
            'rest_vertices': self.rest_pose_vertices,
            'rest_transforms': self.rest_pose_transforms,
            'reference_frame_idx': self.reference_frame_idx,
            'num_vertices': self.skinning_weights.shape[0],
            'num_joints': self.skinning_weights.shape[1]
        }
        
        np.savez_compressed(output_path, **skinning_data)
        print(f"Skinning weights saved to: {output_path}")
        

    def load_skinning_weights(self, input_path):
        """
        Load skinning weights
        
        Args:
            input_path: input file path
        """
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"Error: File Does Not Exist: {input_path}")
            return False
        
        try:
            data = np.load(input_path)
            self.skinning_weights = data['weights']
            self.rest_pose_vertices = data['rest_vertices']
            self.rest_pose_transforms = data['rest_transforms']
            self.reference_frame_idx = data['reference_frame_idx'].item()  # Ensure integer
            
            print(f"Successfully loaded skinning weights:")
            print(f"  - Weight matrix shape: {self.skinning_weights.shape}")
            print(f"  - Rest pose vertices: {len(self.rest_pose_vertices)}")
            print(f"  - Reference frame: {data['reference_frame_idx']}")
            
            return self.skinning_weights
        except Exception as e:
            print(f"加载skinning权重失败: {e}")
            return None

def run_auto_skinning_pipeline(reference_frame_idx = 5):
    """
    Automatic skinning calculation and visualization pipeline
    """
    print("Start automatic skinning calculation Pipeline")
    print("=" * 60)
    
    # Configure paths
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_output_path = f"output/skinning_weights/skinning_weights_{reference_frame_idx}.npz"
    
    # Initialize
    skinner = AutoSkinning(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=reference_frame_idx  # Use the 5th frame as reference
    )
    
    # Load data
    print("Loading mesh sequence...")
    skinner.load_mesh_sequence(mesh_folder_path)
    skinner.rest_pose_transforms = skinner.transforms[skinner.reference_frame_idx]
    skinner.rest_pose_vertices = np.asarray(skinner.reference_mesh.vertices)
    
    print(f"Data loaded:")
    print(f"    Vertices: {len(skinner.rest_pose_vertices):,}")
    print(f"    Joints: {skinner.num_joints}")
    print(f"    Skeleton frames: {skinner.num_frames}")
    print(f"    Mesh files: {len(skinner.mesh_files)}")

    if not os.path.exists(weights_output_path):        
        # Optimize skinning weights
        print("\n🔧 Start optimizing skinning weights...")

        optimization_frames = skinner.calc_optimize_frames(reference_frame_idx - 10, reference_frame_idx + 10, 2)

        skinner.skinning_weights = skinner.optimize_reference_frame_skinning(
            regularization_lambda=0.01,
            optimization_frames=optimization_frames,
            max_iter=100  # Medium number of iterations
        )
        
        if skinner.skinning_weights is None:
            print("Skinning weights optimization failed")
            return
        
        # Save weights
        print(f"\nSave skinning weights to: {weights_output_path}")
        skinner.save_skinning_weights(weights_output_path)
    else:
        print(f"Skinning weights file already exists: {weights_output_path}")
        skinner.load_skinning_weights(weights_output_path)
    
    # Quick validation
    print("\nQuick validation skinning effect...")
    # Calculate available test frames
    max_skeleton_frame = skinner.num_frames - 1
    test_frames = list(range(0, max_skeleton_frame + 1, max_skeleton_frame // 10))
    if skinner.reference_frame_idx in test_frames:
        test_frames.remove(skinner.reference_frame_idx)

    print(f"Plan to test frames: {test_frames}")
    validation_results = skinner.validate_skinning_weights(test_frames=test_frames)
    
    if validation_results:
        print(f"\nValidation completed:")
        print(f"    Average error: {validation_results['average_error']:.6f}")
        print(f"    Error range: [{validation_results['min_error']:.6f}, {validation_results['max_error']:.6f}]")
        
        # Find the best and worst frames
        best_frame = min(validation_results['frame_errors'].items(), 
                        key=lambda x: x[1]['mean_error'])
        worst_frame = max(validation_results['frame_errors'].items(), 
                         key=lambda x: x[1]['mean_error'])
        
        print(f"    Best frame: {best_frame[0]} (error: {best_frame[1]['mean_error']:.6f})")
        print(f"    Worst frame: {worst_frame[0]} (error: {worst_frame[1]['mean_error']:.6f})")
        
        # Select frames to visualize
        viz_frames = [best_frame[0], worst_frame[0]]
        if len(test_frames) > 2:
            # Add a medium quality frame
            middle_frame = sorted(validation_results['frame_errors'].items(), 
                                key=lambda x: x[1]['mean_error'])[len(validation_results['frame_errors'])//2]
            viz_frames.append(middle_frame[0])
        
        viz_frames = sorted(list(set(viz_frames)))[:3]  # Maximum 3 frames
        
        print(f"\nPrepare to visualize reconstruction comparison (frames: {viz_frames})...")
        
        # Automatically run visualization
        run_reconstruction_visualization(skinner, viz_frames, weights_output_path)
    
    print("\nAutomatic skinning Pipeline completed!")
    print(f"Weights file: {weights_output_path}")
    print(f"Test results displayed")

def run_reconstruction_visualization(skinner, viz_frames, weights_path):
    """
    Run reconstruction visualization comparison
    """
    import subprocess
    import sys
    
    print(f"Start reconstruction visualization...")
    
    for frame_idx in viz_frames:
        print(f"    Visualize frame {frame_idx}...")
        try:
            # Run visualization script
            result = subprocess.run([
                sys.executable, "simple_visualize.py", str(frame_idx)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"    Frame {frame_idx} visualization completed")
                # Parse error information in output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Average error:' in line or 'Maximum error:' in line or 'RMSE:' in line:
                        print(f"      {line.strip()}")
            else:
                print(f"    Frame {frame_idx} visualization failed: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print(f"    Frame {frame_idx} visualization timeout")
        except Exception as e:
            print(f"    Frame {frame_idx} visualization error: {e}")
    
    # Run batch export
    print(f"\nExport mesh files for external viewing...")
    try:
        result = subprocess.run([
            sys.executable, "simple_visualize.py", "export"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"    Batch export completed")
            # Find export path
            lines = result.stdout.split('\n')
            for line in lines:
                if 'All files exported to:' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"    Batch export failed")
            
    except Exception as e:
        print(f"    Batch export error: {e}")

def main():
    """
    Main function - Run complete automatic skinning pipeline
    """
    args = sys.argv[1:]
    reference_frame_idx = 5
    if len(args) >= 1:
        reference_frame_idx = int(args[0])
    run_auto_skinning_pipeline(reference_frame_idx)

if __name__ == "__main__":
    main()