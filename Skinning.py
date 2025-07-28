import json
import numpy as np
import trimesh
import os
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
        初始化反向网格统一器
        
        Args:
            skeleton_data_dir: 包含骨骼数据npy文件的文件夹路径
            reference_frame_idx: 参考帧索引（用作统一的目标）
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.reference_frame_idx = reference_frame_idx
        
        # 加载骨骼数据
        self.load_skeleton_data()
        
        self.reference_mesh = None
        self.canonicalization_maps = {}
        
        # 每帧的归一化参数，用于处理每个mesh独立归一化的情况
        self.frame_normalization_params = {}
        
        # LBS相关属性
        self.skinning_weights = None  # [V, J] 顶点到关节的权重矩阵
        self.rest_pose_vertices = None  # 静息姿态顶点坐标
        self.rest_pose_transforms = None  # 静息姿态变换矩阵
        
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
            print(f"  - Keypoints Shape: {self.keypoints.shape} (包含置信度)")
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
        计算单个mesh的归一化参数（模拟episodic_normalization的过程）
        
        Args:
            mesh: Open3D mesh对象
            
        Returns:
            normalization_params: 归一化参数字典
        """
        vertices = np.asarray(mesh.vertices)
        
        # 计算边界框（与episodic_normalization相同的逻辑）
        bmax = np.amax(vertices, axis=0)
        bmin = np.amin(vertices, axis=0)
        blen = (bmax - bmin).max()
        
        # 默认的归一化参数（与episodic_normalization默认值一致）
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
        使用给定的归一化参数将mesh顶点归一化
        
        Args:
            vertices: 原始顶点坐标
            normalization_params: 归一化参数
            
        Returns:
            normalized_vertices: 归一化后的顶点坐标
        """
        params = normalization_params
        
        # 应用与episodic_normalization相同的变换
        # 公式: ((seq - bmin) * scale / (blen + 1e-5)) * 2 - 1 + [x_trans, 0, z_trans]
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
        应用Linear Blend Skinning变换
        
        Args:
            rest_vertices: 静息姿态顶点 [V, 3]
            weights: skinning权重 [V, J]
            transforms: 关节变换矩阵 [J, 4, 4]
            
        Returns:
            transformed_vertices: 变换后的顶点 [V, 3]
        """
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        # 将顶点转换为齐次坐标
        rest_vertices_homo = np.hstack([rest_vertices, np.ones((num_vertices, 1))])  # [V, 4]
        
        # 初始化输出顶点
        transformed_vertices = np.zeros((num_vertices, 3))
        
        # 对每个关节应用变换并混合
        for j in range(num_joints):
            # 获取当前关节的变换矩阵 [4, 4]
            joint_transform = transforms[j]
            
            # 变换所有顶点
            transformed_homo = (joint_transform @ rest_vertices_homo.T).T  # [V, 4]
            transformed_xyz = transformed_homo[:, :3]  # [V, 3]
            
            # 根据权重混合
            joint_weights = weights[:, j:j+1]  # [V, 1]
            transformed_vertices += joint_weights * transformed_xyz
        
        return transformed_vertices
    
    def compute_lbs_loss(self, weights_flat, rest_vertices, target_vertices, transforms, 
                        regularization_lambda=0.01):
        """
        计算LBS损失函数
        
        Args:
            weights_flat: 展平的权重向量 [V*J]
            rest_vertices: 静息姿态顶点 [V, 3]
            target_vertices: 目标顶点 [V, 3]
            transforms: 关节变换矩阵 [J, 4, 4]
            regularization_lambda: 正则化系数
            
        Returns:
            loss: 标量损失值
        """
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        # 重塑权重矩阵
        weights = weights_flat.reshape(num_vertices, num_joints)
        
        # 确保权重非负且归一化
        weights = np.maximum(weights, 0)
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        # 应用LBS变换
        predicted_vertices = self.apply_lbs_transform(rest_vertices, weights, transforms)
        
        # 计算重建损失
        reconstruction_loss = np.mean(np.sum((predicted_vertices - target_vertices)**2, axis=1))
        
        # 添加稀疏性正则化（鼓励每个顶点只受少数关节影响）
        sparsity_loss = np.mean(np.sum(weights**2, axis=1))
        
        # 添加平滑性正则化（可选，需要网格连接信息）
        smoothness_loss = 0.0
        
        total_loss = reconstruction_loss + regularization_lambda * sparsity_loss + smoothness_loss
        
        return total_loss
    
    def optimize_skinning_weights_for_frame(self, target_frame_idx, max_iter=1000, 
                                          init_method='distance_based', regularization_lambda=0.01):
        """
        为特定帧优化skinning权重
        
        Args:
            target_frame_idx: 目标帧索引
            max_iter: 最大迭代次数
            init_method: 初始化方法 ('distance_based', 'uniform', 'random')
            
        Returns:
            optimized_weights: 优化后的权重矩阵 [V, J]
            loss_history: 损失历史
        """
        # 获取数据
        rest_vertices = self.rest_pose_vertices  # 使用reference frame作为rest pose
        target_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[target_frame_idx]))
        target_vertices = np.asarray(target_mesh.vertices)
        
        # 归一化处理（保持与keypoints相同的空间）
        if target_frame_idx not in self.frame_normalization_params:
            self.frame_normalization_params[target_frame_idx] = self.compute_mesh_normalization_params(target_mesh)
        
        target_vertices_norm = self.normalize_mesh_vertices(target_vertices, self.frame_normalization_params[target_frame_idx])
        rest_vertices_norm = self.normalize_mesh_vertices(rest_vertices, self.frame_normalization_params[self.reference_frame_idx])
        
        # 确保rest和target顶点数量匹配
        original_rest_vertices = len(rest_vertices_norm)  # 保存原始rest顶点数
        if len(rest_vertices_norm) != len(target_vertices_norm):
            print(f"警告: rest顶点数 ({len(rest_vertices_norm)}) 与target顶点数 ({len(target_vertices_norm)}) 不匹配")
            # 使用较小的数量进行优化
            min_vertices = min(len(rest_vertices_norm), len(target_vertices_norm))
            rest_vertices_norm_used = rest_vertices_norm[:min_vertices]
            target_vertices_norm_used = target_vertices_norm[:min_vertices]
            print(f"调整为使用前 {min_vertices} 个顶点进行优化")
        else:
            rest_vertices_norm_used = rest_vertices_norm
            target_vertices_norm_used = target_vertices_norm
        
        num_vertices = len(rest_vertices_norm_used)
        
        # 获取变换矩阵
        target_transforms = self.transforms[target_frame_idx]  # [J, 4, 4]
        rest_transforms = self.transforms[self.reference_frame_idx]  # [J, 4, 4]
        
        # 计算相对变换（从rest pose到target pose）
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(self.num_joints):
            if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:  # 检查是否可逆
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            else:
                relative_transforms[j] = np.eye(4)
        
        num_joints = self.num_joints
        
        # 初始化权重
        if init_method == 'distance_based':
            # 基于距离的初始化
            keypoints = self.keypoints[self.reference_frame_idx, :, :3]
            distances = cdist(rest_vertices_norm_used, keypoints)
            weights_init = np.exp(-distances**2 / (2 * 0.1**2))
            weights_init = weights_init / (np.sum(weights_init, axis=1, keepdims=True) + 1e-8)
        elif init_method == 'uniform':
            # 均匀初始化
            weights_init = np.ones((num_vertices, num_joints)) / num_joints
        else:
            # 随机初始化
            weights_init = np.random.rand(num_vertices, num_joints)
            weights_init = weights_init / (np.sum(weights_init, axis=1, keepdims=True) + 1e-8)
        
        # 展平权重用于优化
        weights_flat_init = weights_init.flatten()
        
        # 定义目标函数
        def objective(weights_flat):
            return self.compute_lbs_loss(weights_flat, rest_vertices_norm_used, target_vertices_norm_used, 
                                       relative_transforms, regularization_lambda)
        
        # 使用高效的优化方法：大块并行优化
        print(f"使用高效优化方法...")
        print(f"顶点数: {num_vertices}, 关节数: {num_joints}")
        
        # 大幅减少计算量
        if num_vertices > 10000:
            # 对于大网格，使用采样策略
            # 确保采样大小不超过可用顶点数
            sample_size = min(5000, num_vertices, len(target_vertices_norm_used))
            sample_indices = np.random.choice(min(num_vertices, len(target_vertices_norm_used)), sample_size, replace=False)
            print(f"大网格检测，采样 {sample_size} 个顶点进行优化 (rest: {num_vertices}, target: {len(target_vertices_norm_used)})")
            
            # 采样顶点和目标
            sampled_rest = rest_vertices_norm_used[sample_indices]
            sampled_target = target_vertices_norm_used[sample_indices]
            sampled_weights_init = weights_init[sample_indices]
            
            # 优化采样的权重
            optimized_sampled_weights = self.optimize_sampled_weights(
                sampled_rest, sampled_target, sampled_weights_init, 
                relative_transforms, regularization_lambda, max_iter // 5
            )
            
            # 将优化结果插值到所有顶点
            optimized_weights = weights_init.copy()
            optimized_weights[sample_indices] = optimized_sampled_weights
            
            # 对未采样的顶点使用最近邻插值
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(sampled_rest)
            distances, indices = nbrs.kneighbors(rest_vertices_norm_used)
            
            for i in range(num_vertices):
                if i not in sample_indices:
                    # 使用距离加权平均
                    weights_sum = np.sum(1.0 / (distances[i] + 1e-6))
                    weighted_weights = np.zeros(num_joints)
                    for j, neighbor_idx in enumerate(indices[i]):
                        weight = (1.0 / (distances[i][j] + 1e-6)) / weights_sum
                        weighted_weights += weight * optimized_sampled_weights[neighbor_idx]
                    optimized_weights[i] = weighted_weights
                    # 重新归一化
                    optimized_weights[i] = optimized_weights[i] / (np.sum(optimized_weights[i]) + 1e-8)
        else:
            # 对于小网格，使用标准优化
            optimized_weights = self.optimize_standard_weights(
                rest_vertices_norm_used, target_vertices_norm_used, weights_init,
                relative_transforms, regularization_lambda, max_iter // 5
            )
        
        # 如果优化使用的顶点数少于原始rest顶点数，需要扩展到原始大小
        if num_vertices < original_rest_vertices:
            print(f"扩展权重矩阵: {optimized_weights.shape} -> ({original_rest_vertices}, {num_joints})")
            # 创建完整大小的权重矩阵
            full_optimized_weights = np.zeros((original_rest_vertices, num_joints))
            # 复制优化的权重
            full_optimized_weights[:num_vertices] = optimized_weights
            # 对剩余顶点使用距离加权初始化
            if original_rest_vertices > num_vertices:
                keypoints = self.keypoints[self.reference_frame_idx, :, :3]
                remaining_vertices = rest_vertices_norm[num_vertices:original_rest_vertices]
                distances = cdist(remaining_vertices, keypoints)
                remaining_weights = np.exp(-distances**2 / (2 * 0.1**2))
                remaining_weights = remaining_weights / (np.sum(remaining_weights, axis=1, keepdims=True) + 1e-8)
                full_optimized_weights[num_vertices:] = remaining_weights
            
            optimized_weights = full_optimized_weights
        
        # 计算最终损失（使用优化时的顶点进行计算）
        final_loss = self.compute_lbs_loss(optimized_weights[:num_vertices].flatten(), rest_vertices_norm_used, 
                                         target_vertices_norm_used, relative_transforms, regularization_lambda)
        
        print(f"优化完成，最终损失: {final_loss:.6f}")
        print(f"返回权重矩阵形状: {optimized_weights.shape}")
        
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
        
        print(f"🚀 高效优化采样权重: {num_vertices} 顶点")
        
        # 优化参数
        chunk_size = 1000  # 更大的块以提高并行效率
        learning_rate = 0.03  # 更大的学习率
        num_threads = min(8, (num_vertices + chunk_size - 1) // chunk_size)  # 动态线程数
        
        print(f"  使用 {num_threads} 个线程，块大小: {chunk_size}")
        
        # 预计算变换矩阵的转置，避免重复计算
        transforms_t = relative_transforms.transpose(0, 2, 1)  # [J, 4, 4] -> [J, 4, 4]
        
        def optimize_chunk(chunk_data):
            """优化单个数据块"""
            chunk_indices, chunk_rest, chunk_target, chunk_weights = chunk_data
            
            # 向量化的LBS变换计算
            def fast_apply_lbs(vertices, weights, transforms):
                """快速LBS变换（向量化版本）"""
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
                """快速梯度计算（向量化）"""
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
            for sub_iter in range(3):  # 增加内层迭代次数
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
        
        # 主优化循环
        start_time = time.time()
        for iteration in range(max_iter):
            total_loss = 0.0
            
            # 随机打乱顶点顺序
            perm = np.random.permutation(num_vertices)
            
            # 准备数据块
            chunk_data_list = []
            for start_idx in range(0, num_vertices, chunk_size):
                end_idx = min(start_idx + chunk_size, num_vertices)
                chunk_indices = perm[start_idx:end_idx]
                
                chunk_rest = rest_vertices[chunk_indices]
                chunk_target = target_vertices[chunk_indices]
                chunk_weights = optimized_weights[chunk_indices].copy()
                
                chunk_data_list.append((chunk_indices, chunk_rest, chunk_target, chunk_weights))
            
            # 多线程并行优化
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # 提交所有任务
                future_to_chunk = {executor.submit(optimize_chunk, chunk_data): chunk_data 
                                 for chunk_data in chunk_data_list}
                
                # 收集结果
                for future in as_completed(future_to_chunk):
                    chunk_indices, chunk_weights, chunk_loss = future.result()
                    optimized_weights[chunk_indices] = chunk_weights
                    total_loss += chunk_loss * len(chunk_weights) / num_vertices
            
            # 进度报告
            if iteration % 5 == 0:  # 更频繁的进度报告
                elapsed = time.time() - start_time
                print(f"  🚀 迭代 {iteration}: 损失 = {total_loss:.6f}, 耗时 = {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        print(f"✅ 优化完成，总耗时: {total_time:.2f}s")
        
        return optimized_weights
    
    def optimize_standard_weights(self, rest_vertices, target_vertices, weights_init,
                                relative_transforms, regularization_lambda, max_iter):
        """
        标准权重优化（中等规模）
        """
        num_vertices, num_joints = weights_init.shape
        optimized_weights = weights_init.copy()
        
        print(f"标准优化: {num_vertices} 顶点")
        
        chunk_size = 200  # 适中的块大小
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
                print(f"  标准优化迭代 {iteration}: 损失 = {total_loss:.6f}")
        
        return optimized_weights

    def calc_optimize_frames(self, start_frame_idx, end_frame_idx, step):
        """
        计算优化帧
        """
        total_frames = len(self.mesh_files)
        if start_frame_idx is None:
            start_frame_idx = 0
        if end_frame_idx is None:
            end_frame_idx = total_frames
        if step is None:
            step = 2
        optimization_frames = list(range(start_frame_idx, end_frame_idx, step))
        
        # 移除reference frame
        if self.reference_frame_idx in optimization_frames:
            optimization_frames.remove(self.reference_frame_idx)

        return optimization_frames
    
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
        
        print(f"开始优化reference frame (frame {self.reference_frame_idx}) 的skinning权重...")
        
        # 对所有其他帧进行优化
        all_weights = []
        all_losses = []
        
        # 选择几个代表性帧进行优化
        if optimization_frames is None:
            optimization_frames = self.calc_optimize_frames(None, None, None)

        
        print(f"将使用 {len(optimization_frames)} 帧进行权重优化: {optimization_frames}")
        
        # 为每一帧优化权重
        for frame_idx in tqdm(optimization_frames, desc="优化各帧权重"):
            weights, loss_history = self.optimize_skinning_weights_for_frame(
                frame_idx, max_iter=max_iter, regularization_lambda=regularization_lambda
            )
            all_weights.append(weights)
            all_losses.extend(loss_history)
        
        # 验证所有权重矩阵形状一致
        if all_weights:
            # 检查所有权重矩阵的形状
            shapes = [w.shape for w in all_weights]
            print(f"收集到的权重矩阵形状: {shapes}")
            
            # 确保所有形状相同
            if len(set(shapes)) > 1:
                print("警告: 检测到不同形状的权重矩阵，正在统一形状...")
                # 找到最大的形状
                max_vertices = max(shape[0] for shape in shapes)
                max_joints = max(shape[1] for shape in shapes)
                target_shape = (max_vertices, max_joints)
                print(f"目标形状: {target_shape}")
                
                # 统一所有权重矩阵的形状
                unified_weights = []
                for i, weights in enumerate(all_weights):
                    if weights.shape != target_shape:
                        print(f"  调整权重矩阵 {i}: {weights.shape} -> {target_shape}")
                        unified = np.zeros(target_shape)
                        # 复制现有权重
                        unified[:weights.shape[0], :weights.shape[1]] = weights
                        # 对新增的顶点使用距离初始化
                        if weights.shape[0] < target_shape[0]:
                            keypoints = self.keypoints[self.reference_frame_idx, :, :3]
                            remaining_vertices = self.rest_pose_vertices[weights.shape[0]:target_shape[0]]
                            if len(remaining_vertices) > 0:
                                # 归一化剩余顶点
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
                print(f"形状统一完成，所有权重矩阵形状: {[w.shape for w in all_weights]}")
            
            # 平均所有帧的权重作为最终结果
            self.skinning_weights = np.mean(all_weights, axis=0)
            print(f"权重优化完成，使用了 {len(all_weights)} 帧的平均权重")
            print(f"最终权重矩阵形状: {self.skinning_weights.shape}")
            
            # 计算权重统计信息
            weights_per_vertex = np.sum(self.skinning_weights > 0.01, axis=1)  # 每个顶点受影响的关节数
            print(f"平均每个顶点受 {np.mean(weights_per_vertex):.2f} 个关节影响")
            print(f"权重稀疏度: {np.mean(self.skinning_weights > 0.01):.3f}")
        else:
            print("警告: 没有成功优化任何帧的权重")
            return None
        
        return self.skinning_weights
    
    def validate_skinning_weights(self, test_frames=None):
        """
        验证skinning权重的效果
        
        Args:
            test_frames: 测试帧列表，None表示测试所有帧
            
        Returns:
            validation_results: 验证结果字典
        """
        if self.skinning_weights is None:
            print("错误: 还没有计算skinning权重，请先调用optimize_reference_frame_skinning")
            return None
        
        if test_frames is None:
            test_frames = list(range(min(len(self.mesh_files), 20)))  # 限制测试帧数
        
        results = {
            'frame_errors': {},
            'average_error': 0.0,
            'max_error': 0.0,
            'min_error': float('inf')
        }
        
        print("验证skinning权重效果 on frames:", test_frames)
        
        rest_vertices_norm = self.normalize_mesh_vertices(
            self.rest_pose_vertices, 
            self.frame_normalization_params[self.reference_frame_idx]
        )
        
        total_error = 0.0
        valid_frames = 0
        
        for frame_idx in tqdm(test_frames, desc="验证帧"):
            if frame_idx >= len(self.mesh_files):
                continue
            
            # 加载目标网格
            target_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
            target_vertices = np.asarray(target_mesh.vertices)
            
            # 归一化
            if frame_idx not in self.frame_normalization_params:
                self.frame_normalization_params[frame_idx] = self.compute_mesh_normalization_params(target_mesh)
            
            target_vertices_norm = self.normalize_mesh_vertices(
                target_vertices, 
                self.frame_normalization_params[frame_idx]
            )
            
            # 计算相对变换
            target_transforms = self.transforms[frame_idx]
            rest_transforms = self.transforms[self.reference_frame_idx]
            
            relative_transforms = np.zeros_like(target_transforms)
            for j in range(self.num_joints):
                if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                    rest_inv = np.linalg.inv(rest_transforms[j])
                    relative_transforms[j] = target_transforms[j] @ rest_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # 使用LBS预测顶点位置
            predicted_vertices = self.apply_lbs_transform(
                rest_vertices_norm, self.skinning_weights, relative_transforms
            )
            
            # 处理顶点数量不匹配的问题
            if predicted_vertices.shape[0] != target_vertices_norm.shape[0]:
                print(f"   帧 {frame_idx}: 顶点数不匹配 (predicted: {predicted_vertices.shape[0]}, target: {target_vertices_norm.shape[0]})")
                # 使用较小的数量进行比较
                min_vertices = min(predicted_vertices.shape[0], target_vertices_norm.shape[0])
                predicted_vertices_used = predicted_vertices[:min_vertices]
                target_vertices_used = target_vertices_norm[:min_vertices]
                print(f"   使用前 {min_vertices} 个顶点进行误差计算")
            else:
                predicted_vertices_used = predicted_vertices
                target_vertices_used = target_vertices_norm
            
            # 计算误差
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
            
            print(f"验证完成！")
            print(f"平均重建误差: {results['average_error']:.6f}")
            print(f"最大误差: {results['max_error']:.6f}")
            print(f"最小误差: {results['min_error']:.6f}")
        
        return results
    
    def test_lbs_reconstruction_quality(self, test_frames=None, save_meshes=False, output_dir="output/lbs_test"):
        """
        测试LBS重建质量的详细方法
        
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
            print("错误: 需要先加载或计算skinning权重")
            return None
        
        # 自动选择测试帧
        if test_frames is None:
            total_frames = len(self.mesh_files)
            if total_frames <= 20:
                test_frames = list(range(total_frames))
            else:
                # 选择代表性帧：开始、中间、结束，以及一些随机帧
                test_frames = []
                test_frames.extend([0, 1, 2])  # 开始几帧
                test_frames.extend([total_frames//4, total_frames//2, 3*total_frames//4])  # 中间帧
                test_frames.extend([total_frames-3, total_frames-2, total_frames-1])  # 结束几帧
                # 添加一些随机帧
                import random
                random_frames = random.sample(range(3, total_frames-3), min(6, total_frames-9))
                test_frames.extend(random_frames)
                test_frames = sorted(list(set(test_frames)))  # 去重排序
        
        print(f"🔍 测试LBS重建质量")
        print(f"测试帧: {test_frames}")
        print(f"参考帧: {self.reference_frame_idx}")
        
        # 创建输出目录
        if save_meshes:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 准备rest pose数据
        rest_vertices_norm = self.normalize_mesh_vertices(
            self.rest_pose_vertices, 
            self.frame_normalization_params[self.reference_frame_idx]
        )
        
        # 测试结果
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
        distance_errors = []  # 误差与距离参考帧的关系
        
        print(f"\n开始测试 {len(test_frames)} 帧...")
        
        for i, frame_idx in enumerate(tqdm(test_frames, desc="测试重建质量")):
            if frame_idx >= len(self.mesh_files):
                continue
            
            # 加载目标网格
            target_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
            target_vertices = np.asarray(target_mesh.vertices)
            
            # 归一化
            if frame_idx not in self.frame_normalization_params:
                self.frame_normalization_params[frame_idx] = self.compute_mesh_normalization_params(target_mesh)
            
            target_vertices_norm = self.normalize_mesh_vertices(
                target_vertices, 
                self.frame_normalization_params[frame_idx]
            )
            
            # 计算相对变换
            target_transforms = self.transforms[frame_idx]
            rest_transforms = self.transforms[self.reference_frame_idx]
            
            relative_transforms = np.zeros_like(target_transforms)
            for j in range(self.num_joints):
                if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                    rest_inv = np.linalg.inv(rest_transforms[j])
                    relative_transforms[j] = target_transforms[j] @ rest_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # LBS重建测试
            start_time = time.time()
            predicted_vertices = self.apply_lbs_transform(
                rest_vertices_norm, self.skinning_weights, relative_transforms
            )
            lbs_time = time.time() - start_time
            
            # 计算详细误差指标
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
                'vertices_with_large_error': int(np.sum(vertex_errors > 0.05)),  # 大误差顶点数
                'error_ratio_large': float(np.sum(vertex_errors > 0.05) / len(vertex_errors))  # 大误差比例
            }
            
            detailed_results['frame_results'][frame_idx] = frame_result
            
            # 收集统计数据
            all_errors.extend(vertex_errors)
            all_times.append(lbs_time)
            distance_errors.append((frame_result['distance_from_ref'], frame_result['mean_error']))
            
            # 保存网格（如果需要）
            if save_meshes:
                # 保存重建网格
                reconstructed_mesh = o3d.geometry.TriangleMesh()
                reconstructed_mesh.vertices = o3d.utility.Vector3dVector(predicted_vertices)
                if hasattr(target_mesh, 'triangles') and len(target_mesh.triangles) > 0:
                    reconstructed_mesh.triangles = target_mesh.triangles
                
                mesh_output_path = output_path / f"frame_{frame_idx:06d}_reconstructed.obj"
                o3d.io.write_triangle_mesh(str(mesh_output_path), reconstructed_mesh)
                
                # 保存误差可视化网格
                normalized_errors = vertex_errors / np.max(vertex_errors)
                error_colors = plt.cm.plasma(normalized_errors)[:, :3]  # 使用plasma颜色映射
                
                error_mesh = o3d.geometry.TriangleMesh()
                error_mesh.vertices = target_mesh.vertices
                error_mesh.triangles = target_mesh.triangles
                error_mesh.vertex_colors = o3d.utility.Vector3dVector(error_colors)
                
                error_output_path = output_path / f"frame_{frame_idx:06d}_error_colored.obj"
                o3d.io.write_triangle_mesh(str(error_output_path), error_mesh)
        
        # 计算汇总统计
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
        
        # 性能统计
        if all_times:
            detailed_results['performance_stats'] = {
                'mean_lbs_time': float(np.mean(all_times)),
                'total_lbs_time': float(np.sum(all_times)),
                'min_lbs_time': float(np.min(all_times)),
                'max_lbs_time': float(np.max(all_times)),
                'fps_estimate': float(len(all_times) / np.sum(all_times)) if np.sum(all_times) > 0 else 0
            }
        
        # 分析误差与距离的关系
        if distance_errors:
            distances, errors = zip(*distance_errors)
            if len(set(distances)) > 1:  # 有不同距离的数据点
                correlation = np.corrcoef(distances, errors)[0, 1]
                detailed_results['distance_analysis'] = {
                    'correlation_with_distance': float(correlation),
                    'distance_error_pairs': distance_errors
                }
        
        # 输出总结
        print(f"\n📊 测试完成总结:")
        if 'summary_stats' in detailed_results:
            stats = detailed_results['summary_stats']
            print(f"总体平均误差: {stats['overall_mean_error']:.6f}")
            print(f"总体RMSE: {stats['overall_rmse']:.6f}")
            print(f"误差范围: [{stats['overall_min_error']:.6f}, {stats['overall_max_error']:.6f}]")
            print(f"大误差顶点比例: {stats['large_error_ratio']*100:.2f}%")
        
        if 'performance_stats' in detailed_results:
            perf = detailed_results['performance_stats']
            print(f"平均LBS时间: {perf['mean_lbs_time']:.3f}s")
            print(f"估计帧率: {perf['fps_estimate']:.1f} FPS")
        
        if 'distance_analysis' in detailed_results:
            dist_analysis = detailed_results['distance_analysis']
            print(f"误差与距离参考帧相关性: {dist_analysis['correlation_with_distance']:.3f}")
        
        # 保存详细结果
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
            
            print(f"\n💾 详细结果已保存:")
            print(f"   测试结果: {results_path}")
            print(f"   重建网格: {output_path}/*_reconstructed.obj")
            print(f"   误差可视化: {output_path}/*_error_colored.obj")
        
        return detailed_results
    
    def save_skinning_weights(self, output_path):
        """
        Save self's skinning weights
        """
        if self.skinning_weights is None:
            print("错误: 没有可保存的skinning权重")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存权重和相关信息
        skinning_data = {
            'weights': self.skinning_weights,
            'rest_vertices': self.rest_pose_vertices,
            'rest_transforms': self.rest_pose_transforms,
            'reference_frame_idx': self.reference_frame_idx,
            'num_vertices': self.skinning_weights.shape[0],
            'num_joints': self.skinning_weights.shape[1]
        }
        
        np.savez_compressed(output_path, **skinning_data)
        print(f"Skinning权重已保存到: {output_path}")
        

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
            self.reference_frame_idx = data['reference_frame_idx'].item()  # 确保是整数
            
            print(f"成功加载skinning权重:")
            print(f"  - 权重矩阵形状: {self.skinning_weights.shape}")
            print(f"  - Rest pose顶点数: {len(self.rest_pose_vertices)}")
            print(f"  - Reference frame: {data['reference_frame_idx']}")
            
            return self.skinning_weights
        except Exception as e:
            print(f"加载skinning权重失败: {e}")
            return None

def run_auto_skinning_pipeline(reference_frame_idx = 5):
    """
    自动蒙皮计算和可视化pipeline
    """
    print("🚀 开始自动蒙皮计算Pipeline")
    print("=" * 60)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_output_path = f"output/skinning_weights/skinning_weights_{reference_frame_idx}.npz"
    
    # 初始化
    skinner = AutoSkinning(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=reference_frame_idx  # 使用第5帧作为参考
    )
    
    # 加载数据
    print("📁 加载网格序列...")
    skinner.load_mesh_sequence(mesh_folder_path)
    skinner.rest_pose_transforms = skinner.transforms[skinner.reference_frame_idx]
    skinner.rest_pose_vertices = np.asarray(skinner.reference_mesh.vertices)
    
    print(f"✅ 数据加载完成:")
    print(f"   顶点数: {len(skinner.rest_pose_vertices):,}")
    print(f"   关节数: {skinner.num_joints}")
    print(f"   骨骼帧数: {skinner.num_frames}")
    print(f"   网格文件数: {len(skinner.mesh_files)}")

    if not os.path.exists(weights_output_path):        
        # 优化蒙皮权重
        print("\n🔧 开始优化蒙皮权重...")

        optimization_frames = skinner.calc_optimize_frames(reference_frame_idx - 10, reference_frame_idx + 10, 2)

        skinner.skinning_weights = skinner.optimize_reference_frame_skinning(
            regularization_lambda=0.01,
            optimization_frames=optimization_frames,
            max_iter=100  # 适中的迭代次数
        )
        
        if skinner.skinning_weights is None:
            print("❌ 蒙皮权重优化失败")
            return
        
        # 保存权重
        print(f"\n💾 保存蒙皮权重到: {weights_output_path}")
        skinner.save_skinning_weights(weights_output_path)
    else:
        print(f"✅ 蒙皮权重文件已存在: {weights_output_path}")
        skinner.load_skinning_weights(weights_output_path)
    
    # 快速验证
    print("\n📊 快速验证蒙皮效果...")
    # 计算可用的测试帧
    max_skeleton_frame = skinner.num_frames - 1
    test_frames = list(range(0, max_skeleton_frame + 1, max_skeleton_frame // 10))
    if skinner.reference_frame_idx in test_frames:
        test_frames.remove(skinner.reference_frame_idx)

    print(f"📋 计划测试帧: {test_frames}")
    validation_results = skinner.validate_skinning_weights(test_frames=test_frames)
    
    if validation_results:
        print(f"\n✅ 验证完成:")
        print(f"   平均误差: {validation_results['average_error']:.6f}")
        print(f"   误差范围: [{validation_results['min_error']:.6f}, {validation_results['max_error']:.6f}]")
        
        # 找到最好和最差的帧
        best_frame = min(validation_results['frame_errors'].items(), 
                        key=lambda x: x[1]['mean_error'])
        worst_frame = max(validation_results['frame_errors'].items(), 
                         key=lambda x: x[1]['mean_error'])
        
        print(f"   最佳帧: {best_frame[0]} (误差: {best_frame[1]['mean_error']:.6f})")
        print(f"   最差帧: {worst_frame[0]} (误差: {worst_frame[1]['mean_error']:.6f})")
        
        # 选择要可视化的帧
        viz_frames = [best_frame[0], worst_frame[0]]
        if len(test_frames) > 2:
            # 添加一个中等质量的帧
            middle_frame = sorted(validation_results['frame_errors'].items(), 
                                key=lambda x: x[1]['mean_error'])[len(validation_results['frame_errors'])//2]
            viz_frames.append(middle_frame[0])
        
        viz_frames = sorted(list(set(viz_frames)))[:3]  # 最多3帧
        
        print(f"\n🎨 准备可视化重建对比 (帧: {viz_frames})...")
        
        # 自动运行可视化
        run_reconstruction_visualization(skinner, viz_frames, weights_output_path)
    
    print("\n🎉 自动蒙皮Pipeline完成!")
    print(f"💾 权重文件: {weights_output_path}")
    print(f"📊 测试结果已显示")

def run_reconstruction_visualization(skinner, viz_frames, weights_path):
    """
    运行重建可视化对比
    """
    import subprocess
    import sys
    
    print(f"🖥️  启动重建可视化...")
    
    for frame_idx in viz_frames:
        print(f"   可视化帧 {frame_idx}...")
        try:
            # 运行可视化脚本
            result = subprocess.run([
                sys.executable, "simple_visualize.py", str(frame_idx)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"   ✅ 帧 {frame_idx} 可视化完成")
                # 解析输出中的误差信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if '平均误差:' in line or '最大误差:' in line or 'RMSE:' in line:
                        print(f"      {line.strip()}")
            else:
                print(f"   ⚠️  帧 {frame_idx} 可视化失败: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  帧 {frame_idx} 可视化超时")
        except Exception as e:
            print(f"   ⚠️  帧 {frame_idx} 可视化错误: {e}")
    
    # 运行批量导出
    print(f"\n📦 导出mesh文件用于外部查看...")
    try:
        result = subprocess.run([
            sys.executable, "simple_visualize.py", "export"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"   ✅ 批量导出完成")
            # 查找导出路径
            lines = result.stdout.split('\n')
            for line in lines:
                if '所有文件已导出到:' in line:
                    print(f"   📁 {line.strip()}")
        else:
            print(f"   ⚠️  批量导出失败")
            
    except Exception as e:
        print(f"   ⚠️  批量导出错误: {e}")

def main():
    """
    主函数 - 运行完整的自动蒙皮pipeline
    """
    args = sys.argv[1:]
    reference_frame_idx = 5
    if len(args) >= 1:
        reference_frame_idx = int(args[0])
    run_auto_skinning_pipeline(reference_frame_idx)

if __name__ == "__main__":
    main()