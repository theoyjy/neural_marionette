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

class InverseMeshCanonicalizer:
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
        """加载numpy格式的骨骼数据"""
        try:
            # 加载关键点数据 [num_frames, num_joints, 4] (x, y, z, confidence)
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # 加载变换矩阵 [num_frames, num_joints, 4, 4]
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # 加载父节点关系 [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"成功加载骨骼数据:")
            print(f"  - 帧数: {self.num_frames}")
            print(f"  - 关节数: {self.num_joints}")
            print(f"  - 关键点形状: {self.keypoints.shape} (包含置信度)")
            print(f"  - 变换矩阵形状: {self.transforms.shape}")
            print(f"  - 父节点关系形状: {self.parents.shape}")
            
        except Exception as e:
            raise ValueError(f"无法加载骨骼数据: {e}")
            
        # 尝试加载其他可选数据
        try:
            if (self.skeleton_data_dir / 'affinity.npy').exists():
                self.affinity = np.load(self.skeleton_data_dir / 'affinity.npy')
                print(f"  - 亲和度矩阵形状: {self.affinity.shape}")
            else:
                self.affinity = None
                
            if (self.skeleton_data_dir / 'priority.npy').exists():
                self.priority = np.load(self.skeleton_data_dir / 'priority.npy')
                print(f"  - 优先级形状: {self.priority.shape}")
            else:
                self.priority = None
                
            if (self.skeleton_data_dir / 'A.npy').exists():
                self.A = np.load(self.skeleton_data_dir / 'A.npy')
                print(f"  - A矩阵形状: {self.A.shape}")
            else:
                self.A = None
                
            if (self.skeleton_data_dir / 'rotations.npy').exists():
                self.rotations = np.load(self.skeleton_data_dir / 'rotations.npy')
                print(f"  - 旋转矩阵形状: {self.rotations.shape}")
            else:
                self.rotations = None
        except Exception as e:
            print(f"警告: 无法加载可选数据: {e}")

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

    def load_mesh_sequence(self, mesh_folder_path):
        """
        加载网格序列
        
        Args:
            mesh_folder_path: 包含obj文件的文件夹路径
        """
        self.mesh_folder_path = Path(mesh_folder_path)
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        
        if len(self.mesh_files) != self.num_frames:
            print(f"警告: 网格文件数量 ({len(self.mesh_files)}) 与骨骼帧数 ({self.num_frames}) 不匹配")
        
        # 加载参考网格
        self.reference_mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[self.reference_frame_idx]))
        print(f"参考网格顶点数: {len(self.reference_mesh.vertices)}")
        
        # 预计算参考网格的归一化参数
        self.frame_normalization_params[self.reference_frame_idx] = self.compute_mesh_normalization_params(self.reference_mesh)
        print(f"参考网格归一化参数已计算")
        
        # 预计算参考网格的关节分配（用于加速后续对应关系计算）
        ref_vertices = np.asarray(self.reference_mesh.vertices)
        ref_norm_vertices = self.normalize_mesh_vertices(ref_vertices, self.frame_normalization_params[self.reference_frame_idx])
        ref_joints = self.keypoints[self.reference_frame_idx, :, :3]
        self.reference_vertex_joints = self.assign_dominant_joints(ref_norm_vertices, ref_joints)
        print(f"参考网格关节分配已预计算")

    def compute_bone_influenced_vertices(self, mesh, frame_idx, influence_radius=0.1):
        """
        计算受骨骼影响的顶点
        
        Args:
            mesh
            frame_idx: 帧索引
            influence_radius: 影响半径
            
        Returns:
            vertex_bone_weights: 顶点到骨骼的权重矩阵 [V, J]
        """
        # 获取原始顶点坐标
        vertices = np.asarray(mesh.vertices)
        
        # 获取归一化空间的keypoints，只取前3个坐标（忽略置信度）
        normalized_keypoints = self.keypoints[frame_idx, :, :3]  # [num_joints, 3]
        
        # 计算当前mesh的归一化参数
        if frame_idx not in self.frame_normalization_params:
            self.frame_normalization_params[frame_idx] = self.compute_mesh_normalization_params(mesh)
        
        # 将mesh顶点归一化到与keypoints相同的空间
        normalized_vertices = self.normalize_mesh_vertices(vertices, self.frame_normalization_params[frame_idx])
        
        # 在归一化空间中计算距离
        distances = cdist(normalized_vertices, normalized_keypoints)
        
        # 使用高斯权重
        weights = np.exp(-distances**2 / (2 * influence_radius**2))
        
        # 归一化权重
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        return weights
    
    def compute_vertex_features(self, mesh, frame_idx):
        """
        计算顶点特征用于匹配
        
        Args:
            mesh
            frame_idx: 帧索引
            
        Returns:
            features: 顶点特征矩阵 [V, D]
        """
        vertices = np.asarray(mesh.vertices)
        
        # 几何特征
        geometric_features = []
        
        # 1. 顶点坐标（归一化空间）
        # 计算当前mesh的归一化参数（如果还没有计算过）
        if frame_idx not in self.frame_normalization_params:
            self.frame_normalization_params[frame_idx] = self.compute_mesh_normalization_params(mesh)
        
        # 将顶点归一化到与keypoints相同的空间
        normalized_vertices = self.normalize_mesh_vertices(vertices, self.frame_normalization_params[frame_idx])
        geometric_features.append(normalized_vertices)
        
        # 2. 骨骼空间变换的顶点坐标（可选，作为额外特征）
        keypoints = self.keypoints[frame_idx, :, :3]  # [num_joints, 3] 只取坐标，忽略置信度
        transforms = self.transforms[frame_idx]  # [num_joints, 4, 4]
        
        # 将归一化顶点变换到骨骼空间
        vertices_homogeneous = np.hstack([normalized_vertices, np.ones((len(normalized_vertices), 1))])
        
        # 使用第一个关节的逆变换作为根变换
        if len(transforms) > 0:
            root_inv_transform = np.linalg.inv(transforms[0])
            canonical_vertices = (root_inv_transform @ vertices_homogeneous.T).T[:, :3]
        else:
            canonical_vertices = normalized_vertices
        
        geometric_features.append(canonical_vertices)
        
        # 3. 法向量
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            geometric_features.append(np.asarray(mesh.vertex_normals))
        else:
            # 计算法向量
            mesh.compute_vertex_normals()
            geometric_features.append(np.asarray(mesh.vertex_normals))
        
        # 4. 骨骼权重特征
        bone_weights = self.compute_bone_influenced_vertices(mesh, frame_idx)
        geometric_features.append(bone_weights)
        
        # 5. 曲率特征（简化版，优化性能）
        try:
            # 使用更高效的方法计算曲率特征
            # 方法1：基于顶点的局部密度而不是最近邻
            center = np.mean(vertices, axis=0)
            distances_to_center = np.linalg.norm(vertices - center, axis=1)
            
            # 计算每个顶点在局部区域的密度特征
            vertex_curvature = []
            sample_size = min(len(vertices), 1000)  # 限制样本大小以提高性能
            
            if len(vertices) > sample_size:
                # 如果顶点太多，使用采样
                sample_indices = np.random.choice(len(vertices), sample_size, replace=False)
                sample_vertices = vertices[sample_indices]
            else:
                sample_vertices = vertices
                sample_indices = np.arange(len(vertices))
            
            # 使用KDTree进行高效的最近邻搜索
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(11, len(sample_vertices)), algorithm='kd_tree').fit(sample_vertices)
            
            for i in range(len(vertices)):
                vertex = vertices[i]
                
                # 找到最近的邻居
                distances, indices = nbrs.kneighbors([vertex])
                neighbor_distances = distances[0][1:]  # 排除自己
                
                if len(neighbor_distances) > 0:
                    mean_dist = np.mean(neighbor_distances)
                    var_dist = np.var(neighbor_distances) if len(neighbor_distances) > 1 else 0.0
                else:
                    mean_dist = 0.0
                    var_dist = 0.0
                
                vertex_curvature.append([mean_dist, var_dist])
            
            geometric_features.append(np.array(vertex_curvature))
        except Exception as e:
            print(f"曲率计算失败，使用零特征: {e}")
            # 如果曲率计算失败，使用零特征
            geometric_features.append(np.zeros((len(vertices), 2)))
        
        # 合并所有特征
        features = np.hstack(geometric_features)
        
        return features
    
    def compute_skeleton_driven_correspondence(self, target_mesh, target_frame_idx):
        """
        基于骨骼关节驱动的快速顶点对应关系计算
        
        Args:
            target_mesh: 目标网格
            target_frame_idx: 目标帧索引
            
        Returns:
            correspondence_map: 从目标顶点索引到参考顶点索引的映射
        """
        # 获取参考帧和目标帧的关节点
        ref_joints = self.keypoints[self.reference_frame_idx, :, :3]  # [num_joints, 3]
        target_joints = self.keypoints[target_frame_idx, :, :3]  # [num_joints, 3]
        
        # 获取网格顶点
        ref_vertices = np.asarray(self.reference_mesh.vertices)
        target_vertices = np.asarray(target_mesh.vertices)
        
        # 计算归一化参数
        if target_frame_idx not in self.frame_normalization_params:
            self.frame_normalization_params[target_frame_idx] = self.compute_mesh_normalization_params(target_mesh)
        
        # 归一化顶点到与keypoints相同的空间
        ref_norm_vertices = self.normalize_mesh_vertices(ref_vertices, self.frame_normalization_params[self.reference_frame_idx])
        target_norm_vertices = self.normalize_mesh_vertices(target_vertices, self.frame_normalization_params[target_frame_idx])
        
        # 使用预计算的参考网格关节分配
        ref_vertex_joints = self.reference_vertex_joints
        
        # 为目标网格顶点分配主导关节
        target_vertex_joints = self.assign_dominant_joints(target_norm_vertices, target_joints)
        
        # 基于关节对应关系进行顶点匹配
        correspondence_map = self.match_vertices_by_skeleton(
            ref_norm_vertices, target_norm_vertices, 
            ref_vertex_joints, target_vertex_joints,
            ref_joints, target_joints
        )
        
        return correspondence_map
    
    def assign_dominant_joints(self, vertices, joints, top_k=3):
        """
        为每个顶点分配影响最大的前k个关节
        
        Args:
            vertices: 顶点坐标 [V, 3]
            joints: 关节点坐标 [J, 3] 
            top_k: 保留前k个最近关节
            
        Returns:
            vertex_joints: 每个顶点的主导关节信息 [V, top_k, 2] (joint_idx, weight)
        """
        # 计算顶点到关节的距离
        distances = cdist(vertices, joints)  # [V, J]
        
        # 使用高斯权重
        sigma = 0.1
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        # 归一化权重
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        # 获取每个顶点的前k个最强关节
        vertex_joints = []
        for v in range(len(vertices)):
            top_indices = np.argsort(weights[v])[-top_k:][::-1]  # 降序
            top_weights = weights[v][top_indices]
            vertex_joints.append(list(zip(top_indices, top_weights)))
        
        return vertex_joints
    
    def match_vertices_by_skeleton(self, ref_vertices, target_vertices, 
                                 ref_vertex_joints, target_vertex_joints,
                                 ref_joints, target_joints):
        """
        基于骨骼关节对应关系匹配顶点
        
        Args:
            ref_vertices: 参考顶点 [V_ref, 3]
            target_vertices: 目标顶点 [V_target, 3]
            ref_vertex_joints: 参考顶点的关节分配
            target_vertex_joints: 目标顶点的关节分配
            ref_joints: 参考关节点 [J, 3]
            target_joints: 目标关节点 [J, 3]
            
        Returns:
            correspondence_map: [V_target] -> V_ref
        """
        correspondence_map = np.zeros(len(target_vertices), dtype=int)
        
        # 按关节分组处理顶点
        joint_groups = {}
        
        # 将目标顶点按主导关节分组
        for v_idx, joint_list in enumerate(target_vertex_joints):
            primary_joint = joint_list[0][0]  # 最强关节
            if primary_joint not in joint_groups:
                joint_groups[primary_joint] = []
            joint_groups[primary_joint].append(v_idx)
        
        print(f"按{len(joint_groups)}个关节分组处理顶点...")
        
        for joint_idx, target_vertex_indices in joint_groups.items():
            if len(target_vertex_indices) == 0:
                continue
                
            # 找到参考网格中属于同一关节的顶点
            ref_vertex_indices = []
            for v_idx, joint_list in enumerate(ref_vertex_joints):
                if len(joint_list) > 0 and joint_list[0][0] == joint_idx:
                    ref_vertex_indices.append(v_idx)
            
            if len(ref_vertex_indices) == 0:
                # 如果参考网格中没有对应关节的顶点，使用全局最近邻
                ref_vertex_indices = list(range(len(ref_vertices)))
            
            # 在该关节的局部空间中进行匹配
            target_local_vertices = target_vertices[target_vertex_indices]
            ref_local_vertices = ref_vertices[ref_vertex_indices]
            
            # 转换到关节局部坐标系
            joint_center_target = target_joints[joint_idx]
            joint_center_ref = ref_joints[joint_idx]
            
            target_local_relative = target_local_vertices - joint_center_target
            ref_local_relative = ref_local_vertices - joint_center_ref
            
            # 使用最近邻匹配
            if len(ref_local_vertices) > 0:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ref_local_relative)
                _, indices = nbrs.kneighbors(target_local_relative)
                
                # 映射回全局索引
                for i, target_v_idx in enumerate(target_vertex_indices):
                    ref_v_idx = ref_vertex_indices[indices[i][0]]
                    correspondence_map[target_v_idx] = ref_v_idx
        
        return correspondence_map
    
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
        
        num_vertices = len(rest_vertices_norm)
        num_joints = self.num_joints
        
        # 初始化权重
        if init_method == 'distance_based':
            # 基于距离的初始化
            keypoints = self.keypoints[self.reference_frame_idx, :, :3]
            distances = cdist(rest_vertices_norm, keypoints)
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
            return self.compute_lbs_loss(weights_flat, rest_vertices_norm, target_vertices_norm, 
                                       relative_transforms, regularization_lambda)
        
        # 使用高效的优化方法：大块并行优化
        print(f"使用高效优化方法...")
        print(f"顶点数: {num_vertices}, 关节数: {num_joints}")
        
        # 大幅减少计算量
        if num_vertices > 10000:
            # 对于大网格，使用采样策略
            sample_size = 5000  # 只优化5000个代表性顶点
            sample_indices = np.random.choice(num_vertices, sample_size, replace=False)
            print(f"大网格检测，采样 {sample_size} 个顶点进行优化")
            
            # 采样顶点和目标
            sampled_rest = rest_vertices_norm[sample_indices]
            sampled_target = target_vertices_norm[sample_indices]
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
            distances, indices = nbrs.kneighbors(rest_vertices_norm)
            
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
                rest_vertices_norm, target_vertices_norm, weights_init,
                relative_transforms, regularization_lambda, max_iter // 5
            )
        
        # 计算最终损失
        final_loss = self.compute_lbs_loss(optimized_weights.flatten(), rest_vertices_norm, 
                                         target_vertices_norm, relative_transforms, regularization_lambda)
        
        print(f"优化完成，最终损失: {final_loss:.6f}")
        
        return optimized_weights, [final_loss]
    
    def optimize_sampled_weights(self, rest_vertices, target_vertices, weights_init, 
                               relative_transforms, regularization_lambda, max_iter):
        """
        优化采样的权重（较小规模）
        """
        num_vertices, num_joints = weights_init.shape
        optimized_weights = weights_init.copy()
        
        print(f"优化采样权重: {num_vertices} 顶点")
        
        # 使用块优化，但块更大
        chunk_size = 500  # 更大的块
        learning_rate = 0.02  # 更大的学习率
        
        for iteration in range(max_iter):
            total_loss = 0.0
            
            # 随机打乱顶点顺序
            perm = np.random.permutation(num_vertices)
            
            for start_idx in range(0, num_vertices, chunk_size):
                end_idx = min(start_idx + chunk_size, num_vertices)
                chunk_indices = perm[start_idx:end_idx]
                
                # 提取当前块
                chunk_rest = rest_vertices[chunk_indices]
                chunk_target = target_vertices[chunk_indices]
                chunk_weights = optimized_weights[chunk_indices].copy()
                
                # 简化的梯度下降 - 只做一次内层迭代
                predicted = self.apply_lbs_transform(chunk_rest, chunk_weights, relative_transforms)
                error = predicted - chunk_target
                
                # 简化的梯度计算 - 只计算主要关节的梯度
                gradient = np.zeros_like(chunk_weights)
                eps = 1e-5
                
                for i in range(len(chunk_weights)):
                    # 只优化权重最大的前5个关节
                    top_joints = np.argsort(chunk_weights[i])[-5:]
                    
                    for j in top_joints:
                        chunk_weights_plus = chunk_weights.copy()
                        chunk_weights_plus[i, j] += eps
                        chunk_weights_plus[i] = chunk_weights_plus[i] / (np.sum(chunk_weights_plus[i]) + 1e-8)
                        
                        predicted_plus = self.apply_lbs_transform(chunk_rest, chunk_weights_plus, relative_transforms)
                        error_plus = predicted_plus - chunk_target
                        
                        loss = np.mean(np.sum(error**2, axis=1))
                        loss_plus = np.mean(np.sum(error_plus**2, axis=1))
                        
                        gradient[i, j] = (loss_plus - loss) / eps
                
                # 更新权重
                chunk_weights -= learning_rate * gradient
                chunk_weights = np.maximum(chunk_weights, 0)
                chunk_weights = chunk_weights / (np.sum(chunk_weights, axis=1, keepdims=True) + 1e-8)
                
                optimized_weights[chunk_indices] = chunk_weights
                
                # 计算损失
                predicted = self.apply_lbs_transform(chunk_rest, chunk_weights, relative_transforms)
                chunk_loss = np.mean(np.sum((predicted - chunk_target)**2, axis=1))
                total_loss += chunk_loss * len(chunk_weights) / num_vertices
            
            if iteration % 10 == 0:
                print(f"  采样优化迭代 {iteration}: 损失 = {total_loss:.6f}")
        
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
    
    def optimize_reference_frame_skinning(self, regularization_lambda=0.01, max_iter=1000):
        """
        优化reference frame的skinning权重
        
        Args:
            regularization_lambda: 正则化系数
            max_iter: 最大迭代次数
            
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
        optimization_frames = []
        total_frames = len(self.mesh_files)
        
        # 选择策略：均匀采样 + 包含首尾帧
        if total_frames <= 10:
            optimization_frames = list(range(total_frames))
        else:
            # 均匀采样10帧
            step = total_frames // 10
            optimization_frames = list(range(0, total_frames, step))
            if optimization_frames[-1] != total_frames - 1:
                optimization_frames.append(total_frames - 1)
        
        # 移除reference frame
        if self.reference_frame_idx in optimization_frames:
            optimization_frames.remove(self.reference_frame_idx)
        
        print(f"将使用 {len(optimization_frames)} 帧进行权重优化: {optimization_frames}")
        
        # 为每一帧优化权重
        for frame_idx in tqdm(optimization_frames, desc="优化各帧权重"):
            weights, loss_history = self.optimize_skinning_weights_for_frame(
                frame_idx, max_iter=max_iter, regularization_lambda=regularization_lambda
            )
            all_weights.append(weights)
            all_losses.extend(loss_history)
        
        # 平均所有帧的权重作为最终结果
        if all_weights:
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
        
        print("验证skinning权重效果...")
        
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
            
            # 计算误差
            vertex_errors = np.linalg.norm(predicted_vertices - target_vertices_norm, axis=1)
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
        保存skinning权重
        
        Args:
            output_path: 输出文件路径
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
        加载skinning权重
        
        Args:
            input_path: 输入文件路径
        """
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"错误: 文件不存在: {input_path}")
            return False
        
        try:
            data = np.load(input_path)
            self.skinning_weights = data['weights']
            self.rest_pose_vertices = data['rest_vertices']
            self.rest_pose_transforms = data['rest_transforms']
            
            print(f"成功加载skinning权重:")
            print(f"  - 权重矩阵形状: {self.skinning_weights.shape}")
            print(f"  - Rest pose顶点数: {len(self.rest_pose_vertices)}")
            print(f"  - Reference frame: {data['reference_frame_idx']}")
            
            return True
        except Exception as e:
            print(f"加载skinning权重失败: {e}")
            return False
    
    def find_vertex_correspondence(self, target_mesh, target_frame_idx):
        """
        找到目标网格与参考网格的顶点对应关系（原始特征匹配方法）
        
        Args:
            target_mesh: 目标网格
            target_frame_idx: 目标帧索引
            
        Returns:
            correspondence_map: 从目标顶点索引到参考顶点索引的映射
        """
        # 计算特征
        ref_features = self.compute_vertex_features(self.reference_mesh, self.reference_frame_idx)
        target_features = self.compute_vertex_features(target_mesh, target_frame_idx)
        
        print(f"参考特征维度: {ref_features.shape}, 目标特征维度: {target_features.shape}")
        
        # 使用最近邻匹配
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ref_features)
        distances, indices = nbrs.kneighbors(target_features)
        
        correspondence_map = indices.flatten()
        
        # 计算匹配质量
        avg_distance = np.mean(distances)
        print(f"帧 {target_frame_idx} 平均匹配距离: {avg_distance:.4f}")
        
        return correspondence_map
    
    def reorder_mesh_vertices(self, mesh, correspondence_map):
        """
        根据对应关系重新排序网格顶点
        
        Args:
            mesh: 输入网格
            correspondence_map: 顶点对应关系映射
            
        Returns:
            reordered_mesh: 重新排序的网格
        """
        # 创建新的顶点排列
        new_vertices = np.zeros_like(np.asarray(self.reference_mesh.vertices))
        
        for target_idx, ref_idx in enumerate(correspondence_map):
            if target_idx < len(mesh.vertices):
                new_vertices[ref_idx] = np.asarray(mesh.vertices)[target_idx]
        
        # 创建新的网格
        reordered_mesh = o3d.geometry.TriangleMesh()
        reordered_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        
        # 复制面信息（如果存在）
        if hasattr(mesh, 'triangles') and len(mesh.triangles) > 0:
            reordered_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        
        # 复制其他属性（如果存在）
        if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
            reordered_mesh.vertex_normals = mesh.vertex_normals
        if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
            reordered_mesh.vertex_colors = mesh.vertex_colors
            
        return reordered_mesh
    
    def optimize_correspondence_with_temporal_consistency(self, mesh_sequence_subset=None, max_frames=10, use_skeleton_driven=True):
        """
        使用时间一致性优化对应关系
        
        Args:
            mesh_sequence_subset: 网格序列子集, 如果为None则处理所有
            max_frames: 最大处理帧数
            use_skeleton_driven: 是否使用骨骼驱动的快速方法
        """
        if mesh_sequence_subset is None:
            frame_indices = list(range(min(len(self.mesh_files), max_frames)))
        else:
            frame_indices = mesh_sequence_subset
        
        # 初始化对应关系
        correspondences = {}
        meshes = {}
        
        # 加载网格并计算初始对应关系
        method_name = "骨骼驱动" if use_skeleton_driven else "特征匹配"
        print(f"开始计算基于{method_name}的顶点对应关系...")
        
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="计算初始对应关系")):
            if frame_idx == self.reference_frame_idx:
                # 参考帧使用恒等映射
                correspondences[frame_idx] = np.arange(len(self.reference_mesh.vertices))
                meshes[frame_idx] = self.reference_mesh
            else:
                mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
                meshes[frame_idx] = mesh
                
                if use_skeleton_driven:
                    # 使用骨骼驱动的快速对应关系计算
                    correspondences[frame_idx] = self.compute_skeleton_driven_correspondence(mesh, frame_idx)
                else:
                    # 使用原始的特征匹配方法
                    correspondences[frame_idx] = self.find_vertex_correspondence(mesh, frame_idx)
        
        # 时间一致性优化
        print("进行时间一致性优化...")
        for iteration in range(3):  # 迭代优化
            for i, frame_idx in enumerate(frame_indices):
                if frame_idx == self.reference_frame_idx:
                    continue
                
                # 获取相邻帧
                prev_frame = frame_indices[max(0, i-1)]
                next_frame = frame_indices[min(len(frame_indices)-1, i+1)]
                
                if prev_frame != frame_idx and next_frame != frame_idx:
                    # 使用相邻帧信息调整当前帧的对应关系
                    prev_corr = correspondences[prev_frame]
                    next_corr = correspondences[next_frame]
                    current_corr = correspondences[frame_idx]
                    
                    # 简单的时间一致性约束：当前对应关系应该接近相邻帧的平均
                    # 这里可以实现更复杂的优化算法
                    pass
        
        return correspondences, meshes
    
    def canonicalize_mesh_sequence(self, output_folder, max_frames=None, use_skeleton_driven=True):
        """
        对整个网格序列进行统一化
        
        Args:
            output_folder: 输出文件夹路径
            max_frames: 最大处理帧数，None表示处理所有
            use_skeleton_driven: 是否使用骨骼驱动的快速对应关系计算
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 确定处理的帧数
        if max_frames is None:
            max_frames = len(self.mesh_files)
        else:
            max_frames = min(max_frames, len(self.mesh_files))
        
        frame_indices = list(range(max_frames))
        
        # 优化对应关系
        correspondences, meshes = self.optimize_correspondence_with_temporal_consistency(
            frame_indices, max_frames, use_skeleton_driven
        )
        
        # 保存统一化的网格
        print("保存统一化网格...")
        canonicalized_info = {
            'reference_frame': self.reference_frame_idx,
            'correspondences': {},
            'stats': {}
        }
        
        for frame_idx in tqdm(frame_indices, desc="保存网格"):
            mesh = meshes[frame_idx]
            correspondence = correspondences[frame_idx]
            
            # 重新排序顶点
            if frame_idx != self.reference_frame_idx:
                canonical_mesh = self.reorder_mesh_vertices(mesh, correspondence)
            else:
                canonical_mesh = mesh
            
            # 保存网格
            output_file = output_path / f"canonical_frame_{frame_idx:06d}.obj"
            success = o3d.io.write_triangle_mesh(str(output_file), canonical_mesh)
            if not success:
                print(f"警告: 保存网格文件失败: {output_file}")
            
            # 保存对应关系信息
            canonicalized_info['correspondences'][str(frame_idx)] = correspondence.tolist()
            canonicalized_info['stats'][str(frame_idx)] = {
                'original_vertices': len(mesh.vertices),
                'canonical_vertices': len(canonical_mesh.vertices)
            }
        
        # 保存元信息
        with open(output_path / 'canonicalization_info.json', 'w') as f:
            json.dump(canonicalized_info, f, indent=2)
        
        print(f"统一化完成！结果保存在 {output_path}")
        return canonicalized_info

def main():
    """
    主函数示例
    """
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"    # 包含npy文件的文件夹
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"            # 包含obj文件的文件夹
    output_folder = "output/canonical_meshes"         # 输出文件夹

    # 创建统一化器
    canonicalizer = InverseMeshCanonicalizer(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=5  # 使用第5帧作为参考
    )
    
    # 加载网格序列
    canonicalizer.load_mesh_sequence(mesh_folder_path)
    
    print("=" * 60)
    print("开始LBS权重优化...")
    print("=" * 60)
    
    # 优化LBS权重
    skinning_weights = canonicalizer.optimize_reference_frame_skinning(
        regularization_lambda=0.01,
        max_iter=500
    )
    
    if skinning_weights is not None:
        # 保存权重
        weights_output_path = "output/skinning_weights.npz"
        canonicalizer.save_skinning_weights(weights_output_path)
        
        print("=" * 60)
        print("验证LBS权重效果...")
        print("=" * 60)
        
        # 验证权重效果
        validation_results = canonicalizer.validate_skinning_weights(
            test_frames=list(range(0, min(len(canonicalizer.mesh_files), 15), 2))  # 测试部分帧
        )
        
        if validation_results:
            print(f"\n验证结果摘要:")
            print(f"平均重建误差: {validation_results['average_error']:.6f}")
            print(f"误差范围: [{validation_results['min_error']:.6f}, {validation_results['max_error']:.6f}]")
            
            # 显示每帧的详细误差
            print(f"\n各帧详细误差:")
            for frame_idx, frame_result in validation_results['frame_errors'].items():
                print(f"  帧 {frame_idx:2d}: {frame_result['mean_error']:.6f} "
                      f"(std: {frame_result['std_error']:.6f})")
    
    print("=" * 60)
    print("可选: 执行传统网格统一化...")
    print("=" * 60)
    
    # 可选：也可以执行传统的网格统一化进行比较
    canonicalization_info = canonicalizer.canonicalize_mesh_sequence(
        output_folder=output_folder,
        max_frames=10,  # 限制处理帧数以加快测试
        use_skeleton_driven=True  # 使用骨骼驱动方法
    )
    
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"LBS权重优化结果保存在: output/skinning_weights.npz")
    print(f"传统统一化结果保存在: {output_folder}")
    print(f"LBS方法处理了reference frame的skinning权重优化")
    print(f"传统方法处理了 {len(canonicalization_info['correspondences'])} 帧的网格统一化")


def demo_lbs_only():
    """
    仅演示LBS权重优化的函数
    """
    print("=" * 60)
    print("LBS权重优化演示")
    print("=" * 60)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    # 创建统一化器
    canonicalizer = InverseMeshCanonicalizer(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=5
    )
    
    # 加载网格序列
    canonicalizer.load_mesh_sequence(mesh_folder_path)
    
    # 优化权重
    skinning_weights = canonicalizer.optimize_reference_frame_skinning(
        regularization_lambda=0.01,
        max_iter=300
    )
    
    if skinning_weights is not None:
        # 保存和验证
        canonicalizer.save_skinning_weights("output/skinning_weights.npz")
        validation_results = canonicalizer.validate_skinning_weights()
        
        print(f"\nLBS优化完成！")
        print(f"平均重建误差: {validation_results['average_error']:.6f}")
    
    return canonicalizer

if __name__ == "__main__":
    # 选择运行模式
    mode = "lbs_only"  # 可选: "full", "lbs_only"
    
    if mode == "lbs_only":
        # 仅运行LBS权重优化
        canonicalizer = demo_lbs_only()
    else:
        # 运行完整流程（LBS + 传统统一化）
        main()