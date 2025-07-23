import json
import numpy as np
import trimesh
import os
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import pickle
from tqdm import tqdm
import open3d as o3d

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
        reference_frame_idx=0  # 使用第一帧作为参考
    )
    
    # 加载网格序列
    canonicalizer.load_mesh_sequence(mesh_folder_path)
    
    # 执行统一化
    canonicalization_info = canonicalizer.canonicalize_mesh_sequence(
        output_folder=output_folder,
        max_frames=10  # 限制处理帧数以加快测试
    )
    
    print("统一化完成！")
    print(f"处理了 {len(canonicalization_info['correspondences'])} 帧")

if __name__ == "__main__":
    main()