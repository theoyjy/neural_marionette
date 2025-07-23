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
        
    def load_skeleton_data(self):
        """加载numpy格式的骨骼数据"""
        try:
            # 加载关键点数据 [num_frames, num_joints, 3]
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # 加载变换矩阵 [num_frames, num_joints, 4, 4]
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # 加载父节点关系 [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"成功加载骨骼数据:")
            print(f"  - 帧数: {self.num_frames}")
            print(f"  - 关节数: {self.num_joints}")
            print(f"  - 关键点形状: {self.keypoints.shape}")
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
        vertices = mesh.vertices
        keypoints = self.keypoints[frame_idx]  # [num_joints, 3]
        
        # 计算顶点到关节点的距离
        distances = cdist(vertices, keypoints)
        
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
        vertices = mesh.vertices
        
        # 几何特征
        geometric_features = []
        
        # 1. 顶点坐标（骨骼空间）
        keypoints = self.keypoints[frame_idx]  # [num_joints, 3]
        transforms = self.transforms[frame_idx]  # [num_joints, 4, 4]
        
        # 将顶点变换到骨骼空间
        vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
        
        # 使用第一个关节的逆变换作为根变换
        if len(transforms) > 0:
            root_inv_transform = np.linalg.inv(transforms[0])
            canonical_vertices = (root_inv_transform @ vertices_homogeneous.T).T[:, :3]
        else:
            canonical_vertices = vertices
        
        geometric_features.append(canonical_vertices)
        
        # 2. 法向量
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            geometric_features.append(mesh.vertex_normals)
        else:
            # 计算法向量
            mesh.fix_normals()
            geometric_features.append(mesh.vertex_normals)
        
        # 3. 骨骼权重特征
        bone_weights = self.compute_bone_influenced_vertices(mesh, frame_idx)
        geometric_features.append(bone_weights)
        
        # 4. 曲率特征（简化版）
        try:
            # 计算每个顶点到其邻居的平均距离作为局部曲率的近似
            vertex_curvature = []
            for i in range(len(vertices)):
                vertex = vertices[i]
                # 找到最近的10个顶点
                distances_to_vertex = np.linalg.norm(vertices - vertex, axis=1)
                nearest_indices = np.argsort(distances_to_vertex)[1:11]  # 排除自己
                nearest_vertices = vertices[nearest_indices]
                
                # 计算平均距离和方差作为曲率特征
                mean_dist = np.mean(np.linalg.norm(nearest_vertices - vertex, axis=1))
                var_dist = np.var(np.linalg.norm(nearest_vertices - vertex, axis=1))
                vertex_curvature.append([mean_dist, var_dist])
            
            geometric_features.append(np.array(vertex_curvature))
        except:
            # 如果曲率计算失败，使用零特征
            geometric_features.append(np.zeros((len(vertices), 2)))
        
        # 合并所有特征
        features = np.hstack(geometric_features)
        
        return features
    
    def find_vertex_correspondence(self, target_mesh, target_frame_idx):
        """
        找到目标网格与参考网格的顶点对应关系
        
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
        new_vertices = np.zeros_like(self.reference_mesh.vertices)
        
        for target_idx, ref_idx in enumerate(correspondence_map):
            if target_idx < len(mesh.vertices):
                new_vertices[ref_idx] = mesh.vertices[target_idx]
        
        # 创建新的网格
        # 注意：面可能需要相应调整，这里简化处理
        reordered_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(new_vertices), faces=o3d.utility.Vector3iVector(mesh.faces))
        return reordered_mesh
    
    def optimize_correspondence_with_temporal_consistency(self, mesh_sequence_subset=None, max_frames=10):
        """
        使用时间一致性优化对应关系
        
        Args:
            mesh_sequence_subset: 网格序列子集, 如果为None则处理所有
            max_frames: 最大处理帧数
        """
        if mesh_sequence_subset is None:
            frame_indices = list(range(min(len(self.mesh_files), max_frames)))
        else:
            frame_indices = mesh_sequence_subset
        
        # 初始化对应关系
        correspondences = {}
        meshes = {}
        
        # 加载网格并计算初始对应关系
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="计算初始对应关系")):
            if frame_idx == self.reference_frame_idx:
                # 参考帧使用恒等映射
                correspondences[frame_idx] = np.arange(len(self.reference_mesh.vertices))
                meshes[frame_idx] = self.reference_mesh
            else:
                mesh = o3d.io.read_triangle_mesh(str(self.mesh_files[frame_idx]))
                meshes[frame_idx] = mesh
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
    
    def canonicalize_mesh_sequence(self, output_folder, max_frames=None):
        """
        对整个网格序列进行统一化
        
        Args:
            output_folder: 输出文件夹路径
            max_frames: 最大处理帧数，None表示处理所有
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
            frame_indices, max_frames
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
            canonical_mesh.export(str(output_file))
            
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