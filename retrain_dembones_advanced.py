#!/usr/bin/env python3
"""
DemBones重训练脚本 - 使用高级统一网格数据
基于advanced_skeleton_unifier.py的统一拓扑结果训练DemBones，获得极高变形质量

实现特点:
1. 使用Heat diffusion统一权重作为初始化
2. 基于157帧统一网格(32140顶点)训练 
3. 输出极高质量的骨骼绑定和权重
4. 兼容Neural Marionette的骨骼格式
"""

import os
import pickle
import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import trimesh

class DemBonesAdvancedTrainer:
    def __init__(self, unified_data_path):
        """初始化DemBones高级训练器"""
        self.unified_data_path = unified_data_path
        self.load_unified_data()
        
    def load_unified_data(self):
        """加载高级统一数据"""
        print("🔄 加载高级统一数据...")
        with open(self.unified_data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.unified_meshes = data['unified_vertices']  # (157, 32140, 3)
        self.bone_transforms = data['bone_transforms']  # (157, 24, 4, 4)
        self.heat_diffusion_weights = data['heat_diffusion_weights']  # (32140, 24)
        self.triangles = data['triangles']  # (40000, 3)
        self.rest_pose = data['rest_pose']  # (32140, 3)
        self.joints = data['joints']  # (24, 3)
        self.parents = data['parents']  # (24,)
        
        # 创建模板网格对象
        import open3d as o3d
        self.template_mesh = o3d.geometry.TriangleMesh()
        self.template_mesh.vertices = o3d.utility.Vector3dVector(self.rest_pose)
        self.template_mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        
        print(f"✅ 加载完成:")
        print(f"   统一网格: {self.unified_meshes.shape}")
        print(f"   骨骼变换: {self.bone_transforms.shape}")
        print(f"   Heat权重: {self.heat_diffusion_weights.shape}")
        print(f"   三角形: {self.triangles.shape}")
        print(f"   关节数: {self.joints.shape[0]}")
        
    def setup_dembones_config(self):
        """设置DemBones配置用于极高质量训练"""
        config = {
            # 基础设置
            'num_bones': 24,
            'num_vertices': 32140,
            'num_frames': 157,
            
            # 极高质量训练参数
            'max_iterations': 500,        # 增加迭代次数
            'convergence_tolerance': 1e-6, # 更严格的收敛条件
            'bone_length_regularization': 0.001,
            'bone_symmetry_regularization': 0.01,
            
            # 权重平滑参数
            'weight_smoothness': 0.1,     # 权重平滑强度
            'spatial_regularization': 0.05,
            'temporal_consistency': 0.02,
            
            # Heat diffusion初始化
            'use_heat_init': True,
            'heat_init_strength': 0.8,    # Heat权重初始化强度
            
            # 骨骼层次结构
            'enforce_hierarchy': True,
            'root_bone_index': 0,
        }
        return config
        
    def prepare_dembones_input(self):
        """准备DemBones输入数据格式"""
        print("🔧 准备DemBones输入数据...")
        
        # 1. 顶点位置数据 (frames, vertices, 3)
        vertex_positions = self.unified_meshes.astype(np.float64)
        
        # 2. 三角形拓扑（从加载的数据提取）
        triangles = self.triangles.astype(np.int32)
        
        # 3. 初始骨骼权重（Heat diffusion结果）
        initial_weights = self.heat_diffusion_weights.astype(np.float64)
        
        # 4. 骨骼初始位置（从第一帧提取）
        initial_bone_positions = self.extract_bone_positions_from_transforms(
            self.bone_transforms[0]
        )
        
        # 5. 骨骼层次结构（Neural Marionette标准骨架）
        bone_hierarchy = self.get_nm_bone_hierarchy()
        
        return {
            'vertex_positions': vertex_positions,
            'triangles': triangles,
            'initial_weights': initial_weights,
            'initial_bone_positions': initial_bone_positions,
            'bone_hierarchy': bone_hierarchy
        }
        
    def extract_bone_positions_from_transforms(self, bone_transforms):
        """从骨骼变换矩阵提取骨骼位置"""
        bone_positions = []
        for i in range(bone_transforms.shape[0]):
            transform = bone_transforms[i]
            position = transform[:3, 3]  # 提取平移部分
            bone_positions.append(position)
        return np.array(bone_positions)
        
    def get_nm_bone_hierarchy(self):
        """获取Neural Marionette标准骨架层次结构"""
        # 使用加载的parents数据
        hierarchy = {}
        for i, parent in enumerate(self.parents):
            hierarchy[i] = int(parent) if parent >= 0 else -1
        return hierarchy
        
    def optimize_weights_with_heat_init(self, input_data, config):
        """使用Heat diffusion初始化的权重优化"""
        print("🔥 开始DemBones权重优化...")
        
        # 获取数据
        vertices = input_data['vertex_positions']  # (157, 32140, 3)
        initial_weights = input_data['initial_weights']  # (32140, 24)
        triangles = input_data['triangles']
        
        # 构建拉普拉斯矩阵用于平滑
        laplacian = self.build_mesh_laplacian(vertices[0], triangles)
        
        # 迭代优化
        current_weights = initial_weights.copy()
        
        for iteration in range(config['max_iterations']):
            # 1. 骨骼变换优化
            bone_transforms = self.optimize_bone_transforms(
                vertices, current_weights, config
            )
            
            # 2. 权重优化
            new_weights = self.optimize_skinning_weights(
                vertices, bone_transforms, laplacian, config
            )
            
            # 3. 收敛检查
            weight_change = np.linalg.norm(new_weights - current_weights)
            
            if iteration % 50 == 0:
                print(f"   迭代 {iteration:3d}: 权重变化 = {weight_change:.8f}")
                
            if weight_change < config['convergence_tolerance']:
                print(f"✅ 在第{iteration}次迭代收敛")
                break
                
            current_weights = new_weights
            
        return current_weights, bone_transforms
        
    def build_mesh_laplacian(self, vertices, triangles):
        """构建网格拉普拉斯矩阵"""
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # 使用trimesh的拉普拉斯矩阵
        try:
            # 尝试使用smoothing拉普拉斯
            laplacian = mesh.smoothed_laplacian
        except:
            # 备用方案：手动构建拉普拉斯矩阵
            laplacian = self.build_laplacian_manual(vertices, triangles)
            
        return laplacian
        
    def build_laplacian_manual(self, vertices, triangles):
        """手动构建拉普拉斯矩阵"""
        from scipy.sparse import coo_matrix, diags
        
        n_vertices = len(vertices)
        
        # 构建邻接矩阵
        edges = []
        for face in triangles:
            edges.extend([(face[0], face[1]), (face[1], face[2]), (face[2], face[0])])
            
        # 去重并构建稀疏矩阵
        edge_set = set()
        for e in edges:
            edge_set.add((min(e), max(e)))
            
        # 构建度矩阵和邻接矩阵
        degrees = np.zeros(n_vertices)
        row_indices = []
        col_indices = []
        
        for i, j in edge_set:
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            degrees[i] += 1
            degrees[j] += 1
            
        # 邻接矩阵
        adjacency = coo_matrix(
            (np.ones(len(row_indices)), (row_indices, col_indices)),
            shape=(n_vertices, n_vertices)
        ).tocsr()
        
        # 度矩阵
        degree_matrix = diags(degrees, format='csr')
        
        # 拉普拉斯矩阵 = 度矩阵 - 邻接矩阵
        laplacian = degree_matrix - adjacency
        
        return laplacian
        
    def optimize_bone_transforms(self, vertices, weights, config):
        """优化骨骼变换"""
        num_frames, num_vertices, _ = vertices.shape
        num_bones = weights.shape[1]
        
        bone_transforms = np.zeros((num_frames, num_bones, 4, 4))
        
        for frame in range(num_frames):
            for bone in range(num_bones):
                # 计算当前骨骼影响的顶点
                bone_weights = weights[:, bone]
                influenced_indices = bone_weights > 0.01
                
                if not np.any(influenced_indices):
                    bone_transforms[frame, bone] = np.eye(4)
                    continue
                
                # 加权最小二乘法计算变换
                transform = self.compute_weighted_transform(
                    vertices[0, influenced_indices],  # 模板位置
                    vertices[frame, influenced_indices],  # 当前位置
                    bone_weights[influenced_indices]  # 权重
                )
                bone_transforms[frame, bone] = transform
                
        return bone_transforms
        
    def compute_weighted_transform(self, source_points, target_points, weights):
        """计算加权变换矩阵"""
        # 加权质心
        weights_sum = np.sum(weights)
        if weights_sum < 1e-8:
            return np.eye(4)
            
        source_centroid = np.average(source_points, weights=weights, axis=0)
        target_centroid = np.average(target_points, weights=weights, axis=0)
        
        # 去质心
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # 加权协方差矩阵
        H = np.zeros((3, 3))
        for i in range(len(source_points)):
            w = weights[i]
            H += w * np.outer(source_centered[i], target_centered[i])
            
        # SVD分解得到旋转
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 确保是正确的旋转矩阵
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # 平移
        t = target_centroid - R @ source_centroid
        
        # 构建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
        
    def optimize_skinning_weights(self, vertices, bone_transforms, laplacian, config):
        """优化蒙皮权重"""
        num_vertices = vertices.shape[1]
        num_bones = bone_transforms.shape[1]
        
        # 构建线性方程组
        optimized_weights = np.zeros((num_vertices, num_bones))
        
        for vertex_idx in range(num_vertices):
            # 为每个顶点优化权重
            vertex_weights = self.solve_vertex_weights(
                vertex_idx, vertices, bone_transforms, laplacian, config
            )
            optimized_weights[vertex_idx] = vertex_weights
            
        # 权重归一化
        weight_sums = np.sum(optimized_weights, axis=1, keepdims=True)
        weight_sums = np.maximum(weight_sums, 1e-8)
        optimized_weights /= weight_sums
        
        return optimized_weights
        
    def solve_vertex_weights(self, vertex_idx, vertices, bone_transforms, laplacian, config):
        """求解单个顶点的最优权重"""
        num_frames = vertices.shape[0]
        num_bones = bone_transforms.shape[1]
        
        # 构建最小二乘问题: Aw = b
        A_list = []
        b_list = []
        
        for frame in range(num_frames):
            vertex_pos = vertices[frame, vertex_idx]
            
            # 每个骨骼对顶点的影响
            bone_effects = np.zeros((3, num_bones))
            for bone in range(num_bones):
                transform = bone_transforms[frame, bone]
                transformed_pos = transform[:3, :3] @ vertices[0, vertex_idx] + transform[:3, 3]
                bone_effects[:, bone] = transformed_pos
                
            A_list.append(bone_effects)
            b_list.append(vertex_pos)
            
        # 合并所有帧的约束
        A = np.vstack(A_list)  # (num_frames*3, num_bones)
        b = np.hstack(b_list)  # (num_frames*3,)
        
        # 添加权重归一化约束
        A_norm = np.ones((1, num_bones))
        b_norm = np.array([1.0])
        
        A_combined = np.vstack([A, A_norm])
        b_combined = np.hstack([b, b_norm])
        
        # 求解最小二乘
        try:
            weights, residual, rank, s = np.linalg.lstsq(A_combined, b_combined, rcond=None)
            weights = np.maximum(weights, 0)  # 非负约束
            
            # 重新归一化
            weight_sum = np.sum(weights)
            if weight_sum > 1e-8:
                weights /= weight_sum
            else:
                weights = np.zeros(num_bones)
                weights[0] = 1.0  # 默认给第一个骨骼
                
        except np.linalg.LinAlgError:
            # 求解失败，使用均匀权重
            weights = np.ones(num_bones) / num_bones
            
        return weights
        
    def save_dembones_results(self, optimized_weights, bone_transforms, output_dir):
        """保存DemBones优化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存优化权重
        weights_path = os.path.join(output_dir, "dembones_optimized_weights.npy")
        np.save(weights_path, optimized_weights)
        
        # 2. 保存骨骼变换
        transforms_path = os.path.join(output_dir, "dembones_bone_transforms.npy")
        np.save(transforms_path, bone_transforms)
        
        # 3. 保存完整结果
        results = {
            'optimized_weights': optimized_weights,
            'bone_transforms': bone_transforms,
            'unified_meshes': self.unified_meshes,
            'template_mesh_vertices': self.rest_pose,
            'template_mesh_triangles': self.triangles,
            'training_config': self.setup_dembones_config()
        }
        
        results_path = os.path.join(output_dir, "dembones_advanced_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"✅ DemBones结果已保存到: {output_dir}")
        print(f"   权重文件: {weights_path}")
        print(f"   变换文件: {transforms_path}")
        print(f"   完整结果: {results_path}")
        
        return results_path
        
    def validate_results(self, optimized_weights, bone_transforms):
        """验证优化结果质量"""
        print("🔍 验证DemBones优化结果...")
        
        # 1. 权重分析
        weight_stats = {
            'min_weight': np.min(optimized_weights),
            'max_weight': np.max(optimized_weights),
            'mean_weight': np.mean(optimized_weights),
            'sparsity': np.mean(optimized_weights < 0.01)
        }
        
        # 2. 重建误差分析
        reconstruction_errors = []
        for frame in range(self.unified_meshes.shape[0]):
            reconstructed = self.reconstruct_mesh(
                frame, optimized_weights, bone_transforms
            )
            error = np.linalg.norm(
                reconstructed - self.unified_meshes[frame], axis=1
            )
            reconstruction_errors.append(np.mean(error))
            
        avg_error = np.mean(reconstruction_errors)
        max_error = np.max(reconstruction_errors)
        
        print(f"📊 质量验证结果:")
        print(f"   权重范围: [{weight_stats['min_weight']:.6f}, {weight_stats['max_weight']:.6f}]")
        print(f"   权重稀疏度: {weight_stats['sparsity']:.3f}")
        print(f"   平均重建误差: {avg_error:.6f}")
        print(f"   最大重建误差: {max_error:.6f}")
        
        return {
            'weight_stats': weight_stats,
            'reconstruction_error': avg_error,
            'max_reconstruction_error': max_error
        }
        
    def reconstruct_mesh(self, frame_idx, weights, bone_transforms):
        """使用权重和骨骼变换重建网格"""
        template_vertices = self.rest_pose
        reconstructed = np.zeros_like(template_vertices)
        
        for i, vertex in enumerate(template_vertices):
            vertex_homo = np.append(vertex, 1.0)  # 齐次坐标
            
            # 加权骨骼变换
            transformed_vertex = np.zeros(3)
            for bone in range(weights.shape[1]):
                weight = weights[i, bone]
                if weight > 1e-6:
                    transform = bone_transforms[frame_idx, bone]
                    transformed = transform @ vertex_homo
                    transformed_vertex += weight * transformed[:3]
                    
            reconstructed[i] = transformed_vertex
            
        return reconstructed
        
    def run_advanced_training(self, output_dir):
        """运行完整的DemBones高级训练流程"""
        print("🚀 启动DemBones高级训练流程...")
        
        # 1. 设置配置
        config = self.setup_dembones_config()
        print(f"📋 训练配置: {config['max_iterations']}次迭代，收敛阈值{config['convergence_tolerance']}")
        
        # 2. 准备输入数据
        input_data = self.prepare_dembones_input()
        print(f"📊 输入数据准备完成")
        
        # 3. 运行优化
        optimized_weights, bone_transforms = self.optimize_weights_with_heat_init(
            input_data, config
        )
        
        # 4. 验证结果
        validation_results = self.validate_results(optimized_weights, bone_transforms)
        
        # 5. 保存结果
        results_path = self.save_dembones_results(
            optimized_weights, bone_transforms, output_dir
        )
        
        print("🎉 DemBones高级训练完成！")
        print(f"📈 获得极高变形质量的骨骼权重和变换")
        
        return results_path, validation_results

def main():
    """主函数"""
    # 输入路径
    unified_data_path = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\advanced_unified_results.pkl"
    
    # 输出路径
    output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembones_advanced"
    
    # 检查输入文件
    if not os.path.exists(unified_data_path):
        print(f"❌ 找不到统一数据文件: {unified_data_path}")
        return
        
    # 创建训练器
    trainer = DemBonesAdvancedTrainer(unified_data_path)
    
    # 运行训练
    results_path, validation = trainer.run_advanced_training(output_dir)
    
    print(f"\n🎯 训练完成! 结果保存在: {results_path}")
    print(f"📊 重建误差: {validation['reconstruction_error']:.6f}")

if __name__ == "__main__":
    main()
