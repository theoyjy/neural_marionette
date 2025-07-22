#!/usr/bin/env python3
"""
简化的DemBones重训练脚本 - 专注核心功能
基于advanced_skeleton_unifier.py的统一拓扑结果优化骨骼权重
"""

import os
import pickle
import numpy as np

class SimplifiedDemBonesTrainer:
    def __init__(self, unified_data_path):
        """初始化简化的DemBones训练器"""
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
        
        print(f"✅ 加载完成:")
        print(f"   统一网格: {self.unified_meshes.shape}")
        print(f"   骨骼变换: {self.bone_transforms.shape}")
        print(f"   Heat权重: {self.heat_diffusion_weights.shape}")
        
    def optimize_skinning_weights_iterative(self, max_iterations=200):
        """迭代优化蒙皮权重"""
        print("🔥 开始迭代优化蒙皮权重...")
        
        # 初始化权重
        current_weights = self.heat_diffusion_weights.copy()
        
        num_frames, num_vertices, _ = self.unified_meshes.shape
        num_bones = self.bone_transforms.shape[1]
        
        for iteration in range(max_iterations):
            # 1. 为每个顶点优化权重
            new_weights = np.zeros_like(current_weights)
            
            for vertex_idx in range(num_vertices):
                if vertex_idx % 5000 == 0:
                    print(f"   迭代 {iteration+1}/{max_iterations}, 顶点 {vertex_idx}/{num_vertices}")
                
                # 计算该顶点的最优权重
                vertex_weights = self.optimize_single_vertex_weights(
                    vertex_idx, current_weights[vertex_idx]
                )
                new_weights[vertex_idx] = vertex_weights
                
            # 2. 计算权重变化
            weight_change = np.linalg.norm(new_weights - current_weights)
            
            print(f"   迭代 {iteration+1}: 权重变化 = {weight_change:.8f}")
            
            # 3. 更新权重
            current_weights = new_weights
            
            # 4. 收敛检查
            if weight_change < 1e-6:
                print(f"✅ 在第{iteration+1}次迭代收敛")
                break
                
        return current_weights
        
    def optimize_single_vertex_weights(self, vertex_idx, initial_weights):
        """优化单个顶点的权重"""
        num_frames = self.unified_meshes.shape[0]
        num_bones = self.bone_transforms.shape[1]
        
        # 目标位置（所有帧中该顶点的位置）
        target_positions = self.unified_meshes[:, vertex_idx, :]  # (num_frames, 3)
        rest_position = self.rest_pose[vertex_idx]  # (3,)
        
        # 构建线性方程组 Aw = b
        # 每帧产生3个方程（x, y, z坐标）
        A = np.zeros((num_frames * 3, num_bones))
        b = np.zeros(num_frames * 3)
        
        for frame in range(num_frames):
            # 该帧的目标位置
            target_pos = target_positions[frame]
            
            # 构建该帧的方程
            for bone in range(num_bones):
                # 应用骨骼变换到rest位置
                rest_homo = np.append(rest_position, 1.0)
                transform = self.bone_transforms[frame, bone]
                transformed_pos = (transform @ rest_homo)[:3]
                
                # 填入系数矩阵
                row_start = frame * 3
                A[row_start:row_start+3, bone] = transformed_pos
                
            # 填入右侧向量
            b[row_start:row_start+3] = target_pos
            
        # 添加归一化约束: sum(weights) = 1
        A_constrained = np.vstack([A, np.ones((1, num_bones))])
        b_constrained = np.append(b, 1.0)
        
        # 求解最小二乘问题
        try:
            weights, residual, rank, s = np.linalg.lstsq(
                A_constrained, b_constrained, rcond=None
            )
            
            # 确保权重非负
            weights = np.maximum(weights, 0)
            
            # 重新归一化
            weight_sum = np.sum(weights)
            if weight_sum > 1e-8:
                weights /= weight_sum
            else:
                # 如果权重和为0，使用初始权重
                weights = initial_weights / (np.sum(initial_weights) + 1e-8)
                
        except np.linalg.LinAlgError:
            # 求解失败，使用初始权重
            weights = initial_weights / (np.sum(initial_weights) + 1e-8)
            
        return weights
        
    def compute_reconstruction_error(self, weights):
        """计算重建误差"""
        print("🔍 计算重建误差...")
        
        total_error = 0
        num_frames, num_vertices, _ = self.unified_meshes.shape
        
        for frame in range(num_frames):
            # 重建该帧的网格
            reconstructed = self.reconstruct_mesh_frame(frame, weights)
            
            # 计算误差
            target = self.unified_meshes[frame]
            frame_error = np.linalg.norm(reconstructed - target, axis=1)
            total_error += np.mean(frame_error)
            
        avg_error = total_error / num_frames
        print(f"   平均重建误差: {avg_error:.6f}")
        
        return avg_error
        
    def reconstruct_mesh_frame(self, frame_idx, weights):
        """重建指定帧的网格"""
        num_vertices = self.rest_pose.shape[0]
        num_bones = weights.shape[1]
        
        reconstructed = np.zeros((num_vertices, 3))
        
        for vertex_idx in range(num_vertices):
            rest_pos = self.rest_pose[vertex_idx]
            rest_homo = np.append(rest_pos, 1.0)
            
            # 加权骨骼变换
            final_pos = np.zeros(3)
            for bone in range(num_bones):
                weight = weights[vertex_idx, bone]
                if weight > 1e-6:
                    transform = self.bone_transforms[frame_idx, bone]
                    transformed = (transform @ rest_homo)[:3]
                    final_pos += weight * transformed
                    
            reconstructed[vertex_idx] = final_pos
            
        return reconstructed
        
    def refine_weights_with_regularization(self, weights, lambda_smooth=0.1):
        """使用正则化精细调整权重"""
        print("🎯 应用权重平滑正则化...")
        
        # 构建邻接关系（简化版）
        adjacency = self.build_simple_adjacency()
        
        # 平滑化权重
        refined_weights = weights.copy()
        
        for bone in range(weights.shape[1]):
            bone_weights = weights[:, bone]
            
            # 拉普拉斯平滑
            smoothed = bone_weights.copy()
            for vertex in range(len(bone_weights)):
                neighbors = adjacency[vertex]
                if len(neighbors) > 0:
                    neighbor_avg = np.mean([bone_weights[n] for n in neighbors])
                    smoothed[vertex] = (1 - lambda_smooth) * bone_weights[vertex] + lambda_smooth * neighbor_avg
                    
            refined_weights[:, bone] = smoothed
            
        # 重新归一化
        weight_sums = np.sum(refined_weights, axis=1, keepdims=True)
        weight_sums = np.maximum(weight_sums, 1e-8)
        refined_weights /= weight_sums
        
        return refined_weights
        
    def build_simple_adjacency(self):
        """构建简化的顶点邻接关系"""
        print("   构建顶点邻接关系...")
        
        num_vertices = self.rest_pose.shape[0]
        adjacency = [[] for _ in range(num_vertices)]
        
        # 从三角形构建邻接关系
        for triangle in self.triangles:
            v0, v1, v2 = triangle
            adjacency[v0].extend([v1, v2])
            adjacency[v1].extend([v0, v2])
            adjacency[v2].extend([v0, v1])
            
        # 去重
        for i in range(num_vertices):
            adjacency[i] = list(set(adjacency[i]))
            
        return adjacency
        
    def save_optimized_results(self, optimized_weights, output_dir):
        """保存优化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存优化权重
        weights_path = os.path.join(output_dir, "optimized_skinning_weights.npy")
        np.save(weights_path, optimized_weights)
        
        # 2. 保存完整结果
        results = {
            'optimized_weights': optimized_weights,
            'heat_diffusion_weights': self.heat_diffusion_weights,
            'bone_transforms': self.bone_transforms,
            'unified_meshes': self.unified_meshes,
            'rest_pose': self.rest_pose,
            'triangles': self.triangles,
            'joints': self.joints,
            'parents': self.parents,
        }
        
        results_path = os.path.join(output_dir, "dembones_optimized_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"✅ 优化结果已保存:")
        print(f"   权重文件: {weights_path}")
        print(f"   完整结果: {results_path}")
        
        return results_path
        
    def run_optimization(self, output_dir):
        """运行完整的权重优化流程"""
        print("🚀 启动DemBones权重优化流程...")
        
        # 1. 计算初始误差
        initial_error = self.compute_reconstruction_error(self.heat_diffusion_weights)
        print(f"📊 初始重建误差: {initial_error:.6f}")
        
        # 2. 迭代优化权重
        optimized_weights = self.optimize_skinning_weights_iterative(max_iterations=50)
        
        # 3. 应用正则化
        refined_weights = self.refine_weights_with_regularization(optimized_weights)
        
        # 4. 计算最终误差
        final_error = self.compute_reconstruction_error(refined_weights)
        print(f"📊 最终重建误差: {final_error:.6f}")
        print(f"📈 误差改善: {((initial_error - final_error) / initial_error * 100):.2f}%")
        
        # 5. 保存结果
        results_path = self.save_optimized_results(refined_weights, output_dir)
        
        print("🎉 DemBones权重优化完成！")
        print(f"📁 结果保存在: {results_path}")
        
        return results_path, {
            'initial_error': initial_error,
            'final_error': final_error,
            'improvement_percent': (initial_error - final_error) / initial_error * 100
        }

def main():
    """主函数"""
    # 输入路径
    unified_data_path = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\advanced_unified_results.pkl"
    
    # 输出路径
    output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembones_optimized"
    
    # 检查输入文件
    if not os.path.exists(unified_data_path):
        print(f"❌ 找不到统一数据文件: {unified_data_path}")
        return
        
    # 创建训练器
    trainer = SimplifiedDemBonesTrainer(unified_data_path)
    
    # 运行优化
    results_path, metrics = trainer.run_optimization(output_dir)
    
    print(f"\n🎯 优化完成!")
    print(f"📊 性能提升: {metrics['improvement_percent']:.2f}%")
    print(f"📁 结果文件: {results_path}")

if __name__ == "__main__":
    main()
