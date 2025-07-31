#!/usr/bin/env python3
"""
增强版自适应插值器

实现以下优化策略：
1. 动态参考帧选择：根据相似性动态选择最佳参考帧
2. 多尺度插值：在不同分辨率下进行插值，然后融合
3. 自适应权重调整：根据帧间差异调整插值权重
4. 时序一致性优化：确保插值结果的时序平滑性
5. 质量评估反馈：实时评估插值质量并调整参数
"""

import numpy as np
import open3d as o3d
import torch
import cv2
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import json

class EnhancedAdaptiveInterpolator:
    """
    增强版自适应插值器
    
    主要特性：
    - 动态参考帧选择
    - 多尺度插值融合
    - 自适应权重调整
    - 时序一致性优化
    - 质量评估反馈
    """
    
    def __init__(self, skeleton_data_dir: str, mesh_folder_path: str, weights_path: str):
        """
        初始化增强版自适应插值器
        
        Args:
            skeleton_data_dir: 骨骼数据目录
            mesh_folder_path: 网格文件夹路径
            weights_path: 权重文件路径
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.mesh_folder_path = Path(mesh_folder_path)
        self.weights_path = Path(weights_path)
        self.output_dir = None
        
        # 加载排序后的网格文件列表
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        if len(self.mesh_files) == 0:
            raise ValueError(f"在 {self.mesh_folder_path} 中未找到obj文件")
        
        print(f"✅ 增强版自适应插值器初始化完成")
        print(f"  - 网格文件数量: {len(self.mesh_files)}")
        print(f"  - 相似性阈值: {self.similarity_threshold}")
        print(f"  - 最大参考帧数: {self.max_reference_frames}")
        print(f"  - 质量阈值: {self.quality_threshold}")
        
        # 配置参数
        self.similarity_threshold = 0.8
        self.max_reference_frames = 4
        self.quality_threshold = 0.7
        self.temporal_smoothness_weight = 0.3
        
        # 缓存
        self.frame_cache = {}
        self.similarity_cache = {}
        self.quality_metrics = {}
    
    def load_frame_data(self, frame_idx: int) -> Dict:
        """加载帧数据（网格、骨骼、权重）"""
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]
        
        # 检查帧索引范围
        if frame_idx >= len(self.mesh_files):
            raise ValueError(f"帧索引超出范围: {frame_idx} >= {len(self.mesh_files)}")
        
        # 加载网格 - 使用排序后的文件列表
        mesh_file = self.mesh_files[frame_idx]
        if not mesh_file.exists():
            raise FileNotFoundError(f"网格文件不存在: {mesh_file}")
        
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        
        # 加载骨骼数据
        skeleton_file = self.skeleton_data_dir / f"keypoints_{frame_idx:04d}.npy"
        if skeleton_file.exists():
            keypoints = np.load(str(skeleton_file))
        else:
            keypoints = None
        
        # 加载权重数据
        weights_file = self.weights_path / f"weights_{frame_idx:04d}.npy"
        if weights_file.exists():
            weights = np.load(str(weights_file))
        else:
            weights = None
        
        frame_data = {
            'mesh': mesh,
            'keypoints': keypoints,
            'weights': weights,
            'frame_idx': frame_idx
        }
        
        self.frame_cache[frame_idx] = frame_data
        return frame_data
    
    def compute_frame_similarity(self, frame1: int, frame2: int) -> float:
        """计算两帧之间的相似性"""
        cache_key = (min(frame1, frame2), max(frame1, frame2))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        try:
            data1 = self.load_frame_data(frame1)
            data2 = self.load_frame_data(frame2)
            
            # 计算网格相似性
            mesh_sim = self._compute_mesh_similarity(data1['mesh'], data2['mesh'])
            
            # 计算骨骼相似性
            skeleton_sim = 0.0
            if data1['keypoints'] is not None and data2['keypoints'] is not None:
                skeleton_sim = self._compute_skeleton_similarity(data1['keypoints'], data2['keypoints'])
            
            # 综合相似性
            similarity = 0.7 * mesh_sim + 0.3 * skeleton_sim
            self.similarity_cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            print(f"⚠️ 计算相似性失败 {frame1}-{frame2}: {e}")
            return 0.0
    
    def _compute_mesh_similarity(self, mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh) -> float:
        """计算网格相似性"""
        # 提取顶点
        vertices1 = np.asarray(mesh1.vertices)
        vertices2 = np.asarray(mesh2.vertices)
        
        # 计算Hausdorff距离
        distances = cdist(vertices1, vertices2)
        hausdorff_dist = max(np.min(distances, axis=1).max(), np.min(distances, axis=0).max())
        
        # 转换为相似性分数
        similarity = np.exp(-hausdorff_dist / 0.1)
        return similarity
    
    def _compute_skeleton_similarity(self, keypoints1: np.ndarray, keypoints2: np.ndarray) -> float:
        """计算骨骼相似性"""
        if keypoints1.shape != keypoints2.shape:
            return 0.0
        
        # 计算关键点距离
        distances = np.linalg.norm(keypoints1[:, :3] - keypoints2[:, :3], axis=1)
        mean_distance = np.mean(distances)
        
        # 转换为相似性分数
        similarity = np.exp(-mean_distance / 0.05)
        return similarity
    
    def select_optimal_reference_frames(self, target_frame: int, available_frames: List[int]) -> List[int]:
        """选择最优参考帧"""
        similarities = []
        for frame in available_frames:
            if frame != target_frame:
                sim = self.compute_frame_similarity(target_frame, frame)
                similarities.append((frame, sim))
        
        # 按相似性排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最相似的帧作为参考
        selected_frames = []
        for frame, sim in similarities:
            if sim >= self.similarity_threshold and len(selected_frames) < self.max_reference_frames:
                selected_frames.append(frame)
        
        print(f"🎯 目标帧 {target_frame} 的参考帧选择:")
        for frame, sim in similarities[:5]:
            status = "✅" if frame in selected_frames else "❌"
            print(f"  {status} 帧 {frame}: 相似性 {sim:.3f}")
        
        return selected_frames
    
    def multi_scale_interpolation(self, reference_frames: List[int], target_time: float) -> o3d.geometry.TriangleMesh:
        """多尺度插值"""
        print(f"🔄 多尺度插值: 参考帧 {reference_frames}, 目标时间 {target_time:.3f}")
        
        # 不同尺度的插值结果
        scale_results = []
        
        for scale in [1.0, 0.5, 0.25]:
            print(f"  - 尺度 {scale}: 开始插值...")
            
            # 缩放网格
            scaled_meshes = []
            for frame in reference_frames:
                data = self.load_frame_data(frame)
                mesh = data['mesh']
                
                # 缩放网格
                if scale != 1.0:
                    mesh.scale(scale, scale, scale)
                
                scaled_meshes.append(mesh)
            
            # 在缩放尺度下进行插值
            interpolated_mesh = self._interpolate_meshes(scaled_meshes, target_time)
            
            # 恢复原始尺度
            if scale != 1.0:
                interpolated_mesh.scale(1/scale, 1/scale, 1/scale)
            
            scale_results.append(interpolated_mesh)
        
        # 融合多尺度结果
        final_mesh = self._fuse_multi_scale_results(scale_results)
        
        return final_mesh
    
    def _interpolate_meshes(self, meshes: List[o3d.geometry.TriangleMesh], target_time: float) -> o3d.geometry.TriangleMesh:
        """插值多个网格"""
        if len(meshes) == 1:
            return meshes[0]
        
        # 提取顶点
        vertices_list = [np.asarray(mesh.vertices) for mesh in meshes]
        
        # 计算插值权重
        weights = self._compute_interpolation_weights(len(meshes), target_time)
        
        # 插值顶点
        interpolated_vertices = np.zeros_like(vertices_list[0])
        for i, weight in enumerate(weights):
            interpolated_vertices += weight * vertices_list[i]
        
        # 创建插值网格
        interpolated_mesh = o3d.geometry.TriangleMesh()
        interpolated_mesh.vertices = o3d.utility.Vector3dVector(interpolated_vertices)
        interpolated_mesh.triangles = meshes[0].triangles  # 使用第一个网格的面片
        
        return interpolated_mesh
    
    def _compute_interpolation_weights(self, num_frames: int, target_time: float) -> np.ndarray:
        """计算插值权重"""
        # 线性插值权重
        weights = np.zeros(num_frames)
        
        if num_frames == 2:
            weights[0] = 1 - target_time
            weights[1] = target_time
        else:
            # 多帧插值，使用距离加权
            frame_times = np.linspace(0, 1, num_frames)
            distances = np.abs(frame_times - target_time)
            weights = 1 / (distances + 1e-6)
            weights = weights / np.sum(weights)
        
        return weights
    
    def _fuse_multi_scale_results(self, scale_results: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
        """融合多尺度结果"""
        if len(scale_results) == 1:
            return scale_results[0]
        
        # 加权融合
        weights = [0.5, 0.3, 0.2]  # 高分辨率权重更高
        
        final_vertices = np.zeros_like(np.asarray(scale_results[0].vertices))
        
        for i, (mesh, weight) in enumerate(zip(scale_results, weights)):
            vertices = np.asarray(mesh.vertices)
            final_vertices += weight * vertices
        
        final_mesh = o3d.geometry.TriangleMesh()
        final_mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
        final_mesh.triangles = scale_results[0].triangles
        
        return final_mesh
    
    def evaluate_interpolation_quality(self, interpolated_mesh: o3d.geometry.TriangleMesh, 
                                    reference_frames: List[int]) -> float:
        """评估插值质量"""
        try:
            # 计算网格质量指标
            vertices = np.asarray(interpolated_mesh.vertices)
            
            # 1. 网格完整性
            completeness = len(vertices) / 1000  # 假设正常网格有1000+顶点
            completeness = min(completeness, 1.0)
            
            # 2. 几何一致性
            reference_meshes = [self.load_frame_data(frame)['mesh'] for frame in reference_frames]
            consistency = self._compute_geometric_consistency(interpolated_mesh, reference_meshes)
            
            # 3. 平滑性
            smoothness = self._compute_mesh_smoothness(interpolated_mesh)
            
            # 综合质量分数
            quality = 0.4 * completeness + 0.4 * consistency + 0.2 * smoothness
            
            print(f"📊 插值质量评估:")
            print(f"  - 完整性: {completeness:.3f}")
            print(f"  - 一致性: {consistency:.3f}")
            print(f"  - 平滑性: {smoothness:.3f}")
            print(f"  - 综合质量: {quality:.3f}")
            
            return quality
            
        except Exception as e:
            print(f"⚠️ 质量评估失败: {e}")
            return 0.5
    
    def _compute_geometric_consistency(self, interpolated_mesh: o3d.geometry.TriangleMesh, 
                                     reference_meshes: List[o3d.geometry.TriangleMesh]) -> float:
        """计算几何一致性"""
        try:
            # 计算与参考网格的平均距离
            total_distance = 0.0
            count = 0
            
            for ref_mesh in reference_meshes:
                distances = cdist(np.asarray(interpolated_mesh.vertices), 
                                np.asarray(ref_mesh.vertices))
                min_distances = np.min(distances, axis=1)
                total_distance += np.mean(min_distances)
                count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                consistency = np.exp(-avg_distance / 0.1)
                return consistency
            
            return 0.5
            
        except Exception as e:
            print(f"⚠️ 几何一致性计算失败: {e}")
            return 0.5
    
    def _compute_mesh_smoothness(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """计算网格平滑性"""
        try:
            # 计算顶点法向量的变化
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            
            # 计算法向量的标准差
            normal_std = np.std(normals, axis=0)
            smoothness = 1.0 / (1.0 + np.sum(normal_std))
            
            return smoothness
            
        except Exception as e:
            print(f"⚠️ 平滑性计算失败: {e}")
            return 0.5
    
    def optimize_temporal_consistency(self, interpolated_frames: List[o3d.geometry.TriangleMesh]) -> List[o3d.geometry.TriangleMesh]:
        """优化时序一致性"""
        print(f"🔄 优化时序一致性: {len(interpolated_frames)} 帧")
        
        if len(interpolated_frames) <= 2:
            return interpolated_frames
        
        optimized_frames = []
        
        for i, mesh in enumerate(interpolated_frames):
            if i == 0 or i == len(interpolated_frames) - 1:
                # 首尾帧保持不变
                optimized_frames.append(mesh)
            else:
                # 中间帧进行平滑
                smoothed_mesh = self._smooth_frame(mesh, interpolated_frames, i)
                optimized_frames.append(smoothed_mesh)
        
        return optimized_frames
    
    def _smooth_frame(self, current_mesh: o3d.geometry.TriangleMesh, 
                     all_frames: List[o3d.geometry.TriangleMesh], 
                     current_idx: int) -> o3d.geometry.TriangleMesh:
        """平滑单个帧"""
        try:
            # 获取相邻帧
            prev_mesh = all_frames[current_idx - 1]
            next_mesh = all_frames[current_idx + 1]
            
            # 计算平滑权重
            alpha = self.temporal_smoothness_weight
            
            # 平滑顶点
            current_vertices = np.asarray(current_mesh.vertices)
            prev_vertices = np.asarray(prev_mesh.vertices)
            next_vertices = np.asarray(next_mesh.vertices)
            
            # 确保顶点数量一致
            min_vertices = min(len(current_vertices), len(prev_vertices), len(next_vertices))
            
            smoothed_vertices = (1 - 2*alpha) * current_vertices[:min_vertices] + \
                              alpha * prev_vertices[:min_vertices] + \
                              alpha * next_vertices[:min_vertices]
            
            # 创建平滑后的网格
            smoothed_mesh = o3d.geometry.TriangleMesh()
            smoothed_mesh.vertices = o3d.utility.Vector3dVector(smoothed_vertices)
            smoothed_mesh.triangles = current_mesh.triangles
            
            return smoothed_mesh
            
        except Exception as e:
            print(f"⚠️ 帧平滑失败: {e}")
            return current_mesh
    
    def generate_interpolated_frames(self, frame_start: int, frame_end: int, num_interpolate: int,
                                   max_optimize_frames: int = 5, optimize_weights: bool = True,
                                   output_dir: str = None, debug_frames: List[int] = None,
                                   smooth_mesh: bool = False, subdivide_iter: int = 3,
                                   use_vertex_colors: bool = False) -> Dict:
        """生成插值帧"""
        print(f"🚀 增强版自适应插值开始")
        print(f"  - 起始帧索引: {frame_start}")
        print(f"  - 结束帧索引: {frame_end}")
        print(f"  - 插值帧数: {num_interpolate}")
        
        # 检查帧索引范围（frame_start和frame_end是排序后文件列表的索引）
        if frame_start >= len(self.mesh_files) or frame_end >= len(self.mesh_files):
            raise ValueError(f"帧索引超出范围: start_frame={frame_start}, end_frame={frame_end}, 可用帧数={len(self.mesh_files)}")
        
        # 检查帧索引是否相等（不允许相等）
        if frame_start == frame_end:
            raise ValueError(f"起始帧不能等于结束帧: {frame_start} == {frame_end}")
        
        # 确定实际的起始和结束帧（支持反向插值）
        actual_start = min(frame_start, frame_end)
        actual_end = max(frame_start, frame_end)
        is_reverse = frame_start > frame_end
        
        # 打印实际使用的文件信息
        start_file = self.mesh_files[actual_start].name
        end_file = self.mesh_files[actual_end].name
        print(f"  - 使用文件: {start_file} (索引 {actual_start}) -> {end_file} (索引 {actual_end})")
        
        start_time = time.time()
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取可用帧
        available_frames = list(range(actual_start, actual_end + 1))
        print(f"  - 可用帧: {available_frames}")
        
        # 生成插值帧
        interpolated_frames = []
        quality_scores = []
        
        for i in range(num_interpolate):
            target_time = (i + 1) / (num_interpolate + 1)
            target_frame = actual_start + target_time * (actual_end - actual_start)
            
            print(f"\n🎯 生成插值帧 {i+1}/{num_interpolate} (时间: {target_time:.3f})")
            
            # 1. 选择最优参考帧
            reference_frames = self.select_optimal_reference_frames(int(target_frame), available_frames)
            
            if not reference_frames:
                print(f"❌ 无法找到合适的参考帧")
                continue
            
            # 2. 多尺度插值
            interpolated_mesh = self.multi_scale_interpolation(reference_frames, target_time)
            
            # 3. 质量评估
            quality = self.evaluate_interpolation_quality(interpolated_mesh, reference_frames)
            quality_scores.append(quality)
            
            # 4. 如果质量不达标，尝试优化
            if quality < self.quality_threshold:
                print(f"⚠️ 质量不达标 ({quality:.3f} < {self.quality_threshold})，尝试优化...")
                interpolated_mesh = self._optimize_low_quality_frame(interpolated_mesh, reference_frames)
            
            interpolated_frames.append(interpolated_mesh)
            
            # 保存插值帧
            if self.output_dir:
                output_file = self.output_dir / f"enhanced_interpolated_frame_{i:04d}.obj"
                o3d.io.write_triangle_mesh(str(output_file), interpolated_mesh)
                print(f"💾 保存插值帧: {output_file}")
        
        # 5. 时序一致性优化
        if len(interpolated_frames) > 2:
            print(f"\n🔄 应用时序一致性优化...")
            interpolated_frames = self.optimize_temporal_consistency(interpolated_frames)
        
        # 6. 最终质量评估
        final_quality = np.mean(quality_scores) if quality_scores else 0.0
        print(f"\n📊 最终质量评估: {final_quality:.3f}")
        
        # 生成结果
        results = {
            'interpolated_frames': interpolated_frames,
            'quality_scores': quality_scores,
            'average_quality': final_quality,
            'processing_time': time.time() - start_time,
            'method': 'enhanced_adaptive'
        }
        
        print(f"✅ 增强版自适应插值完成")
        print(f"  - 处理时间: {results['processing_time']:.2f}秒")
        print(f"  - 平均质量: {final_quality:.3f}")
        print(f"  - 生成帧数: {len(interpolated_frames)}")
        
        return results
    
    def _optimize_low_quality_frame(self, mesh: o3d.geometry.TriangleMesh, 
                                  reference_frames: List[int]) -> o3d.geometry.TriangleMesh:
        """优化低质量帧"""
        try:
            # 尝试不同的优化策略
            optimized_mesh = mesh
            
            # 1. 网格平滑
            optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=2)
            
            # 2. 网格简化（如果顶点过多）
            if len(optimized_mesh.vertices) > 10000:
                optimized_mesh = optimized_mesh.simplify_quadric_decimation(5000)
            
            # 3. 重新计算法向量
            optimized_mesh.compute_vertex_normals()
            
            return optimized_mesh
            
        except Exception as e:
            print(f"⚠️ 帧优化失败: {e}")
            return mesh


def main():
    """测试函数"""
    print("🧪 增强版自适应插值器测试")
    
    # 测试参数
    skeleton_dir = "output/skeleton_prediction"
    mesh_folder = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_path = "output/skinning_weights"
    
    # 创建插值器
    interpolator = EnhancedAdaptiveInterpolator(skeleton_dir, mesh_folder, weights_path)
    
    # 测试插值
    results = interpolator.generate_interpolated_frames(
        frame_start=1,
        frame_end=10,
        num_interpolate=5,
        output_dir="output/enhanced_interpolation"
    )
    
    print(f"✅ 测试完成: {results}")


if __name__ == "__main__":
    main() 