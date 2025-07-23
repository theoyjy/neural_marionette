#!/usr/bin/env python3
"""
LBS权重验证和可视化测试脚本

测试已优化的LBS权重在不同帧上的重建质量，并生成详细的分析报告和可视化结果。
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import time

from Skinning import InverseMeshCanonicalizer


class LBSValidator:
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path, reference_frame_idx=5):
        """
        初始化LBS验证器
        
        Args:
            skeleton_data_dir: 骨骼数据文件夹
            mesh_folder_path: 网格文件夹
            weights_path: 权重文件路径
            reference_frame_idx: 参考帧索引
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.mesh_folder_path = Path(mesh_folder_path)
        self.weights_path = Path(weights_path)
        self.reference_frame_idx = reference_frame_idx
        
        # 初始化canonicalizer
        self.canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=reference_frame_idx
        )
        
        # 加载网格序列
        self.canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 加载权重
        if not self.canonicalizer.load_skinning_weights(weights_path):
            raise ValueError(f"无法加载权重文件: {weights_path}")
        
        self.results = {}
        
    def test_single_frame(self, frame_idx, verbose=False):
        """
        测试单个帧的重建质量
        
        Args:
            frame_idx: 帧索引
            verbose: 是否输出详细信息
            
        Returns:
            frame_result: 帧测试结果字典
        """
        if frame_idx >= len(self.canonicalizer.mesh_files):
            return None
            
        # 加载目标网格
        target_mesh = o3d.io.read_triangle_mesh(str(self.canonicalizer.mesh_files[frame_idx]))
        target_vertices = np.asarray(target_mesh.vertices)
        
        # 获取rest pose顶点
        rest_vertices = self.canonicalizer.rest_pose_vertices
        
        # 归一化处理
        if frame_idx not in self.canonicalizer.frame_normalization_params:
            self.canonicalizer.frame_normalization_params[frame_idx] = \
                self.canonicalizer.compute_mesh_normalization_params(target_mesh)
        
        target_vertices_norm = self.canonicalizer.normalize_mesh_vertices(
            target_vertices, 
            self.canonicalizer.frame_normalization_params[frame_idx]
        )
        rest_vertices_norm = self.canonicalizer.normalize_mesh_vertices(
            rest_vertices, 
            self.canonicalizer.frame_normalization_params[self.reference_frame_idx]
        )
        
        # 计算相对变换
        target_transforms = self.canonicalizer.transforms[frame_idx]
        rest_transforms = self.canonicalizer.transforms[self.reference_frame_idx]
        
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(self.canonicalizer.num_joints):
            if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            else:
                relative_transforms[j] = np.eye(4)
        
        # 使用LBS预测顶点位置
        start_time = time.time()
        predicted_vertices = self.canonicalizer.apply_lbs_transform(
            rest_vertices_norm, 
            self.canonicalizer.skinning_weights, 
            relative_transforms
        )
        lbs_time = time.time() - start_time
        
        # 计算误差指标
        vertex_errors = np.linalg.norm(predicted_vertices - target_vertices_norm, axis=1)
        
        frame_result = {
            'frame_idx': frame_idx,
            'mean_error': np.mean(vertex_errors),
            'max_error': np.max(vertex_errors),
            'min_error': np.min(vertex_errors),
            'std_error': np.std(vertex_errors),
            'median_error': np.median(vertex_errors),
            'rmse': np.sqrt(np.mean(vertex_errors**2)),
            'lbs_time': lbs_time,
            'num_vertices': len(target_vertices),
            'vertex_errors': vertex_errors,
            'predicted_vertices': predicted_vertices,
            'target_vertices_norm': target_vertices_norm,
            'relative_transforms': relative_transforms
        }
        
        # 计算百分位数误差
        frame_result['p95_error'] = np.percentile(vertex_errors, 95)
        frame_result['p99_error'] = np.percentile(vertex_errors, 99)
        
        # 计算误差分布
        frame_result['error_histogram'] = np.histogram(vertex_errors, bins=50)
        
        if verbose:
            print(f"帧 {frame_idx:3d}: 平均误差={frame_result['mean_error']:.6f}, "
                  f"RMSE={frame_result['rmse']:.6f}, "
                  f"最大误差={frame_result['max_error']:.6f}, "
                  f"时间={lbs_time:.3f}s")
        
        return frame_result
    
    def test_frame_range(self, start_frame=0, end_frame=None, step=1, verbose=True):
        """
        测试帧范围的重建质量
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧，None表示到最后一帧
            step: 步长
            verbose: 是否输出详细信息
            
        Returns:
            results: 测试结果字典
        """
        if end_frame is None:
            end_frame = len(self.canonicalizer.mesh_files)
        
        end_frame = min(end_frame, len(self.canonicalizer.mesh_files))
        frame_indices = list(range(start_frame, end_frame, step))
        
        print(f"测试帧范围: {start_frame} 到 {end_frame-1} (步长={step})")
        print(f"总共测试 {len(frame_indices)} 帧")
        
        results = {
            'frame_results': {},
            'summary': {},
            'test_params': {
                'start_frame': start_frame,
                'end_frame': end_frame,
                'step': step,
                'total_frames': len(frame_indices),
                'reference_frame': self.reference_frame_idx
            }
        }
        
        # 测试各帧
        total_errors = []
        total_times = []
        
        for frame_idx in tqdm(frame_indices, desc="测试帧重建质量"):
            frame_result = self.test_single_frame(frame_idx, verbose=False)
            if frame_result is not None:
                results['frame_results'][frame_idx] = frame_result
                total_errors.append(frame_result['mean_error'])
                total_times.append(frame_result['lbs_time'])
        
        # 计算总体统计
        if total_errors:
            results['summary'] = {
                'overall_mean_error': np.mean(total_errors),
                'overall_std_error': np.std(total_errors),
                'overall_max_error': np.max(total_errors),
                'overall_min_error': np.min(total_errors),
                'overall_median_error': np.median(total_errors),
                'average_lbs_time': np.mean(total_times),
                'total_lbs_time': np.sum(total_times),
                'frames_per_second': len(total_times) / np.sum(total_times) if np.sum(total_times) > 0 else 0
            }
            
            if verbose:
                print(f"\n📊 总体测试结果:")
                print(f"平均重建误差: {results['summary']['overall_mean_error']:.6f} ± {results['summary']['overall_std_error']:.6f}")
                print(f"误差范围: [{results['summary']['overall_min_error']:.6f}, {results['summary']['overall_max_error']:.6f}]")
                print(f"平均LBS时间: {results['summary']['average_lbs_time']:.3f}s")
                print(f"处理速度: {results['summary']['frames_per_second']:.1f} FPS")
        
        self.results = results
        return results
    
    def analyze_error_distribution(self, frame_idx_list=None):
        """
        分析误差分布
        
        Args:
            frame_idx_list: 要分析的帧列表，None表示所有已测试的帧
        """
        if not self.results or 'frame_results' not in self.results:
            print("请先运行测试")
            return
        
        if frame_idx_list is None:
            frame_idx_list = list(self.results['frame_results'].keys())
        
        print(f"\n📈 误差分布分析 (基于 {len(frame_idx_list)} 帧):")
        
        all_errors = []
        for frame_idx in frame_idx_list:
            if frame_idx in self.results['frame_results']:
                frame_result = self.results['frame_results'][frame_idx]
                all_errors.extend(frame_result['vertex_errors'])
        
        if not all_errors:
            print("没有可分析的误差数据")
            return
        
        all_errors = np.array(all_errors)
        
        print(f"总顶点数: {len(all_errors):,}")
        print(f"误差统计:")
        print(f"  平均: {np.mean(all_errors):.6f}")
        print(f"  标准差: {np.std(all_errors):.6f}")
        print(f"  中位数: {np.median(all_errors):.6f}")
        print(f"  25%分位: {np.percentile(all_errors, 25):.6f}")
        print(f"  75%分位: {np.percentile(all_errors, 75):.6f}")
        print(f"  95%分位: {np.percentile(all_errors, 95):.6f}")
        print(f"  99%分位: {np.percentile(all_errors, 99):.6f}")
        print(f"  最大值: {np.max(all_errors):.6f}")
        
        # 分析高误差顶点比例
        thresholds = [0.01, 0.02, 0.05, 0.1]
        print(f"\n高误差顶点比例:")
        for threshold in thresholds:
            ratio = np.sum(all_errors > threshold) / len(all_errors) * 100
            print(f"  误差 > {threshold:.3f}: {ratio:.2f}%")
    
    def generate_visualization_plots(self, output_dir="output/lbs_validation", 
                                   frame_idx_list=None, max_frames_to_plot=20):
        """
        生成可视化图表
        
        Args:
            output_dir: 输出目录
            frame_idx_list: 要可视化的帧列表
            max_frames_to_plot: 最大绘制帧数
        """
        if not self.results or 'frame_results' not in self.results:
            print("请先运行测试")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if frame_idx_list is None:
            frame_idx_list = list(self.results['frame_results'].keys())
        
        # 限制绘制的帧数
        if len(frame_idx_list) > max_frames_to_plot:
            step = len(frame_idx_list) // max_frames_to_plot
            frame_idx_list = frame_idx_list[::step]
        
        print(f"生成可视化图表到: {output_path}")
        
        # 1. 误差随帧变化的趋势图
        plt.figure(figsize=(15, 10))
        
        # 准备数据
        frames = []
        mean_errors = []
        max_errors = []
        rmse_errors = []
        
        for frame_idx in sorted(frame_idx_list):
            if frame_idx in self.results['frame_results']:
                result = self.results['frame_results'][frame_idx]
                frames.append(frame_idx)
                mean_errors.append(result['mean_error'])
                max_errors.append(result['max_error'])
                rmse_errors.append(result['rmse'])
        
        # 子图1: 误差趋势
        plt.subplot(2, 3, 1)
        plt.plot(frames, mean_errors, 'b-o', label='Mean Error', markersize=4)
        plt.plot(frames, rmse_errors, 'r-s', label='RMSE', markersize=4)
        plt.xlabel('Frame Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction Error vs Frame')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 最大误差
        plt.subplot(2, 3, 2)
        plt.plot(frames, max_errors, 'g-^', label='Max Error', markersize=4)
        plt.xlabel('Frame Index')
        plt.ylabel('Max Error')
        plt.title('Maximum Error vs Frame')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图3: 误差分布直方图
        plt.subplot(2, 3, 3)
        all_errors = []
        for frame_idx in frame_idx_list:
            if frame_idx in self.results['frame_results']:
                all_errors.extend(self.results['frame_results'][frame_idx]['vertex_errors'])
        
        plt.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Vertex Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution (All Vertices)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 子图4: 误差散点图（帧vs平均误差）
        plt.subplot(2, 3, 4)
        plt.scatter(frames, mean_errors, alpha=0.6, c=mean_errors, cmap='viridis')
        plt.colorbar(label='Mean Error')
        plt.xlabel('Frame Index')
        plt.ylabel('Mean Error')
        plt.title('Error Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        # 子图5: 性能分析
        plt.subplot(2, 3, 5)
        lbs_times = [self.results['frame_results'][f]['lbs_time'] for f in frames]
        plt.plot(frames, lbs_times, 'purple', marker='o', markersize=4)
        plt.xlabel('Frame Index')
        plt.ylabel('LBS Time (seconds)')
        plt.title('Performance vs Frame')
        plt.grid(True, alpha=0.3)
        
        # 子图6: 误差箱型图（按帧分组）
        plt.subplot(2, 3, 6)
        if len(frames) <= 10:  # 只有当帧数不太多时才画箱型图
            error_data = []
            frame_labels = []
            for frame_idx in frames:
                if frame_idx in self.results['frame_results']:
                    error_data.append(self.results['frame_results'][frame_idx]['vertex_errors'])
                    frame_labels.append(str(frame_idx))
            
            plt.boxplot(error_data, labels=frame_labels)
            plt.xlabel('Frame Index')
            plt.ylabel('Vertex Error')
            plt.title('Error Distribution by Frame')
            plt.xticks(rotation=45)
        else:
            # 太多帧时，显示误差统计
            p25_errors = [np.percentile(self.results['frame_results'][f]['vertex_errors'], 25) for f in frames]
            p75_errors = [np.percentile(self.results['frame_results'][f]['vertex_errors'], 75) for f in frames]
            p95_errors = [np.percentile(self.results['frame_results'][f]['vertex_errors'], 95) for f in frames]
            
            plt.plot(frames, p25_errors, label='25th percentile', alpha=0.7)
            plt.plot(frames, p75_errors, label='75th percentile', alpha=0.7)
            plt.plot(frames, p95_errors, label='95th percentile', alpha=0.7)
            plt.xlabel('Frame Index')
            plt.ylabel('Error Percentiles')
            plt.title('Error Percentiles vs Frame')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'lbs_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 图表已保存: {output_path / 'lbs_validation_analysis.png'}")
        
        # 2. 生成详细的误差分析图
        self._generate_detailed_error_analysis(output_path, frame_idx_list)
    
    def _generate_detailed_error_analysis(self, output_path, frame_idx_list):
        """生成详细的误差分析图"""
        plt.figure(figsize=(12, 8))
        
        # 收集所有误差数据
        all_frame_errors = {}
        for frame_idx in frame_idx_list:
            if frame_idx in self.results['frame_results']:
                all_frame_errors[frame_idx] = self.results['frame_results'][frame_idx]['vertex_errors']
        
        if not all_frame_errors:
            return
        
        # 子图1: 误差热力图（如果帧数合适）
        plt.subplot(2, 2, 1)
        if len(all_frame_errors) <= 20:
            error_matrix = []
            frame_indices = sorted(all_frame_errors.keys())
            
            # 计算每帧的误差分位数
            for frame_idx in frame_indices:
                errors = all_frame_errors[frame_idx]
                percentiles = [np.percentile(errors, p) for p in [10, 25, 50, 75, 90, 95, 99]]
                error_matrix.append(percentiles)
            
            error_matrix = np.array(error_matrix).T
            im = plt.imshow(error_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(im, label='Error Value')
            plt.yticks(range(len(['10%', '25%', '50%', '75%', '90%', '95%', '99%'])), 
                      ['10%', '25%', '50%', '75%', '90%', '95%', '99%'])
            plt.xticks(range(len(frame_indices)), [str(f) for f in frame_indices], rotation=45)
            plt.xlabel('Frame Index')
            plt.ylabel('Error Percentile')
            plt.title('Error Heatmap by Frame')
        else:
            # 太多帧时显示统计信息
            plt.text(0.5, 0.5, f'Too many frames ({len(all_frame_errors)}) for heatmap\nUse frame range analysis instead', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Error Heatmap (Too Many Frames)')
        
        # 子图2: 累积误差分布
        plt.subplot(2, 2, 2)
        all_errors_combined = []
        for errors in all_frame_errors.values():
            all_errors_combined.extend(errors)
        
        sorted_errors = np.sort(all_errors_combined)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative)
        plt.xlabel('Error Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 误差与距离参考帧的关系
        plt.subplot(2, 2, 3)
        distances_from_ref = []
        frame_mean_errors = []
        
        for frame_idx in sorted(all_frame_errors.keys()):
            distance = abs(frame_idx - self.reference_frame_idx)
            distances_from_ref.append(distance)
            frame_mean_errors.append(np.mean(all_frame_errors[frame_idx]))
        
        plt.scatter(distances_from_ref, frame_mean_errors, alpha=0.6)
        plt.xlabel('Distance from Reference Frame')
        plt.ylabel('Mean Reconstruction Error')
        plt.title('Error vs Distance from Reference')
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(distances_from_ref) > 1:
            z = np.polyfit(distances_from_ref, frame_mean_errors, 1)
            p = np.poly1d(z)
            plt.plot(distances_from_ref, p(distances_from_ref), "r--", alpha=0.8, label=f'Trend: y={z[0]:.6f}x+{z[1]:.6f}')
            plt.legend()
        
        # 子图4: 误差方差分析
        plt.subplot(2, 2, 4)
        frame_std_errors = [np.std(all_frame_errors[f]) for f in sorted(all_frame_errors.keys())]
        frame_indices = sorted(all_frame_errors.keys())
        
        plt.plot(frame_indices, frame_std_errors, 'o-', color='orange')
        plt.xlabel('Frame Index')
        plt.ylabel('Error Standard Deviation')
        plt.title('Error Variance by Frame')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'lbs_detailed_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 详细分析图已保存: {output_path / 'lbs_detailed_error_analysis.png'}")
    
    def save_results(self, output_path="output/lbs_validation/validation_results.json"):
        """
        保存测试结果到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        if not self.results:
            print("没有可保存的结果")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备可序列化的结果
        serializable_results = {
            'summary': self.results.get('summary', {}),
            'test_params': self.results.get('test_params', {}),
            'frame_results': {}
        }
        
        # 转换帧结果（移除numpy数组）
        for frame_idx, frame_result in self.results.get('frame_results', {}).items():
            serializable_frame_result = {
                'frame_idx': frame_result['frame_idx'],
                'mean_error': float(frame_result['mean_error']),
                'max_error': float(frame_result['max_error']),
                'min_error': float(frame_result['min_error']),
                'std_error': float(frame_result['std_error']),
                'median_error': float(frame_result['median_error']),
                'rmse': float(frame_result['rmse']),
                'p95_error': float(frame_result['p95_error']),
                'p99_error': float(frame_result['p99_error']),
                'lbs_time': float(frame_result['lbs_time']),
                'num_vertices': int(frame_result['num_vertices'])
            }
            serializable_results['frame_results'][str(frame_idx)] = serializable_frame_result
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存: {output_path}")
    
    def export_mesh_comparison(self, frame_idx, output_dir="output/lbs_validation/meshes"):
        """
        导出特定帧的网格比较（原始vs重建）
        
        Args:
            frame_idx: 帧索引
            output_dir: 输出目录
        """
        if frame_idx not in self.results.get('frame_results', {}):
            print(f"帧 {frame_idx} 未测试，请先运行测试")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_result = self.results['frame_results'][frame_idx]
        
        # 创建原始网格
        original_mesh = o3d.io.read_triangle_mesh(str(self.canonicalizer.mesh_files[frame_idx]))
        
        # 创建重建网格
        reconstructed_mesh = o3d.geometry.TriangleMesh()
        reconstructed_mesh.vertices = o3d.utility.Vector3dVector(frame_result['predicted_vertices'])
        if hasattr(original_mesh, 'triangles') and len(original_mesh.triangles) > 0:
            reconstructed_mesh.triangles = original_mesh.triangles
        
        # 保存网格
        o3d.io.write_triangle_mesh(str(output_path / f"frame_{frame_idx:06d}_original.obj"), original_mesh)
        o3d.io.write_triangle_mesh(str(output_path / f"frame_{frame_idx:06d}_reconstructed.obj"), reconstructed_mesh)
        
        # 创建误差颜色编码的网格
        vertex_errors = frame_result['vertex_errors']
        error_colors = plt.cm.viridis(vertex_errors / np.max(vertex_errors))[:, :3]
        
        error_mesh = o3d.geometry.TriangleMesh()
        error_mesh.vertices = original_mesh.vertices
        error_mesh.triangles = original_mesh.triangles
        error_mesh.vertex_colors = o3d.utility.Vector3dVector(error_colors)
        
        o3d.io.write_triangle_mesh(str(output_path / f"frame_{frame_idx:06d}_error_colored.obj"), error_mesh)
        
        print(f"✅ 网格比较已导出到: {output_path}")
        print(f"   - frame_{frame_idx:06d}_original.obj")
        print(f"   - frame_{frame_idx:06d}_reconstructed.obj") 
        print(f"   - frame_{frame_idx:06d}_error_colored.obj")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LBS权重验证测试')
    parser.add_argument('--skeleton_dir', default='output/skeleton_prediction', 
                       help='骨骼数据目录')
    parser.add_argument('--mesh_dir', default='D:/Code/VVEditor/Rafa_Approves_hd_4k',
                       help='网格文件目录')
    parser.add_argument('--weights_path', default='output/skinning_weights_fast.npz',
                       help='权重文件路径')
    parser.add_argument('--reference_frame', type=int, default=5,
                       help='参考帧索引')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='测试起始帧')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='测试结束帧')
    parser.add_argument('--step', type=int, default=1,
                       help='测试步长')
    parser.add_argument('--output_dir', default='output/lbs_validation',
                       help='输出目录')
    parser.add_argument('--export_meshes', nargs='*', type=int,
                       help='导出特定帧的网格比较')
    
    args = parser.parse_args()
    
    print("🔍 LBS权重验证测试")
    print("=" * 60)
    
    # 初始化验证器
    try:
        validator = LBSValidator(
            skeleton_data_dir=args.skeleton_dir,
            mesh_folder_path=args.mesh_dir,
            weights_path=args.weights_path,
            reference_frame_idx=args.reference_frame
        )
        print(f"✅ 验证器初始化成功")
        print(f"   权重文件: {args.weights_path}")
        print(f"   参考帧: {args.reference_frame}")
    except Exception as e:
        print(f"❌ 验证器初始化失败: {e}")
        return
    
    # 运行测试
    print(f"\n🧪 开始测试...")
    try:
        results = validator.test_frame_range(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            verbose=True
        )
        
        # 分析误差分布
        validator.analyze_error_distribution()
        
        # 生成可视化
        print(f"\n📊 生成可视化图表...")
        validator.generate_visualization_plots(output_dir=args.output_dir)
        
        # 保存结果
        validator.save_results(f"{args.output_dir}/validation_results.json")
        
        # 导出特定帧的网格
        if args.export_meshes is not None:
            print(f"\n💾 导出网格比较...")
            for frame_idx in args.export_meshes:
                if frame_idx in results['frame_results']:
                    validator.export_mesh_comparison(frame_idx, f"{args.output_dir}/meshes")
                else:
                    print(f"⚠️  帧 {frame_idx} 未在测试范围内")
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 验证测试完成！")
    print(f"📁 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
