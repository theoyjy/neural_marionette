#!/usr/bin/env python3
"""
LBS重建网格可视化工具

可视化原始mesh与LBS重建mesh的对比，支持多种显示模式
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from Skinning import InverseMeshCanonicalizer

class LBSMeshVisualizer:
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path, reference_frame_idx=5):
        """
        初始化可视化器
        
        Args:
            skeleton_data_dir: 骨骼数据目录
            mesh_folder_path: 网格文件目录
            weights_path: 权重文件路径
            reference_frame_idx: 参考帧索引
        """
        self.skeleton_data_dir = skeleton_data_dir
        self.mesh_folder_path = mesh_folder_path
        self.weights_path = weights_path
        self.reference_frame_idx = reference_frame_idx
        
        # 初始化canonicalizer
        print("🔧 初始化LBS系统...")
        self.canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=reference_frame_idx
        )
        
        # 加载网格序列和权重
        self.canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        if not self.canonicalizer.load_skinning_weights(weights_path):
            raise ValueError(f"无法加载权重文件: {weights_path}")
        
        print("✅ 初始化完成")
    
    def reconstruct_frame(self, frame_idx):
        """
        重建指定帧的mesh
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            original_mesh, reconstructed_mesh, vertex_errors
        """
        if frame_idx >= len(self.canonicalizer.mesh_files):
            raise ValueError(f"帧索引 {frame_idx} 超出范围")
        
        # 加载原始mesh
        original_mesh = o3d.io.read_triangle_mesh(str(self.canonicalizer.mesh_files[frame_idx]))
        original_vertices = np.asarray(original_mesh.vertices)
        
        # 获取rest pose顶点
        rest_vertices = self.canonicalizer.rest_pose_vertices
        
        # 归一化处理
        if frame_idx not in self.canonicalizer.frame_normalization_params:
            self.canonicalizer.frame_normalization_params[frame_idx] = \
                self.canonicalizer.compute_mesh_normalization_params(original_mesh)
        
        target_vertices_norm = self.canonicalizer.normalize_mesh_vertices(
            original_vertices, 
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
        
        # LBS重建
        predicted_vertices_norm = self.canonicalizer.apply_lbs_transform(
            rest_vertices_norm, 
            self.canonicalizer.skinning_weights, 
            relative_transforms
        )
        
        # 计算误差
        vertex_errors = np.linalg.norm(predicted_vertices_norm - target_vertices_norm, axis=1)
        
        # 创建重建mesh（使用归一化后的顶点进行显示）
        reconstructed_mesh = o3d.geometry.TriangleMesh()
        reconstructed_mesh.vertices = o3d.utility.Vector3dVector(predicted_vertices_norm)
        
        # 复制面信息
        if hasattr(original_mesh, 'triangles') and len(original_mesh.triangles) > 0:
            reconstructed_mesh.triangles = original_mesh.triangles
        
        # 为了对比，也将原始mesh转换到归一化空间
        original_mesh_normalized = o3d.geometry.TriangleMesh()
        original_mesh_normalized.vertices = o3d.utility.Vector3dVector(target_vertices_norm)
        original_mesh_normalized.triangles = original_mesh.triangles
        
        return original_mesh_normalized, reconstructed_mesh, vertex_errors
    
    def create_error_colored_mesh(self, mesh, vertex_errors, colormap='plasma'):
        """
        创建误差颜色编码的mesh
        
        Args:
            mesh: 网格对象
            vertex_errors: 顶点误差数组
            colormap: 颜色映射名称
            
        Returns:
            error_colored_mesh: 带颜色的网格
        """
        # 归一化误差到0-1范围
        if np.max(vertex_errors) > 0:
            normalized_errors = vertex_errors / np.max(vertex_errors)
        else:
            normalized_errors = vertex_errors
        
        # 应用颜色映射
        if colormap == 'plasma':
            colors = plt.cm.plasma(normalized_errors)[:, :3]
        elif colormap == 'viridis':
            colors = plt.cm.viridis(normalized_errors)[:, :3]
        elif colormap == 'hot':
            colors = plt.cm.hot(normalized_errors)[:, :3]
        elif colormap == 'jet':
            colors = plt.cm.jet(normalized_errors)[:, :3]
        else:
            colors = plt.cm.plasma(normalized_errors)[:, :3]
        
        # 创建带颜色的mesh
        error_mesh = o3d.geometry.TriangleMesh()
        error_mesh.vertices = mesh.vertices
        error_mesh.triangles = mesh.triangles
        error_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        return error_mesh
    
    def visualize_single_frame(self, frame_idx, mode='side_by_side', colormap='plasma', save_path=None):
        """
        可视化单帧对比
        
        Args:
            frame_idx: 帧索引
            mode: 显示模式 ('side_by_side', 'overlay', 'error_only')
            colormap: 误差颜色映射
            save_path: 保存路径（可选）
        """
        print(f"🔍 重建帧 {frame_idx}...")
        original_mesh, reconstructed_mesh, vertex_errors = self.reconstruct_frame(frame_idx)
        
        # 计算误差统计
        mean_error = np.mean(vertex_errors)
        max_error = np.max(vertex_errors)
        rmse = np.sqrt(np.mean(vertex_errors**2))
        
        print(f"📊 重建质量统计:")
        print(f"   平均误差: {mean_error:.6f}")
        print(f"   最大误差: {max_error:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        
        # 准备可视化
        vis = o3d.visualization.Visualizer()
        vis.create_window(f"LBS重建对比 - 帧 {frame_idx}", width=1200, height=800)
        
        if mode == 'side_by_side':
            # 并排显示原始和重建mesh
            # 设置原始mesh颜色（蓝色）
            original_mesh.paint_uniform_color([0.1, 0.1, 0.9])  # 蓝色
            
            # 设置重建mesh颜色（红色）
            reconstructed_mesh.paint_uniform_color([0.9, 0.1, 0.1])  # 红色
            
            # 将重建mesh平移一点距离以便并排显示
            bbox = original_mesh.get_axis_aligned_bounding_box()
            width = bbox.max_bound[0] - bbox.min_bound[0]
            
            # 平移重建mesh
            reconstructed_mesh.translate([width * 1.2, 0, 0])
            
            # 添加到可视化
            vis.add_geometry(original_mesh)
            vis.add_geometry(reconstructed_mesh)
            
            print("  显示模式: 并排对比 (蓝色=原始, 红色=重建)")
            
        elif mode == 'overlay':
            # 叠加显示，原始mesh半透明
            original_mesh.paint_uniform_color([0.1, 0.1, 0.9])  # 蓝色
            reconstructed_mesh.paint_uniform_color([0.9, 0.1, 0.1])  # 红色
            
            vis.add_geometry(original_mesh)
            vis.add_geometry(reconstructed_mesh)
            
            print("  显示模式: 叠加对比 (蓝色=原始, 红色=重建)")
            
        elif mode == 'error_only':
            # 只显示误差颜色编码的mesh
            error_mesh = self.create_error_colored_mesh(original_mesh, vertex_errors, colormap)
            vis.add_geometry(error_mesh)
            
            print(f"  显示模式: 误差可视化 (颜色映射: {colormap})")
            print(f"   颜色含义: 深色=低误差, 亮色=高误差")
            
        elif mode == 'triple':
            # 三个mesh并排：原始、重建、误差
            bbox = original_mesh.get_axis_aligned_bounding_box()
            width = bbox.max_bound[0] - bbox.min_bound[0]
            
            # 原始mesh（蓝色）
            original_mesh.paint_uniform_color([0.1, 0.1, 0.9])
            vis.add_geometry(original_mesh)
            
            # 重建mesh（红色）
            reconstructed_mesh.paint_uniform_color([0.9, 0.1, 0.1])
            reconstructed_mesh.translate([width * 1.2, 0, 0])
            vis.add_geometry(reconstructed_mesh)
            
            # 误差mesh（颜色编码）
            error_mesh = self.create_error_colored_mesh(original_mesh, vertex_errors, colormap)
            error_mesh.translate([width * 2.4, 0, 0])
            vis.add_geometry(error_mesh)
            
            print("  显示模式: 三重对比 (蓝色=原始, 红色=重建, 彩色=误差)")
        
        # 设置相机参数以便更好地观察
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.8)
        
        # 运行可视化
        print("🖥️  可视化窗口已打开，按Q退出")
        print("💡 提示: 可以用鼠标旋转、缩放视图")
        
        vis.run()
        
        # 保存截图（可选）
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(str(save_path))
            print(f"📷 截图已保存: {save_path}")
        
        vis.destroy_window()
    
    def visualize_multiple_frames(self, frame_list, mode='error_only', colormap='plasma', save_dir=None):
        """
        可视化多个帧的重建结果
        
        Args:
            frame_list: 帧索引列表
            mode: 显示模式
            colormap: 颜色映射
            save_dir: 保存目录
        """
        print(f"🔍 批量可视化 {len(frame_list)} 帧...")
        
        for i, frame_idx in enumerate(frame_list):
            print(f"\n--- 帧 {frame_idx} ({i+1}/{len(frame_list)}) ---")
            
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"frame_{frame_idx:06d}_comparison.png"
            
            try:
                self.visualize_single_frame(frame_idx, mode=mode, colormap=colormap, save_path=save_path)
            except Exception as e:
                print(f"❌ 可视化帧 {frame_idx} 失败: {e}")
                continue
        
        print(f"\n✅ 批量可视化完成")
    
    def create_error_heatmap(self, frame_idx, save_path=None):
        """
        创建误差热力图
        
        Args:
            frame_idx: 帧索引
            save_path: 保存路径
        """
        print(f"📊 创建帧 {frame_idx} 的误差热力图...")
        
        original_mesh, reconstructed_mesh, vertex_errors = self.reconstruct_frame(frame_idx)
        
        # 创建热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 误差直方图
        ax1.hist(vertex_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Vertex Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Frame {frame_idx} - Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 误差统计信息
        stats_text = f"""Statistics:
Mean: {np.mean(vertex_errors):.6f}
Std:  {np.std(vertex_errors):.6f}
Max:  {np.max(vertex_errors):.6f}
RMSE: {np.sqrt(np.mean(vertex_errors**2)):.6f}
P95:  {np.percentile(vertex_errors, 95):.6f}"""
        
        ax1.text(0.65, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 累积分布函数
        sorted_errors = np.sort(vertex_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax2.plot(sorted_errors, cumulative, 'b-', linewidth=2)
        ax2.set_xlabel('Vertex Error')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f'Frame {frame_idx} - Cumulative Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 添加关键百分位线
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            error_p = np.percentile(vertex_errors, p)
            ax2.axvline(error_p, color='red', linestyle='--', alpha=0.7, 
                       label=f'P{p}: {error_p:.6f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 热力图已保存: {save_path}")
        
        plt.show()
    
    def export_meshes(self, frame_idx, output_dir="output/mesh_comparison"):
        """
        导出mesh文件用于外部软件查看
        
        Args:
            frame_idx: 帧索引
            output_dir: 输出目录
        """
        print(f"💾 导出帧 {frame_idx} 的mesh文件...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        original_mesh, reconstructed_mesh, vertex_errors = self.reconstruct_frame(frame_idx)
        
        # 导出原始mesh
        original_path = output_path / f"frame_{frame_idx:06d}_original.obj"
        o3d.io.write_triangle_mesh(str(original_path), original_mesh)
        
        # 导出重建mesh
        reconstructed_path = output_path / f"frame_{frame_idx:06d}_reconstructed.obj"
        o3d.io.write_triangle_mesh(str(reconstructed_path), reconstructed_mesh)
        
        # 导出误差颜色编码mesh
        error_mesh = self.create_error_colored_mesh(original_mesh, vertex_errors)
        error_path = output_path / f"frame_{frame_idx:06d}_error_colored.obj"
        o3d.io.write_triangle_mesh(str(error_path), error_mesh)
        
        # 保存误差数据
        error_data_path = output_path / f"frame_{frame_idx:06d}_errors.npy"
        np.save(error_data_path, vertex_errors)
        
        print(f"✅ 文件已导出到: {output_path}")
        print(f"   - {original_path.name} (原始)")
        print(f"   - {reconstructed_path.name} (重建)")
        print(f"   - {error_path.name} (误差可视化)")
        print(f"   - {error_data_path.name} (误差数据)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LBS重建网格可视化工具')
    parser.add_argument('--skeleton_dir', default='output/skeleton_prediction', 
                       help='骨骼数据目录')
    parser.add_argument('--mesh_dir', default='D:/Code/VVEditor/Rafa_Approves_hd_4k',
                       help='网格文件目录')
    parser.add_argument('--weights_path', default='output/skinning_weights_fast.npz',
                       help='权重文件路径')
    parser.add_argument('--reference_frame', type=int, default=5,
                       help='参考帧索引')
    parser.add_argument('--frame', type=int, default=10,
                       help='要可视化的帧索引')
    parser.add_argument('--mode', default='triple',
                       choices=['side_by_side', 'overlay', 'error_only', 'triple'],
                       help='显示模式')
    parser.add_argument('--colormap', default='plasma',
                       choices=['plasma', 'viridis', 'hot', 'jet'],
                       help='误差颜色映射')
    parser.add_argument('--export', action='store_true',
                       help='导出mesh文件')
    parser.add_argument('--heatmap', action='store_true',
                       help='显示误差热力图')
    parser.add_argument('--frames', nargs='*', type=int,
                       help='批量可视化多个帧')
    parser.add_argument('--output_dir', default='output/visualization',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("  LBS重建网格可视化工具")
    print("=" * 50)
    
    try:
        # 初始化可视化器
        visualizer = LBSMeshVisualizer(
            skeleton_data_dir=args.skeleton_dir,
            mesh_folder_path=args.mesh_dir,
            weights_path=args.weights_path,
            reference_frame_idx=args.reference_frame
        )
        
        # 根据参数执行不同操作
        if args.frames:
            # 批量可视化
            visualizer.visualize_multiple_frames(
                frame_list=args.frames,
                mode=args.mode,
                colormap=args.colormap,
                save_dir=args.output_dir if args.export else None
            )
        else:
            # 单帧可视化
            frame_idx = args.frame
            
            if args.heatmap:
                visualizer.create_error_heatmap(
                    frame_idx, 
                    save_path=f"{args.output_dir}/frame_{frame_idx:06d}_heatmap.png"
                )
            
            if args.export:
                visualizer.export_meshes(frame_idx, args.output_dir)
            
            # 主要可视化
            visualizer.visualize_single_frame(
                frame_idx,
                mode=args.mode,
                colormap=args.colormap,
                save_path=f"{args.output_dir}/frame_{frame_idx:06d}_comparison.png" if args.export else None
            )
    
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
