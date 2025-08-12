#!/usr/bin/env python3
"""
3D文件可视化脚本
加载OBJ和FBX文件并生成三种可视化：点云、网格、体素网格
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
import argparse
import os
from pathlib import Path

def load_3d_file(file_path):
    """
    加载3D文件（支持OBJ和FBX）
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        trimesh.Trimesh: 加载的网格对象
    """
    print(f"Loading 3D file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"3D file not found: {file_path}")
    
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.fbx':
        print("Detected FBX file, loading with trimesh...")
        # FBX文件需要额外的依赖，如果加载失败会给出提示
        try:
            mesh = trimesh.load(file_path)
        except Exception as e:
            print(f"Error loading FBX file: {e}")
            print("Note: FBX support may require additional dependencies.")
            print("Try installing: pip install trimesh[easy]")
            raise
    elif file_extension == '.obj':
        print("Detected OBJ file, loading with trimesh...")
        mesh = trimesh.load(file_path)
    else:
        print(f"Attempting to load {file_extension} file with trimesh...")
        # 尝试用trimesh加载其他格式
        mesh = trimesh.load(file_path)
    
    if isinstance(mesh, trimesh.Scene):
        # 如果是场景，提取第一个网格
        if len(mesh.geometry) == 0:
            raise ValueError("No geometry found in the 3D file scene")
        
        # 获取第一个几何体或合并所有几何体
        geometries = list(mesh.geometry.values())
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            print(f"Found {len(geometries)} geometries, combining them...")
            mesh = mesh.dump(concatenate=True)
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    return mesh

def orient_human_model(mesh):
    """
    调整人体模型方向，让其站立
    智能检测模型当前方向并进行适当的调整
    
    Args:
        mesh (trimesh.Trimesh): 网格对象
        
    Returns:
        trimesh.Trimesh: 调整方向后的网格对象
    """
    mesh_copy = mesh.copy()
    bounds = mesh_copy.bounds
    
    # 计算各轴的尺寸
    x_size = bounds[1, 0] - bounds[0, 0]
    y_size = bounds[1, 1] - bounds[0, 1] 
    z_size = bounds[1, 2] - bounds[0, 2]
    
    print(f"Original model dimensions - X: {x_size:.3f}, Y: {y_size:.3f}, Z: {z_size:.3f}")
    
    # 如果Z轴是最大的，模型可能已经是站立的
    if z_size > max(x_size, y_size):
        print("Model appears to be already upright (Z is the largest dimension)")
    # 如果Y轴是最大的，可能需要绕X轴旋转90度
    elif y_size > max(x_size, z_size):
        print("Model appears to be lying on XZ plane, rotating around X-axis")
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        mesh_copy.apply_transform(rotation_matrix)
    # 如果X轴是最大的，可能需要绕Z轴旋转90度
    elif x_size > max(y_size, z_size):
        print("Model appears to be lying on YZ plane, rotating around Z-axis")
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1])
        mesh_copy.apply_transform(rotation_matrix)
    
    # 将模型移到地面上（最低点设为0）
    bounds = mesh_copy.bounds
    z_offset = -bounds[0, 2]  # 最低点的Z坐标的负值
    translation = [0, 0, z_offset]
    mesh_copy.apply_translation(translation)
    
    print(f"Model adjusted, moved up by Z offset: {z_offset:.3f}")
    
    return mesh_copy

def sample_vertices(vertices, max_points=10000):
    """
    对顶点进行采样以提高性能
    
    Args:
        vertices (np.ndarray): 原始顶点数组
        max_points (int): 最大点数
        
    Returns:
        np.ndarray: 采样后的顶点数组
    """
    if len(vertices) <= max_points:
        return vertices
    
    # 随机采样
    indices = np.random.choice(len(vertices), max_points, replace=False)
    sampled_vertices = vertices[indices]
    
    print(f"Sampled {len(sampled_vertices)} points from {len(vertices)} total vertices")
    return sampled_vertices

def visualize_point_cloud(mesh, output_dir="output", filename_prefix="obj_vis", max_points=10000):
    """
    生成点云可视化
    
    Args:
        mesh (trimesh.Trimesh): 网格对象
        output_dir (str): 输出目录
        filename_prefix (str): 文件名前缀
        max_points (int): 最大点数，用于采样
    """
    print("Generating point cloud visualization...")
    
    # 调整人体模型方向
    mesh_oriented = orient_human_model(mesh)
    
    # 从网格顶点创建点云并采样
    vertices = mesh_oriented.vertices
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云 - 使用灰色
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c='gray', s=1, alpha=0.6)
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # 隐藏坐标轴框架
    ax.grid(False)
    ax.axis('off')
    
    # 设置标题在图的下方
    ax.text2D(0.5, 0.1, 'Point Cloud', transform=ax.transAxes, 
              ha='center', va='top', fontsize=14)
    
    # 设置相等的轴比例 (使用原始顶点来确保正确的边界)
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{filename_prefix}_point_cloud.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Point cloud visualization saved to: {output_path}")
    
    plt.show()
    plt.close()

def sample_mesh_faces(vertices, faces, max_faces=20000):
    """
    对网格面进行采样以提高性能
    
    Args:
        vertices (np.ndarray): 顶点数组
        faces (np.ndarray): 面数组
        max_faces (int): 最大面数
        
    Returns:
        tuple: (采样后的顶点, 采样后的面, 顶点映射)
    """
    if len(faces) <= max_faces:
        return vertices, faces
    
    # 随机采样面
    face_indices = np.random.choice(len(faces), max_faces, replace=False)
    sampled_faces = faces[face_indices]
    
    # 找到用到的顶点
    used_vertices = np.unique(sampled_faces.flatten())
    
    # 重新映射顶点索引
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    remapped_faces = np.array([[vertex_map[v] for v in face] for face in sampled_faces])
    sampled_vertices = vertices[used_vertices]
    
    print(f"Sampled {len(sampled_faces)} faces from {len(faces)} total faces")
    print(f"Using {len(sampled_vertices)} vertices from {len(vertices)} total vertices")
    
    return sampled_vertices, remapped_faces

def visualize_mesh(mesh, output_dir="output", filename_prefix="obj_vis", max_faces=20000):
    """
    生成网格可视化
    
    Args:
        mesh (trimesh.Trimesh): 网格对象
        output_dir (str): 输出目录
        filename_prefix (str): 文件名前缀
        max_faces (int): 最大面数，用于采样
    """
    print("Generating mesh visualization...")
    
    # 调整人体模型方向
    mesh_oriented = orient_human_model(mesh)
    
    # 获取顶点和面并采样
    vertices = mesh_oriented.vertices
    faces = mesh_oriented.faces
    vertices_sampled, faces_sampled = sample_mesh_faces(vertices, faces, max_faces)
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制网格 - 使用灰色
    ax.plot_trisurf(vertices_sampled[:, 0], vertices_sampled[:, 1], vertices_sampled[:, 2], 
                   triangles=faces_sampled, alpha=0.7, color='lightgray', 
                   linewidth=0.1, edgecolor='black')
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # 隐藏坐标轴框架
    ax.grid(False)
    ax.axis('off')
    
    ax.text2D(0.5, 0.1, 'Mesh', transform=ax.transAxes, 
              ha='center', va='top', fontsize=14)
    
    # 设置相等的轴比例
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{filename_prefix}_mesh.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Mesh visualization saved to: {output_path}")
    
    plt.show()
    plt.close()

def visualize_voxel_grid(mesh, voxel_size=0.3, output_dir="output", filename_prefix="obj_vis"):
    """
    生成体素网格可视化
    
    Args:
        mesh (trimesh.Trimesh): 网格对象
        voxel_size (float): 体素大小
        output_dir (str): 输出目录
        filename_prefix (str): 文件名前缀
    """
    print("Generating voxel grid visualization...")
    
    # 调整人体模型方向
    mesh_oriented = orient_human_model(mesh)
    
    # 将trimesh转换为open3d mesh
    vertices = mesh_oriented.vertices
    faces = mesh_oriented.faces
    
    # 创建Open3D网格
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # 体素化
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size)
    
    # 获取体素中心点
    voxels = voxel_grid.get_voxels()
    voxel_centers = []
    
    for voxel in voxels:
        center = voxel_grid.origin + (voxel.grid_index + 0.5) * voxel_grid.voxel_size
        voxel_centers.append(center)
    
    if len(voxel_centers) == 0:
        print("Warning: No voxels generated. Try reducing voxel_size.")
        return
    
    voxel_centers = np.array(voxel_centers)
    print(f"Generated {len(voxel_centers)} voxels total")
    
    # 直接使用所有生成的voxel，不进行采样
    voxel_centers_to_draw = voxel_centers
    
    # 创建3D图形 - 调整图形大小
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Creating voxel grid visualization with all {len(voxel_centers_to_draw)} voxels...")
    
    # 计算网格范围
    min_coords = voxel_centers_to_draw.min(axis=0)
    max_coords = voxel_centers_to_draw.max(axis=0)
    
    # 创建网格索引
    grid_size = ((max_coords - min_coords) / voxel_size + 1).astype(int)
    
    # 限制网格大小以避免内存问题
    max_grid_size = 50  # 减小网格大小
    if any(grid_size > max_grid_size):
        print(f"Grid size {grid_size} too large, using simplified bar visualization...")
        
        # 使用简化的scatter plot而不是bar3d来提高性能
        x_coords = voxel_centers_to_draw[:, 0]
        y_coords = voxel_centers_to_draw[:, 1]
        z_coords = voxel_centers_to_draw[:, 2]
        
        # 绘制为小一点的点来模拟体素 - 使用灰色
        ax.scatter(x_coords, y_coords, z_coords, 
                  c='gray', s=20, alpha=0.8, 
                  marker='s', edgecolors='black', linewidth=0.3)
    
    else:
        # 使用voxels功能
        # 创建布尔数组
        voxel_array = np.zeros(grid_size, dtype=bool)
        
        for center in voxel_centers_to_draw:
            # 计算网格索引
            grid_idx = ((center - min_coords) / voxel_size).astype(int)
            # 确保索引在范围内
            grid_idx = np.clip(grid_idx, 0, grid_size - 1)
            
            # 确保grid_idx是整数元组
            grid_idx_tuple = tuple(grid_idx.astype(int))
            
            voxel_array[grid_idx_tuple] = True
        
        # 绘制体素 - 使用灰色
        ax.voxels(voxel_array, facecolors='lightgray', alpha=0.7, edgecolors='black', linewidth=0.1)
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # 隐藏坐标轴框架
    ax.grid(False)
    ax.axis('off')
    
    ax.text2D(0.5, 0, 'Voxel Grid', transform=ax.transAxes, 
            ha='center', va='top', fontsize=14)
    
    # 设置相等的轴比例和更好的视图
    # 计算模型的实际边界
    model_min = voxel_centers_to_draw.min(axis=0)
    model_max = voxel_centers_to_draw.max(axis=0)
    model_center = (model_min + model_max) / 2
    model_size = (model_max - model_min).max()
    
    # 设置合适的视图边界 - 进一步增加视图范围
    margin = model_size * 1.5  # 增加150%的边距
    view_range = model_size * 3.5  # 进一步扩大视图范围
    ax.set_xlim(model_center[0] - view_range, model_center[0] + view_range)
    ax.set_ylim(model_center[1] - view_range, model_center[1] + view_range)
    ax.set_zlim(model_center[2] - view_range, model_center[2] + view_range)
    
    # 设置更好的视角 - 稍微远一点的角度
    ax.view_init(elev=25, azim=45)
    
    # 设置图形的距离和比例 - 进一步增加相机距离
    ax.dist = 20  # 进一步增加相机距离，让模型看起来更小
    
    # 设置背景色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{filename_prefix}_voxel_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Voxel grid visualization saved to: {output_path}")
    print(f"Generated {len(voxel_centers)} voxels")
    
    plt.show()
    plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3D文件可视化工具 (支持OBJ和FBX)')
    parser.add_argument('file_path', type=str, help='3D文件路径 (支持.obj, .fbx等格式)')
    parser.add_argument('--output_dir', type=str, default='output/visualizations', 
                       help='输出目录 (默认: output/visualizations)')
    parser.add_argument('--voxel_size', type=float, default=0.15, 
                       help='体素大小 (默认: 0.15)')
    parser.add_argument('--prefix', type=str, default=None, 
                       help='输出文件名前缀 (默认: 使用文件名)')
    parser.add_argument('--max_points', type=int, default=10000,
                       help='点云可视化的最大点数 (默认: 10000)')
    parser.add_argument('--max_faces', type=int, default=20000,
                       help='网格可视化的最大面数 (默认: 20000)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置文件名前缀
    if args.prefix is None:
        args.prefix = Path(args.file_path).stem
    
    try:
        # 加载3D文件
        mesh = load_3d_file(args.file_path)
        
        print(f"Mesh bounds: {mesh.bounds}")
        print(f"Mesh center: {mesh.center_mass}")
        print(f"Total vertices: {len(mesh.vertices)}, Total faces: {len(mesh.faces)}")
        
        # 生成三种可视化（使用采样）
        visualize_point_cloud(mesh, args.output_dir, args.prefix, args.max_points)
        # visualize_mesh(mesh, args.output_dir, args.prefix, args.max_faces)
        # visualize_voxel_grid(mesh, args.voxel_size, args.output_dir, args.prefix)
        
        print("\n所有可视化已完成！")
        print(f"输出目录: {args.output_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 示例使用（如果没有命令行参数）
    import sys
    if len(sys.argv) == 1:
        print("使用示例:")
        print("python obj_visualization.py path/to/your/model.obj")
        print("python obj_visualization.py path/to/your/model.fbx")
        print("python obj_visualization.py path/to/your/model.obj --voxel_size 0.05 --output_dir custom_output")
        print("python obj_visualization.py path/to/your/model.fbx --max_points 5000 --max_faces 10000")
        print("\n性能优化参数:")
        print("--max_points: 点云最大点数 (默认: 10000)")
        print("--max_faces: 网格最大面数 (默认: 20000)")
        print("--voxel_size: 体素大小，越大性能越好 (默认: 0.02)")
        print("\n支持的文件格式: .obj, .fbx, .ply, .stl 等")
        print("如果您有3D文件，请提供文件路径作为参数")
        
        # 检查是否有示例文件
        example_paths = [
            "data/demo/sample.obj",
            "data/demo/sample.fbx",
            "dataset/dfaust/sample.obj",
            "output/demo/sample.obj",
            "output/demo/sample.fbx"
        ]
        
        for path in example_paths:
            if os.path.exists(path):
                print(f"\n找到示例文件: {path}")
                print(f"运行命令: python obj_visualization.py {path}")
                break
    else:
        exit(main())
