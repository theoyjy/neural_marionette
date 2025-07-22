#!/usr/bin/env python3
"""
修复版enhanced_interpolation.py - 使用原始高质量数据而不是降质量的统一数据
解决模糊问题的关键：直接使用interpolate.py的数据源
"""

import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse

def load_high_quality_mesh_data(data_folder):
    """
    加载原始的高质量mesh数据 (和interpolate.py相同的数据源)
    这是解决模糊问题的关键！
    """
    
    # 加载DemBones结果
    dembone_path = os.path.join(data_folder, 'dembone_results.pkl')
    if not os.path.exists(dembone_path):
        raise FileNotFoundError(f"DemBones results not found: {dembone_path}")
    
    with open(dembone_path, 'rb') as f:
        dembone_results = pickle.load(f)
    
    # 加载个体mesh数据文件 - 这些包含高质量的pts_norm数据
    data_files = [f for f in os.listdir(data_folder) if f.endswith('_data.pkl')]
    data_files.sort()
    
    mesh_data = {}
    for data_file in data_files:
        frame_name = data_file.replace('_data.pkl', '')
        with open(os.path.join(data_folder, data_file), 'rb') as f:
            mesh_data[frame_name] = pickle.load(f)
    
    frame_names = sorted(list(mesh_data.keys()))
    
    print(f"📊 加载了 {len(mesh_data)} 个高质量原始mesh数据")
    print(f"🔍 数据质量检查 - 第一帧：")
    first_frame = mesh_data[frame_names[0]]
    print(f"   顶点数: {first_frame['pts_norm'].shape[0]}")
    print(f"   坐标范围: [{first_frame['pts_norm'].min():.6f}, {first_frame['pts_norm'].max():.6f}]")
    
    return dembone_results, mesh_data, frame_names

def interpolate_high_quality_direct(verts_a, verts_b, t):
    """
    高质量直接插值 - 和interpolate.py的direct方法完全相同
    保持原始精度，避免模糊
    """
    # 处理不同顶点数量
    if verts_a.shape[0] != verts_b.shape[0]:
        tree = cKDTree(verts_b)
        _, indices = tree.query(verts_a, k=1)
        verts_b_mapped = verts_b[indices]
    else:
        verts_b_mapped = verts_b
    
    return (1 - t) * verts_a + t * verts_b_mapped

def interpolate_skeleton_driven_dembone(frame_a_name, frame_b_name, t, dembone_results, mesh_data):
    """
    骨骼驱动插值 - 使用DemBones的高质量结果
    这个版本应该比之前的统一方法质量更高
    """
    data_a = mesh_data[frame_a_name]
    data_b = mesh_data[frame_b_name]
    
    # 使用高质量的joint数据
    joints_a = data_a['joints']
    joints_b = data_b['joints']
    
    # 插值joint位置
    joints_interp = (1 - t) * joints_a + t * joints_b
    
    # 使用DemBones的高质量rest pose和skinning weights
    rest_pose = dembone_results['rest_pose']  # 这是高质量的normalized数据
    skinning_weights = dembone_results['skinning_weights']
    parents = dembone_results['parents']
    
    # 应用骨骼变形
    deformed_vertices = np.zeros_like(rest_pose)
    
    for k in range(len(parents)):
        if k < len(joints_a) and k < len(joints_b):
            # 简单的translation-based变形
            bone_translation = joints_interp[k] - joints_a[k]
            
            # 应用加权translation
            for v in range(len(rest_pose)):
                weight = skinning_weights[v, k]
                deformed_vertices[v] += weight * bone_translation
    
    # 添加基础rest pose
    final_vertices = rest_pose + deformed_vertices
    
    return final_vertices, joints_interp

def interpolate_nearest_neighbor_hq(verts_a, verts_b, t):
    """
    高质量最近邻插值 - 使用多点加权而不是单点映射
    """
    if verts_a.shape[0] != verts_b.shape[0]:
        tree = cKDTree(verts_b)
        distances, indices = tree.query(verts_a, k=3)  # 使用3个最近邻
        
        # 距离加权插值
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        verts_b_mapped = np.sum(verts_b[indices] * weights[:, :, np.newaxis], axis=1)
    else:
        verts_b_mapped = verts_b
    
    return (1 - t) * verts_a + t * verts_b_mapped

def create_interpolated_mesh(vertices, triangles, scale_params):
    """创建插值mesh - 和interpolate.py完全相同的转换"""
    # 从normalized space转换回世界坐标
    bmin, blen = scale_params['bmin'], scale_params['blen']
    vertices_world = (vertices + 1) * 0.5 * blen + bmin
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices_world),
        triangles=o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    return mesh

def enhanced_interpolate_fixed(data_folder, frame_a, frame_b, num_steps=10, 
                              method='high_quality_direct', output_dir=None):
    """
    修复的高质量插值函数
    
    Methods:
    - 'high_quality_direct': 原始高精度数据 + 直接插值 (和interpolate.py的direct相同)
    - 'skeleton_driven_hq': 高质量骨骼驱动插值 (使用DemBones原始结果)
    - 'nearest_neighbor_hq': 高质量最近邻插值
    """
    
    # 加载高质量原始数据
    dembone_results, mesh_data, frame_names = load_high_quality_mesh_data(data_folder)
    
    # 转换frame名称
    def parse_frame_name(frame_input):
        if frame_input.isdigit():
            frame_num = int(frame_input)
            if 1 <= frame_num <= len(frame_names):
                return frame_names[frame_num - 1]
            else:
                raise ValueError(f"Frame number {frame_num} out of range (1-{len(frame_names)})")
        else:
            if frame_input not in frame_names:
                raise ValueError(f"Frame '{frame_input}' not found")
            return frame_input
    
    frame_a_name = parse_frame_name(frame_a)
    frame_b_name = parse_frame_name(frame_b)
    
    data_a = mesh_data[frame_a_name]
    data_b = mesh_data[frame_b_name]
    
    print(f"🎯 高质量插值：{frame_a_name} → {frame_b_name}")
    print(f"📊 方法：{method}, 步数：{num_steps}")
    print(f"🔍 Frame A: {data_a['pts_norm'].shape[0]} 顶点, 范围: [{data_a['pts_norm'].min():.6f}, {data_a['pts_norm'].max():.6f}]")
    print(f"🔍 Frame B: {data_b['pts_norm'].shape[0]} 顶点, 范围: [{data_b['pts_norm'].min():.6f}, {data_b['pts_norm'].max():.6f}]")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(data_folder, f'enhanced_fixed_{method}_{frame_a}_{frame_b}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取三角形拓扑
    triangles = data_a['mesh_triangles']
    
    # 生成插值序列
    interpolated_meshes = []
    
    for i in range(num_steps + 1):
        t = i / num_steps
        
        if method == 'high_quality_direct':
            # 高质量直接插值 - 和interpolate.py的direct方法相同
            verts_a = data_a['pts_norm']
            verts_b = data_b['pts_norm']
            interp_vertices = interpolate_high_quality_direct(verts_a, verts_b, t)
            scale_params = data_a
            
        elif method == 'skeleton_driven_hq':
            # 高质量骨骼驱动插值
            interp_vertices, _ = interpolate_skeleton_driven_dembone(
                frame_a_name, frame_b_name, t, dembone_results, mesh_data)
            scale_params = data_a
            
        elif method == 'nearest_neighbor_hq':
            # 高质量最近邻插值
            verts_a = data_a['pts_norm']
            verts_b = data_b['pts_norm']
            interp_vertices = interpolate_nearest_neighbor_hq(verts_a, verts_b, t)
            scale_params = data_a
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 创建mesh
        mesh = create_interpolated_mesh(interp_vertices, triangles, scale_params)
        interpolated_meshes.append(mesh)
        
        # 保存mesh
        filename = f'enhanced_fixed_{method}_{i:03d}_t{t:.3f}.obj'
        filepath = os.path.join(output_dir, filename)
        o3d.io.write_triangle_mesh(filepath, mesh)
        
        print(f"✅ 保存: {filename} (t={t:.3f})")
    
    print(f"\n🎉 高质量插值完成！")
    print(f"📁 生成了 {len(interpolated_meshes)} 个mesh在: {output_dir}")
    
    return interpolated_meshes, output_dir

def compare_fixed_methods(data_folder, frame_a, frame_b, steps=5):
    """对比修复后的所有插值方法"""
    
    methods = [
        'high_quality_direct',      # 和interpolate.py的direct完全相同
        'skeleton_driven_hq',       # 使用DemBones高质量结果的骨骼插值
        'nearest_neighbor_hq'       # 高质量最近邻
    ]
    
    results = {}
    
    print("🔍 === 高质量插值方法对比 ===")
    print("解决方案：使用原始高精度数据而不是降质量的统一数据")
    
    for method in methods:
        try:
            print(f"\n🔄 测试方法: {method}")
            meshes, output_dir = enhanced_interpolate_fixed(
                data_folder, frame_a, frame_b, steps, method)
            results[method] = {
                'meshes': meshes,
                'output_dir': output_dir,
                'success': True
            }
            print(f"✅ {method} 成功生成高质量结果")
        except Exception as e:
            print(f"❌ {method} 失败: {e}")
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    # 输出对比总结
    print(f"\n📊 === 高质量方法对比总结 ===")
    print(f"💡 这些方法现在都使用原始高精度数据，应该和interpolate.py的质量相同或更好")
    
    for method, result in results.items():
        if result['success']:
            print(f"✅ {method}: 成功生成 {len(result['meshes'])} 个高质量mesh")
            print(f"   📁 输出目录: {result['output_dir']}")
        else:
            print(f"❌ {method}: 失败 - {result['error']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="修复版高质量mesh插值系统 - 解决模糊问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 修复说明:
原问题：enhanced_interpolation.py使用了降质量的统一数据，导致模糊
解决方案：直接使用interpolate.py的原始高精度数据源

示例用法:
  # 高质量直接插值 (和interpolate.py的direct方法完全相同)
  python enhanced_interpolation_fixed.py 1 20 --method high_quality_direct --steps 10
  
  # 高质量骨骼驱动插值
  python enhanced_interpolation_fixed.py 1 20 --method skeleton_driven_hq --steps 10
  
  # 对比所有高质量方法
  python enhanced_interpolation_fixed.py 1 20 --compare --steps 5
        """
    )
    
    parser.add_argument('frame_a', nargs='?', help='起始帧 (1-157 或完整名称)')
    parser.add_argument('frame_b', nargs='?', help='结束帧 (1-157 或完整名称)')
    
    parser.add_argument('--data_folder', 
                       default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
                       help='generated_skeletons文件夹路径')
    parser.add_argument('--steps', type=int, default=10, help='插值步数')
    parser.add_argument('--method', 
                       choices=['high_quality_direct', 'skeleton_driven_hq', 'nearest_neighbor_hq'],
                       default='high_quality_direct',
                       help='插值方法')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--compare', action='store_true', help='对比所有高质量方法')
    parser.add_argument('--list', action='store_true', help='列出所有可用帧')
    
    args = parser.parse_args()
    
    # 检查数据文件夹
    if not os.path.exists(args.data_folder):
        print(f"❌ 数据文件夹不存在: {args.data_folder}")
        return
    
    # 列出帧
    if args.list:
        try:
            _, mesh_data, frame_names = load_high_quality_mesh_data(args.data_folder)
            print(f"\n📋 可用帧 ({len(frame_names)} 个):")
            for i, name in enumerate(frame_names, 1):
                print(f"  {i:3d}: {name}")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
        return
    
    # 验证参数
    if not args.frame_a or not args.frame_b:
        print("❌ 请指定起始帧和结束帧")
        print("使用 --list 查看可用帧")
        return
    
    # 执行插值
    try:
        if args.compare:
            print("🔄 对比所有高质量插值方法...")
            compare_fixed_methods(args.data_folder, args.frame_a, args.frame_b, args.steps)
        else:
            print(f"🚀 使用高质量 {args.method} 方法进行插值...")
            enhanced_interpolate_fixed(
                args.data_folder, args.frame_a, args.frame_b, 
                args.steps, args.method, args.output)
        
    except Exception as e:
        print(f"❌ 插值过程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
