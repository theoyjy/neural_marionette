#!/usr/bin/env python3
"""
高质量混合插值系统 - 结合interpolate.py的高精度数据和enhanced系统的拓扑统一
"""

import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse

def load_original_mesh_data(data_folder):
    """加载原始的高精度mesh数据 (interpolate.py使用的数据)"""
    
    # 加载DemBones结果
    dembone_path = os.path.join(data_folder, 'dembone_results.pkl')
    if not os.path.exists(dembone_path):
        raise FileNotFoundError(f"DemBones results not found: {dembone_path}")
    
    with open(dembone_path, 'rb') as f:
        dembone_results = pickle.load(f)
    
    # 加载个体mesh数据文件
    data_files = [f for f in os.listdir(data_folder) if f.endswith('_data.pkl')]
    data_files.sort()
    
    mesh_data = {}
    for data_file in data_files:
        frame_name = data_file.replace('_data.pkl', '')
        with open(os.path.join(data_folder, data_file), 'rb') as f:
            mesh_data[frame_name] = pickle.load(f)
    
    print(f"📊 加载了 {len(mesh_data)} 个高精度原始mesh数据")
    return dembone_results, mesh_data

def interpolate_vertices_direct_high_quality(verts_a, verts_b, t):
    """
    高质量的直接顶点插值 (interpolate.py的direct方法)
    保持原始精度
    """
    # 处理不同顶点数量：使用最近邻映射
    if verts_a.shape[0] != verts_b.shape[0]:
        tree = cKDTree(verts_b)
        _, indices = tree.query(verts_a, k=1)
        verts_b_mapped = verts_b[indices]
    else:
        verts_b_mapped = verts_b
    
    return (1 - t) * verts_a + t * verts_b_mapped

def interpolate_vertices_unified_topology(frame_a_idx, frame_b_idx, t, unified_results):
    """
    使用unified topology但保持高精度的插值方法
    """
    unified_vertices = unified_results['unified_vertices']
    
    # 直接在统一拓扑上进行插值
    vertices_a = unified_vertices[frame_a_idx]
    vertices_b = unified_vertices[frame_b_idx]
    
    return (1 - t) * vertices_a + t * vertices_b

def create_interpolated_mesh(vertices, triangles, scale_params):
    """将插值顶点转换为世界坐标的mesh"""
    # 从normalized space转换回世界坐标
    bmin, blen = scale_params['bmin'], scale_params['blen']
    vertices_world = (vertices + 1) * 0.5 * blen + bmin
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices_world),
        triangles=o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    return mesh

def hybrid_interpolate_meshes(data_folder, frame_a, frame_b, num_steps=10, 
                             method='high_quality_direct', output_dir=None):
    """
    混合高质量插值系统
    
    方法选项:
    - 'high_quality_direct': 使用原始高精度数据 + 直接插值 (推荐)
    - 'unified_topology': 使用统一拓扑数据 + 直接插值
    - 'nearest_neighbor_hq': 高质量最近邻方法
    """
    
    # 加载原始高精度数据
    dembone_results, mesh_data = load_original_mesh_data(data_folder)
    frame_names = sorted(list(mesh_data.keys()))
    
    # 转换frame名称为实际名称
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
    
    # 获取数据
    data_a = mesh_data[frame_a_name]
    data_b = mesh_data[frame_b_name]
    
    print(f"🎯 混合插值：{frame_a_name} → {frame_b_name}")
    print(f"📊 方法：{method}, 步数：{num_steps}")
    print(f"🔍 Frame A 顶点数：{data_a['pts_norm'].shape[0]}")
    print(f"🔍 Frame B 顶点数：{data_b['pts_norm'].shape[0]}")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(data_folder, f'hybrid_interpolation_{frame_a}_{frame_b}_{method}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取三角形拓扑（使用第一个mesh的拓扑）
    triangles = data_a['mesh_triangles']
    
    # 加载unified数据（如果需要）
    unified_results = None
    if method == 'unified_topology':
        try:
            unified_path = os.path.join(data_folder, 'skeleton_driven_results.pkl')
            if os.path.exists(unified_path):
                with open(unified_path, 'rb') as f:
                    unified_data = pickle.load(f)
                unified_results = {
                    'unified_vertices': unified_data['unified_vertices'],
                    'frame_names': unified_data['frame_names']
                }
                
                # 找到对应的帧索引
                unified_frame_names = unified_results['frame_names']
                frame_a_idx = unified_frame_names.index(frame_a_name) if frame_a_name in unified_frame_names else 0
                frame_b_idx = unified_frame_names.index(frame_b_name) if frame_b_name in unified_frame_names else min(1, len(unified_frame_names)-1)
                
                print(f"🔗 使用统一拓扑，Frame A idx: {frame_a_idx}, Frame B idx: {frame_b_idx}")
            else:
                print("⚠️  未找到unified数据，降级为high_quality_direct方法")
                method = 'high_quality_direct'
        except Exception as e:
            print(f"⚠️  加载unified数据失败: {e}，降级为high_quality_direct方法")
            method = 'high_quality_direct'
    
    # 生成插值mesh序列
    interpolated_meshes = []
    
    for i in range(num_steps + 1):
        t = i / num_steps
        
        if method == 'high_quality_direct':
            # 使用原始高精度数据进行直接插值
            verts_a = data_a['pts_norm']
            verts_b = data_b['pts_norm']
            interp_vertices = interpolate_vertices_direct_high_quality(verts_a, verts_b, t)
            scale_params = data_a  # 使用Frame A的scale参数
            
        elif method == 'unified_topology':
            # 使用统一拓扑数据
            interp_vertices = interpolate_vertices_unified_topology(
                frame_a_idx, frame_b_idx, t, unified_results)
            scale_params = data_a
            # 注意：这里可能需要调整triangles来匹配unified拓扑
            
        elif method == 'nearest_neighbor_hq':
            # 高质量最近邻方法
            verts_a = data_a['pts_norm']
            verts_b = data_b['pts_norm']
            
            # 更精确的最近邻映射
            if verts_a.shape[0] != verts_b.shape[0]:
                tree = cKDTree(verts_b)
                distances, indices = tree.query(verts_a, k=3)  # 使用k=3进行更好的插值
                
                # 使用距离加权的插值而不是简单的最近邻
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                verts_b_mapped = np.sum(verts_b[indices] * weights[:, :, np.newaxis], axis=1)
            else:
                verts_b_mapped = verts_b
            
            interp_vertices = (1 - t) * verts_a + t * verts_b_mapped
            scale_params = data_a
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 创建mesh
        mesh = create_interpolated_mesh(interp_vertices, triangles, scale_params)
        interpolated_meshes.append(mesh)
        
        # 保存mesh
        filename = f'hybrid_{method}_{i:03d}_t{t:.3f}.obj'
        filepath = os.path.join(output_dir, filename)
        o3d.io.write_triangle_mesh(filepath, mesh)
        
        print(f"✅ 保存: {filename} (t={t:.3f})")
    
    # 创建summary文件
    summary = {
        'method': f'hybrid_{method}',
        'frame_a': frame_a_name,
        'frame_b': frame_b_name,
        'num_steps': num_steps,
        'total_meshes': len(interpolated_meshes),
        'data_source': 'original_high_quality',
        'vertex_counts': {
            'frame_a': data_a['pts_norm'].shape[0],
            'frame_b': data_b['pts_norm'].shape[0]
        }
    }
    
    summary_path = os.path.join(output_dir, 'hybrid_interpolation_info.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n🎉 混合插值完成！")
    print(f"📁 生成了 {len(interpolated_meshes)} 个高质量mesh在: {output_dir}")
    
    return interpolated_meshes, output_dir

def compare_all_methods(data_folder, frame_a, frame_b, steps=5):
    """对比所有插值方法的效果"""
    
    methods = [
        'high_quality_direct',      # 原始高精度数据 + 直接插值
        'unified_topology',         # 统一拓扑数据 + 直接插值  
        'nearest_neighbor_hq'       # 高质量最近邻
    ]
    
    results = {}
    
    for method in methods:
        try:
            print(f"\n🔄 测试方法: {method}")
            meshes, output_dir = hybrid_interpolate_meshes(
                data_folder, frame_a, frame_b, steps, method)
            results[method] = {
                'meshes': meshes,
                'output_dir': output_dir,
                'success': True
            }
            print(f"✅ {method} 成功")
        except Exception as e:
            print(f"❌ {method} 失败: {e}")
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    # 输出对比总结
    print(f"\n📊 === 所有方法对比总结 ===")
    for method, result in results.items():
        if result['success']:
            print(f"✅ {method}: 成功生成 {len(result['meshes'])} 个mesh")
            print(f"   📁 输出目录: {result['output_dir']}")
        else:
            print(f"❌ {method}: 失败 - {result['error']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="混合高质量mesh插值系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用原始高精度数据进行插值 (推荐)
  python hybrid_interpolation.py 1 20 --method high_quality_direct --steps 10
  
  # 使用统一拓扑数据
  python hybrid_interpolation.py 1 20 --method unified_topology --steps 10
  
  # 对比所有方法
  python hybrid_interpolation.py 1 20 --compare --steps 5
  
  # 列出可用帧
  python hybrid_interpolation.py --list
        """
    )
    
    parser.add_argument('frame_a', nargs='?', help='起始帧 (1-157 或完整名称)')
    parser.add_argument('frame_b', nargs='?', help='结束帧 (1-157 或完整名称)')
    
    parser.add_argument('--data_folder', 
                       default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
                       help='generated_skeletons文件夹路径')
    parser.add_argument('--steps', type=int, default=10, help='插值步数')
    parser.add_argument('--method', 
                       choices=['high_quality_direct', 'unified_topology', 'nearest_neighbor_hq'],
                       default='high_quality_direct',
                       help='插值方法')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--compare', action='store_true', help='对比所有方法')
    parser.add_argument('--list', action='store_true', help='列出所有可用帧')
    
    args = parser.parse_args()
    
    # 检查数据文件夹
    if not os.path.exists(args.data_folder):
        print(f"❌ 数据文件夹不存在: {args.data_folder}")
        return
    
    # 列出帧
    if args.list:
        try:
            _, mesh_data = load_original_mesh_data(args.data_folder)
            frame_names = sorted(list(mesh_data.keys()))
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
            print("🔄 对比所有插值方法...")
            compare_all_methods(args.data_folder, args.frame_a, args.frame_b, args.steps)
        else:
            print(f"🚀 使用 {args.method} 方法进行插值...")
            hybrid_interpolate_meshes(
                args.data_folder, args.frame_a, args.frame_b, 
                args.steps, args.method, args.output)
        
    except Exception as e:
        print(f"❌ 插值过程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
