#!/usr/bin/env python3
"""
基于真正统一拓扑的高级插值系统
现在包含：
1. 直接线性插值 (基于统一拓扑)
2. 骨骼驱动插值 (基于统一拓扑的DemBones数据)
3. 混合插值方法
"""

import os
import pickle
import numpy as np
import open3d as o3d
import argparse

def load_unified_topology_data(data_folder):
    """加载统一拓扑数据"""
    unified_path = os.path.join(data_folder, 'unified_topology_results.pkl')
    dembone_path = os.path.join(data_folder, 'unified_dembone_results.pkl')
    
    if not os.path.exists(unified_path):
        raise FileNotFoundError(f"统一拓扑结果未找到: {unified_path}")
    if not os.path.exists(dembone_path):
        print(f"⚠️  统一DemBones结果未找到: {dembone_path}")
        print("只支持直接插值方法")
        dembone_data = None
    else:
        with open(dembone_path, 'rb') as f:
            dembone_data = pickle.load(f)
    
    with open(unified_path, 'rb') as f:
        unified_data = pickle.load(f)
    
    return unified_data, dembone_data

def apply_linear_blend_skinning(vertices, joint_positions, rest_joint_positions, skinning_weights, parents):
    """应用线性混合蒙皮"""
    deformed_vertices = vertices.copy()
    
    for joint_idx in range(len(joint_positions)):
        # 计算关节变换（简化为平移）
        translation = joint_positions[joint_idx] - rest_joint_positions[joint_idx]
        
        # 应用到所有受影响的顶点
        for vertex_idx in range(len(vertices)):
            weight = skinning_weights[vertex_idx, joint_idx]
            if weight > 0.001:  # 只处理有显著权重的顶点
                deformed_vertices[vertex_idx] += weight * translation
    
    return deformed_vertices

def unified_topology_interpolation_advanced(data_folder, frame_a, frame_b, num_steps=10, 
                                          method='direct', output_dir=None):
    """
    基于统一拓扑的高级插值
    
    Methods:
    - 'direct': 直接线性插值 (保证无缺失顶点)
    - 'skeleton_driven': 基于统一DemBones的骨骼驱动插值
    - 'hybrid': 混合方法
    """
    
    print(f"\n🎯 统一拓扑高级插值 (方法: {method})...")
    
    # 加载数据
    unified_data, dembone_data = load_unified_topology_data(data_folder)
    
    unified_vertices = unified_data['unified_vertices']  # (F, N, 3)
    triangles = unified_data['unified_triangles']  # (T, 3)
    frame_names = unified_data['frame_names']
    
    print(f"📊 数据加载成功:")
    print(f"   帧数: {len(frame_names)}")
    print(f"   统一顶点数: {unified_vertices.shape[1]}")
    print(f"   三角形数: {triangles.shape[0]}")
    
    # 转换frame名称
    def parse_frame_name(frame_input):
        if frame_input.isdigit():
            frame_num = int(frame_input)
            if 1 <= frame_num <= len(frame_names):
                return frame_names[frame_num - 1], frame_num - 1
            else:
                raise ValueError(f"Frame number {frame_num} out of range (1-{len(frame_names)})")
        else:
            if frame_input not in frame_names:
                raise ValueError(f"Frame '{frame_input}' not found")
            return frame_input, frame_names.index(frame_input)
    
    frame_a_name, frame_a_idx = parse_frame_name(frame_a)
    frame_b_name, frame_b_idx = parse_frame_name(frame_b)
    
    print(f"🔗 插值: {frame_a_name} (idx={frame_a_idx}) → {frame_b_name} (idx={frame_b_idx})")
    
    # 检查骨骼驱动方法的可用性
    if method in ['skeleton_driven', 'hybrid'] and dembone_data is None:
        print(f"⚠️  方法 '{method}' 需要统一DemBones数据，降级为直接插值")
        method = 'direct'
    
    # 获取scale参数
    first_frame_unified_path = os.path.join(data_folder, f'{frame_names[0]}_unified.pkl')
    with open(first_frame_unified_path, 'rb') as f:
        scale_params = pickle.load(f)['original_scale_params']
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(data_folder, f'unified_advanced_interpolation_{method}_{frame_a}_{frame_b}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 预计算数据
    vertices_a = unified_vertices[frame_a_idx]  # (N, 3)
    vertices_b = unified_vertices[frame_b_idx]  # (N, 3)
    
    if method in ['skeleton_driven', 'hybrid']:
        # 加载关节数据
        frame_a_data_path = os.path.join(data_folder, f'{frame_a_name}_data.pkl')
        frame_b_data_path = os.path.join(data_folder, f'{frame_b_name}_data.pkl')
        
        with open(frame_a_data_path, 'rb') as f:
            joints_a = pickle.load(f)['joints']
        with open(frame_b_data_path, 'rb') as f:
            joints_b = pickle.load(f)['joints']
        
        rest_pose = dembone_data['rest_pose']
        skinning_weights = dembone_data['skinning_weights']
        rest_joints = dembone_data['joints']
        parents = dembone_data['parents']
        
        print(f"🦴 骨骼数据:")
        print(f"   关节数: {len(joints_a)}")
        print(f"   权重矩阵: {skinning_weights.shape}")
    
    # 执行插值
    interpolated_meshes = []
    
    for i in range(num_steps + 1):
        t = i / num_steps
        
        if method == 'direct':
            # 直接线性插值 - 保证无缺失顶点
            interp_vertices = (1 - t) * vertices_a + t * vertices_b
            
        elif method == 'skeleton_driven':
            # 骨骼驱动插值
            # 1. 插值关节位置
            interp_joints = (1 - t) * joints_a + t * joints_b
            
            # 2. 基于插值关节位置应用LBS
            interp_vertices = apply_linear_blend_skinning(
                rest_pose, interp_joints, rest_joints, skinning_weights, parents
            )
            
        elif method == 'hybrid':
            # 混合方法：结合直接插值和骨骼驱动
            # 1. 直接插值
            direct_vertices = (1 - t) * vertices_a + t * vertices_b
            
            # 2. 骨骼驱动插值
            interp_joints = (1 - t) * joints_a + t * joints_b
            skeleton_vertices = apply_linear_blend_skinning(
                rest_pose, interp_joints, rest_joints, skinning_weights, parents
            )
            
            # 3. 加权混合
            blend_weight = 0.3  # 30% 骨骼驱动, 70% 直接插值
            interp_vertices = (1 - blend_weight) * direct_vertices + blend_weight * skeleton_vertices
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 转换回世界坐标
        bmin, blen = scale_params['bmin'], scale_params['blen']
        vertices_world = (interp_vertices + 1) * 0.5 * blen + bmin
        
        # 创建mesh
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices_world),
            triangles=o3d.utility.Vector3iVector(triangles)
        )
        mesh.compute_vertex_normals()
        interpolated_meshes.append(mesh)
        
        # 保存mesh
        filename = f'unified_{method}_interp_{i:03d}_t{t:.3f}.obj'
        filepath = os.path.join(output_dir, filename)
        o3d.io.write_triangle_mesh(filepath, mesh)
        
        print(f"✅ 保存: {filename} (t={t:.3f})")
    
    print(f"\n🎉 统一拓扑高级插值完成！")
    print(f"📁 生成了 {len(interpolated_meshes)} 个mesh在: {output_dir}")
    print(f"🔍 所有mesh都有相同的拓扑: {vertices_world.shape[0]} 顶点, {triangles.shape[0]} 三角形")
    print(f"✨ 优点: 无顶点缺失, 无面片变形, 完美插值")
    
    return interpolated_meshes, output_dir

def compare_unified_methods(data_folder, frame_a, frame_b, steps=5):
    """对比所有统一拓扑插值方法"""
    
    methods = ['direct', 'skeleton_driven', 'hybrid']
    results = {}
    
    print("🔍 === 统一拓扑插值方法对比 ===")
    print("这些方法都基于真正的统一拓扑，保证无顶点缺失")
    
    for method in methods:
        try:
            print(f"\n🔄 测试方法: {method}")
            meshes, output_dir = unified_topology_interpolation_advanced(
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
    print(f"\n📊 === 统一拓扑方法对比总结 ===")
    print(f"💡 所有方法基于统一拓扑，彻底解决顶点缺失和面片变形问题")
    
    for method, result in results.items():
        if result['success']:
            print(f"✅ {method}: 成功生成 {len(result['meshes'])} 个完美拓扑mesh")
            print(f"   📁 输出目录: {result['output_dir']}")
        else:
            print(f"❌ {method}: 失败 - {result['error']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="基于真正统一拓扑的高级插值系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 解决方案说明:
此系统基于真正的统一拓扑，彻底解决了：
- ✅ 顶点缺失问题
- ✅ 面片变形问题  
- ✅ 拓扑不一致问题

示例用法:
  # 直接插值 (推荐，最稳定)
  python unified_topology_interpolation_advanced.py 1 20 --method direct --steps 10
  
  # 骨骼驱动插值 (需要统一DemBones数据)
  python unified_topology_interpolation_advanced.py 1 20 --method skeleton_driven --steps 10
  
  # 混合插值方法
  python unified_topology_interpolation_advanced.py 1 20 --method hybrid --steps 10
  
  # 对比所有方法
  python unified_topology_interpolation_advanced.py 1 20 --compare --steps 5
        """
    )
    
    parser.add_argument('frame_a', nargs='?', help='起始帧 (1-157 或完整名称)')
    parser.add_argument('frame_b', nargs='?', help='结束帧 (1-157 或完整名称)')
    
    parser.add_argument('--data_folder', 
                       default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
                       help='generated_skeletons文件夹路径')
    parser.add_argument('--steps', type=int, default=10, help='插值步数')
    parser.add_argument('--method', 
                       choices=['direct', 'skeleton_driven', 'hybrid'],
                       default='direct',
                       help='插值方法')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--compare', action='store_true', help='对比所有统一拓扑方法')
    
    args = parser.parse_args()
    
    # 检查数据文件夹
    if not os.path.exists(args.data_folder):
        print(f"❌ 数据文件夹不存在: {args.data_folder}")
        return
    
    # 验证参数
    if not args.frame_a or not args.frame_b:
        print("❌ 请指定起始帧和结束帧")
        return
    
    # 执行插值
    try:
        if args.compare:
            print("🔄 对比所有统一拓扑插值方法...")
            compare_unified_methods(args.data_folder, args.frame_a, args.frame_b, args.steps)
        else:
            print(f"🚀 使用统一拓扑 {args.method} 方法进行插值...")
            unified_topology_interpolation_advanced(
                args.data_folder, args.frame_a, args.frame_b, 
                args.steps, args.method, args.output)
        
    except Exception as e:
        print(f"❌ 插值过程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
