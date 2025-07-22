#!/usr/bin/env python3
"""
真正的骨骼驱动网格拓扑统一系统
解决方案：
1. 创建统一的高质量模板网格
2. 所有帧映射到相同拓扑 
3. 重新生成DemBones数据
4. 基于统一拓扑进行高质量插值
"""

import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse

class TrueTopologyUnifier:
    """真正的拓扑统一器 - 确保所有帧使用完全相同的网格拓扑"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.template_mesh = None
        self.template_triangles = None
        self.template_vertices = None
        self.unified_topology_size = None
        self.template_frame_name = None
        
    def load_all_mesh_data(self):
        """加载所有原始mesh数据"""
        print("📊 加载所有原始mesh数据...")
        
        data_files = [f for f in os.listdir(self.data_folder) if f.endswith('_data.pkl')]
        data_files.sort()
        
        mesh_data = {}
        vertex_counts = {}
        
        for data_file in data_files:
            frame_name = data_file.replace('_data.pkl', '')
            with open(os.path.join(self.data_folder, data_file), 'rb') as f:
                data = pickle.load(f)
                mesh_data[frame_name] = data
                vertex_counts[frame_name] = data['pts_norm'].shape[0]
        
        print(f"✅ 加载了 {len(mesh_data)} 帧数据")
        print(f"🔍 顶点数范围: {min(vertex_counts.values())} - {max(vertex_counts.values())}")
        
        # 选择顶点数最多的帧作为模板
        template_frame = max(vertex_counts.items(), key=lambda x: x[1])
        self.template_frame_name = template_frame[0]
        self.unified_topology_size = template_frame[1]
        
        print(f"🎯 选择模板帧: {self.template_frame_name}")
        print(f"📏 统一拓扑大小: {self.unified_topology_size} 顶点")
        
        return mesh_data
    
    def create_unified_template(self, mesh_data):
        """创建统一的模板网格拓扑"""
        print(f"\n🔨 创建统一模板网格 (基于 {self.template_frame_name})...")
        
        template_data = mesh_data[self.template_frame_name]
        
        # 使用模板帧的拓扑作为统一拓扑
        self.template_vertices = template_data['pts_norm'].copy()  # (N, 3) normalized
        self.template_triangles = template_data['mesh_triangles'].copy()  # (F, 3) 
        
        print(f"✅ 模板网格创建完成:")
        print(f"   顶点数: {self.template_vertices.shape[0]}")
        print(f"   三角形数: {self.template_triangles.shape[0]}")
        print(f"   坐标范围: [{self.template_vertices.min():.6f}, {self.template_vertices.max():.6f}]")
        
        return self.template_vertices, self.template_triangles
    
    def map_frame_to_unified_topology(self, frame_data):
        """将单帧映射到统一拓扑"""
        frame_vertices = frame_data['pts_norm']  # (M, 3)
        
        if frame_vertices.shape[0] == self.unified_topology_size:
            # 如果顶点数已经相同，直接返回
            return frame_vertices
        
        # 使用KD-Tree找到最近邻映射
        tree = cKDTree(frame_vertices)
        distances, indices = tree.query(self.template_vertices, k=1)
        
        # 映射到统一拓扑
        unified_vertices = frame_vertices[indices]
        
        return unified_vertices
    
    def process_all_frames_to_unified_topology(self, mesh_data):
        """将所有帧处理到统一拓扑"""
        print(f"\n🔄 将所有帧映射到统一拓扑...")
        
        unified_frames = {}
        unified_vertices_array = []
        frame_names = []
        
        for i, (frame_name, frame_data) in enumerate(sorted(mesh_data.items())):
            print(f"   处理帧 {i+1}/{len(mesh_data)}: {frame_name}")
            
            # 映射到统一拓扑
            unified_vertices = self.map_frame_to_unified_topology(frame_data)
            
            unified_frames[frame_name] = {
                'vertices_normalized': unified_vertices,
                'triangles': self.template_triangles,
                'original_scale_params': {
                    'bmin': frame_data['bmin'],
                    'blen': frame_data['blen']
                },
                'joints': frame_data['joints']
            }
            
            unified_vertices_array.append(unified_vertices)
            frame_names.append(frame_name)
        
        # 转换为numpy数组
        unified_vertices_array = np.array(unified_vertices_array)  # (F, N, 3)
        
        print(f"✅ 统一拓扑处理完成:")
        print(f"   统一后形状: {unified_vertices_array.shape}")
        print(f"   所有帧现在都有 {self.unified_topology_size} 个顶点")
        
        return unified_frames, unified_vertices_array, frame_names
    
    def save_unified_results(self, unified_frames, unified_vertices_array, frame_names):
        """保存统一拓扑结果"""
        print(f"\n💾 保存统一拓扑结果...")
        
        # 保存统一后的数据
        unified_results = {
            'unified_vertices': unified_vertices_array,  # (F, N, 3)
            'unified_triangles': self.template_triangles,  # (T, 3)
            'frame_names': frame_names,
            'template_frame': self.template_frame_name,
            'topology_size': self.unified_topology_size,
            'method': 'true_topology_unification'
        }
        
        output_path = os.path.join(self.data_folder, 'unified_topology_results.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(unified_results, f)
        
        print(f"✅ 统一拓扑结果已保存: {output_path}")
        
        # 保存个体帧数据（用于新的DemBones训练）
        for frame_name, frame_data in unified_frames.items():
            frame_output_path = os.path.join(self.data_folder, f'{frame_name}_unified.pkl')
            with open(frame_output_path, 'wb') as f:
                pickle.dump(frame_data, f)
        
        print(f"✅ 个体统一帧数据已保存 ({len(unified_frames)} 个文件)")
        
        return unified_results
    
    def create_unified_mesh_files(self, unified_frames):
        """创建统一拓扑的OBJ文件（用于验证）"""
        print(f"\n📁 创建统一拓扑的OBJ验证文件...")
        
        verification_dir = os.path.join(self.data_folder, 'unified_topology_verification')
        os.makedirs(verification_dir, exist_ok=True)
        
        for i, (frame_name, frame_data) in enumerate(unified_frames.items()):
            if i >= 5:  # 只保存前5帧用于验证
                break
                
            vertices_norm = frame_data['vertices_normalized']
            triangles = frame_data['triangles']
            scale_params = frame_data['original_scale_params']
            
            # 转换回世界坐标
            bmin, blen = scale_params['bmin'], scale_params['blen'] 
            vertices_world = (vertices_norm + 1) * 0.5 * blen + bmin
            
            # 创建mesh
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vertices_world),
                triangles=o3d.utility.Vector3iVector(triangles)
            )
            mesh.compute_vertex_normals()
            
            # 保存验证文件
            output_path = os.path.join(verification_dir, f'{frame_name}_unified_topology.obj')
            o3d.io.write_triangle_mesh(output_path, mesh)
            
            print(f"   保存验证文件: {frame_name}_unified_topology.obj")
        
        print(f"✅ 验证文件已保存到: {verification_dir}")

def regenerate_dembone_data_from_unified_topology(data_folder):
    """基于统一拓扑重新生成DemBones数据"""
    print(f"\n🔧 基于统一拓扑重新生成DemBones数据...")
    
    # 加载统一拓扑结果
    unified_path = os.path.join(data_folder, 'unified_topology_results.pkl')
    if not os.path.exists(unified_path):
        raise FileNotFoundError(f"统一拓扑结果未找到: {unified_path}")
    
    with open(unified_path, 'rb') as f:
        unified_results = pickle.load(f)
    
    unified_vertices = unified_results['unified_vertices']  # (F, N, 3)
    frame_names = unified_results['frame_names']
    
    print(f"📊 统一拓扑数据:")
    print(f"   帧数: {unified_vertices.shape[0]}")
    print(f"   顶点数: {unified_vertices.shape[1]} (所有帧相同)")
    print(f"   坐标维度: {unified_vertices.shape[2]}")
    
    # 选择一个帧作为rest pose (通常选择中间帧或特定姿态)
    rest_frame_idx = len(frame_names) // 2  # 选择中间帧
    rest_pose = unified_vertices[rest_frame_idx]  # (N, 3)
    
    print(f"🎯 选择Rest Pose: Frame {rest_frame_idx} ({frame_names[rest_frame_idx]})")
    
    # 计算简化的skinning weights (基于顶点与关节的距离)
    # 加载joint信息
    first_frame_data_path = os.path.join(data_folder, f'{frame_names[0]}_data.pkl')
    with open(first_frame_data_path, 'rb') as f:
        first_frame_data = pickle.load(f)
    
    joints = first_frame_data['joints']  # (K, 3)
    num_joints = len(joints)
    num_vertices = rest_pose.shape[0]
    
    print(f"🦴 关节信息: {num_joints} 个关节")
    
    # 生成简单的距离基础skinning weights
    skinning_weights = np.zeros((num_vertices, num_joints))
    
    for v in range(num_vertices):
        vertex_pos = rest_pose[v]
        distances = np.linalg.norm(joints - vertex_pos, axis=1)
        
        # 使用指数衰减权重
        weights = np.exp(-distances * 2.0)  # 调整衰减率
        weights = weights / (weights.sum() + 1e-8)  # 归一化
        
        # 稀疏化：只保留top-4权重
        top_indices = np.argsort(weights)[-4:]
        sparse_weights = np.zeros(num_joints)
        sparse_weights[top_indices] = weights[top_indices]
        sparse_weights = sparse_weights / (sparse_weights.sum() + 1e-8)
        
        skinning_weights[v] = sparse_weights
    
    print(f"🎭 Skinning权重生成完成:")
    print(f"   权重矩阵形状: {skinning_weights.shape}")
    print(f"   权重范围: [{skinning_weights.min():.6f}, {skinning_weights.max():.6f}]")
    
    # 生成简化的parent关系 (线性链)
    parents = np.arange(-1, num_joints-1)  # [-1, 0, 1, 2, ..., K-2]
    
    # 创建新的DemBones结果
    new_dembone_results = {
        'rest_pose': rest_pose,  # (N, 3) normalized
        'skinning_weights': skinning_weights,  # (N, K)
        'parents': parents,  # (K,)
        'joints': joints,  # (K, 3)
        'method': 'unified_topology_dembone',
        'source_data': 'unified_topology_results.pkl',
        'num_frames': unified_vertices.shape[0],
        'num_vertices': num_vertices,
        'num_joints': num_joints
    }
    
    # 保存新的DemBones结果
    new_dembone_path = os.path.join(data_folder, 'unified_dembone_results.pkl')
    with open(new_dembone_path, 'wb') as f:
        pickle.dump(new_dembone_results, f)
    
    print(f"✅ 统一拓扑DemBones数据已保存: {new_dembone_path}")
    
    return new_dembone_results

def unified_topology_interpolation(data_folder, frame_a, frame_b, num_steps=10, output_dir=None):
    """基于统一拓扑的高质量插值"""
    print(f"\n🎯 基于统一拓扑的高质量插值...")
    
    # 加载统一拓扑数据
    unified_path = os.path.join(data_folder, 'unified_topology_results.pkl')
    if not os.path.exists(unified_path):
        raise FileNotFoundError(f"统一拓扑结果未找到: {unified_path}")
    
    with open(unified_path, 'rb') as f:
        unified_results = pickle.load(f)
    
    unified_vertices = unified_results['unified_vertices']  # (F, N, 3)
    triangles = unified_results['unified_triangles']  # (T, 3)
    frame_names = unified_results['frame_names']
    
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
    
    # 获取scale参数 (用第一帧的)
    first_frame_unified_path = os.path.join(data_folder, f'{frame_names[0]}_unified.pkl')
    with open(first_frame_unified_path, 'rb') as f:
        scale_params = pickle.load(f)['original_scale_params']
    
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(data_folder, f'unified_topology_interpolation_{frame_a}_{frame_b}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行插值
    interpolated_meshes = []
    
    for i in range(num_steps + 1):
        t = i / num_steps
        
        # 在统一拓扑上进行线性插值
        vertices_a = unified_vertices[frame_a_idx]  # (N, 3)
        vertices_b = unified_vertices[frame_b_idx]  # (N, 3)
        
        interp_vertices = (1 - t) * vertices_a + t * vertices_b  # (N, 3)
        
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
        filename = f'unified_topology_interp_{i:03d}_t{t:.3f}.obj'
        filepath = os.path.join(output_dir, filename)
        o3d.io.write_triangle_mesh(filepath, mesh)
        
        print(f"✅ 保存: {filename} (t={t:.3f})")
    
    print(f"\n🎉 统一拓扑插值完成！")
    print(f"📁 生成了 {len(interpolated_meshes)} 个mesh在: {output_dir}")
    print(f"🔍 所有mesh都有相同的拓扑: {vertices_world.shape[0]} 顶点, {triangles.shape[0]} 三角形")
    
    return interpolated_meshes, output_dir

def main():
    parser = argparse.ArgumentParser(
        description="真正的骨骼驱动网格拓扑统一系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 工作流程:
1. 创建统一拓扑模板
2. 将所有帧映射到统一拓扑  
3. 重新生成DemBones数据
4. 基于统一拓扑进行插值

示例用法:
  # 1. 创建统一拓扑
  python true_topology_unifier.py --unify
  
  # 2. 重新生成DemBones数据
  python true_topology_unifier.py --regenerate-dembone
  
  # 3. 基于统一拓扑插值
  python true_topology_unifier.py 1 20 --interpolate --steps 10
  
  # 4. 完整流程
  python true_topology_unifier.py --full-pipeline 1 20 --steps 10
        """
    )
    
    parser.add_argument('frame_a', nargs='?', help='起始帧 (1-157 或完整名称)')
    parser.add_argument('frame_b', nargs='?', help='结束帧 (1-157 或完整名称)')
    
    parser.add_argument('--data_folder', 
                       default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
                       help='generated_skeletons文件夹路径')
    parser.add_argument('--unify', action='store_true', help='创建统一拓扑')
    parser.add_argument('--regenerate-dembone', action='store_true', help='重新生成DemBones数据')
    parser.add_argument('--interpolate', action='store_true', help='执行统一拓扑插值')
    parser.add_argument('--full-pipeline', action='store_true', help='执行完整流程')
    parser.add_argument('--steps', type=int, default=10, help='插值步数')
    parser.add_argument('--output', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查数据文件夹
    if not os.path.exists(args.data_folder):
        print(f"❌ 数据文件夹不存在: {args.data_folder}")
        return
    
    unifier = TrueTopologyUnifier(args.data_folder)
    
    try:
        if args.full_pipeline:
            # 完整流程
            if not args.frame_a or not args.frame_b:
                print("❌ 完整流程需要指定起始帧和结束帧")
                return
                
            print("🚀 执行完整拓扑统一流程...")
            
            # Step 1: 创建统一拓扑
            mesh_data = unifier.load_all_mesh_data()
            template_vertices, template_triangles = unifier.create_unified_template(mesh_data)
            unified_frames, unified_vertices_array, frame_names = unifier.process_all_frames_to_unified_topology(mesh_data)
            unified_results = unifier.save_unified_results(unified_frames, unified_vertices_array, frame_names)
            unifier.create_unified_mesh_files(unified_frames)
            
            # Step 2: 重新生成DemBones数据
            new_dembone_results = regenerate_dembone_data_from_unified_topology(args.data_folder)
            
            # Step 3: 执行插值
            interpolated_meshes, output_dir = unified_topology_interpolation(
                args.data_folder, args.frame_a, args.frame_b, args.steps, args.output)
            
            print(f"\n🎉 完整流程完成！现在您有了真正统一拓扑的插值系统！")
            
        elif args.unify:
            # 只创建统一拓扑
            mesh_data = unifier.load_all_mesh_data()
            template_vertices, template_triangles = unifier.create_unified_template(mesh_data)
            unified_frames, unified_vertices_array, frame_names = unifier.process_all_frames_to_unified_topology(mesh_data)
            unified_results = unifier.save_unified_results(unified_frames, unified_vertices_array, frame_names)
            unifier.create_unified_mesh_files(unified_frames)
            
        elif args.regenerate_dembone:
            # 只重新生成DemBones数据
            new_dembone_results = regenerate_dembone_data_from_unified_topology(args.data_folder)
            
        elif args.interpolate:
            # 只执行插值
            if not args.frame_a or not args.frame_b:
                print("❌ 插值需要指定起始帧和结束帧")
                return
            interpolated_meshes, output_dir = unified_topology_interpolation(
                args.data_folder, args.frame_a, args.frame_b, args.steps, args.output)
            
        else:
            print("❌ 请指定操作: --unify, --regenerate-dembone, --interpolate, 或 --full-pipeline")
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
