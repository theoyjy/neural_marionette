#!/usr/bin/env python3
"""
DemBones数据格式深度调试
专注于理解DemBones的正确数据输入格式
"""

import numpy as np
import py_dem_bones
import time
import traceback

def debug_demBones_data_format():
    """深度调试DemBones的数据格式要求"""
    print("🔧 DemBones数据格式深度调试")
    print("=" * 50)
    
    # 创建最小测试数据
    n_frames = 3
    n_vertices = 4  
    n_bones = 3
    
    # 顶点位置：简单的变形序列
    vertices = np.array([
        [[-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],    # frame 0
        [[-0.8, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0, 0.8]], # frame 1  
        [[-0.6, 0, 0], [0.6, 0, 0], [0, 0, 0], [0, 0, 0.6]]  # frame 2
    ], dtype=np.float64)
    
    # 骨骼层次：简单链式结构
    parents = np.array([-1, 0, 1], dtype=np.int32)
    
    print(f"测试数据: {n_frames} frames, {n_vertices} vertices, {n_bones} bones")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Parents: {parents}")
    
    # 方法1：按照文档的标准格式
    print("\n📋 方法1：标准格式")
    try:
        dem1 = py_dem_bones.DemBones()
        
        # rest pose: (3, N) - 第一帧作为rest pose
        rest_pose = vertices[0].T  # (4,3) -> (3,4)
        print(f"Rest pose shape: {rest_pose.shape}")
        
        # animated poses: (3, F*N) - 所有帧flatten
        animated_poses = vertices.transpose(0,2,1).reshape(3, -1)  # (3, 12)
        print(f"Animated poses shape: {animated_poses.shape}")
        
        # 设置数据
        dem1.set_rest_pose(rest_pose)
        print(f"After set_rest_pose: num_vertices={dem1.num_vertices}")
        
        # 尝试不同的target设置方法
        dem1.animated_poses = animated_poses
        print(f"After animated_poses: num_vertices={dem1.num_vertices}")
        
        # 骨骼设置
        bone_names = [f"bone_{i}" for i in range(n_bones)]
        dem1.set_bone_names(*bone_names)
        
        for i in range(n_bones):
            if parents[i] >= 0:
                dem1.set_parent_bone(i, parents[i])
            else:
                dem1.set_parent_bone(i, None)
        
        print(f"Final status: vertices={dem1.num_vertices}, bones={dem1.num_bones}, frames={dem1.num_frames}")
        
        # 参数设置
        dem1.num_iterations = 3
        dem1.max_nonzeros_per_vertex = 2
        
        # 计算
        print("计算中...")
        start_time = time.time()
        success = dem1.compute()
        elapsed = time.time() - start_time
        print(f"方法1结果: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
        
        if success:
            print("方法1成功！")
            return True
            
    except Exception as e:
        print(f"方法1异常: {e}")
        traceback.print_exc()
    
    # 方法2：逐帧添加target
    print("\n📋 方法2：逐帧添加")
    try:
        dem2 = py_dem_bones.DemBones()
        
        # rest pose
        rest_pose = vertices[0].T
        dem2.set_rest_pose(rest_pose)
        print(f"Rest pose set: num_vertices={dem2.num_vertices}")
        
        # 逐帧添加
        for frame_idx in range(n_frames):
            frame_data = vertices[frame_idx].T  # (3, 4)
            target_name = f"frame_{frame_idx}"
            dem2.set_target_vertices(target_name, frame_data)
            print(f"Added {target_name}: num_targets={dem2.num_targets}")
        
        # 骨骼设置
        bone_names = [f"bone_{i}" for i in range(n_bones)]
        dem2.set_bone_names(*bone_names)
        
        for i in range(n_bones):
            if parents[i] >= 0:
                dem2.set_parent_bone(i, parents[i])
            else:
                dem2.set_parent_bone(i, None)
        
        print(f"Final status: vertices={dem2.num_vertices}, bones={dem2.num_bones}, targets={dem2.num_targets}")
        
        # 参数设置
        dem2.num_iterations = 3
        dem2.max_nonzeros_per_vertex = 2
        
        # 计算
        print("计算中...")
        start_time = time.time()
        success = dem2.compute()
        elapsed = time.time() - start_time
        print(f"方法2结果: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
        
        if success:
            print("方法2成功！")
            return True
            
    except Exception as e:
        print(f"方法2异常: {e}")
        traceback.print_exc()
    
    # 方法3：单一target，正确的数据格式
    print("\n📋 方法3：单一target")
    try:
        dem3 = py_dem_bones.DemBones()
        
        # rest pose
        rest_pose = vertices[0].T  # (3, 4)
        dem3.set_rest_pose(rest_pose)
        print(f"Rest pose set: num_vertices={dem3.num_vertices}")
        
        # 所有动画帧作为一个target - 但格式要正确
        # 尝试(3, F*N)格式
        all_frames = vertices.transpose(0,2,1).reshape(3, -1)  # (3, 12)
        dem3.set_target_vertices("animated", all_frames)
        print(f"Target set: num_vertices={dem3.num_vertices}, num_targets={dem3.num_targets}")
        
        # 骨骼设置
        bone_names = [f"bone_{i}" for i in range(n_bones)]
        dem3.set_bone_names(*bone_names)
        
        for i in range(n_bones):
            if parents[i] >= 0:
                dem3.set_parent_bone(i, parents[i])
            else:
                dem3.set_parent_bone(i, None)
        
        print(f"Final status: vertices={dem3.num_vertices}, bones={dem3.num_bones}, targets={dem3.num_targets}")
        
        # 参数设置
        dem3.num_iterations = 3
        dem3.max_nonzeros_per_vertex = 2
        
        # 计算
        print("计算中...")
        start_time = time.time()
        success = dem3.compute()
        elapsed = time.time() - start_time
        print(f"方法3结果: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
        
        if success:
            print("方法3成功！")
            return True
            
    except Exception as e:
        print(f"方法3异常: {e}")
        traceback.print_exc()
    
    # 方法4：检查是否需要显式设置num_vertices
    print("\n📋 方法4：显式设置顶点数")
    try:
        dem4 = py_dem_bones.DemBones()
        
        # 先显式设置顶点数
        if hasattr(dem4, 'set_num_vertices'):
            dem4.set_num_vertices(n_vertices)
            print(f"Set num_vertices: {dem4.num_vertices}")
        
        # rest pose
        rest_pose = vertices[0].T  # (3, 4)
        dem4.set_rest_pose(rest_pose)
        print(f"Rest pose set: num_vertices={dem4.num_vertices}")
        
        # 单个目标，但每帧分别处理
        animated_poses = vertices.transpose(0,2,1).reshape(3, -1)
        dem4.set_target_vertices("animated", animated_poses)
        print(f"Target set: num_vertices={dem4.num_vertices}")
        
        # 骨骼设置
        bone_names = [f"bone_{i}" for i in range(n_bones)]
        dem4.set_bone_names(*bone_names)
        
        for i in range(n_bones):
            if parents[i] >= 0:
                dem4.set_parent_bone(i, parents[i])
            else:
                dem4.set_parent_bone(i, None)
        
        print(f"Final status: vertices={dem4.num_vertices}, bones={dem4.num_bones}")
        
        # 参数设置
        dem4.num_iterations = 3
        dem4.max_nonzeros_per_vertex = 2
        
        # 计算
        print("计算中...")
        start_time = time.time()
        success = dem4.compute()
        elapsed = time.time() - start_time
        print(f"方法4结果: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
        
        if success:
            print("方法4成功！")
            return True
            
    except Exception as e:
        print(f"方法4异常: {e}")
        traceback.print_exc()
    
    print("\n❌ 所有方法都失败了")
    return False

if __name__ == "__main__":
    success = debug_demBones_data_format()
    if not success:
        print("\n🔍 建议进一步调查：")
        print("1. 检查DemBones版本和依赖")
        print("2. 查看DemBones源码和示例")
        print("3. 尝试不同的数据类型和精度")
        print("4. 检查是否需要额外的初始化步骤")
