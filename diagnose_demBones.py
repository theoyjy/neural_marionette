#!/usr/bin/env python3
"""
深度诊断DemBonesExtWrapper的数据需求
"""

import numpy as np
import py_dem_bones as pdb
import time

def diagnose_data_format():
    """诊断DemBones数据格式问题"""
    print("🔬 DemBonesExtWrapper 数据格式诊断")
    print("=" * 50)
    
    # 超简单测试：只有1个顶点，1根骨骼
    print("📋 测试1：最简单情况 - 1顶点1骨骼")
    try:
        dem1 = pdb.DemBonesExtWrapper()
        
        # 1个顶点，1帧，静止不动
        vertices_1v = np.array([[[0, 0, 0]]], dtype=np.float64)  # (1, 1, 3)
        rest_pose_1v = vertices_1v[0].T  # (3, 1)
        
        print(f"Rest pose shape: {rest_pose_1v.shape}")
        
        dem1.num_vertices = 1
        dem1.num_bones = 1
        dem1.num_iterations = 1
        dem1.max_nonzeros_per_vertex = 1
        
        dem1.set_rest_pose(rest_pose_1v)
        print(f"After set_rest_pose: num_vertices={dem1.num_vertices}")
        
        # 不设置animated poses，只有rest pose
        # 设置骨骼
        dem1.set_bone_names("root")
        dem1.set_parent_bone(0, None)
        
        print("尝试计算...")
        success = dem1.compute()
        print(f"结果: {'✅' if success else '❌'}")
        
        if not success:
            print("连最简单的情况都失败了")
        
    except Exception as e:
        print(f"最简单测试失败: {e}")
    
    # 测试2：检查是否需要设置animated_poses
    print("\n📋 测试2：检查animated_poses设置")
    try:
        dem2 = pdb.DemBonesExtWrapper()
        
        # 2个顶点，2帧
        vertices_2v = np.array([
            [[0, 0, 0], [1, 0, 0]],
            [[0.1, 0, 0], [0.9, 0, 0]]
        ], dtype=np.float64)
        
        rest_pose_2v = vertices_2v[0].T  # (3, 2)
        print(f"Rest pose shape: {rest_pose_2v.shape}")
        
        dem2.num_vertices = 2
        dem2.num_bones = 1
        dem2.num_iterations = 1
        dem2.max_nonzeros_per_vertex = 1
        
        dem2.set_rest_pose(rest_pose_2v)
        print(f"After set_rest_pose: num_vertices={dem2.num_vertices}")
        
        # 检查是否有animated_poses属性
        if hasattr(dem2, 'animated_poses'):
            print("✅ 有animated_poses属性")
            # 尝试设置animated_poses
            anim_poses_2v = vertices_2v.transpose(0,2,1).reshape(3, -1)  # (3, 4)
            dem2.animated_poses = anim_poses_2v
            print(f"设置animated_poses后: num_vertices={dem2.num_vertices}")
        else:
            print("❌ 没有animated_poses属性")
        
        # 设置骨骼
        dem2.set_bone_names("root")
        dem2.set_parent_bone(0, None)
        
        print("尝试计算...")
        success = dem2.compute()
        print(f"结果: {'✅' if success else '❌'}")
        
    except Exception as e:
        print(f"animated_poses测试失败: {e}")
    
    # 测试3：尝试逐帧设置target
    print("\n📋 测试3：逐帧设置target")
    try:
        dem3 = pdb.DemBonesExtWrapper()
        
        vertices_test = np.array([
            [[0, 0, 0], [1, 0, 0]],
            [[0.1, 0, 0], [0.9, 0, 0]]
        ], dtype=np.float64)
        
        rest_pose = vertices_test[0].T  # (3, 2)
        
        dem3.num_vertices = 2
        dem3.num_bones = 1
        dem3.num_iterations = 1
        dem3.max_nonzeros_per_vertex = 1
        
        dem3.set_rest_pose(rest_pose)
        print(f"After set_rest_pose: num_vertices={dem3.num_vertices}")
        
        # 逐帧添加target
        for i, frame in enumerate(vertices_test):
            frame_data = frame.T  # (3, 2)
            target_name = f"frame_{i}"
            dem3.set_target_vertices(target_name, frame_data)
            print(f"Added {target_name}: num_targets={dem3.num_targets}, num_vertices={dem3.num_vertices}")
        
        # 设置骨骼
        dem3.set_bone_names("root")
        dem3.set_parent_bone(0, None)
        
        print("尝试计算...")
        success = dem3.compute()
        print(f"结果: {'✅' if success else '❌'}")
        
    except Exception as e:
        print(f"逐帧设置测试失败: {e}")
    
    # 测试4：检查num_vertices是否应该手动设置
    print("\n📋 测试4：不手动设置num_vertices")
    try:
        dem4 = pdb.DemBonesExtWrapper()
        
        vertices_test = np.array([
            [[0, 0, 0], [1, 0, 0]]
        ], dtype=np.float64)
        
        rest_pose = vertices_test[0].T  # (3, 2)
        
        # 不设置num_vertices，让它自动推断
        dem4.num_bones = 1
        dem4.num_iterations = 1
        dem4.max_nonzeros_per_vertex = 1
        
        dem4.set_rest_pose(rest_pose)
        print(f"Auto-detected num_vertices: {dem4.num_vertices}")
        
        # 设置骨骼
        dem4.set_bone_names("root")
        dem4.set_parent_bone(0, None)
        
        print("尝试计算...")
        success = dem4.compute()
        print(f"结果: {'✅' if success else '❌'}")
        
        if success:
            print("🎉 成功！问题在于不应该手动设置num_vertices")
            return True
        
    except Exception as e:
        print(f"自动推断测试失败: {e}")
    
    return False

if __name__ == "__main__":
    success = diagnose_data_format()
    if success:
        print("\n✅ 找到了解决方案！")
    else:
        print("\n❌ 仍需进一步调试")
