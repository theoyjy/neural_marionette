#!/usr/bin/env python3
"""
DemBones API 探索
检查可用的方法和属性
"""

import py_dem_bones
import numpy as np

def explore_demBones_api():
    """探索DemBones的可用API"""
    print("🔍 DemBones API 探索")
    print("=" * 50)
    
    # 创建实例
    dem = py_dem_bones.DemBones()
    
    # 检查所有属性和方法
    print("📋 可用属性和方法:")
    all_attrs = [attr for attr in dir(dem) if not attr.startswith('_')]
    for attr in sorted(all_attrs):
        try:
            value = getattr(dem, attr)
            if callable(value):
                print(f"  方法: {attr}()")
            else:
                print(f"  属性: {attr} = {value}")
        except Exception as e:
            print(f"  属性: {attr} (无法访问: {e})")
    
    print(f"\n总共找到 {len(all_attrs)} 个公开属性/方法")
    
    # 测试基本设置
    print("\n🧪 测试基本设置:")
    
    # 创建简单测试数据
    vertices = np.array([
        [[-1, 0, 0], [1, 0, 0]],    # frame 0: 2个顶点
        [[-0.8, 0, 0], [0.8, 0, 0]] # frame 1: 2个顶点
    ], dtype=np.float64)
    
    rest_pose = vertices[0].T  # (3, 2)
    print(f"Rest pose shape: {rest_pose.shape}")
    
    try:
        dem.set_rest_pose(rest_pose)
        print("✅ set_rest_pose 成功")
    except Exception as e:
        print(f"❌ set_rest_pose 失败: {e}")
    
    # 检查状态
    print("\n📊 设置后的状态:")
    status_attrs = ['num_vertices', 'num_bones', 'num_frames', 'num_targets']
    for attr in status_attrs:
        try:
            value = getattr(dem, attr)
            print(f"  {attr}: {value}")
        except:
            print(f"  {attr}: 不可用")
    
    # 测试骨骼设置
    print("\n🦴 测试骨骼设置:")
    try:
        dem.set_bone_names("bone_0", "bone_1")
        print("✅ set_bone_names 成功")
        
        dem.set_parent_bone(0, None)  # 根骨骼
        dem.set_parent_bone(1, 0)     # 子骨骼
        print("✅ set_parent_bone 成功")
        
    except Exception as e:
        print(f"❌ 骨骼设置失败: {e}")
    
    # 测试target设置
    print("\n🎯 测试target设置:")
    try:
        animated_data = vertices.transpose(0,2,1).reshape(3, -1)  # (3, 4)
        print(f"Animated data shape: {animated_data.shape}")
        
        dem.set_target_vertices("animated", animated_data)
        print("✅ set_target_vertices 成功")
        
    except Exception as e:
        print(f"❌ set_target_vertices 失败: {e}")
    
    # 最终状态
    print("\n📊 最终状态:")
    for attr in status_attrs:
        try:
            value = getattr(dem, attr)
            print(f"  {attr}: {value}")
        except:
            print(f"  {attr}: 不可用")
    
    # 尝试计算
    print("\n⚡ 测试计算:")
    try:
        # 设置基本参数
        if hasattr(dem, 'num_iterations'):
            dem.num_iterations = 1
            print("  设置 num_iterations = 1")
        
        if hasattr(dem, 'max_nonzeros_per_vertex'):
            dem.max_nonzeros_per_vertex = 2
            print("  设置 max_nonzeros_per_vertex = 2")
        
        # 计算
        success = dem.compute()
        print(f"  计算结果: {'✅ 成功' if success else '❌ 失败'}")
        
        if success:
            print("🎉 DemBones计算成功！")
            
            # 获取结果
            if hasattr(dem, 'get_weights'):
                weights = dem.get_weights()
                print(f"  权重矩阵shape: {weights.shape}")
            
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_demBones_api()
