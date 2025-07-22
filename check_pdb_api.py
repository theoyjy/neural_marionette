#!/usr/bin/env python3
"""
检查py_dem_bones包的完整API
"""

import py_dem_bones as pdb
import numpy as np

def check_py_dem_bones_api():
    """检查py_dem_bones的所有可用类和函数"""
    print("🔍 py_dem_bones 包API检查")
    print("=" * 50)
    
    # 检查包级别的属性
    print("📦 包级别的类和函数:")
    all_attrs = [attr for attr in dir(pdb) if not attr.startswith('_')]
    for attr in sorted(all_attrs):
        obj = getattr(pdb, attr)
        if hasattr(obj, '__call__'):
            if hasattr(obj, '__doc__') and obj.__doc__:
                print(f"  类/函数: {attr} - {obj.__doc__.split('.')[0] if obj.__doc__ else 'No doc'}")
            else:
                print(f"  类/函数: {attr}")
        else:
            print(f"  其他: {attr} = {obj}")
    
    # 检查是否有DemBonesExtWrapper
    print(f"\n🎯 检查DemBonesExtWrapper:")
    if hasattr(pdb, 'DemBonesExtWrapper'):
        print("✅ DemBonesExtWrapper 存在!")
        
        # 创建实例并检查其API
        try:
            wrapper = pdb.DemBonesExtWrapper()
            print("✅ 成功创建DemBonesExtWrapper实例")
            
            # 检查wrapper的属性和方法
            print("\n📋 DemBonesExtWrapper 的属性和方法:")
            wrapper_attrs = [attr for attr in dir(wrapper) if not attr.startswith('_')]
            for attr in sorted(wrapper_attrs):
                try:
                    value = getattr(wrapper, attr)
                    if callable(value):
                        print(f"  方法: {attr}()")
                    else:
                        print(f"  属性: {attr} = {value}")
                except Exception as e:
                    print(f"  属性: {attr} (无法访问: {e})")
                    
        except Exception as e:
            print(f"❌ 创建DemBonesExtWrapper失败: {e}")
    else:
        print("❌ DemBonesExtWrapper 不存在")
        
        # 检查其他可能的包装器
        potential_wrappers = ['DemBonesWrapper', 'ExtendedDemBones', 'DemBonesExt']
        for wrapper_name in potential_wrappers:
            if hasattr(pdb, wrapper_name):
                print(f"✅ 找到替代包装器: {wrapper_name}")
            else:
                print(f"❌ {wrapper_name} 不存在")
    
    # 测试基本的DemBones类
    print(f"\n🧪 测试基本DemBones类:")
    try:
        basic_dem = pdb.DemBones()
        print("✅ 成功创建基本DemBones实例")
        
        # 尝试使用真实API
        print("\n📋 尝试正确的API调用:")
        
        # 创建测试数据
        vertices = np.array([
            [[-1, 0, 0], [1, 0, 0]],    # frame 0
            [[-0.8, 0, 0], [0.8, 0, 0]] # frame 1
        ], dtype=np.float64)
        
        parents = np.array([-1, 0], dtype=np.int32)
        
        # 按照真实API设置
        rest_pose = vertices[0].T  # (3, 2)
        anim_poses = vertices.transpose(0,2,1).reshape(3, -1)  # (3, 4)
        
        print(f"Rest pose: {rest_pose.shape}")
        print(f"Animated poses: {anim_poses.shape}")
        
        # 设置基本属性
        basic_dem.nV = vertices.shape[1]  # 顶点数
        basic_dem.nB = len(parents)       # 骨骼数
        basic_dem.nF = vertices.shape[0]  # 帧数
        basic_dem.nS = 1                  # 主题数
        
        print(f"设置: nV={basic_dem.nV}, nB={basic_dem.nB}, nF={basic_dem.nF}, nS={basic_dem.nS}")
        
        # 设置数据
        basic_dem.set_rest_pose(rest_pose)
        basic_dem.set_animated_poses(anim_poses)
        
        print("✅ 数据设置成功")
        
        # 设置参数
        basic_dem.nIters = 3
        basic_dem.nnz = 2
        basic_dem.weightsSmooth = 0.001
        
        print("✅ 参数设置成功")
        
        # 尝试计算
        print("🚀 尝试计算...")
        success = basic_dem.compute()
        print(f"计算结果: {'✅ 成功' if success else '❌ 失败'}")
        
        if success:
            weights = basic_dem.get_weights()
            print(f"权重矩阵shape: {weights.shape}")
            
    except Exception as e:
        print(f"❌ 基本DemBones测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_py_dem_bones_api()
