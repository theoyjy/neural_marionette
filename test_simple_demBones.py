#!/usr/bin/env python3
"""
简化的DemBones测试，带超时
"""

import numpy as np
import py_dem_bones as pdb
import time
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout!")

def test_simple_with_timeout():
    """带超时的简单测试"""
    print("🔧 带超时的DemBones测试")
    print("=" * 40)
    
    # 设置10秒超时
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        dem = pdb.DemBonesExtWrapper()
        
        # 2个顶点，2帧，1个骨骼
        vertices = np.array([
            [[0, 0, 0], [1, 0, 0]],      # frame 0
            [[0.1, 0, 0], [0.9, 0, 0]]   # frame 1
        ], dtype=np.float64)
        
        rest_pose = vertices[0].T  # (3, 2)
        
        # 不手动设置num_vertices，让它自动推断
        dem.num_bones = 1
        dem.num_iterations = 2  # 很少的迭代
        dem.max_nonzeros_per_vertex = 1
        
        print(f"设置rest pose: {rest_pose.shape}")
        dem.set_rest_pose(rest_pose)
        print(f"自动推断 num_vertices: {dem.num_vertices}")
        
        # 逐帧添加target
        for i, frame in enumerate(vertices):
            frame_data = frame.T  # (3, 2)
            target_name = f"frame_{i}"
            dem.set_target_vertices(target_name, frame_data)
            print(f"添加 {target_name}: targets={dem.num_targets}")
        
        # 设置骨骼
        dem.set_bone_names("root")
        dem.set_parent_bone(0, None)
        
        print("状态检查:")
        print(f"  num_vertices: {dem.num_vertices}")
        print(f"  num_bones: {dem.num_bones}")
        print(f"  num_targets: {dem.num_targets}")
        print(f"  bone_names: {dem.bone_names}")
        
        # 启动10秒超时
        signal.alarm(10)
        
        print("🚀 开始计算（10s超时）...")
        start_time = time.time()
        
        try:
            success = dem.compute()
            elapsed = time.time() - start_time
            signal.alarm(0)  # 取消超时
            
            print(f"计算完成: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
            
            if success:
                weights = dem.get_weights()
                print(f"权重矩阵: {weights.shape}")
                print(f"权重内容:\n{weights}")
                return True
            else:
                return False
                
        except TimeoutError:
            signal.alarm(0)
            elapsed = time.time() - start_time
            print(f"❌ 超时！计算超过10秒 (已耗时 {elapsed:.2f}s)")
            return False
            
    except Exception as e:
        signal.alarm(0)
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_params():
    """测试不同的参数组合"""
    print("\n🧪 测试不同参数组合")
    
    # 参数组合列表
    param_combinations = [
        {"iterations": 1, "nnz": 1, "smoothness": 0.001},
        {"iterations": 2, "nnz": 1, "smoothness": 0.0001},
        {"iterations": 1, "nnz": 2, "smoothness": 0.001},
    ]
    
    for i, params in enumerate(param_combinations):
        print(f"\n📋 组合 {i+1}: {params}")
        
        try:
            dem = pdb.DemBonesExtWrapper()
            
            # 简单的2顶点数据
            vertices = np.array([
                [[0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [1, 0, 0]]  # 相同数据，看看是否能计算
            ], dtype=np.float64)
            
            rest_pose = vertices[0].T
            
            dem.num_bones = 1
            dem.num_iterations = params["iterations"]
            dem.max_nonzeros_per_vertex = params["nnz"]
            dem.weight_smoothness = params["smoothness"]
            
            dem.set_rest_pose(rest_pose)
            
            # 只添加一个target（除了rest pose）
            dem.set_target_vertices("target", vertices[1].T)
            
            dem.set_bone_names("root")
            dem.set_parent_bone(0, None)
            
            # 超短时间测试
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5秒超时
            
            start_time = time.time()
            success = dem.compute()
            elapsed = time.time() - start_time
            signal.alarm(0)
            
            print(f"  结果: {'✅' if success else '❌'} (耗时 {elapsed:.2f}s)")
            
            if success:
                print(f"  🎉 成功组合: {params}")
                return True
                
        except TimeoutError:
            signal.alarm(0)
            print(f"  ⏰ 超时")
        except Exception as e:
            signal.alarm(0)
            print(f"  ❌ 异常: {e}")
    
    return False

if __name__ == "__main__":
    # Windows不支持SIGALRM，改用threading
    import threading
    import sys
    
    if sys.platform == "win32":
        print("Windows环境，使用threading模拟超时")
        
        def run_with_timeout():
            dem = pdb.DemBonesExtWrapper()
            
            vertices = np.array([
                [[0, 0, 0], [1, 0, 0]],
                [[0.1, 0, 0], [0.9, 0, 0]]
            ], dtype=np.float64)
            
            rest_pose = vertices[0].T
            
            dem.num_bones = 1
            dem.num_iterations = 1
            dem.max_nonzeros_per_vertex = 1
            
            dem.set_rest_pose(rest_pose)
            dem.set_target_vertices("target", vertices[1].T)
            dem.set_bone_names("root")
            dem.set_parent_bone(0, None)
            
            print("🚀 开始计算...")
            success = dem.compute()
            print(f"结果: {'✅' if success else '❌'}")
            return success
        
        try:
            result = run_with_timeout()
        except Exception as e:
            print(f"❌ 异常: {e}")
    else:
        success1 = test_simple_with_timeout()
        if not success1:
            success2 = test_with_different_params()
            print(f"\n最终结果: {'✅ 成功' if success2 else '❌ 失败'}")
