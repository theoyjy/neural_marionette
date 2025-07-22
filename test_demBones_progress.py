#!/usr/bin/env python3
"""
带进度监控的DemBones测试
"""

import numpy as np
import py_dem_bones as pdb
import time

def test_with_progress_callback():
    """使用进度回调监控DemBones计算"""
    print("🔧 带进度监控的DemBones测试")
    print("=" * 50)
    
    try:
        dem = pdb.DemBonesExtWrapper()
        
        # 最简单的测试数据
        vertices = np.array([
            [[0, 0, 0]],      # frame 0: 1个顶点
            [[0.1, 0, 0]]     # frame 1: 轻微移动
        ], dtype=np.float64)
        
        rest_pose = vertices[0].T  # (3, 1)
        
        # 最保守的设置
        dem.num_iterations = 1  # 只做1次迭代
        dem.max_influences = 1  # 每个顶点最多1个骨骼影响
        dem.weight_smoothness = 0.0  # 不要平滑
        
        dem.set_rest_pose(rest_pose)
        dem.set_target_vertices("target", vertices[1].T)
        
        dem.set_bone_names("root")
        dem.set_parent_bone(0, None)
        
        print(f"设置完成: {dem.num_vertices}v, {dem.num_bones}b, {dem.num_targets}t")
        print(f"参数: iterations={dem.num_iterations}, max_influences={dem.max_influences}")
        
        # 进度回调函数
        progress_info = {"last_update": time.time(), "progress": 0.0}
        
        def progress_callback(progress):
            current_time = time.time()
            if current_time - progress_info["last_update"] > 1.0:  # 每秒更新一次
                print(f"  进度: {progress*100:.1f}%")
                progress_info["last_update"] = current_time
                progress_info["progress"] = progress
        
        print("🚀 开始计算（带进度监控）...")
        start_time = time.time()
        
        try:
            success = dem.compute(callback=progress_callback)
            elapsed = time.time() - start_time
            
            print(f"\n计算完成: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
            
            if success:
                weights = dem.get_weights()
                print(f"权重矩阵: {weights.shape}")
                print(f"权重内容:\n{weights}")
                return True
            else:
                print("计算返回失败状态")
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 计算异常 (耗时 {elapsed:.2f}s): {e}")
            return False
            
    except Exception as e:
        print(f"❌ 设置异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_data_sizes():
    """测试不同数据规模，找到可工作的配置"""
    print("\n🧪 测试不同数据规模")
    print("=" * 30)
    
    test_cases = [
        {"vertices": 1, "bones": 1, "frames": 2},
        {"vertices": 2, "bones": 1, "frames": 2},
        {"vertices": 2, "bones": 2, "frames": 2},
        {"vertices": 4, "bones": 2, "frames": 3},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📋 测试 {i+1}: {case['vertices']}v, {case['bones']}b, {case['frames']}f")
        
        try:
            dem = pdb.DemBonesExtWrapper()
            
            # 生成测试数据
            n_v, n_b, n_f = case['vertices'], case['bones'], case['frames']
            
            # 创建简单的变形序列
            vertices = []
            for f in range(n_f):
                frame = []
                for v in range(n_v):
                    # 简单的线性变形
                    factor = f * 0.1
                    pos = [v * 1.0 + factor, 0, 0]
                    frame.append(pos)
                vertices.append(frame)
            
            vertices = np.array(vertices, dtype=np.float64)
            rest_pose = vertices[0].T
            
            # 最小参数
            dem.num_iterations = 1
            dem.max_influences = min(2, n_b)
            dem.weight_smoothness = 0.0
            
            dem.set_rest_pose(rest_pose)
            
            # 添加目标帧
            for f in range(1, n_f):
                dem.set_target_vertices(f"frame_{f}", vertices[f].T)
            
            # 设置骨骼
            bone_names = [f"bone_{j}" for j in range(n_b)]
            dem.set_bone_names(*bone_names)
            
            # 简单的链式骨骼结构
            for j in range(n_b):
                if j == 0:
                    dem.set_parent_bone(j, None)  # 根骨骼
                else:
                    dem.set_parent_bone(j, j-1)   # 父骨骼是前一个
            
            # 快速测试（5秒超时）
            print(f"  开始计算...")
            
            progress_count = [0]
            
            def quick_callback(progress):
                progress_count[0] += 1
                if progress_count[0] % 10 == 0:  # 每10次更新显示一次
                    print(f"    进度: {progress*100:.0f}%")
            
            start_time = time.time()
            
            # 使用线程进行超时控制
            import threading
            result = [None]
            exception = [None]
            
            def compute_thread():
                try:
                    result[0] = dem.compute(callback=quick_callback)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=compute_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # 10秒超时
            
            elapsed = time.time() - start_time
            
            if thread.is_alive():
                print(f"  ⏰ 超时 (10s)")
            elif exception[0]:
                print(f"  ❌ 异常: {exception[0]}")
            elif result[0]:
                print(f"  ✅ 成功 (耗时 {elapsed:.2f}s)")
                
                weights = dem.get_weights()
                print(f"  权重矩阵: {weights.shape}")
                
                # 这个配置成功了，返回它
                return case
            else:
                print(f"  ❌ 失败 (耗时 {elapsed:.2f}s)")
                
        except Exception as e:
            print(f"  ❌ 设置异常: {e}")
    
    return None

if __name__ == "__main__":
    print("开始带监控的DemBones测试...")
    
    # 先测试最简单的情况
    basic_success = test_with_progress_callback()
    
    if basic_success:
        print("\n✅ 基本测试成功！")
    else:
        print("\n基本测试失败，尝试找到可工作的配置...")
        working_config = test_different_data_sizes()
        
        if working_config:
            print(f"\n✅ 找到可工作的配置: {working_config}")
            print("现在可以基于这个配置扩展到完整数据")
        else:
            print("\n❌ 没有找到任何可工作的配置")
            print("DemBones可能存在根本性问题或需要特殊设置")
