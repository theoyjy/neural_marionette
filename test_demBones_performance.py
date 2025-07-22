#!/usr/bin/env python3
"""
DemBones性能测试 - 找到最佳顶点数配置
"""

import numpy as np
import time
import py_dem_bones as pdb

def test_demBones_performance():
    """测试不同顶点数下的DemBones性能"""
    
    # 测试配置
    test_configs = [
        {'vertices': 50, 'name': '50顶点'},
        {'vertices': 100, 'name': '100顶点'},
        {'vertices': 200, 'name': '200顶点'},
        {'vertices': 500, 'name': '500顶点'},
        {'vertices': 1000, 'name': '1000顶点'},
    ]
    
    # 固定参数
    K = 10  # 骨骼数
    F = 2   # 动画帧数
    
    # 创建简单的父子关系
    parents = [i-1 if i > 0 else -1 for i in range(K)]
    
    print("🧪 DemBones性能测试开始...")
    print(f"骨骼数: {K}, 动画帧数: {F}")
    print("-" * 50)
    
    results = []
    
    for config in test_configs:
        N = config['vertices']
        name = config['name']
        
        print(f"\n测试: {name}")
        
        # 创建测试数据
        rest_pose = np.random.rand(N, 3).astype(np.float32)
        animated_pose = rest_pose + np.random.rand(N, 3) * 0.1  # 小幅变形
        
        # 测试DemBones
        start_time = time.time()
        success = False
        
        try:
            dem_bones = pdb.DemBones()
            
            # 最简配置
            dem_bones.nIters = 5
            dem_bones.nInitIters = 1
            dem_bones.nTransIters = 1
            dem_bones.nWeightsIters = 1
            dem_bones.nnz = 3
            dem_bones.weightsSmooth = 1e-3
            
            # 设置数据
            dem_bones.nV = N
            dem_bones.nB = K
            dem_bones.nF = 1
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(1, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_pose
            
            print(f"  开始计算...")
            dem_bones.compute()
            
            elapsed = time.time() - start_time
            print(f"  ✅ 成功! 耗时: {elapsed:.2f}秒")
            
            # 验证结果
            weights = dem_bones.get_weights()
            print(f"  权重矩阵: {weights.shape}")
            
            results.append({
                'vertices': N,
                'time': elapsed,
                'success': True,
                'rate': N / elapsed
            })
            
            success = True
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ 失败! 耗时: {elapsed:.2f}秒, 错误: {e}")
            results.append({
                'vertices': N,
                'time': elapsed,
                'success': False,
                'rate': 0
            })
        
        # 如果耗时超过30秒，停止测试更大的配置
        if elapsed > 30:
            print(f"  ⚠️ 耗时过长，停止后续测试")
            break
    
    # 输出结果总结
    print("\n" + "="*50)
    print("📊 性能测试结果总结:")
    print("-" * 50)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print("成功的配置:")
        for r in successful_results:
            print(f"  {r['vertices']:4d} 顶点: {r['time']:6.2f}秒 ({r['rate']:6.1f} 顶点/秒)")
        
        # 推荐配置
        best_rate = max(successful_results, key=lambda x: x['rate'])
        max_vertices = max(r['vertices'] for r in successful_results if r['time'] <= 10)
        
        print(f"\n🎯 推荐配置:")
        print(f"  最佳性能: {best_rate['vertices']} 顶点 ({best_rate['rate']:.1f} 顶点/秒)")
        print(f"  10秒内最大: {max_vertices} 顶点")
        
        return max_vertices
    else:
        print("❌ 没有成功的配置!")
        return 50  # 回退到已知安全值


if __name__ == "__main__":
    optimal_vertices = test_demBones_performance()
    print(f"\n🚀 建议的最大顶点数: {optimal_vertices}")
