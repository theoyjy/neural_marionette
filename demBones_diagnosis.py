#!/usr/bin/env python3
"""
DemBones深度诊断和修复脚本
============================

专门分析和解决DemBones失败的问题
"""

import numpy as np
import time
import threading
import queue
import py_dem_bones as pdb

def test_demBones_basic():
    """最基本的DemBones功能测试"""
    print("=== DemBones基本功能测试 ===")
    
    # 创建最简单的测试数据
    nV = 100  # 100个顶点
    nB = 4    # 4个骨骼
    nF = 2    # 2帧动画
    
    print(f"测试参数: {nV}顶点, {nB}骨骼, {nF}帧")
    
    # Rest pose (100, 3)
    rest_pose = np.random.rand(nV, 3).astype(np.float64)
    
    # Animated poses (200, 3) = (2*100, 3)
    # 需要将rest_pose重复nF次然后添加扰动
    rest_repeated = np.tile(rest_pose, (nF, 1))  # (200, 3)
    animated_poses = rest_repeated + np.random.rand(nF * nV, 3).astype(np.float64) * 0.1
    
    print(f"Rest pose shape: {rest_pose.shape}, dtype: {rest_pose.dtype}")
    print(f"Animated poses shape: {animated_poses.shape}, dtype: {animated_poses.dtype}")
    
    try:
        dem_bones = pdb.DemBones()
        
        # 最基本的参数
        dem_bones.nIters = 5
        dem_bones.nInitIters = 1
        dem_bones.nTransIters = 1
        dem_bones.nWeightsIters = 1
        dem_bones.nnz = 2
        dem_bones.weightsSmooth = 1e-2
        
        # 设置数据
        dem_bones.nV = nV
        dem_bones.nB = nB
        dem_bones.nF = nF
        dem_bones.nS = 1
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        print("开始基本计算...")
        start_time = time.time()
        
        dem_bones.compute()
        
        elapsed = time.time() - start_time
        print(f"✓ 基本测试成功! 耗时: {elapsed:.2f}秒")
        
        # 获取结果
        weights = dem_bones.get_weights()
        print(f"权重矩阵shape: {weights.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
        return False

def test_demBones_data_formats():
    """测试不同的数据格式"""
    print("\n=== DemBones数据格式测试 ===")
    
    nV, nB, nF = 50, 3, 2
    
    # 测试不同的数据类型和布局
    data_configs = [
        ("float32, C-contiguous", np.float32, 'C'),
        ("float64, C-contiguous", np.float64, 'C'), 
        ("float32, F-contiguous", np.float32, 'F'),
        ("float64, F-contiguous", np.float64, 'F'),
    ]
    
    for name, dtype, order in data_configs:
        print(f"\n测试 {name}:")
        
        try:
            # 创建数据
            rest_pose = np.random.rand(nV, 3).astype(dtype, order=order)
            rest_repeated = np.tile(rest_pose, (nF, 1))
            animated_poses = np.asarray(rest_repeated + np.random.rand(nF * nV, 3) * 0.1, dtype=dtype, order=order)
            
            print(f"  Rest pose: {rest_pose.flags}")
            print(f"  Animated: {animated_poses.flags}")
            
            dem_bones = pdb.DemBones()
            
            # 超简单参数
            dem_bones.nIters = 3
            dem_bones.nInitIters = 1
            dem_bones.nTransIters = 1
            dem_bones.nWeightsIters = 1
            dem_bones.nnz = 2
            dem_bones.weightsSmooth = 1e-2
            
            dem_bones.nV = nV
            dem_bones.nB = nB
            dem_bones.nF = nF
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_poses
            
            # 带超时的计算
            result = compute_with_timeout(dem_bones, timeout=30)
            
            if result:
                print(f"  ✓ {name} 成功!")
            else:
                print(f"  ❌ {name} 超时或失败")
                
        except Exception as e:
            print(f"  ❌ {name} 异常: {e}")

def compute_with_timeout(dem_bones, timeout=30):
    """带超时的DemBones计算"""
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def worker():
        try:
            dem_bones.compute()
            weights = dem_bones.get_weights()
            result_queue.put(weights)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        return False
    
    if not exception_queue.empty():
        return False
        
    return not result_queue.empty()

def test_demBones_parameters():
    """测试不同的参数组合"""
    print("\n=== DemBones参数组合测试 ===")
    
    nV, nB, nF = 100, 4, 3
    
    # 创建测试数据
    rest_pose = np.random.rand(nV, 3).astype(np.float64)
    rest_repeated = np.tile(rest_pose, (nF, 1))
    animated_poses = rest_repeated + np.random.rand(nF * nV, 3).astype(np.float64) * 0.1
    
    # 不同的参数配置
    param_configs = [
        {
            'name': '最小配置',
            'nIters': 1, 'nInitIters': 1, 'nTransIters': 1, 
            'nWeightsIters': 1, 'nnz': 1, 'weightsSmooth': 1e-1
        },
        {
            'name': '保守配置',
            'nIters': 3, 'nInitIters': 1, 'nTransIters': 1, 
            'nWeightsIters': 1, 'nnz': 2, 'weightsSmooth': 1e-2
        },
        {
            'name': '标准配置',
            'nIters': 5, 'nInitIters': 2, 'nTransIters': 1, 
            'nWeightsIters': 1, 'nnz': 4, 'weightsSmooth': 1e-3
        },
        {
            'name': '完整配置',
            'nIters': 10, 'nInitIters': 3, 'nTransIters': 2, 
            'nWeightsIters': 2, 'nnz': 6, 'weightsSmooth': 1e-4
        }
    ]
    
    for config in param_configs:
        print(f"\n测试 {config['name']}:")
        print(f"  参数: iters={config['nIters']}, nnz={config['nnz']}, smooth={config['weightsSmooth']}")
        
        try:
            dem_bones = pdb.DemBones()
            
            # 设置参数
            for key, value in config.items():
                if key != 'name':
                    setattr(dem_bones, key, value)
            
            # 设置数据
            dem_bones.nV = nV
            dem_bones.nB = nB
            dem_bones.nF = nF
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_poses
            
            result = compute_with_timeout(dem_bones, timeout=60)
            
            if result:
                print(f"  ✓ {config['name']} 成功!")
            else:
                print(f"  ❌ {config['name']} 超时或失败")
                
        except Exception as e:
            print(f"  ❌ {config['name']} 异常: {e}")

def test_large_data():
    """测试大规模数据"""
    print("\n=== 大规模数据测试 ===")
    
    sizes = [
        (500, 8, 2),
        (1000, 12, 2),
        (2000, 16, 2),
        (4000, 20, 2),
        (6000, 24, 2)
    ]
    
    for nV, nB, nF in sizes:
        print(f"\n测试规模: {nV}顶点, {nB}骨骼, {nF}帧")
        
        try:
            # 创建数据
            rest_pose = np.random.rand(nV, 3).astype(np.float64)
            rest_repeated = np.tile(rest_pose, (nF, 1))
            animated_poses = rest_repeated + np.random.rand(nF * nV, 3).astype(np.float64) * 0.1
            
            dem_bones = pdb.DemBones()
            
            # 保守参数
            dem_bones.nIters = 3
            dem_bones.nInitIters = 1
            dem_bones.nTransIters = 1
            dem_bones.nWeightsIters = 1
            dem_bones.nnz = min(4, nB)
            dem_bones.weightsSmooth = 1e-2
            
            dem_bones.nV = nV
            dem_bones.nB = nB
            dem_bones.nF = nF
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_poses
            
            start_time = time.time()
            result = compute_with_timeout(dem_bones, timeout=120)
            elapsed = time.time() - start_time
            
            if result:
                print(f"  ✓ 成功! 耗时: {elapsed:.2f}秒")
            else:
                print(f"  ❌ 失败或超时 (>{120}秒)")
                break  # 如果这个规模失败了，更大的肯定也会失败
                
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            break

def analyze_demBones_limits():
    """分析DemBones的限制"""
    print("\n=== DemBones限制分析 ===")
    
    print("1. 测试最大顶点数限制...")
    max_vertices = 100
    step = 100
    
    while max_vertices <= 10000:
        try:
            rest_pose = np.random.rand(max_vertices, 3).astype(np.float64)
            rest_repeated = np.tile(rest_pose, (2, 1))
            animated_poses = rest_repeated + np.random.rand(2 * max_vertices, 3).astype(np.float64) * 0.1
            
            dem_bones = pdb.DemBones()
            dem_bones.nIters = 1
            dem_bones.nInitIters = 1
            dem_bones.nTransIters = 1
            dem_bones.nWeightsIters = 1
            dem_bones.nnz = 2
            dem_bones.weightsSmooth = 1e-2
            
            dem_bones.nV = max_vertices
            dem_bones.nB = 4
            dem_bones.nF = 2
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(2, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_poses
            
            result = compute_with_timeout(dem_bones, timeout=30)
            
            if result:
                print(f"  ✓ {max_vertices} 顶点: 成功")
                max_vertices += step
            else:
                print(f"  ❌ {max_vertices} 顶点: 失败")
                print(f"  建议最大顶点数: {max_vertices - step}")
                break
                
        except Exception as e:
            print(f"  ❌ {max_vertices} 顶点: 异常 {e}")
            print(f"  建议最大顶点数: {max_vertices - step}")
            break

def main():
    """运行所有诊断测试"""
    print("🔍 DemBones深度诊断开始...")
    print("=" * 50)
    
    # 基本功能测试
    basic_success = test_demBones_basic()
    
    if not basic_success:
        print("\n❌ 基本功能测试失败，DemBones可能有根本性问题")
        return
    
    # 数据格式测试
    test_demBones_data_formats()
    
    # 参数组合测试  
    test_demBones_parameters()
    
    # 大规模数据测试
    test_large_data()
    
    # 限制分析
    analyze_demBones_limits()
    
    print("\n" + "=" * 50)
    print("🎯 DemBones诊断完成!")
    print("\n建议:")
    print("1. 使用float64类型和C-contiguous内存布局")
    print("2. 保持顶点数 < 5000，骨骼数 < 20")
    print("3. 使用保守的参数设置")
    print("4. 确保数据数值范围合理")

if __name__ == "__main__":
    main()
