#!/usr/bin/env python3
"""
极简DemBones测试 - 找出最基本的工作配置
"""

import numpy as np
import py_dem_bones as pdb
import time

def minimal_test():
    """最小可能的DemBones测试"""
    print("=== 极简DemBones测试 ===")
    
    # 最小数据: 4个顶点，2个骨骼，1帧动画
    nV = 4   # 4个顶点
    nB = 2   # 2个骨骼
    nF = 1   # 1帧动画
    
    print(f"极简配置: {nV}顶点, {nB}骨骼, {nF}帧")
    
    # Rest pose: 4个顶点
    rest_pose = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ], dtype=np.float64)
    
    # 1帧动画: 稍微移动
    animated_poses = np.array([
        [0.1, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [0.0, 1.1, 0.0], 
        [1.0, 1.1, 0.0]
    ], dtype=np.float64)
    
    print(f"Rest pose shape: {rest_pose.shape}")
    print(f"Animated poses shape: {animated_poses.shape}")
    
    try:
        dem_bones = pdb.DemBones()
        
        # 绝对最小参数
        dem_bones.nIters = 1
        dem_bones.nInitIters = 1
        dem_bones.nTransIters = 1
        dem_bones.nWeightsIters = 1
        dem_bones.nnz = 1
        dem_bones.weightsSmooth = 0.1
        
        # 设置数据
        dem_bones.nV = nV
        dem_bones.nB = nB
        dem_bones.nF = nF
        dem_bones.nS = 1
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        print("开始极简计算...")
        start = time.time()
        
        dem_bones.compute()
        
        elapsed = time.time() - start
        print(f"✓ 极简测试成功! 耗时: {elapsed:.3f}秒")
        
        weights = dem_bones.get_weights()
        print(f"权重矩阵: {weights.shape}")
        print(f"权重内容:\n{weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ 极简测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_demBones_version():
    """检查DemBones版本和基本信息"""
    print("=== DemBones版本信息 ===")
    
    try:
        dem_bones = pdb.DemBones()
        print(f"DemBones对象创建成功")
        
        # 尝试访问一些属性
        print(f"默认参数:")
        print(f"  nIters: {getattr(dem_bones, 'nIters', 'N/A')}")
        print(f"  nnz: {getattr(dem_bones, 'nnz', 'N/A')}")
        print(f"  weightsSmooth: {getattr(dem_bones, 'weightsSmooth', 'N/A')}")
        
    except Exception as e:
        print(f"❌ DemBones对象创建失败: {e}")

def test_different_sizes():
    """测试不同大小的数据"""
    print("\n=== 不同规模测试 ===")
    
    test_cases = [
        (3, 2, 1),   # 3顶点，2骨骼，1帧
        (4, 2, 1),   # 4顶点，2骨骼，1帧
        (5, 2, 1),   # 5顶点，2骨骼，1帧
        (10, 3, 1),  # 10顶点，3骨骼，1帧
        (10, 3, 2),  # 10顶点，3骨骼，2帧
    ]
    
    for nV, nB, nF in test_cases:
        print(f"\n测试: {nV}顶点, {nB}骨骼, {nF}帧")
        
        try:
            # 简单数据
            rest_pose = np.random.rand(nV, 3).astype(np.float64) * 2 - 1  # [-1, 1]范围
            
            if nF == 1:
                animated_poses = rest_pose + (np.random.rand(nV, 3).astype(np.float64) - 0.5) * 0.2
            else:
                rest_repeated = np.tile(rest_pose, (nF, 1))
                animated_poses = rest_repeated + (np.random.rand(nF * nV, 3).astype(np.float64) - 0.5) * 0.2
            
            dem_bones = pdb.DemBones()
            
            # 最简参数
            dem_bones.nIters = 1
            dem_bones.nInitIters = 1
            dem_bones.nTransIters = 1
            dem_bones.nWeightsIters = 1
            dem_bones.nnz = min(2, nB)
            dem_bones.weightsSmooth = 0.1
            
            dem_bones.nV = nV
            dem_bones.nB = nB
            dem_bones.nF = nF
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_poses
            
            start = time.time()
            dem_bones.compute()
            elapsed = time.time() - start
            
            weights = dem_bones.get_weights()
            print(f"  ✓ 成功! 耗时: {elapsed:.3f}秒, 权重: {weights.shape}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            break  # 如果这个失败了，不用测试更大的

def main():
    check_demBones_version()
    
    if minimal_test():
        test_different_sizes()
    else:
        print("\n❌ 连最基本的测试都失败了，DemBones可能有严重问题")

if __name__ == "__main__":
    main()
