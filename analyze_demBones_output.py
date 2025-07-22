#!/usr/bin/env python3
"""
DemBones输出格式分析
"""

import numpy as np
import py_dem_bones as pdb

def analyze_demBones_output():
    """深入分析DemBones的输出格式"""
    print("=== DemBones输出格式分析 ===")
    
    nV = 6   # 6个顶点
    nB = 3   # 3个骨骼
    nF = 2   # 2帧动画
    
    print(f"测试参数: {nV}顶点, {nB}骨骼, {nF}帧")
    
    # Rest pose
    rest_pose = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0]
    ], dtype=np.float64)
    
    # 2帧动画
    frame1 = rest_pose + 0.1
    frame2 = rest_pose + 0.2
    animated_poses = np.vstack([frame1, frame2])  # (12, 3)
    
    print(f"Rest pose shape: {rest_pose.shape}")
    print(f"Animated poses shape: {animated_poses.shape}")
    
    try:
        dem_bones = pdb.DemBones()
        
        # 设置参数
        dem_bones.nIters = 5
        dem_bones.nInitIters = 2
        dem_bones.nTransIters = 1
        dem_bones.nWeightsIters = 1
        dem_bones.nnz = 3
        dem_bones.weightsSmooth = 1e-3
        
        # 设置数据
        dem_bones.nV = nV
        dem_bones.nB = nB
        dem_bones.nF = nF
        dem_bones.nS = 1
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        print("\n开始计算...")
        dem_bones.compute()
        
        # 获取结果
        weights = dem_bones.get_weights()
        transformations = dem_bones.get_transformations()
        
        print(f"\n✓ 计算成功!")
        print(f"权重矩阵shape: {weights.shape}")
        print(f"权重矩阵内容:\n{weights}")
        print(f"权重数据类型: {weights.dtype}")
        
        print(f"\n变换矩阵shape: {transformations.shape}")
        print(f"变换数据类型: {transformations.dtype}")
        
        # 分析权重矩阵
        if weights.shape[0] == nB:
            print(f"\n✓ 权重矩阵格式: (nB={nB}, nV={nV}) - 每行是一个骨骼的权重")
        elif weights.shape[1] == nB:
            print(f"\n✓ 权重矩阵格式: (nV={nV}, nB={nB}) - 每行是一个顶点的权重")
        else:
            print(f"\n? 未知的权重矩阵格式: {weights.shape}")
        
        # 检查权重归一化
        if weights.shape[0] == nB:
            row_sums = weights.sum(axis=0)
            print(f"每个顶点的权重和: {row_sums}")
        else:
            row_sums = weights.sum(axis=1)
            print(f"每个顶点的权重和: {row_sums}")
        
        return weights, transformations
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_realistic_size():
    """测试更现实的数据规模"""
    print("\n=== 现实规模测试 ===")
    
    nV = 1000  # 1000个顶点
    nB = 12    # 12个骨骼 
    nF = 3     # 3帧动画
    
    print(f"现实测试: {nV}顶点, {nB}骨骼, {nF}帧")
    
    try:
        # 创建数据
        rest_pose = np.random.rand(nV, 3).astype(np.float64)
        rest_repeated = np.tile(rest_pose, (nF, 1))
        animated_poses = rest_repeated + np.random.rand(nF * nV, 3).astype(np.float64) * 0.1
        
        dem_bones = pdb.DemBones()
        
        # 中等参数
        dem_bones.nIters = 10
        dem_bones.nInitIters = 3
        dem_bones.nTransIters = 2
        dem_bones.nWeightsIters = 1
        dem_bones.nnz = 6
        dem_bones.weightsSmooth = 1e-4
        
        dem_bones.nV = nV
        dem_bones.nB = nB
        dem_bones.nF = nF
        dem_bones.nS = 1
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        print("开始现实规模计算...")
        import time
        start = time.time()
        
        dem_bones.compute()
        
        elapsed = time.time() - start
        
        weights = dem_bones.get_weights()
        print(f"✓ 现实规模测试成功! 耗时: {elapsed:.2f}秒")
        print(f"权重矩阵shape: {weights.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 现实规模测试失败: {e}")
        return False

def main():
    weights, transforms = analyze_demBones_output()
    
    if weights is not None:
        print("\n" + "="*50)
        print("🎯 DemBones输出格式理解:")
        
        if weights.shape[0] < weights.shape[1]:
            print("权重矩阵格式: (nB, nV) - 需要转置为(nV, nB)")
        else:
            print("权重矩阵格式: (nV, nB) - 已经是正确格式")
            
        # 测试现实规模
        test_realistic_size()

if __name__ == "__main__":
    main()
