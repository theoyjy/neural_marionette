#!/usr/bin/env python3
"""
DemBones正确API使用方法研究
根据官方示例重新实现
"""

import numpy as np
import py_dem_bones as pdb

def test_correct_demBones_api():
    """基于官方API文档的正确使用方法"""
    print("=== DemBones正确API测试 ===")
    
    # 创建简单测试数据
    nV = 8   # 8个顶点
    nB = 4   # 4个骨骼
    nF = 3   # 3帧动画
    
    print(f"参数: {nV}顶点, {nB}骨骼, {nF}帧")
    
    # Rest pose (nV, 3)
    rest_pose = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]
    ], dtype=np.float64)
    
    # 动画序列 (nF * nV, 3) 
    # 每一帧都是rest_pose加上一些变形
    animated_frames = []
    for f in range(nF):
        frame = rest_pose + np.random.rand(nV, 3) * 0.2 * (f + 1)
        animated_frames.append(frame)
    
    animated_poses = np.vstack(animated_frames)  # (nF * nV, 3)
    
    print(f"Rest pose: {rest_pose.shape}")
    print(f"Animated poses: {animated_poses.shape}")
    
    try:
        # 创建DemBones对象
        dem_bones = pdb.DemBones()
        
        # 设置算法参数
        dem_bones.nIters = 15
        dem_bones.nInitIters = 3
        dem_bones.nTransIters = 2
        dem_bones.nWeightsIters = 2
        dem_bones.nnz = 4  # 每个顶点最多受4个骨骼影响
        dem_bones.weightsSmooth = 1e-4
        
        # 设置数据维度
        dem_bones.nV = nV  # 顶点数
        dem_bones.nB = nB  # 骨骼数
        dem_bones.nF = nF  # 帧数
        dem_bones.nS = 1   # 主体数（通常为1）
        
        # 设置帧起始索引和主体ID
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
        
        # 设置mesh数据
        dem_bones.u = rest_pose        # Rest pose
        dem_bones.v = animated_poses   # 动画序列
        
        print("开始DemBones计算...")
        import time
        start = time.time()
        
        # 执行计算
        dem_bones.compute()
        
        elapsed = time.time() - start
        print(f"✓ 计算完成! 耗时: {elapsed:.2f}秒")
        
        # 获取结果
        print("\n获取结果...")
        
        # 获取蒙皮权重
        weights = dem_bones.get_weights()
        print(f"蒙皮权重 shape: {weights.shape}")
        print(f"蒙皮权重 dtype: {weights.dtype}")
        
        # 获取骨骼变换
        transforms = dem_bones.get_transformations()
        print(f"骨骼变换 shape: {transforms.shape}")
        print(f"骨骼变换 dtype: {transforms.dtype}")
        
        # 分析权重矩阵
        print(f"\n权重矩阵分析:")
        print(f"权重矩阵:\n{weights}")
        
        # 检查权重矩阵的正确格式
        if weights.shape == (nB, nV):
            print(f"✓ 权重矩阵格式正确: (nB={nB}, nV={nV})")
            print("每行代表一个骨骼对所有顶点的影响权重")
            
            # 转置为(nV, nB)格式用于后续处理
            weights_transposed = weights.T
            print(f"转置后的权重矩阵: {weights_transposed.shape}")
            
            # 检查权重归一化
            vertex_weight_sums = weights_transposed.sum(axis=1)
            print(f"每个顶点的权重和: {vertex_weight_sums}")
            
        elif weights.shape == (nV, nB):
            print(f"✓ 权重矩阵已经是(nV={nV}, nB={nB})格式")
            vertex_weight_sums = weights.sum(axis=1)
            print(f"每个顶点的权重和: {vertex_weight_sums}")
            
        else:
            print(f"❌ 意外的权重矩阵格式: {weights.shape}")
        
        # 分析变换矩阵
        print(f"\n变换矩阵分析:")
        if transforms.shape == (nF, nB, 4, 4):
            print(f"✓ 变换矩阵格式正确: (nF={nF}, nB={nB}, 4, 4)")
            print("transforms[f][b] 是第f帧第b个骨骼的4x4变换矩阵")
        else:
            print(f"变换矩阵shape: {transforms.shape}")
        
        return weights, transforms
        
    except Exception as e:
        print(f"❌ DemBones计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_larger_scale():
    """测试更大规模的数据"""
    print("\n=== 大规模数据测试 ===")
    
    scales = [
        (500, 8, 2),
        (1000, 12, 2), 
        (2000, 16, 2),
        (3000, 20, 2)
    ]
    
    for nV, nB, nF in scales:
        print(f"\n测试规模: {nV}顶点, {nB}骨骼, {nF}帧")
        
        try:
            # 创建随机数据
            rest_pose = np.random.rand(nV, 3).astype(np.float64)
            
            # 创建动画帧
            animated_frames = []
            for f in range(nF):
                frame = rest_pose + np.random.rand(nV, 3).astype(np.float64) * 0.1
                animated_frames.append(frame)
            animated_poses = np.vstack(animated_frames)
            
            # DemBones计算
            dem_bones = pdb.DemBones()
            
            # 保守参数设置
            dem_bones.nIters = 10
            dem_bones.nInitIters = 2
            dem_bones.nTransIters = 1
            dem_bones.nWeightsIters = 1
            dem_bones.nnz = min(4, nB)
            dem_bones.weightsSmooth = 1e-3
            
            dem_bones.nV = nV
            dem_bones.nB = nB
            dem_bones.nF = nF
            dem_bones.nS = 1
            dem_bones.fStart = np.array([0], dtype=np.int32)
            dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
            dem_bones.u = rest_pose
            dem_bones.v = animated_poses
            
            import time
            start = time.time()
            dem_bones.compute()
            elapsed = time.time() - start
            
            weights = dem_bones.get_weights()
            
            print(f"  ✓ 成功! 耗时: {elapsed:.2f}秒, 权重: {weights.shape}")
            
            # 如果这个规模成功了，继续测试下一个
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            print(f"  建议最大规模: {nV}顶点以下")
            break

def main():
    weights, transforms = test_correct_demBones_api()
    
    if weights is not None:
        print("\n" + "="*50)
        print("🎯 DemBones正确使用方法确认!")
        print("1. 权重矩阵格式应该是 (nB, nV) 或 (nV, nB)")
        print("2. 需要转置权重矩阵为 (nV, nB) 用于蒙皮")
        print("3. 变换矩阵格式是 (nF, nB, 4, 4)")
        
        test_larger_scale()

if __name__ == "__main__":
    main()
