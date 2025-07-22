#!/usr/bin/env python3
"""
基于DemBones官方示例的实现
参考: https://github.com/electronicarts/dem-bones
"""

import numpy as np
import py_dem_bones as pdb

def official_example_test():
    """基于官方示例的DemBones使用"""
    print("=== 官方示例风格测试 ===")
    
    # 创建一个简单的立方体网格
    # 8个顶点，组成一个立方体
    nV = 8
    nB = 2  # 2个骨骼
    nF = 2  # 2帧动画
    
    # 立方体的8个顶点
    rest_pose = np.array([
        [-1, -1, -1],  # 顶点0
        [ 1, -1, -1],  # 顶点1  
        [ 1,  1, -1],  # 顶点2
        [-1,  1, -1],  # 顶点3
        [-1, -1,  1],  # 顶点4
        [ 1, -1,  1],  # 顶点5
        [ 1,  1,  1],  # 顶点6
        [-1,  1,  1]   # 顶点7
    ], dtype=np.float64)
    
    # 创建2帧动画 - 简单的拉伸变形
    frame1 = rest_pose.copy()
    frame1[:, 0] *= 1.2  # X方向拉伸
    
    frame2 = rest_pose.copy() 
    frame2[:, 1] *= 1.3  # Y方向拉伸
    
    # 组合成animated_poses (16, 3)
    animated_poses = np.vstack([frame1, frame2])
    
    print(f"顶点数: {nV}, 骨骼数: {nB}, 帧数: {nF}")
    print(f"Rest pose: {rest_pose.shape}")
    print(f"Animated poses: {animated_poses.shape}")
    
    try:
        # 创建DemBones实例
        dem_bones = pdb.DemBones()
        
        # 设置基本参数
        dem_bones.nV = nV
        dem_bones.nB = nB  
        dem_bones.nF = nF
        dem_bones.nS = 1  # 一个主体
        
        # 设置帧索引
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(nF, dtype=np.int32)
        
        # 设置网格数据
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        # 设置算法参数 - 使用默认值或简单值
        dem_bones.nIters = 20
        dem_bones.nInitIters = 3
        dem_bones.nTransIters = 3
        dem_bones.nWeightsIters = 3
        dem_bones.nnz = 4
        dem_bones.weightsSmooth = 1e-4
        
        # 可能需要设置的其他参数
        try:
            dem_bones.bindUpdate = 1
            dem_bones.transAffine = 1
            dem_bones.transAffineNorm = 1
        except:
            pass  # 如果这些属性不存在就忽略
        
        print("开始计算...")
        dem_bones.compute()
        print("✓ 计算完成!")
        
        # 获取结果
        weights = dem_bones.get_weights()
        transforms = dem_bones.get_transformations()
        
        print(f"\n结果分析:")
        print(f"权重矩阵 shape: {weights.shape}")
        print(f"变换矩阵 shape: {transforms.shape}")
        
        # 详细分析权重
        print(f"\n权重矩阵内容:")
        print(weights)
        
        # 尝试理解权重矩阵的实际含义
        print(f"\n权重矩阵的可能解释:")
        
        # 可能的情况1: (nB, nV) 格式
        if weights.shape == (nB, nV):
            print(f"✓ 格式是 (nB={nB}, nV={nV}) - 标准格式")
            
        # 可能的情况2: (nV, nB) 格式  
        elif weights.shape == (nV, nB):
            print(f"✓ 格式是 (nV={nV}, nB={nB}) - 转置格式")
            
        # 可能的情况3: 稀疏格式或压缩格式
        else:
            print(f"? 未知格式: {weights.shape}")
            print("可能是稀疏矩阵或压缩格式")
            
            # 尝试检查是否是nnz相关的格式
            if weights.shape[1] == nV:
                print(f"可能每行代表一个非零权重组")
            elif weights.shape[0] == nV:
                print(f"可能每列代表一个顶点的权重")
                
        return weights, transforms
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_demBones_api():
    """调试DemBones API的具体用法"""
    print("\n=== DemBones API调试 ===")
    
    # 检查DemBones对象的所有属性
    dem_bones = pdb.DemBones()
    
    print("DemBones对象的属性:")
    attributes = [attr for attr in dir(dem_bones) if not attr.startswith('_')]
    for attr in attributes:
        try:
            value = getattr(dem_bones, attr)
            if callable(value):
                print(f"  {attr}(): 方法")
            else:
                print(f"  {attr} = {value} ({type(value).__name__})")
        except:
            print(f"  {attr}: 无法访问")

def main():
    debug_demBones_api()
    weights, transforms = official_example_test()
    
    if weights is not None:
        print("\n" + "="*50)
        print("🔍 DemBones权重矩阵分析结果:")
        print(f"实际格式: {weights.shape}")
        
        # 根据实际格式提供解决方案
        if weights.shape[1] == 8 and weights.shape[0] == 1:
            print("可能的原因:")
            print("1. 只有一个有效的权重组")
            print("2. 算法收敛到了一个平凡解")
            print("3. 数据不足以产生有意义的蒙皮权重")

if __name__ == "__main__":
    main()
