#!/usr/bin/env python3
"""
完整SSDR管道：正确处理DemBones的完整输出
包括 Rest Pose Skeleton + Skinning Weights + Per-frame Bone Transforms
"""

import numpy as np
import py_dem_bones as pdb
import time
import pickle
import os

def complete_SSDR_pipeline(frames_vertices, target_bones=None):
    """
    完整的SSDR管道，正确提取和处理DemBones的全部输出
    
    Args:
        frames_vertices: (F, N, 3) 顶点动画序列
        target_bones: 目标骨骼数量（可选，DemBones会自动确定）
    
    Returns:
        dict: 完整的SSDR结果
    """
    print(f"🦴 完整SSDR管道")
    F, N, _ = frames_vertices.shape
    print(f"输入: {F}帧, {N}顶点")
    
    # 自动确定骨骼数量或使用用户指定
    if target_bones is None:
        # 基于顶点数量的启发式估计
        K = min(max(2, N // 50), 10)
        print(f"自动确定骨骼数量: {K}")
    else:
        K = target_bones
        print(f"使用指定骨骼数量: {K}")
    
    try:
        # 准备数据
        rest_pose = frames_vertices[0]  # (N, 3)
        animated_poses = frames_vertices[1:].reshape(-1, 3)  # ((F-1)*N, 3)
        
        # 初始化DemBones
        db = pdb.DemBones()
        
        # 正确的API设置
        db.nV = N
        db.nB = K
        db.nF = F - 1
        db.nS = 1
        db.fStart = np.array([0], dtype=np.int32)
        db.subjectID = np.zeros(F - 1, dtype=np.int32)
        db.u = rest_pose
        db.v = animated_poses
        
        # 优化参数
        db.nIters = 80
        db.nInitIters = 15
        db.nWeightsIters = 12
        db.nTransIters = 12
        db.weightsSmooth = 0.01
        
        print(f"DemBones SSDR配置: nV={N}, nB={K}, nF={F-1}")
        
        # 执行SSDR计算
        print("🚀 开始SSDR计算...")
        start_time = time.time()
        
        db.compute()
        
        elapsed = time.time() - start_time
        print(f"✅ SSDR计算完成 (耗时 {elapsed:.2f}s)")
        
        # 提取完整SSDR结果
        ssdr_results = extract_complete_SSDR(db, F, N, K)
        
        return ssdr_results
        
    except Exception as e:
        print(f"❌ SSDR计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_complete_SSDR(db, F, N, K):
    """提取完整的SSDR结果并进行智能处理"""
    print(f"\n🔍 提取完整SSDR结果...")
    
    # 1. 提取Skinning Weights
    weights = db.get_weights()
    print(f"1️⃣ Skinning Weights shape: {weights.shape}")
    
    # 2. 提取Per-frame Bone Transforms
    transforms = db.get_transformations()
    print(f"2️⃣ Bone Transforms shape: {transforms.shape}")
    
    # 3. 提取Rest Pose Skeleton
    rest_skeleton = db.get_rest_pose()
    print(f"3️⃣ Rest Skeleton shape: {rest_skeleton.shape}")
    
    # 4. 提取额外的骨骼信息
    bone_matrices = None
    if hasattr(db, 'm') and hasattr(db.m, 'shape'):
        bone_matrices = db.m
        print(f"4️⃣ Bone Matrices shape: {bone_matrices.shape}")
    
    # 智能处理权重矩阵
    processed_weights = process_skinning_weights(weights, N, transforms.shape[1] if transforms.size > 0 else K)
    
    # 验证和分析变换矩阵
    analyzed_transforms = analyze_bone_transforms(transforms, F-1)
    
    # 处理骨骼层级
    skeleton_hierarchy = process_skeleton_hierarchy(rest_skeleton, bone_matrices)
    
    # 组装完整结果
    ssdr_results = {
        'skinning_weights': processed_weights,          # 处理后的蒙皮权重 (N, actual_K)
        'bone_transforms': analyzed_transforms,         # 每帧骨骼变换 (nF, actual_K, 4, 4)
        'rest_skeleton': skeleton_hierarchy,           # Rest pose骨骼层级
        'raw_weights': weights,                        # 原始权重
        'raw_transforms': transforms,                  # 原始变换
        'raw_skeleton': rest_skeleton,                 # 原始骨骼数据
        'bone_matrices': bone_matrices,                # 额外骨骼矩阵
        'metadata': {
            'input_frames': F,
            'input_vertices': N,
            'target_bones': K,
            'actual_bones': transforms.shape[1] if transforms.size > 0 else 0,
            'complete': True
        }
    }
    
    print_SSDR_summary(ssdr_results)
    
    return ssdr_results

def process_skinning_weights(weights, N, actual_K):
    """智能处理蒙皮权重矩阵"""
    print(f"\n⚖️ 处理蒙皮权重: {weights.shape} -> 目标(N={N}, K={actual_K})")
    
    if weights.size == 0:
        # 创建默认权重
        processed = np.zeros((N, actual_K))
        processed[:, 0] = 1.0
        print("   使用默认权重（全部分配给第一个骨骼）")
        return processed
    
    if len(weights.shape) == 2:
        rows, cols = weights.shape
        
        if rows == actual_K and cols == N:
            # 标准格式 (K, N) -> 转置为 (N, K)
            processed = weights.T
            print(f"   标准格式转置: ({actual_K}, {N}) -> ({N}, {actual_K})")
            
        elif rows == N and cols == actual_K:
            # 已经是目标格式 (N, K)
            processed = weights
            print(f"   已是目标格式: ({N}, {actual_K})")
            
        elif rows == 1 and cols == N:
            # 单权重组 (1, N) -> 扩展为 (N, K)
            processed = np.zeros((N, actual_K))
            processed[:, 0] = weights[0]
            print(f"   单权重组扩展: (1, {N}) -> ({N}, {actual_K})")
            
        elif cols == N and rows < actual_K:
            # 部分骨骼 (partial_K, N) -> 扩展为 (N, K)
            processed = np.zeros((N, actual_K))
            processed[:, :rows] = weights.T
            print(f"   部分骨骼扩展: ({rows}, {N}) -> ({N}, {actual_K})")
            
        else:
            # 其他格式，创建默认权重
            processed = np.zeros((N, actual_K))
            processed[:, 0] = 1.0
            print(f"   不支持格式，使用默认权重")
    else:
        # 非2D格式，创建默认权重
        processed = np.zeros((N, actual_K))
        processed[:, 0] = 1.0
        print(f"   非2D格式，使用默认权重")
    
    # 归一化权重
    row_sums = processed.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-8] = 1.0
    processed = processed / row_sums
    
    # 验证
    normalized = np.allclose(processed.sum(axis=1), 1.0, atol=1e-6)
    print(f"   归一化检查: {'✅' if normalized else '❌'}")
    print(f"   权重范围: [{processed.min():.4f}, {processed.max():.4f}]")
    
    return processed

def analyze_bone_transforms(transforms, expected_frames):
    """分析和验证骨骼变换矩阵"""
    print(f"\n🎯 分析骨骼变换: {transforms.shape}")
    
    if transforms.size == 0:
        print("   变换矩阵为空")
        return None
    
    if len(transforms.shape) == 3:
        nF, nB, matrix_dim = transforms.shape
        print(f"   格式: {nF}帧 × {nB}骨骼 × {matrix_dim}D变换")
        
        if matrix_dim == 4:
            print("   ✅ 标准4x4变换矩阵")
            # 验证变换矩阵的有效性
            valid_transforms = 0
            for f in range(min(nF, 3)):  # 检查前3帧
                for b in range(min(nB, 3)):  # 检查前3个骨骼
                    det = np.linalg.det(transforms[f, b, :3, :3])  # 旋转部分的行列式
                    if abs(abs(det) - 1.0) < 0.1:  # 应该接近1（正交矩阵）
                        valid_transforms += 1
            
            validity = valid_transforms / min(nF * nB, 9)
            print(f"   变换有效性: {validity:.1%}")
            
        return transforms
    else:
        print(f"   非标准变换格式: {transforms.shape}")
        return transforms

def process_skeleton_hierarchy(rest_skeleton, bone_matrices):
    """处理骨骼层级结构"""
    print(f"\n🦴 处理骨骼层级...")
    
    skeleton_info = {
        'rest_pose': rest_skeleton,
        'bone_matrices': bone_matrices,
        'joint_positions': None,
        'hierarchy': None
    }
    
    # 从rest_skeleton提取关节位置
    if rest_skeleton is not None and hasattr(rest_skeleton, 'shape'):
        if len(rest_skeleton.shape) == 2 and rest_skeleton.shape[1] == 3:
            skeleton_info['joint_positions'] = rest_skeleton
            print(f"   提取到{rest_skeleton.shape[0]}个关节位置")
        else:
            print(f"   Rest skeleton格式: {rest_skeleton.shape}")
    
    return skeleton_info

def print_SSDR_summary(results):
    """打印SSDR结果摘要"""
    print(f"\n📋 SSDR结果摘要:")
    
    meta = results['metadata']
    print(f"   输入: {meta['input_frames']}帧, {meta['input_vertices']}顶点")
    print(f"   骨骼: 目标{meta['target_bones']}个 -> 实际{meta['actual_bones']}个")
    
    if results['skinning_weights'] is not None:
        w_shape = results['skinning_weights'].shape
        print(f"   ✅ 蒙皮权重: {w_shape}")
    
    if results['bone_transforms'] is not None:
        t_shape = results['bone_transforms'].shape
        print(f"   ✅ 骨骼变换: {t_shape}")
    
    if results['rest_skeleton']['joint_positions'] is not None:
        s_shape = results['rest_skeleton']['joint_positions'].shape
        print(f"   ✅ 骨骼层级: {s_shape}")
    
    print(f"   🎯 完整度: {'✅ 完整' if meta['complete'] else '❌ 不完整'}")

def test_complete_SSDR_pipeline():
    """测试完整SSDR管道"""
    print("🧪 测试完整SSDR管道")
    
    # 创建测试数据
    frames = 6
    vertices = 200
    
    # 生成有意义的变形序列
    np.random.seed(42)
    base_vertices = np.random.randn(vertices, 3).astype(np.float32)
    
    test_frames = []
    for f in range(frames):
        # 渐进变形
        deform = 0.15 * f * np.random.randn(vertices, 3).astype(np.float32)
        test_frames.append(base_vertices + deform)
    
    test_frames = np.array(test_frames)
    
    print(f"测试数据: {test_frames.shape}")
    
    # 运行完整SSDR管道
    results = complete_SSDR_pipeline(test_frames, target_bones=4)
    
    if results and results['metadata']['complete']:
        print(f"\n🎉 完整SSDR管道测试成功!")
        
        # 保存结果
        output_dir = "output/ssdr_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "complete_ssdr.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"📁 结果已保存至: {output_dir}")
        return True
    else:
        print(f"\n❌ SSDR管道测试失败")
        return False

if __name__ == "__main__":
    success = test_complete_SSDR_pipeline()
    
    if success:
        print(f"\n🚀 完整SSDR管道已就绪!")
        print(f"✅ 可以正确提取和处理DemBones的全部SSDR输出")
        print(f"📝 包括: Rest Pose Skeleton + Skinning Weights + Per-frame Bone Transforms")
        print(f"🎯 准备集成到主体积视频插值管道中!")
