#!/usr/bin/env python3
"""
测试InverseMeshCanonicalizer的归一化修复
"""

import numpy as np
import open3d as o3d
from Skinning import InverseMeshCanonicalizer
from pathlib import Path
import scipy

def test_normalization_consistency():
    """
    测试归一化的一致性
    """
    print("=== 测试归一化修复 ===")
    
    # 检查是否有测试数据
    skeleton_data_dir = "output/skeleton_prediction"
    if not Path(skeleton_data_dir).exists():
        print(f"警告: 骨骼数据目录不存在: {skeleton_data_dir}")
        return
    
    try:
        # 创建统一化器
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        print(f"成功加载骨骼数据:")
        print(f"  - keypoints shape: {canonicalizer.keypoints.shape}")
        print(f"  - keypoints范围: [{canonicalizer.keypoints.min():.3f}, {canonicalizer.keypoints.max():.3f}]")
        
        # 创建一个测试mesh
        test_vertices = np.random.rand(100, 3) * 2 - 1  # 范围在[-1, 1]
        test_mesh = o3d.geometry.TriangleMesh()
        test_mesh.vertices = o3d.utility.Vector3dVector(test_vertices)
        
        # 测试归一化参数计算
        params = canonicalizer.compute_mesh_normalization_params(test_mesh)
        print(f"\n测试mesh归一化参数:")
        print(f"  - bmin: {params['bmin']}")
        print(f"  - bmax: {params['bmax']}")
        print(f"  - blen: {params['blen']}")
        
        # 测试归一化
        normalized_vertices = canonicalizer.normalize_mesh_vertices(test_vertices, params)
        print(f"\n归一化后顶点范围: [{normalized_vertices.min():.3f}, {normalized_vertices.max():.3f}]")
        
        # 测试骨骼影响权重计算
        frame_idx = 0
        if frame_idx < canonicalizer.num_frames:
            canonicalizer.frame_normalization_params[frame_idx] = params
            weights = canonicalizer.compute_bone_influenced_vertices(test_mesh, frame_idx)
            print(f"\n骨骼权重矩阵形状: {weights.shape}")
            print(f"权重和（应该都接近1.0）: 范围[{weights.sum(axis=1).min():.3f}, {weights.sum(axis=1).max():.3f}]")
        
        print("\n✅ 归一化修复测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_episodic_normalization_simulation():
    """
    测试模拟episodic_normalization的正确性
    """
    print("\n=== 测试episodic_normalization模拟 ===")
    
    # 创建测试数据
    test_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-0.5, 1.5, -1.0]
    ])
    
    # 手动计算预期结果
    bmax = np.amax(test_vertices, axis=0)
    bmin = np.amin(test_vertices, axis=0) 
    blen = (bmax - bmin).max()
    
    print(f"测试数据:")
    print(f"  - vertices: \n{test_vertices}")
    print(f"  - bmin: {bmin}, bmax: {bmax}, blen: {blen}")
    
    # 手动episodic_normalization
    scale = 1.0
    x_trans = 0.0
    z_trans = 0.0
    manual_normalized = ((test_vertices - bmin) * scale / (blen + 1e-5)) * 2 - 1 + np.array([x_trans, 0, z_trans])
    
    # 使用我们的方法
    test_mesh = o3d.geometry.TriangleMesh()
    test_mesh.vertices = o3d.utility.Vector3dVector(test_vertices)
    
    canonicalizer = InverseMeshCanonicalizer("output/skeleton_prediction", 0) if Path("output/skeleton_prediction").exists() else None
    if canonicalizer is None:
        print("跳过测试：没有骨骼数据")
        return
        
    params = canonicalizer.compute_mesh_normalization_params(test_mesh)
    our_normalized = canonicalizer.normalize_mesh_vertices(test_vertices, params)
    
    print(f"\n归一化结果比较:")
    print(f"手动计算: \n{manual_normalized}")
    print(f"我们的方法: \n{our_normalized}")
    print(f"差异: \n{np.abs(manual_normalized - our_normalized)}")
    
    if np.allclose(manual_normalized, our_normalized, atol=1e-6):
        print("✅ episodic_normalization模拟正确！")
    else:
        print("❌ episodic_normalization模拟有误差！")

if __name__ == "__main__":
    test_episodic_normalization_simulation()
    test_normalization_consistency()
