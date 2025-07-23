#!/usr/bin/env python3
"""
超快速LBS权重优化测试 - 专门针对大网格优化
"""

import numpy as np
import sys
from pathlib import Path
from Skinning import InverseMeshCanonicalizer

def test_fast_lbs():
    """
    超快速LBS测试 - 使用最小化的计算
    """
    print("=" * 80)
    print("超快速LBS权重优化测试")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 设置rest pose
        canonicalizer.rest_pose_vertices = np.asarray(canonicalizer.reference_mesh.vertices)
        canonicalizer.rest_pose_transforms = canonicalizer.transforms[0]
        
        print(f"网格顶点数: {len(canonicalizer.rest_pose_vertices)}")
        print(f"关节数: {canonicalizer.num_joints}")
        
        # 极简版本：只使用很少的顶点和很少的迭代
        test_frame = 1
        print(f"\n测试帧 {test_frame} (极简版)...")
        
        # 修改优化函数使其更快
        weights, loss_history = canonicalizer.optimize_skinning_weights_for_frame(
            test_frame, 
            max_iter=20,  # 极少的迭代
            regularization_lambda=0.01,
            init_method='distance_based'
        )
        
        print(f"\n✅ 极简优化完成！")
        print(f"权重矩阵形状: {weights.shape}")
        print(f"最终损失: {loss_history[0]:.6f}")
        
        # 快速分析
        non_zero_weights = weights > 0.01
        sparsity = np.mean(non_zero_weights)
        influences_per_vertex = np.sum(non_zero_weights, axis=1)
        
        print(f"\n权重分析:")
        print(f"稀疏度: {sparsity:.3f}")
        print(f"平均每顶点受影响关节数: {np.mean(influences_per_vertex):.2f}")
        
        # 保存结果
        canonicalizer.skinning_weights = weights
        canonicalizer.save_skinning_weights("output/skinning_weights_fast.npz")
        print(f"权重已保存到: output/skinning_weights_fast.npz")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_validation():
    """
    最小化验证测试
    """
    print("=" * 80)
    print("最小化验证测试")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 尝试加载已保存的权重
        if canonicalizer.load_skinning_weights("output/skinning_weights_fast.npz"):
            print("成功加载已保存的权重")
            
            # 快速验证
            validation_results = canonicalizer.validate_skinning_weights(test_frames=[1])
            
            if validation_results:
                avg_error = validation_results['average_error']
                print(f"验证结果: 平均误差 = {avg_error:.6f}")
                
                if avg_error < 0.1:
                    print("✅ 权重质量可接受")
                else:
                    print("⚠️ 权重质量需要改进")
            
            return True
        else:
            print("无法加载权重文件")
            return False
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def create_simple_baseline_weights():
    """
    创建简单的基准权重（不进行复杂优化）
    """
    print("=" * 80)
    print("创建简单基准权重")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 设置rest pose
        rest_vertices = np.asarray(canonicalizer.reference_mesh.vertices)
        canonicalizer.rest_pose_vertices = rest_vertices
        canonicalizer.rest_pose_transforms = canonicalizer.transforms[0]
        
        print(f"创建基于距离的简单权重...")
        
        # 计算归一化
        rest_norm_params = canonicalizer.frame_normalization_params[0]
        rest_vertices_norm = canonicalizer.normalize_mesh_vertices(rest_vertices, rest_norm_params)
        
        # 获取关节点
        keypoints = canonicalizer.keypoints[0, :, :3]
        
        # 计算距离权重
        distances = np.linalg.norm(rest_vertices_norm[:, None, :] - keypoints[None, :, :], axis=2)
        weights = np.exp(-distances**2 / (2 * 0.1**2))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        # 稀疏化权重（只保留前3个最强的关节）
        for i in range(len(weights)):
            top_3_indices = np.argsort(weights[i])[-3:]
            sparse_weights = np.zeros(canonicalizer.num_joints)
            sparse_weights[top_3_indices] = weights[i][top_3_indices]
            sparse_weights = sparse_weights / (np.sum(sparse_weights) + 1e-8)
            weights[i] = sparse_weights
        
        canonicalizer.skinning_weights = weights
        
        print(f"基准权重创建完成:")
        print(f"权重矩阵形状: {weights.shape}")
        
        # 分析权重
        non_zero_weights = weights > 0.01
        sparsity = np.mean(non_zero_weights)
        influences_per_vertex = np.sum(non_zero_weights, axis=1)
        
        print(f"稀疏度: {sparsity:.3f}")
        print(f"平均每顶点受影响关节数: {np.mean(influences_per_vertex):.2f}")
        
        # 保存基准权重
        canonicalizer.save_skinning_weights("output/skinning_weights_baseline.npz")
        
        # 快速验证基准权重
        print(f"\n验证基准权重...")
        validation_results = canonicalizer.validate_skinning_weights(test_frames=[1, 2])
        
        if validation_results:
            avg_error = validation_results['average_error']
            print(f"基准权重平均误差: {avg_error:.6f}")
            
            for frame_idx, frame_result in validation_results['frame_errors'].items():
                print(f"  帧 {frame_idx}: {frame_result['mean_error']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建基准权重失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 超快速LBS优化测试")
    print("2. 验证已保存的权重")
    print("3. 创建简单基准权重")
    
    try:
        choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == "2":
            success = test_minimal_validation()
        elif choice == "3":
            success = create_simple_baseline_weights()
        else:
            success = test_fast_lbs()
        
        if success:
            print("\n🎉 测试成功完成！")
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试出错: {e}")
        sys.exit(1)
