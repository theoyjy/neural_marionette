#!/usr/bin/env python3
"""
修复后的LBS权重优化测试
"""

import numpy as np
import sys
from pathlib import Path
from Skinning import InverseMeshCanonicalizer

def test_fixed_optimization():
    """
    测试修复后的优化函数
    """
    print("=" * 80)
    print("测试修复后的LBS权重优化")
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
        
        # 测试单帧优化
        test_frame = 1
        print(f"\n测试帧 {test_frame}...")
        
        weights, loss_history = canonicalizer.optimize_skinning_weights_for_frame(
            test_frame, 
            max_iter=100,  # 适中的迭代次数
            regularization_lambda=0.01,
            init_method='distance_based'
        )
        
        print(f"\n✅ 单帧优化成功！")
        print(f"权重矩阵形状: {weights.shape}")
        print(f"最终损失: {loss_history[0]:.6f}")
        
        # 分析权重
        non_zero_weights = weights > 0.01
        sparsity = np.mean(non_zero_weights)
        influences_per_vertex = np.sum(non_zero_weights, axis=1)
        
        print(f"\n权重分析:")
        print(f"稀疏度: {sparsity:.3f}")
        print(f"平均每顶点受影响关节数: {np.mean(influences_per_vertex):.2f}")
        print(f"权重统计: min={np.min(weights):.4f}, max={np.max(weights):.4f}, mean={np.mean(weights):.4f}")
        
        # 验证权重效果
        print(f"\n验证权重效果...")
        canonicalizer.skinning_weights = weights
        validation_results = canonicalizer.validate_skinning_weights(test_frames=[test_frame])
        
        if validation_results:
            avg_error = validation_results['average_error']
            print(f"验证结果:")
            print(f"平均重建误差: {avg_error:.6f}")
            
            if avg_error < 0.01:
                print("🎉 优秀！")
            elif avg_error < 0.05:
                print("👍 良好！") 
            elif avg_error < 0.1:
                print("⚠️  一般")
            else:
                print("❌ 需要改进")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_frame_optimization():
    """
    测试多帧优化
    """
    print("=" * 80)
    print("测试多帧LBS权重优化")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        print(f"网格顶点数: {len(canonicalizer.reference_mesh.vertices)}")
        print(f"关节数: {canonicalizer.num_joints}")
        print(f"总帧数: {len(canonicalizer.mesh_files)}")
        
        # 使用非常保守的设置
        skinning_weights = canonicalizer.optimize_reference_frame_skinning(
            regularization_lambda=0.005,
            max_iter=50  # 较少的迭代次数
        )
        
        if skinning_weights is not None:
            print(f"\n✅ 多帧优化成功！")
            print(f"权重矩阵形状: {skinning_weights.shape}")
            
            # 保存权重
            canonicalizer.save_skinning_weights("output/skinning_weights_fixed.npz")
            
            # 验证效果
            validation_results = canonicalizer.validate_skinning_weights(
                test_frames=[1, 2, 3]  # 只验证前几帧
            )
            
            if validation_results:
                avg_error = validation_results['average_error']
                print(f"\n验证结果:")
                print(f"平均重建误差: {avg_error:.6f}")
                
                for frame_idx, frame_result in validation_results['frame_errors'].items():
                    print(f"  帧 {frame_idx}: {frame_result['mean_error']:.6f}")
            
            return True
        else:
            print("❌ 多帧优化失败")
            return False
        
    except Exception as e:
        print(f"❌ 多帧测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 单帧优化测试")
    print("2. 多帧优化测试")
    
    try:
        choice = input("请输入选择 (1/2): ").strip()
        
        if choice == "2":
            success = test_multi_frame_optimization()
        else:
            success = test_fixed_optimization()
        
        if success:
            print("\n🎉 测试成功完成！")
            print("LBS权重优化现在应该可以正常工作了。")
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试出错: {e}")
        sys.exit(1)
