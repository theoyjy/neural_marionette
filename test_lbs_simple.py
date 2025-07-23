#!/usr/bin/env python3
"""
简化版LBS权重优化测试

这个版本使用更保守的设置来避免内存问题
"""

import numpy as np
import sys
from pathlib import Path
from Skinning import InverseMeshCanonicalizer

def test_lbs_simple():
    """
    简化版LBS权重优化测试
    """
    print("=" * 80)
    print("简化版 Linear Blend Skinning (LBS) 权重优化测试")
    print("=" * 80)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_output_path = "output/skinning_weights_simple.npz"
    
    # 检查路径是否存在
    if not Path(skeleton_data_dir).exists():
        print(f"错误: 骨骼数据目录不存在: {skeleton_data_dir}")
        return False
    
    if not Path(mesh_folder_path).exists():
        print(f"错误: 网格数据目录不存在: {mesh_folder_path}")
        return False
    
    try:
        # 创建统一化器
        print("1. 初始化InverseMeshCanonicalizer...")
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0  # 使用第0帧作为reference frame
        )
        
        # 加载网格序列
        print("2. 加载网格序列...")
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        print(f"   - 骨骼帧数: {canonicalizer.num_frames}")
        print(f"   - 关节数: {canonicalizer.num_joints}")
        print(f"   - 网格文件数: {len(canonicalizer.mesh_files)}")
        print(f"   - Reference frame: {canonicalizer.reference_frame_idx}")
        print(f"   - Reference mesh顶点数: {len(canonicalizer.reference_mesh.vertices)}")
        
        # 限制处理的帧数以避免内存问题
        max_optimization_frames = 3  # 只优化3帧
        
        print(f"\n3. 开始简化版LBS权重优化...")
        print(f"   - 限制优化帧数: {max_optimization_frames}")
        print(f"   - 使用分块优化策略")
        
        # 设置rest pose
        canonicalizer.rest_pose_vertices = np.asarray(canonicalizer.reference_mesh.vertices)
        canonicalizer.rest_pose_transforms = canonicalizer.transforms[canonicalizer.reference_frame_idx]
        
        # 手动选择几个测试帧
        total_frames = len(canonicalizer.mesh_files)
        if total_frames > max_optimization_frames:
            test_frames = [1, total_frames//2, total_frames-1]  # 选择首、中、尾帧
            test_frames = [f for f in test_frames if f != canonicalizer.reference_frame_idx]
            test_frames = test_frames[:max_optimization_frames]
        else:
            test_frames = [i for i in range(total_frames) if i != canonicalizer.reference_frame_idx]
        
        print(f"   - 测试帧: {test_frames}")
        
        all_weights = []
        
        # 为每一帧优化权重
        for i, frame_idx in enumerate(test_frames):
            print(f"\n   优化帧 {frame_idx} ({i+1}/{len(test_frames)})...")
            try:
                weights, loss_history = canonicalizer.optimize_skinning_weights_for_frame(
                    frame_idx, 
                    max_iter=50,  # 减少迭代次数
                    regularization_lambda=0.005,  # 较小的正则化
                    init_method='distance_based'
                )
                all_weights.append(weights)
                print(f"     ✅ 帧 {frame_idx} 优化成功，损失: {loss_history[0]:.6f}")
            except Exception as e:
                print(f"     ❌ 帧 {frame_idx} 优化失败: {e}")
                continue
        
        if not all_weights:
            print("❌ 所有帧优化都失败了")
            return False
        
        # 平均权重
        canonicalizer.skinning_weights = np.mean(all_weights, axis=0)
        print(f"\n✅ LBS权重优化成功完成！")
        print(f"   - 使用了 {len(all_weights)} 帧的平均权重")
        print(f"   - 权重矩阵形状: {canonicalizer.skinning_weights.shape}")
        
        # 分析权重特性
        print("\n4. 分析权重特性...")
        analyze_skinning_weights_simple(canonicalizer.skinning_weights)
        
        # 保存权重
        print("\n5. 保存权重...")
        canonicalizer.save_skinning_weights(weights_output_path)
        
        # 简化版验证
        print("\n6. 验证权重效果...")
        validation_frames = test_frames[:3]  # 只验证前3帧
        validation_results = canonicalizer.validate_skinning_weights(test_frames=validation_frames)
        
        if validation_results:
            print("✅ 验证完成！")
            avg_error = validation_results['average_error']
            print(f"   平均重建误差: {avg_error:.6f}")
            
            if avg_error < 0.01:
                print("🎉 优秀！平均重建误差 < 0.01")
            elif avg_error < 0.05:
                print("👍 良好！平均重建误差 < 0.05") 
            elif avg_error < 0.1:
                print("⚠️  一般，平均重建误差 < 0.1")
            else:
                print("❌ 较差，平均重建误差 >= 0.1")
                
            # 显示每帧详细结果
            for frame_idx, frame_result in validation_results['frame_errors'].items():
                print(f"     帧 {frame_idx}: {frame_result['mean_error']:.6f}")
        
        print("\n" + "=" * 80)
        print("简化版LBS权重优化测试完成！")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_skinning_weights_simple(weights):
    """
    简化版权重分析
    
    Args:
        weights: 权重矩阵 [V, J]
    """
    num_vertices, num_joints = weights.shape
    
    # 计算稀疏度
    non_zero_threshold = 0.01
    non_zero_weights = weights > non_zero_threshold
    sparsity = np.mean(non_zero_weights)
    
    # 每个顶点的影响关节数
    influences_per_vertex = np.sum(non_zero_weights, axis=1)
    
    # 权重分布统计
    weight_stats = {
        'mean': np.mean(weights),
        'std': np.std(weights),
        'min': np.min(weights),
        'max': np.max(weights)
    }
    
    print(f"   权重矩阵分析:")
    print(f"   - 顶点数: {num_vertices}")
    print(f"   - 关节数: {num_joints}")
    print(f"   - 稀疏度: {sparsity:.3f}")
    print(f"   - 平均每顶点受影响关节数: {np.mean(influences_per_vertex):.2f}")
    print(f"   - 权重统计: mean={weight_stats['mean']:.4f}, std={weight_stats['std']:.4f}")

def test_single_frame():
    """
    测试单帧优化
    """
    print("=" * 80)
    print("单帧LBS权重优化测试")
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
        canonicalizer.rest_pose_transforms = canonicalizer.transforms[canonicalizer.reference_frame_idx]
        
        # 测试单帧
        test_frame = 1
        print(f"测试帧 {test_frame}...")
        
        weights, loss_history = canonicalizer.optimize_skinning_weights_for_frame(
            test_frame, 
            max_iter=20,  # 很少的迭代
            regularization_lambda=0.01,
            init_method='distance_based'
        )
        
        print(f"单帧优化完成！")
        print(f"权重形状: {weights.shape}")
        print(f"损失: {loss_history[0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"单帧测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 简化版完整测试")
    print("2. 单帧测试")
    
    try:
        choice = input("请输入选择 (1/2): ").strip()
        
        if choice == "2":
            success = test_single_frame()
        else:
            success = test_lbs_simple()
        
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
