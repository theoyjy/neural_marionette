#!/usr/bin/env python3
"""
测试LBS权重优化功能

这个脚本专门测试Linear Blend Skinning权重优化，
通过最小化 ||V_target - LBS(V_rest, weights, transforms)||² 来求解最优权重
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from Skinning import InverseMeshCanonicalizer

def test_lbs_optimization():
    """
    测试LBS权重优化功能
    """
    print("=" * 80)
    print("Linear Blend Skinning (LBS) 权重优化测试")
    print("=" * 80)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_output_path = "output/skinning_weights_test.npz"
    
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
            reference_frame_idx=5  # 使用第5帧作为reference frame
        )
        
        # 加载网格序列
        print("2. 加载网格序列...")
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        print(f"   - 骨骼帧数: {canonicalizer.num_frames}")
        print(f"   - 关节数: {canonicalizer.num_joints}")
        print(f"   - 网格文件数: {len(canonicalizer.mesh_files)}")
        print(f"   - Reference frame: {canonicalizer.reference_frame_idx}")
        print(f"   - Reference mesh顶点数: {len(canonicalizer.reference_mesh.vertices)}")
        
        # 开始LBS权重优化
        print("\n3. 开始LBS权重优化...")
        print("   优化目标: minimize ||V_target - LBS(V_rest, weights, transforms)||²")
        
        skinning_weights = canonicalizer.optimize_reference_frame_skinning(
            regularization_lambda=0.005,  # 较小的正则化系数
            max_iter=200  # 减少迭代次数用于测试
        )
        
        if skinning_weights is None:
            print("❌ LBS权重优化失败！")
            return False
        
        print("✅ LBS权重优化成功完成！")
        print(f"   - 权重矩阵形状: {skinning_weights.shape}")
        
        # 分析权重特性
        print("\n4. 分析权重特性...")
        analyze_skinning_weights(skinning_weights)
        
        # 保存权重
        print("\n5. 保存权重...")
        canonicalizer.save_skinning_weights(weights_output_path)
        
        # 验证权重效果
        print("\n6. 验证权重效果...")
        validation_results = canonicalizer.validate_skinning_weights(
            test_frames=list(range(0, min(canonicalizer.num_frames, 20), 3))
        )
        
        if validation_results:
            print("✅ 验证完成！")
            
            # 绘制误差分析图
            plot_validation_results(validation_results)
            
            # 误差阈值检查
            avg_error = validation_results['average_error']
            if avg_error < 0.01:
                print("🎉 优秀！平均重建误差 < 0.01")
            elif avg_error < 0.05:
                print("👍 良好！平均重建误差 < 0.05")
            elif avg_error < 0.1:
                print("⚠️  一般，平均重建误差 < 0.1")
            else:
                print("❌ 较差，平均重建误差 >= 0.1")
        
        # 测试加载权重
        print("\n7. 测试权重加载...")
        new_canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=5
        )
        
        if new_canonicalizer.load_skinning_weights(weights_output_path):
            print("✅ 权重加载测试成功！")
        else:
            print("❌ 权重加载测试失败！")
        
        print("\n" + "=" * 80)
        print("LBS权重优化测试完成！")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_skinning_weights(weights):
    """
    分析skinning权重的特性
    
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
    
    # 每个关节影响的顶点数
    vertices_per_joint = np.sum(non_zero_weights, axis=0)
    
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
    print(f"   - 稀疏度: {sparsity:.3f} (权重 > {non_zero_threshold})")
    print(f"   - 平均每顶点受影响关节数: {np.mean(influences_per_vertex):.2f}")
    print(f"   - 影响关节数分布: min={np.min(influences_per_vertex)}, max={np.max(influences_per_vertex)}")
    print(f"   - 平均每关节影响顶点数: {np.mean(vertices_per_joint):.2f}")
    print(f"   - 权重统计: mean={weight_stats['mean']:.4f}, std={weight_stats['std']:.4f}")
    print(f"   - 权重范围: [{weight_stats['min']:.4f}, {weight_stats['max']:.4f}]")

def plot_validation_results(validation_results):
    """
    绘制验证结果图表
    
    Args:
        validation_results: 验证结果字典
    """
    try:
        import matplotlib.pyplot as plt
        
        frame_indices = list(validation_results['frame_errors'].keys())
        mean_errors = [validation_results['frame_errors'][idx]['mean_error'] 
                      for idx in frame_indices]
        std_errors = [validation_results['frame_errors'][idx]['std_error'] 
                     for idx in frame_indices]
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 误差趋势
        plt.subplot(2, 2, 1)
        plt.plot(frame_indices, mean_errors, 'b-o', label='Mean Error')
        plt.fill_between(frame_indices, 
                        [m - s for m, s in zip(mean_errors, std_errors)],
                        [m + s for m, s in zip(mean_errors, std_errors)],
                        alpha=0.3, label='±1 Std')
        plt.xlabel('Frame Index')
        plt.ylabel('Reconstruction Error')
        plt.title('LBS Reconstruction Error vs Frame')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 误差分布
        plt.subplot(2, 2, 2)
        plt.hist(mean_errors, bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 误差统计
        plt.subplot(2, 2, 3)
        stats = ['Average', 'Min', 'Max']
        values = [validation_results['average_error'],
                 validation_results['min_error'],
                 validation_results['max_error']]
        colors = ['blue', 'green', 'red']
        
        bars = plt.bar(stats, values, color=colors, alpha=0.7)
        plt.ylabel('Error Value')
        plt.title('Error Statistics')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 子图4: 帧距离vs误差
        plt.subplot(2, 2, 4)
        reference_frame = 5  # 假设reference frame是5
        frame_distances = [abs(idx - reference_frame) for idx in frame_indices]
        plt.scatter(frame_distances, mean_errors, c='red', alpha=0.7)
        plt.xlabel('Distance from Reference Frame')
        plt.ylabel('Mean Error')
        plt.title('Error vs Distance from Reference')
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(frame_distances) > 1:
            z = np.polyfit(frame_distances, mean_errors, 1)
            p = np.poly1d(z)
            plt.plot(sorted(frame_distances), p(sorted(frame_distances)), "r--", alpha=0.8, label='Trend')
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_path = "output/lbs_validation_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   验证结果图表已保存: {output_path}")
        
        # 可选：显示图表
        # plt.show()
        
    except ImportError:
        print("   警告: matplotlib不可用，跳过图表绘制")
    except Exception as e:
        print(f"   警告: 绘制图表时出错: {e}")

def compare_methods():
    """
    比较不同方法的效果（如果需要）
    """
    print("=" * 80)
    print("方法比较（可选功能）")
    print("=" * 80)
    print("这里可以实现:")
    print("1. LBS方法 vs 骨骼驱动方法 vs 特征匹配方法的比较")
    print("2. 不同正则化参数的效果比较")
    print("3. 不同初始化方法的效果比较")
    print("4. 不同reference frame选择的影响")

if __name__ == "__main__":
    print("开始LBS权重优化测试...")
    
    success = test_lbs_optimization()
    
    if success:
        print("\n🎉 所有测试通过！")
        
        # 可选：运行比较测试
        user_input = input("\n是否运行方法比较？(y/n): ").strip().lower()
        if user_input == 'y':
            compare_methods()
    else:
        print("\n❌ 测试失败，请检查错误信息")
        sys.exit(1)
