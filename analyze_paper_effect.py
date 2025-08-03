"""
分析"纸片化"效果的原因

对比单帧BBW vs 多帧BBW的权重分布和效果
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time

def analyze_weight_distribution(weights, title="Weight Analysis"):
    """分析权重分布"""
    print(f"\n=== {title} ===")
    
    if weights is None:
        print("❌ Weights is None")
        return
    
    # 基本统计
    print(f"Weights shape: {weights.shape}")
    print(f"Range: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"Mean: {weights.mean():.6f}")
    print(f"Std: {weights.std():.6f}")
    
    # 权重分布分析
    weight_sums = np.sum(weights, axis=1)
    print(f"Weight sums: min={weight_sums.min():.6f}, max={weight_sums.max():.6f}")
    
    # 稀疏性分析
    significant_weights = weights > 0.01  # 显著权重
    sparsity_per_vertex = np.sum(significant_weights, axis=1)
    
    print(f"Significant weights per vertex:")
    print(f"  - Mean: {sparsity_per_vertex.mean():.1f}")
    print(f"  - Min: {sparsity_per_vertex.min()}")
    print(f"  - Max: {sparsity_per_vertex.max()}")
    
    # 权重方差 (检查是否过度平滑)
    weight_variance = np.var(weights, axis=1)
    print(f"Weight variance per vertex:")
    print(f"  - Mean: {weight_variance.mean():.6f}")
    print(f"  - Std: {weight_variance.std():.6f}")
    
    # 检查权重集中度
    max_weights_per_vertex = np.max(weights, axis=1)
    print(f"Max weight per vertex:")
    print(f"  - Mean: {max_weights_per_vertex.mean():.6f}")
    print(f"  - Min: {max_weights_per_vertex.min():.6f}")
    
    # 如果权重过于平均，可能导致纸片化
    if weight_variance.mean() < 0.01:
        print("⚠️  权重方差较低 - 可能导致过度平滑/纸片化")
    
    if max_weights_per_vertex.mean() < 0.3:
        print("⚠️  最大权重较低 - 权重过于分散")
    
    return {
        'variance_mean': weight_variance.mean(),
        'max_weight_mean': max_weights_per_vertex.mean(),
        'sparsity_mean': sparsity_per_vertex.mean()
    }

def compare_single_vs_multi_frame():
    """对比单帧vs多帧BBW效果"""
    print("🔍 对比单帧vs多帧BBW效果")
    
    test_mesh_folder = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    test_skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
    
    if not all([os.path.exists(test_mesh_folder), os.path.exists(test_skeleton_dir)]):
        print("❌ 测试数据不可用")
        return False
    
    try:
        from bbw_enhanced_interpolator import BBWEnhancedInterpolator
        
        results = {}
        
        # 测试1: 单帧BBW
        print("\n🔸 测试单帧BBW...")
        single_interpolator = BBWEnhancedInterpolator(
            skeleton_data_dir=test_skeleton_dir,
            mesh_folder_path=test_mesh_folder,
            use_bbw=True,
            use_multi_frame=False,
            bbw_reference_frame=5
        )
        
        success = single_interpolator.initialize_skinning()
        if success and single_interpolator.skinning_weights is not None:
            single_stats = analyze_weight_distribution(
                single_interpolator.skinning_weights, 
                "单帧BBW权重分析"
            )
            results['single'] = {
                'interpolator': single_interpolator,
                'stats': single_stats
            }
        
        # 测试2: 多帧BBW
        print("\n🔸 测试多帧BBW...")
        multi_interpolator = BBWEnhancedInterpolator(
            skeleton_data_dir=test_skeleton_dir,
            mesh_folder_path=test_mesh_folder,
            use_bbw=True,
            use_multi_frame=True,
            num_key_frames=5,
            frame_selection_method="pose_diversity"
        )
        
        success = multi_interpolator.initialize_skinning(start_frame=5, end_frame=15)
        if success and multi_interpolator.skinning_weights is not None:
            multi_stats = analyze_weight_distribution(
                multi_interpolator.skinning_weights,
                "多帧BBW权重分析"
            )
            results['multi'] = {
                'interpolator': multi_interpolator,
                'stats': multi_stats
            }
        
        # 对比分析
        if 'single' in results and 'multi' in results:
            print(f"\n📊 单帧vs多帧对比:")
            
            single_stats = results['single']['stats']
            multi_stats = results['multi']['stats']
            
            print(f"权重方差均值:")
            print(f"  - 单帧: {single_stats['variance_mean']:.6f}")
            print(f"  - 多帧: {multi_stats['variance_mean']:.6f}")
            print(f"  - 变化: {(multi_stats['variance_mean'] - single_stats['variance_mean']):.6f}")
            
            print(f"最大权重均值:")
            print(f"  - 单帧: {single_stats['max_weight_mean']:.6f}")
            print(f"  - 多帧: {multi_stats['max_weight_mean']:.6f}")
            print(f"  - 变化: {(multi_stats['max_weight_mean'] - single_stats['max_weight_mean']):.6f}")
            
            print(f"稀疏性均值:")
            print(f"  - 单帧: {single_stats['sparsity_mean']:.1f}")
            print(f"  - 多帧: {multi_stats['sparsity_mean']:.1f}")
            print(f"  - 变化: {(multi_stats['sparsity_mean'] - single_stats['sparsity_mean']):.1f}")
            
            # 判断纸片化原因
            if multi_stats['variance_mean'] < single_stats['variance_mean']:
                print("💡 发现问题：多帧融合导致权重方差降低 -> 过度平滑 -> 纸片化")
            
            if multi_stats['max_weight_mean'] < single_stats['max_weight_mean']:
                print("💡 发现问题：多帧融合导致最大权重降低 -> 权重分散 -> 缺乏细节")
        
        return results
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interpolation_comparison():
    """测试插值结果对比"""
    print("\n🔧 测试插值结果对比")
    
    test_mesh_folder = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    test_skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
    
    try:
        from bbw_enhanced_interpolator import BBWEnhancedInterpolator
        
        # 输出目录
        single_output = Path("paper_analysis/single_frame_output")
        multi_output = Path("paper_analysis/multi_frame_output")
        
        single_output.mkdir(parents=True, exist_ok=True)
        multi_output.mkdir(parents=True, exist_ok=True)
        
        # 测试参数
        start_frame, end_frame = 5, 7
        num_interpolate = 1
        
        # 单帧BBW测试
        print("🔸 生成单帧BBW结果...")
        single_interpolator = BBWEnhancedInterpolator(
            skeleton_data_dir=test_skeleton_dir,
            mesh_folder_path=test_mesh_folder,
            use_bbw=True,
            use_multi_frame=False,
            bbw_reference_frame=5
        )
        
        single_interpolator.initialize_skinning()
        single_results = single_interpolator.generate_interpolated_frames(
            frame_start=start_frame,
            frame_end=end_frame,
            num_interpolate=num_interpolate,
            output_dir=str(single_output),
            save_standard_obj=True
        )
        
        # 多帧BBW测试
        print("🔸 生成多帧BBW结果...")
        multi_interpolator = BBWEnhancedInterpolator(
            skeleton_data_dir=test_skeleton_dir,
            mesh_folder_path=test_mesh_folder,
            use_bbw=True,
            use_multi_frame=True,
            num_key_frames=3,  # 减少关键帧数
            frame_selection_method="pose_diversity"
        )
        
        multi_interpolator.initialize_skinning(start_frame=start_frame, end_frame=end_frame)
        multi_results = multi_interpolator.generate_interpolated_frames(
            frame_start=start_frame,
            frame_end=end_frame,
            num_interpolate=num_interpolate,
            output_dir=str(multi_output),
            save_standard_obj=True
        )
        
        # 分析生成结果
        single_files = list(single_output.glob("*.obj"))
        multi_files = list(multi_output.glob("*.obj"))
        
        if single_files and multi_files:
            print(f"\n📊 结果对比:")
            
            # 分析单帧结果
            single_mesh = o3d.io.read_triangle_mesh(str(single_files[0]))
            single_vertices = np.asarray(single_mesh.vertices)
            
            print(f"单帧BBW结果:")
            print(f"  - 顶点数: {len(single_vertices)}")
            print(f"  - 坐标范围: [{single_vertices.min():.3f}, {single_vertices.max():.3f}]")
            
            # 计算mesh"厚度"(检查纸片化)
            bbox = single_vertices.max(axis=0) - single_vertices.min(axis=0)
            thickness = bbox.min()  # 最小维度作为厚度指标
            print(f"  - '厚度'指标: {thickness:.6f}")
            
            # 分析多帧结果
            multi_mesh = o3d.io.read_triangle_mesh(str(multi_files[0]))
            multi_vertices = np.asarray(multi_mesh.vertices)
            
            print(f"多帧BBW结果:")
            print(f"  - 顶点数: {len(multi_vertices)}")
            print(f"  - 坐标范围: [{multi_vertices.min():.3f}, {multi_vertices.max():.3f}]")
            
            multi_bbox = multi_vertices.max(axis=0) - multi_vertices.min(axis=0)
            multi_thickness = multi_bbox.min()
            print(f"  - '厚度'指标: {multi_thickness:.6f}")
            
            # 厚度对比
            thickness_ratio = multi_thickness / thickness
            print(f"\n厚度对比:")
            print(f"  - 比率 (多帧/单帧): {thickness_ratio:.3f}")
            
            if thickness_ratio < 0.9:
                print("⚠️  多帧结果确实更'扁'- 纸片化问题确认")
            
            return True
        else:
            print("❌ 生成结果文件不完整")
            return False
            
    except Exception as e:
        print(f"❌ 插值对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔍 分析'纸片化'效果")
    print("=" * 50)
    
    # 分析1: 权重分布对比
    comparison_results = compare_single_vs_multi_frame()
    
    # 分析2: 插值结果对比
    interpolation_success = test_interpolation_comparison()
    
    print(f"\n💡 分析结论:")
    if comparison_results:
        print("1. 多帧BBW权重融合可能导致:")
        print("   - 权重方差降低 → 过度平滑")
        print("   - 最大权重分散 → 细节丢失")
        print("   - 整体权重趋于平均 → 纸片化效果")
    
    print("\n🎯 建议解决方案:")
    print("1. 改进权重融合策略 - 保持权重锐度")
    print("2. 调整关键帧选择 - 减少过度平滑")
    print("3. 增强权重后处理 - 恢复局部细节")
    print("4. 或者针对特定场景使用单帧BBW")

if __name__ == "__main__":
    main()