#!/usr/bin/env python3
"""
测试插值修复效果
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Interpolate import VolumetricInterpolator

def test_interpolation_fix():
    """测试插值修复效果"""
    print("🧪 测试插值修复效果")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_path = "output/skinning_weights_auto.npz"
    output_dir = "output/interpolation_test_results"
    
    # 检查数据是否存在
    if not os.path.exists(skeleton_data_dir):
        print(f"❌ 骨骼数据目录不存在: {skeleton_data_dir}")
        print("请先运行 SkelSequencePrediction.py 生成骨骼数据")
        return
    
    if not os.path.exists(mesh_folder_path):
        print(f"❌ 网格文件夹不存在: {mesh_folder_path}")
        return
    
    # 初始化插值器
    print("🔧 初始化插值器...")
    interpolator = VolumetricInterpolator(
        skeleton_data_dir=skeleton_data_dir,
        mesh_folder_path=mesh_folder_path,
        weights_path=weights_path if os.path.exists(weights_path) else None
    )
    
    # 设置测试参数
    frame_start = 10
    frame_end = 30  # 使用较小的范围进行测试
    num_interpolate = 10
    max_optimize_frames = 5
    
    print(f"📋 测试参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    print(f"  - 最大优化帧数: {max_optimize_frames}")
    
    # 执行测试
    try:
        # 生成插值帧
        print("\n🎬 生成插值帧...")
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=max_optimize_frames,
            optimize_weights=True,
            output_dir=output_dir
        )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return
        
        # 验证插值质量
        print("\n🔍 验证插值质量...")
        quality_metrics = interpolator.validate_interpolation_quality(
            frame_start=frame_start,
            frame_end=frame_end,
            interpolated_frames=interpolated_frames
        )
        
        # 检查质量指标
        print("\n📊 质量检查结果:")
        
        # 体积稳定性检查
        volume_variance = quality_metrics['volume_stability']['volume_variance']
        if volume_variance < 0.01:
            print("✅ 体积稳定性良好")
        elif volume_variance < 0.1:
            print("⚠️  体积稳定性一般")
        else:
            print("❌ 体积稳定性较差")
        
        # 连续性检查
        mean_displacement = quality_metrics['continuity']['mean_displacement']
        if mean_displacement < 0.01:
            print("✅ 网格连续性良好")
        elif mean_displacement < 0.1:
            print("⚠️  网格连续性一般")
        else:
            print("❌ 网格连续性较差")
        
        # 姿态自然性检查
        mean_bone_variance = quality_metrics['pose_naturality']['mean_bone_length_variance']
        if mean_bone_variance < 0.01:
            print("✅ 姿态自然性良好")
        elif mean_bone_variance < 0.1:
            print("⚠️  姿态自然性一般")
        else:
            print("❌ 姿态自然性较差")
        
        print(f"\n🎉 测试完成！")
        print(f"📁 结果保存在: {output_dir}")
        
        return quality_metrics
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_interpolation_fix() 