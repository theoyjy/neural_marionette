#!/usr/bin/env python3
"""
演示直接使用Skinning.py方法的插值系统
"""

import os
import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Interpolate import VolumetricInterpolator

def demo_skinning_integration():
    """演示Skinning.py集成"""
    print("🎬 演示Skinning.py集成")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = "output/skinning_integration_demo"
    
    # 检查数据是否存在
    if not os.path.exists(skeleton_data_dir):
        print(f"❌ 骨骼数据目录不存在: {skeleton_data_dir}")
        return False
    
    if not os.path.exists(mesh_folder_path):
        print(f"❌ 网格文件夹不存在: {mesh_folder_path}")
        return False
    
    # 初始化插值器
    print("🔧 初始化插值器...")
    interpolator = VolumetricInterpolator(
        skeleton_data_dir=skeleton_data_dir,
        mesh_folder_path=mesh_folder_path,
        weights_path=None  # 不使用预加载权重
    )
    
    # 设置演示参数
    frame_start = 10
    frame_end = 20
    num_interpolate = 10
    max_optimize_frames = 5
    
    print(f"📋 演示参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    print(f"  - 最大优化帧数: {max_optimize_frames}")
    
    # 执行演示
    try:
        print("\n🎬 开始插值演示...")
        start_time = time.time()
        
        # 生成插值帧
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=max_optimize_frames,
            optimize_weights=True,
            output_dir=output_dir
        )
        
        total_time = time.time() - start_time
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return False
        
        print(f"\n✅ 插值演示完成!")
        print(f"📊 性能统计:")
        print(f"  - 总耗时: {total_time:.2f}秒")
        print(f"  - 生成帧数: {len(interpolated_frames)}")
        print(f"  - 平均每帧耗时: {total_time / len(interpolated_frames):.3f}秒")
        print(f"  - 处理速度: {len(interpolated_frames) / total_time:.1f} 帧/秒")
        
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
        
        print(f"\n🎉 演示完成！")
        print(f"📁 结果保存在: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_approaches():
    """比较不同方法的性能"""
    print("\n🔍 比较不同方法...")
    print("=" * 50)
    
    # 这里可以添加不同方法的性能比较
    print("✅ 直接使用Skinning.py方法:")
    print("  - 复用成熟的优化算法")
    print("  - 减少代码重复")
    print("  - 更好的维护性")
    print("  - 利用Skinning.py的优化经验")
    
    print("\n❌ 自定义权重优化方法:")
    print("  - 需要重新实现优化逻辑")
    print("  - 代码重复")
    print("  - 维护困难")
    print("  - 可能引入bug")

if __name__ == "__main__":
    success = demo_skinning_integration()
    if success:
        compare_approaches()
        print("\n✅ 演示成功！")
    else:
        print("\n❌ 演示失败！") 