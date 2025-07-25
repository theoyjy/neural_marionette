#!/usr/bin/env python3
"""
测试修复后的插值效果
"""

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Interpolate import VolumetricInterpolator

def test_fixed_interpolation():
    """测试修复后的插值"""
    print("🧪 测试修复后的插值效果")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = "output/test_fixed_interpolation"
    
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
        weights_path=None  # 强制重新优化权重
    )
    
    # 设置测试参数
    frame_start = 10
    frame_end = 20
    num_interpolate = 5  # 减少插值帧数以便观察
    
    print(f"📋 测试参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    
    # 执行测试
    try:
        print("\n🎬 开始测试修复后的插值...")
        
        # 生成插值帧
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=3,
            optimize_weights=True,
            output_dir=output_dir,
            debug_frames=[0, 2, 4]  # 调试第0、2、4个插值帧
        )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return False
        
        # 分析结果
        print("\n📊 修复效果分析:")
        for i, frame_data in enumerate(interpolated_frames):
            vertices = frame_data['vertices']
            transforms = frame_data['transforms']
            
            # 分析网格
            mesh_volume = np.prod(np.max(vertices, axis=0) - np.min(vertices, axis=0))
            vertex_norms = np.linalg.norm(vertices, axis=1)
            max_norm = np.max(vertex_norms)
            
            # 分析骨骼
            bone_lengths = []
            for j in range(1, interpolator.num_joints):
                parent_idx = interpolator.parents[j]
                if parent_idx >= 0:
                    bone_length = np.linalg.norm(
                        transforms[j][:3, 3] - transforms[parent_idx][:3, 3]
                    )
                    bone_lengths.append(bone_length)
            
            print(f"\n  帧 {i}:")
            print(f"    - 网格体积: {mesh_volume:.6f}")
            print(f"    - 最大顶点距离: {max_norm:.6f}")
            if bone_lengths:
                print(f"    - 平均骨骼长度: {np.mean(bone_lengths):.6f}")
                print(f"    - 骨骼长度方差: {np.var(bone_lengths):.6f}")
            
            # 检查是否有异常
            if max_norm > 10.0:
                print(f"    ⚠️  警告: 顶点距离过大")
            if bone_lengths and np.var(bone_lengths) > 0.1:
                print(f"    ⚠️  警告: 骨骼长度变化过大")
        
        # 生成可视化对比
        print("\n🎨 生成可视化对比...")
        
        # 可视化起始帧
        start_frame_data = {
            'mesh': o3d.io.read_triangle_mesh(str(interpolator.mesh_files[frame_start])),
            'transforms': interpolator.transforms[frame_start],
            'keypoints': interpolator.keypoints[frame_start]
        }
        interpolator.visualize_skeleton_with_mesh(
            start_frame_data, 
            str(Path(output_dir) / "fixed_start_frame.png"), 
            "start"
        )
        
        # 可视化结束帧
        end_frame_data = {
            'mesh': o3d.io.read_triangle_mesh(str(interpolator.mesh_files[frame_end])),
            'transforms': interpolator.transforms[frame_end],
            'keypoints': interpolator.keypoints[frame_end]
        }
        interpolator.visualize_skeleton_with_mesh(
            end_frame_data, 
            str(Path(output_dir) / "fixed_end_frame.png"), 
            "end"
        )
        
        # 可视化修复后的插值帧
        for i in [0, 2, 4]:  # 显示第0、2、4个插值帧
            if i < len(interpolated_frames):
                frame_data = interpolated_frames[i]
                interpolator.visualize_skeleton_with_mesh(
                    frame_data, 
                    str(Path(output_dir) / f"fixed_interpolated_frame_{i}.png"), 
                    i
                )
        
        print(f"\n✅ 修复测试完成！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"🔍 请对比PNG文件，检查:")
        print(f"  1. 骨骼是否不再扭曲")
        print(f"  2. 网格和骨骼是否对齐")
        print(f"  3. 插值是否自然")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_interpolation()
    if success:
        print("\n✅ 修复测试成功！")
    else:
        print("\n❌ 修复测试失败！") 