#!/usr/bin/env python3
"""
调试插值问题的脚本
"""

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Interpolate import VolumetricInterpolator

def debug_interpolation():
    """调试插值问题"""
    print("🔍 调试插值问题")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = "output/debug_interpolation"
    
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
        weights_path=None  # 不使用预加载权重，强制重新优化
    )
    
    # 设置调试参数
    frame_start = 10
    frame_end = 20
    num_interpolate = 10
    max_optimize_frames = 5
    
    # 选择要调试的帧（中间帧）
    debug_frames = [4, 5, 6]  # 调试第4、5、6个插值帧
    
    print(f"📋 调试参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    print(f"  - 调试帧: {debug_frames}")
    
    # 执行调试
    try:
        print("\n🎬 开始调试插值...")
        
        # 生成插值帧（包含调试）
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=max_optimize_frames,
            optimize_weights=True,
            output_dir=output_dir,
            debug_frames=debug_frames
        )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return False
        
        # 分析调试帧
        print("\n📊 调试帧分析:")
        for frame_idx in debug_frames:
            if frame_idx < len(interpolated_frames):
                frame_data = interpolated_frames[frame_idx]
                
                # 分析网格变形
                vertices = frame_data['vertices']
                mesh_volume = np.prod(np.max(vertices, axis=0) - np.min(vertices, axis=0))
                
                print(f"\n  帧 {frame_idx}:")
                print(f"    - 网格体积: {mesh_volume:.6f}")
                print(f"    - 顶点范围: {np.min(vertices, axis=0)} -> {np.max(vertices, axis=0)}")
                print(f"    - 顶点标准差: {np.std(vertices, axis=0)}")
                
                # 检查异常值
                vertex_norms = np.linalg.norm(vertices, axis=1)
                max_norm = np.max(vertex_norms)
                min_norm = np.min(vertex_norms)
                print(f"    - 顶点距离范围: {min_norm:.6f} -> {max_norm:.6f}")
                
                if max_norm > 10.0:  # 如果顶点距离过大
                    print(f"    ⚠️  警告: 顶点距离过大，可能存在变形问题")
                
                # 分析骨骼
                transforms = frame_data['transforms']
                keypoints = frame_data['keypoints']
                
                # 检查骨骼长度
                bone_lengths = []
                for j in range(1, interpolator.num_joints):
                    parent_idx = interpolator.parents[j]
                    if parent_idx >= 0:
                        bone_length = np.linalg.norm(
                            transforms[j][:3, 3] - transforms[parent_idx][:3, 3]
                        )
                        bone_lengths.append(bone_length)
                
                if bone_lengths:
                    print(f"    - 平均骨骼长度: {np.mean(bone_lengths):.6f}")
                    print(f"    - 骨骼长度方差: {np.var(bone_lengths):.6f}")
        
        # 生成对比可视化
        print("\n🎨 生成对比可视化...")
        
        # 可视化起始帧
        start_frame_data = {
            'mesh': o3d.io.read_triangle_mesh(str(interpolator.mesh_files[frame_start])),
            'transforms': interpolator.transforms[frame_start],
            'keypoints': interpolator.keypoints[frame_start]
        }
        interpolator.visualize_skeleton_with_mesh(
            start_frame_data, 
            str(Path(output_dir) / "debug_start_frame.png"), 
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
            str(Path(output_dir) / "debug_end_frame.png"), 
            "end"
        )
        
        # 可视化问题帧
        for frame_idx in debug_frames:
            if frame_idx < len(interpolated_frames):
                frame_data = interpolated_frames[frame_idx]
                interpolator.visualize_skeleton_with_mesh(
                    frame_data, 
                    str(Path(output_dir) / f"debug_problem_frame_{frame_idx}.png"), 
                    frame_idx
                )
        
        print(f"\n✅ 调试完成！")
        print(f"📁 调试结果保存在: {output_dir}")
        print(f"🔍 请查看生成的PNG文件对比分析问题")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_interpolation()
    if success:
        print("\n✅ 调试成功！")
    else:
        print("\n❌ 调试失败！") 