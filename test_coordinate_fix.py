#!/usr/bin/env python3
"""
测试坐标系修复效果
验证骨骼和网格是否在同一个坐标系中
"""

import os
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Interpolate import VolumetricInterpolator

def test_coordinate_fix():
    """测试坐标系修复效果"""
    print("🧪 测试坐标系修复效果")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = "output/test_coordinate_fix"
    
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
    num_interpolate = 2  # 减少插值帧数以便观察
    
    print(f"📋 测试参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    
    # 执行测试
    try:
        print("\n🎬 开始测试坐标系修复...")
        
        # 生成插值帧
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames=3,
            optimize_weights=True,
            output_dir=output_dir,
            debug_frames=[0, 1]  # 调试所有插值帧
        )
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return False
        
        # 分析坐标系对齐效果
        print("\n📊 坐标系对齐效果分析:")
        for i, frame_data in enumerate(interpolated_frames):
            vertices = frame_data['vertices']
            transforms = frame_data['transforms']
            
            # 计算网格和骨骼的中心
            mesh_center = np.mean(vertices, axis=0)
            joint_positions = transforms[:, :3, 3]
            joint_center = np.mean(joint_positions, axis=0)
            
            # 计算中心距离
            center_distance = np.linalg.norm(joint_center - mesh_center)
            
            print(f"\n  帧 {i}:")
            print(f"    - 网格中心: {mesh_center}")
            print(f"    - 骨骼中心: {joint_center}")
            print(f"    - 中心距离: {center_distance:.6f}")
            
            # 检查对齐效果
            if center_distance < 0.1:
                print(f"    ✅ 坐标系对齐良好")
            elif center_distance < 0.5:
                print(f"    ⚠️  坐标系对齐一般")
            else:
                print(f"    ❌ 坐标系对齐较差")
            
            # 分析网格和骨骼的尺度
            mesh_scale = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            joint_scale = np.max(joint_positions, axis=0) - np.min(joint_positions, axis=0)
            
            print(f"    - 网格尺度: {mesh_scale}")
            print(f"    - 骨骼尺度: {joint_scale}")
            
            # 检查尺度匹配
            scale_ratio = np.mean(joint_scale) / np.mean(mesh_scale)
            print(f"    - 尺度比例: {scale_ratio:.3f}")
            
            if 0.5 < scale_ratio < 2.0:
                print(f"    ✅ 尺度匹配良好")
            else:
                print(f"    ⚠️  尺度匹配需要调整")
        
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
            str(Path(output_dir) / "coordinate_fix_start_frame.png"), 
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
            str(Path(output_dir) / "coordinate_fix_end_frame.png"), 
            "end"
        )
        
        # 可视化修复后的插值帧
        for i in range(len(interpolated_frames)):
            frame_data = interpolated_frames[i]
            interpolator.visualize_skeleton_with_mesh(
                frame_data, 
                str(Path(output_dir) / f"coordinate_fix_interpolated_frame_{i}.png"), 
                i
            )
        
        print(f"\n✅ 坐标系修复测试完成！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"🔍 请检查PNG文件，验证:")
        print(f"  1. 红色关节球体是否与灰色网格重叠")
        print(f"  2. 绿色骨骼线是否在网格内部")
        print(f"  3. 整体比例是否合理")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coordinate_fix()
    if success:
        print("\n✅ 坐标系修复测试成功！")
    else:
        print("\n❌ 坐标系修复测试失败！") 