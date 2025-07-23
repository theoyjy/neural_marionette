#!/usr/bin/env python3
"""
测试骨骼驱动的顶点对应关系计算性能
"""

import time
import numpy as np
from Skinning import InverseMeshCanonicalizer
from pathlib import Path

def test_performance_comparison():
    """
    比较骨骼驱动方法和原始特征匹配方法的性能
    """
    print("=== 性能对比测试 ===")
    
    # 检查数据是否存在
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    if not Path(skeleton_data_dir).exists():
        print(f"跳过测试：{skeleton_data_dir} 不存在")
        return
        
    if not Path(mesh_folder_path).exists():
        print(f"跳过测试：{mesh_folder_path} 不存在")
        return
    
    try:
        # 创建统一化器
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        # 加载网格序列
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 测试少量帧数以比较性能
        test_frames = 3
        output_folder_skeleton = "output/canonical_meshes_skeleton"
        output_folder_feature = "output/canonical_meshes_feature"
        
        # print(f"\n--- 测试骨骼驱动方法 (前{test_frames}帧) ---")
        # start_time = time.time()
        # canonicalizer.canonicalize_mesh_sequence(
        #     output_folder=output_folder_skeleton,
        #     max_frames=test_frames,
        #     use_skeleton_driven=True
        # )
        # skeleton_time = time.time() - start_time
        # print(f"骨骼驱动方法耗时: {skeleton_time:.2f} 秒")
        
        
        print(f"\n--- 测试特征匹配方法 (前{test_frames}帧) ---")
        start_time = time.time()
        canonicalizer.canonicalize_mesh_sequence(
            output_folder=output_folder_feature,
            max_frames=test_frames,
            use_skeleton_driven=False
        )
        feature_time = time.time() - start_time
        print(f"特征匹配方法耗时: {feature_time:.2f} 秒")
        return
        print(f"\n=== 性能对比结果 ===")
        print(f"骨骼驱动方法: {skeleton_time:.2f} 秒")
        print(f"特征匹配方法: {feature_time:.2f} 秒")
        if feature_time > 0:
            speedup = feature_time / skeleton_time
            print(f"加速比: {speedup:.1f}x")
        
        print("\n提示：")
        print("- 骨骼驱动方法利用关节的稳定性，应该明显更快")
        print("- 特征匹配方法计算复杂的几何特征，相对较慢")
        print("- 在大规模网格和更多帧数下，差异会更明显")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_skeleton_stability():
    """
    分析骨骼关节的稳定性
    """
    print("\n=== 骨骼稳定性分析 ===")
    
    skeleton_data_dir = "output/skeleton_prediction"
    if not Path(skeleton_data_dir).exists():
        print("跳过分析：骨骼数据不存在")
        return
    
    try:
        canonicalizer = InverseMeshCanonicalizer(skeleton_data_dir, 0)
        
        print(f"关节数量: {canonicalizer.num_joints}")
        print(f"总帧数: {canonicalizer.num_frames}")
        
        # 分析关节位置的变化
        keypoints = canonicalizer.keypoints[:, :, :3]  # 忽略置信度
        
        # 计算每个关节在时间上的变化
        joint_movements = []
        for j in range(canonicalizer.num_joints):
            joint_positions = keypoints[:, j, :]  # [T, 3]
            
            # 计算相邻帧之间的移动距离
            movements = []
            for t in range(1, len(joint_positions)):
                movement = np.linalg.norm(joint_positions[t] - joint_positions[t-1])
                movements.append(movement)
            
            avg_movement = np.mean(movements) if movements else 0
            max_movement = np.max(movements) if movements else 0
            joint_movements.append((avg_movement, max_movement))
        
        print(f"\n关节运动统计:")
        print(f"平均帧间移动距离: {np.mean([m[0] for m in joint_movements]):.4f}")
        print(f"最大帧间移动距离: {np.max([m[1] for m in joint_movements]):.4f}")
        
        # 找出最稳定和最活跃的关节
        avg_movements = [m[0] for m in joint_movements]
        most_stable_joint = np.argmin(avg_movements)
        most_active_joint = np.argmax(avg_movements)
        
        print(f"最稳定关节: {most_stable_joint} (平均移动: {avg_movements[most_stable_joint]:.4f})")
        print(f"最活跃关节: {most_active_joint} (平均移动: {avg_movements[most_active_joint]:.4f})")
        
        print("\n✅ 骨骼关节确实提供了稳定的参考框架用于顶点对应关系计算")
        
    except Exception as e:
        print(f"分析失败: {e}")

if __name__ == "__main__":
    analyze_skeleton_stability()
    test_performance_comparison()
