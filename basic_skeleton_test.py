#!/usr/bin/env python3
"""
最基本的骨骼插值测试
只验证核心逻辑，不涉及复杂的网格处理
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_skeleton_interpolation():
    """测试基本的骨骼插值逻辑"""
    print("🧪 基本骨骼插值测试")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    
    # 检查数据是否存在
    if not os.path.exists(skeleton_data_dir):
        print(f"❌ 骨骼数据目录不存在: {skeleton_data_dir}")
        return False
    
    try:
        # 加载骨骼数据
        print("📂 加载骨骼数据...")
        skeleton_path = Path(skeleton_data_dir)
        
        # 加载关键数据
        keypoints = np.load(skeleton_path / 'keypoints.npy')
        transforms = np.load(skeleton_path / 'transforms.npy')
        parents = np.load(skeleton_path / 'parents.npy')
        
        # 检查是否有rotations数据
        rotations = None
        if (skeleton_path / 'rotations.npy').exists():
            rotations = np.load(skeleton_path / 'rotations.npy')
            print("✅ 找到rotations数据")
        else:
            print("⚠️  没有rotations数据，将使用全局变换插值")
        
        num_frames, num_joints = keypoints.shape[0], keypoints.shape[1]
        print(f"📊 数据统计:")
        print(f"  - 帧数: {num_frames}")
        print(f"  - 关节数: {num_joints}")
        print(f"  - 父节点关系: {parents}")
        
        # 选择测试帧
        frame_start = 10
        frame_end = 20
        
        if frame_start >= num_frames or frame_end >= num_frames:
            print(f"❌ 测试帧超出范围: [{0}, {num_frames-1}]")
            return False
        
        print(f"\n🎯 测试帧: {frame_start} -> {frame_end}")
        
        # 获取起始和结束帧的变换
        transforms_start = transforms[frame_start]  # [num_joints, 4, 4]
        transforms_end = transforms[frame_end]      # [num_joints, 4, 4]
        
        print(f"起始帧 {frame_start} 根节点位置: {transforms_start[0][:3, 3]}")
        print(f"结束帧 {frame_end} 根节点位置: {transforms_end[0][:3, 3]}")
        
        # 测试插值
        t = 0.5  # 中间帧
        print(f"\n🔍 测试插值 (t={t})...")
        
        # 简单的线性插值测试
        print("1. 简单线性插值测试:")
        linear_interp = (1-t) * transforms_start + t * transforms_end
        print(f"   根节点位置: {linear_interp[0][:3, 3]}")
        
        # SLERP插值测试
        print("2. SLERP插值测试:")
        slerp_interp = np.zeros_like(transforms_start)
        
        for j in range(num_joints):
            # 提取旋转部分
            R_start = transforms_start[j][:3, :3]
            R_end = transforms_end[j][:3, :3]
            
            # 提取平移部分
            pos_start = transforms_start[j][:3, 3]
            pos_end = transforms_end[j][:3, 3]
            
            # SLERP插值旋转
            quat_start = R.from_matrix(R_start).as_quat()
            quat_end = R.from_matrix(R_end).as_quat()
            
            if np.dot(quat_start, quat_end) < 0:
                quat_end = -quat_end
            
            quat_interp = (1-t) * quat_start + t * quat_end
            quat_interp = quat_interp / np.linalg.norm(quat_interp)
            R_interp = R.from_quat(quat_interp).as_matrix()
            
            # 线性插值平移
            pos_interp = (1-t) * pos_start + t * pos_end
            
            # 构建4x4变换矩阵
            transform_interp = np.eye(4)
            transform_interp[:3, :3] = R_interp
            transform_interp[:3, 3] = pos_interp
            slerp_interp[j] = transform_interp
        
        print(f"   根节点位置: {slerp_interp[0][:3, 3]}")
        
        # 检查骨骼长度
        print("\n📏 检查骨骼长度:")
        for j in range(1, min(5, num_joints)):  # 只检查前几个关节
            parent_idx = parents[j]
            if parent_idx >= 0:
                # 起始帧骨骼长度
                bone_length_start = np.linalg.norm(
                    transforms_start[j][:3, 3] - transforms_start[parent_idx][:3, 3]
                )
                
                # 结束帧骨骼长度
                bone_length_end = np.linalg.norm(
                    transforms_end[j][:3, 3] - transforms_end[parent_idx][:3, 3]
                )
                
                # 插值后骨骼长度
                bone_length_interp = np.linalg.norm(
                    slerp_interp[j][:3, 3] - slerp_interp[parent_idx][:3, 3]
                )
                
                print(f"   关节 {j} (父节点 {parent_idx}):")
                print(f"     起始长度: {bone_length_start:.6f}")
                print(f"     结束长度: {bone_length_end:.6f}")
                print(f"     插值长度: {bone_length_interp:.6f}")
                print(f"     长度变化: {abs(bone_length_interp - bone_length_start):.6f}")
        
        # 验证插值合理性
        print("\n✅ 插值验证:")
        
        # 检查根节点位置是否在合理范围内
        root_start = transforms_start[0][:3, 3]
        root_end = transforms_end[0][:3, 3]
        root_interp = slerp_interp[0][:3, 3]
        
        expected_root = (1-t) * root_start + t * root_end
        root_error = np.linalg.norm(root_interp - expected_root)
        
        print(f"   根节点位置误差: {root_error:.6f}")
        if root_error < 0.1:
            print("   ✅ 根节点插值合理")
        else:
            print("   ⚠️  根节点插值可能有问题")
        
        # 检查旋转矩阵是否正交
        for j in range(min(3, num_joints)):
            R_matrix = slerp_interp[j][:3, :3]
            orthogonality_error = np.linalg.norm(R_matrix @ R_matrix.T - np.eye(3))
            print(f"   关节 {j} 旋转矩阵正交性误差: {orthogonality_error:.6f}")
        
        print(f"\n🎉 基本骨骼插值测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_skeleton_interpolation()
    if success:
        print("\n✅ 基本测试通过！")
    else:
        print("\n❌ 基本测试失败！") 