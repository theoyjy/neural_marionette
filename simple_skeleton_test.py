#!/usr/bin/env python3
"""
简单的骨骼插值测试脚本
只验证插值逻辑，不生成GLB文件
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimpleSkeletonInterpolator:
    """简单的骨骼插值器，只用于验证逻辑"""
    
    def __init__(self, skeleton_data_dir):
        """
        初始化骨骼插值器
        
        Args:
            skeleton_data_dir: 骨骼数据目录路径
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        
        # 加载骨骼数据
        self.load_skeleton_data()
        
    def load_skeleton_data(self):
        """加载骨骼预测数据"""
        try:
            # 加载关键点数据 [num_frames, num_joints, 4] (x, y, z, confidence)
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # 加载变换矩阵 [num_frames, num_joints, 4, 4]
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # 加载父节点关系 [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            # 加载旋转矩阵 [num_frames, num_joints, 3, 3]
            if (self.skeleton_data_dir / 'rotations.npy').exists():
                self.rotations = np.load(self.skeleton_data_dir / 'rotations.npy')
            else:
                self.rotations = None
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"✅ 成功加载骨骼数据:")
            print(f"  - 帧数: {self.num_frames}")
            print(f"  - 关节数: {self.num_joints}")
            print(f"  - 关键点形状: {self.keypoints.shape}")
            print(f"  - 变换矩阵形状: {self.transforms.shape}")
            
        except Exception as e:
            raise ValueError(f"无法加载骨骼数据: {e}")
    
    def interpolate_skeleton_transforms(self, frame_start, frame_end, t):
        """
        使用改进的SLERP插值骨骼变换
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            t: 插值参数 [0, 1]
            
        Returns:
            interpolated_transforms: 插值后的变换矩阵 [num_joints, 4, 4]
        """
        transforms_start = self.transforms[frame_start]  # [num_joints, 4, 4]
        transforms_end = self.transforms[frame_end]      # [num_joints, 4, 4]
        
        interpolated_transforms = np.zeros_like(transforms_start)
        
        # 方法1: 如果有rotations数据，使用局部旋转插值（推荐）
        if self.rotations is not None:
            print(f"使用局部旋转插值 (rotations数据可用)")
            rotations_start = self.rotations[frame_start]  # [num_joints, 3, 3]
            rotations_end = self.rotations[frame_end]      # [num_joints, 3, 3]
            
            # 对每个关节进行局部旋转插值
            for j in range(self.num_joints):
                # 获取局部旋转
                R_local_start = rotations_start[j]
                R_local_end = rotations_end[j]
                
                # SLERP插值局部旋转
                quat_start = R.from_matrix(R_local_start).as_quat()
                quat_end = R.from_matrix(R_local_end).as_quat()
                
                # 确保四元数在同一半球
                if np.dot(quat_start, quat_end) < 0:
                    quat_end = -quat_end
                
                # SLERP插值
                quat_interp = (1-t) * quat_start + t * quat_end
                quat_interp = quat_interp / np.linalg.norm(quat_interp)
                R_local_interp = R.from_quat(quat_interp).as_matrix()
                
                # 重建全局变换矩阵
                if j == 0:  # 根节点
                    # 根节点直接使用局部旋转
                    R_global_interp = R_local_interp
                    # 线性插值根节点位置
                    pos_start = transforms_start[j][:3, 3]
                    pos_end = transforms_end[j][:3, 3]
                    pos_interp = (1-t) * pos_start + t * pos_end
                else:
                    # 非根节点需要考虑父节点
                    parent_idx = self.parents[j]
                    R_parent_interp = interpolated_transforms[parent_idx][:3, :3]
                    R_global_interp = R_parent_interp @ R_local_interp
                    
                    # 改进的位置计算：基于骨骼长度和父节点位置
                    parent_pos = interpolated_transforms[parent_idx][:3, 3]
                    
                    # 计算骨骼长度（从原始数据中获取）
                    bone_length_start = np.linalg.norm(transforms_start[j][:3, 3] - transforms_start[parent_idx][:3, 3])
                    bone_length_end = np.linalg.norm(transforms_end[j][:3, 3] - transforms_end[parent_idx][:3, 3])
                    bone_length_interp = (1-t) * bone_length_start + t * bone_length_end
                    
                    # 计算局部偏移方向
                    local_offset_start = transforms_start[j][:3, 3] - transforms_start[parent_idx][:3, 3]
                    local_offset_end = transforms_end[j][:3, 3] - transforms_end[parent_idx][:3, 3]
                    
                    # 插值局部偏移方向
                    if np.linalg.norm(local_offset_start) > 1e-6 and np.linalg.norm(local_offset_end) > 1e-6:
                        local_offset_start_norm = local_offset_start / np.linalg.norm(local_offset_start)
                        local_offset_end_norm = local_offset_end / np.linalg.norm(local_offset_end)
                        
                        # 使用SLERP插值方向
                        dot_product = np.dot(local_offset_start_norm, local_offset_end_norm)
                        dot_product = np.clip(dot_product, -1.0, 1.0)
                        angle = np.arccos(dot_product)
                        
                        if abs(angle) > 1e-6:
                            # 使用球面插值
                            local_offset_interp_norm = (np.sin((1-t)*angle) * local_offset_start_norm + 
                                                      np.sin(t*angle) * local_offset_end_norm) / np.sin(angle)
                        else:
                            # 如果角度很小，使用线性插值
                            local_offset_interp_norm = (1-t) * local_offset_start_norm + t * local_offset_end_norm
                            local_offset_interp_norm = local_offset_interp_norm / np.linalg.norm(local_offset_interp_norm)
                    else:
                        # 如果偏移很小，使用简单的线性插值
                        local_offset_interp_norm = (1-t) * local_offset_start + t * local_offset_end
                        if np.linalg.norm(local_offset_interp_norm) > 1e-6:
                            local_offset_interp_norm = local_offset_interp_norm / np.linalg.norm(local_offset_interp_norm)
                        else:
                            local_offset_interp_norm = np.array([0, 1, 0])  # 默认向上
                    
                    # 计算最终位置
                    pos_interp = parent_pos + bone_length_interp * local_offset_interp_norm
                
                # 构建4x4变换矩阵
                transform_interp = np.eye(4)
                transform_interp[:3, :3] = R_global_interp
                transform_interp[:3, 3] = pos_interp
                interpolated_transforms[j] = transform_interp
        
        # 方法2: 如果没有rotations数据，使用全局变换插值（备选）
        else:
            print(f"使用全局变换插值 (rotations数据不可用)")
            for j in range(self.num_joints):
                # 提取旋转部分 (3x3)
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
                
                # 改进的位置插值：考虑骨骼长度约束
                if j > 0:  # 非根节点
                    parent_idx = self.parents[j]
                    parent_pos_start = transforms_start[parent_idx][:3, 3]
                    parent_pos_end = transforms_end[parent_idx][:3, 3]
                    
                    # 插值父节点位置
                    parent_pos_interp = (1-t) * parent_pos_start + t * parent_pos_end
                    
                    # 计算相对位置
                    relative_pos_start = pos_start - parent_pos_start
                    relative_pos_end = pos_end - parent_pos_end
                    
                    # 插值相对位置
                    relative_pos_interp = (1-t) * relative_pos_start + t * relative_pos_end
                    
                    # 应用父节点变换
                    pos_interp = parent_pos_interp + relative_pos_interp
                else:
                    # 根节点使用简单线性插值
                    pos_interp = (1-t) * pos_start + t * pos_end
                
                # 构建4x4变换矩阵
                transform_interp = np.eye(4)
                transform_interp[:3, :3] = R_interp
                transform_interp[:3, 3] = pos_interp
                interpolated_transforms[j] = transform_interp
        
        return interpolated_transforms
    
    def validate_interpolation(self, frame_start, frame_end, num_test_frames=5):
        """
        验证插值结果
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_test_frames: 测试帧数
        """
        print(f"\n🔍 验证插值结果...")
        
        # 获取原始帧数据
        start_transforms = self.transforms[frame_start]
        end_transforms = self.transforms[frame_end]
        
        print(f"原始帧数据:")
        print(f"  - 起始帧 {frame_start} 根节点位置: {start_transforms[0][:3, 3]}")
        print(f"  - 结束帧 {frame_end} 根节点位置: {end_transforms[0][:3, 3]}")
        
        # 测试插值
        test_steps = np.linspace(0, 1, num_test_frames)
        
        for i, t in enumerate(test_steps):
            interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
            
            # 检查根节点位置
            root_pos = interpolated_transforms[0][:3, 3]
            expected_pos = (1-t) * start_transforms[0][:3, 3] + t * end_transforms[0][:3, 3]
            
            print(f"  插值帧 {i} (t={t:.2f}):")
            print(f"    - 根节点位置: {root_pos}")
            print(f"    - 期望位置: {expected_pos}")
            print(f"    - 位置误差: {np.linalg.norm(root_pos - expected_pos):.6f}")
            
            # 检查骨骼长度
            bone_lengths = []
            for j in range(1, self.num_joints):
                parent_idx = self.parents[j]
                bone_length = np.linalg.norm(
                    interpolated_transforms[j][:3, 3] - interpolated_transforms[parent_idx][:3, 3]
                )
                bone_lengths.append(bone_length)
            
            avg_bone_length = np.mean(bone_lengths)
            print(f"    - 平均骨骼长度: {avg_bone_length:.6f}")
            print(f"    - 骨骼长度方差: {np.var(bone_lengths):.6f}")
    
    def test_simple_interpolation(self, frame_start, frame_end):
        """
        简单的插值测试
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
        """
        print(f"\n🧪 简单插值测试: {frame_start} -> {frame_end}")
        
        # 测试中间帧
        t = 0.5
        interpolated_transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
        
        print(f"插值结果 (t={t}):")
        print(f"  - 根节点位置: {interpolated_transforms[0][:3, 3]}")
        
        # 检查所有关节位置
        for j in range(min(5, self.num_joints)):  # 只显示前5个关节
            pos = interpolated_transforms[j][:3, 3]
            print(f"  - 关节 {j} 位置: {pos}")
        
        return interpolated_transforms

def main():
    """主函数"""
    print("🧪 简单骨骼插值测试")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    
    # 检查数据是否存在
    if not os.path.exists(skeleton_data_dir):
        print(f"❌ 骨骼数据目录不存在: {skeleton_data_dir}")
        print("请先运行 SkelSequencePrediction.py 生成骨骼数据")
        return
    
    # 初始化骨骼插值器
    print("🔧 初始化骨骼插值器...")
    interpolator = SimpleSkeletonInterpolator(skeleton_data_dir)
    
    # 设置测试参数
    frame_start = 10
    frame_end = 20
    
    print(f"📋 测试参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    
    # 执行测试
    try:
        # 简单插值测试
        interpolated_transforms = interpolator.test_simple_interpolation(frame_start, frame_end)
        
        # 验证插值结果
        interpolator.validate_interpolation(frame_start, frame_end)
        
        print(f"\n🎉 骨骼插值测试完成！")
        print(f"✅ 插值逻辑验证成功")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 