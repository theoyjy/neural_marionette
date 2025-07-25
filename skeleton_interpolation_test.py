#!/usr/bin/env python3
"""
骨骼姿态插值测试脚本
使用SkelVisualizer.py的方法生成骨骼可视化
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SkelVisualizer import SkeletonGLBVisualizer

class SkeletonInterpolator:
    """专门用于骨骼姿态插值的类"""
    
    def __init__(self, skeleton_data_dir):
        """
        初始化骨骼插值器
        
        Args:
            skeleton_data_dir: 骨骼数据目录路径
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        
        # 加载骨骼数据
        self.load_skeleton_data()
        
        # 初始化可视化器
        self.visualizer = SkeletonGLBVisualizer()
        
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
            print("⚠️  没有rotations数据，使用全局变换插值")
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
    
    def create_interpolated_keypoints(self, frame_start, frame_end, t):
        """
        插值关键点位置和置信度
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            t: 插值参数 [0, 1]
            
        Returns:
            interpolated_keypoints: 插值后的关键点 [num_joints, 4]
        """
        keypoints_start = self.keypoints[frame_start]  # [num_joints, 4]
        keypoints_end = self.keypoints[frame_end]      # [num_joints, 4]
        
        # 线性插值位置和置信度
        positions_start = keypoints_start[:, :3]
        positions_end = keypoints_end[:, :3]
        positions_interp = (1-t) * positions_start + t * positions_end
        
        # 置信度取最小值（保守策略）
        confidences_start = keypoints_start[:, 3]
        confidences_end = keypoints_end[:, 3]
        confidences_interp = np.minimum(confidences_start, confidences_end)
        
        interpolated_keypoints = np.column_stack([positions_interp, confidences_interp])
        
        return interpolated_keypoints
    
    def generate_skeleton_interpolation(self, frame_start, frame_end, num_interpolate, output_dir):
        """
        生成骨骼插值序列
        
        Args:
            frame_start: 起始帧索引
            frame_end: 结束帧索引
            num_interpolate: 插值帧数
            output_dir: 输出目录
        """
        print(f"🎬 生成骨骼插值序列: {frame_start} -> {frame_end} (插值 {num_interpolate} 帧)")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成插值帧
        interpolation_steps = np.linspace(0, 1, num_interpolate + 2)[1:-1]  # 排除起始和结束帧
        
        print(f"🎨 生成 {len(interpolation_steps)} 个插值帧...")
        
        # 准备插值后的完整序列数据
        interpolated_keypoints = []
        interpolated_transforms = []
        
        # 添加起始帧
        interpolated_keypoints.append(self.keypoints[frame_start])
        interpolated_transforms.append(self.transforms[frame_start])
        
        # 生成插值帧
        for i, t in enumerate(interpolation_steps):
            # 插值骨骼变换
            transforms = self.interpolate_skeleton_transforms(frame_start, frame_end, t)
            
            # 插值关键点
            keypoints = self.create_interpolated_keypoints(frame_start, frame_end, t)
            
            interpolated_keypoints.append(keypoints)
            interpolated_transforms.append(transforms)
            
            # 保存变换数据
            transform_filename = f"skeleton_interpolated_{i:04d}_transforms.npy"
            transform_path = output_path / transform_filename
            np.save(transform_path, transforms)
        
        # 添加结束帧
        interpolated_keypoints.append(self.keypoints[frame_end])
        interpolated_transforms.append(self.transforms[frame_end])
        
        # 转换为numpy数组
        interpolated_keypoints = np.stack(interpolated_keypoints, axis=0)  # [T, K, 4]
        interpolated_transforms = np.stack(interpolated_transforms, axis=0)  # [T, K, 4, 4]
        
        # 创建插值后的骨骼数据
        interpolated_skeleton_data = {
            'keypoints': interpolated_keypoints,
            'transforms': interpolated_transforms,
            'parents': self.parents,
            'num_frames': len(interpolated_keypoints),
            'num_joints': self.num_joints
        }
        
        # 使用SkelVisualizer生成GLB文件
        print("🔧 使用SkelVisualizer生成GLB文件...")
        
        # 为每个插值帧生成单独的GLB
        for i in range(len(interpolated_keypoints)):
            frame_keypoints = interpolated_keypoints[i:i+1]  # 保持维度
            frame_skeleton_data = {
                'keypoints': frame_keypoints,
                'transforms': interpolated_transforms[i:i+1],
                'parents': self.parents,
                'num_frames': 1,
                'num_joints': self.num_joints
            }
            
            # 生成GLB文件
            glb_filename = f"skeleton_interpolated_{i:04d}.glb"
            glb_path = output_path / glb_filename
            
            try:
                self.visualizer.create_animated_glb(frame_skeleton_data, str(glb_path))
                print(f"✅ 生成GLB文件: {glb_filename}")
            except Exception as e:
                print(f"❌ 生成GLB文件失败 {glb_filename}: {e}")
        
        # 生成完整的动画GIF
        print("🎬 生成动画GIF...")
        gif_path = output_path / "skeleton_interpolation_animation.gif"
        
        try:
            self.visualizer.create_frame_sequence_gif(
                interpolated_skeleton_data, 
                str(gif_path), 
                fps=10, 
                image_size=(800, 600)
            )
            print(f"✅ 生成动画GIF: {gif_path}")
        except Exception as e:
            print(f"❌ 生成GIF失败: {e}")
        
        # 保存元数据
        metadata = {
            'frame_start': frame_start,
            'frame_end': frame_end,
            'num_interpolate': num_interpolate,
            'total_frames': len(interpolated_keypoints),
            'interpolation_method': 'skeleton_slerp',
            'joint_count': self.num_joints,
            'parents': self.parents.tolist(),
            'gif_path': str(gif_path)
        }
        
        metadata_path = output_path / "skeleton_interpolation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 骨骼插值序列生成完成:")
        print(f"  - 输出目录: {output_path}")
        print(f"  - 总帧数: {len(interpolated_keypoints)}")
        print(f"  - 元数据: {metadata_path}")
        print(f"  - GIF动画: {gif_path}")

def main():
    """主函数"""
    print("🎬 骨骼姿态插值测试")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    output_dir = "output/skeleton_interpolation_test"
    
    # 检查数据是否存在
    if not os.path.exists(skeleton_data_dir):
        print(f"❌ 骨骼数据目录不存在: {skeleton_data_dir}")
        print("请先运行 SkelSequencePrediction.py 生成骨骼数据")
        return
    
    # 初始化骨骼插值器
    print("🔧 初始化骨骼插值器...")
    interpolator = SkeletonInterpolator(skeleton_data_dir)
    
    # 设置测试参数
    frame_start = 10
    frame_end = 20
    num_interpolate = 10
    
    print(f"📋 测试参数:")
    print(f"  - 起始帧: {frame_start}")
    print(f"  - 结束帧: {frame_end}")
    print(f"  - 插值帧数: {num_interpolate}")
    
    # 执行测试
    try:
        # 生成骨骼插值序列
        interpolator.generate_skeleton_interpolation(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            output_dir=output_dir
        )
        
        print(f"\n🎉 骨骼插值测试完成！")
        print(f"📁 结果保存在: {output_dir}")
        print(f"🔍 请查看生成的GLB文件和GIF动画验证姿势插值是否正确")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 