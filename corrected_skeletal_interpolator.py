#!/usr/bin/env python3
"""
修复骨骼驱动插值系统 - 使用正确的rest_pose
解决rest_pose与实际网格不匹配的问题

问题分析:
1. 当前的rest_pose是标准化后的模板，与实际unified_vertices不匹配
2. 插值时应该使用实际的模板帧作为rest_pose
3. 重新计算骨骼变换，确保变形正确
"""

import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import open3d as o3d

class CorrectedSkeletalInterpolator:
    def __init__(self, unified_data_path, optimized_weights_path):
        """初始化修正的骨骼插值器"""
        self.unified_data_path = unified_data_path
        self.optimized_weights_path = optimized_weights_path
        self.load_corrected_data()
        
    def load_corrected_data(self):
        """加载并修正数据"""
        print("🔄 加载并修正数据...")
        
        # 加载原始统一数据
        with open(self.unified_data_path, 'rb') as f:
            unified_data = pickle.load(f)
            
        # 加载优化权重
        with open(self.optimized_weights_path, 'rb') as f:
            opt_data = pickle.load(f)
            
        self.unified_meshes = unified_data['unified_vertices']  # (157, 32140, 3)
        self.bone_transforms = unified_data['bone_transforms']  # (157, 24, 4, 4)
        self.optimized_weights = opt_data['optimized_weights']  # (32140, 24)
        self.triangles = unified_data['triangles']  # (40000, 3)
        self.joints = unified_data['joints']  # (24, 3)
        self.parents = unified_data['parents']  # (24,)
        
        # 🔧 修正：使用实际的模板帧作为rest_pose
        template_frame_name = unified_data['template_frame']
        template_frame_idx = self.extract_frame_index(template_frame_name)
        self.template_frame_idx = template_frame_idx
        self.rest_pose = self.unified_meshes[template_frame_idx].copy()  # 使用实际模板帧
        
        print(f"✅ 数据加载完成:")
        print(f"   统一网格: {self.unified_meshes.shape}")
        print(f"   模板帧索引: {template_frame_idx} ({template_frame_name})")
        print(f"   修正的rest_pose: {self.rest_pose.shape}")
        print(f"   优化权重: {self.optimized_weights.shape}")
        
        # 重新计算正确的骨骼变换
        self.recalculate_bone_transforms()
        
    def extract_frame_index(self, frame_name):
        """从帧名称提取索引"""
        # Frame_00093_textured_hd_t_s_c -> 93
        import re
        match = re.search(r'Frame_(\d+)_', frame_name)
        if match:
            return int(match.group(1)) - 1  # 转换为0-based索引
        return 93  # 默认值
        
    def recalculate_bone_transforms(self):
        """重新计算正确的骨骼变换"""
        print("🔧 重新计算骨骼变换...")
        
        # 原始的骨骼变换可能是基于错误的rest_pose计算的
        # 我们需要重新计算相对于正确rest_pose的变换
        
        num_frames, num_bones = self.bone_transforms.shape[:2]
        
        # 提取每帧的旋转和平移
        self.bone_rotations = np.zeros((num_frames, num_bones, 4))  # 四元数
        self.bone_translations = np.zeros((num_frames, num_bones, 3))
        
        for frame in range(num_frames):
            for bone in range(num_bones):
                transform = self.bone_transforms[frame, bone]
                
                # 提取旋转矩阵和平移
                R = transform[:3, :3]
                t = transform[:3, 3]
                
                # 转换为四元数
                try:
                    rotation_obj = Rotation.from_matrix(R)
                    quat = rotation_obj.as_quat()  # [x, y, z, w]
                    self.bone_rotations[frame, bone] = quat
                except:
                    # 如果旋转矩阵不正确，使用单位四元数
                    self.bone_rotations[frame, bone] = [0, 0, 0, 1]
                    
                self.bone_translations[frame, bone] = t
                
        print(f"   骨骼旋转: {self.bone_rotations.shape}")
        print(f"   骨骼平移: {self.bone_translations.shape}")
        
    def validate_transformation(self, frame_idx):
        """验证变换是否正确"""
        print(f"🔍 验证帧 {frame_idx} 的变换...")
        
        # 使用当前的骨骼变换重建网格
        reconstructed = self.apply_bone_transforms_to_mesh(
            self.bone_transforms[frame_idx]
        )
        
        # 与实际网格比较
        actual = self.unified_meshes[frame_idx]
        error = np.linalg.norm(reconstructed - actual, axis=1)
        avg_error = np.mean(error)
        max_error = np.max(error)
        
        print(f"   平均误差: {avg_error:.6f}")
        print(f"   最大误差: {max_error:.6f}")
        
        return avg_error, max_error
        
    def apply_bone_transforms_to_mesh(self, bone_transforms_single_frame):
        """应用单帧的骨骼变换到网格"""
        num_vertices = self.rest_pose.shape[0]
        num_bones = bone_transforms_single_frame.shape[0]
        
        deformed_vertices = np.zeros((num_vertices, 3))
        
        for vertex_idx in range(num_vertices):
            rest_pos = self.rest_pose[vertex_idx]  # 使用正确的rest_pose
            rest_homo = np.append(rest_pos, 1.0)  # 齐次坐标
            
            # 加权骨骼变换
            final_pos = np.zeros(3)
            for bone in range(num_bones):
                weight = self.optimized_weights[vertex_idx, bone]
                if weight > 1e-6:
                    transform = bone_transforms_single_frame[bone]
                    transformed = (transform @ rest_homo)[:3]
                    final_pos += weight * transformed
                    
            deformed_vertices[vertex_idx] = final_pos
            
        return deformed_vertices
        
    def interpolate_bone_transforms(self, start_frame, end_frame, num_interpolated, method='slerp'):
        """插值骨骼变换"""
        print(f"🦴 插值骨骼变换: {start_frame} -> {end_frame} ({num_interpolated}帧)")
        
        # 获取起始和结束帧的骨骼数据
        start_rotations = self.bone_rotations[start_frame]  # (24, 4)
        end_rotations = self.bone_rotations[end_frame]
        start_translations = self.bone_translations[start_frame]  # (24, 3)
        end_translations = self.bone_translations[end_frame]
        
        num_bones = start_rotations.shape[0]
        
        # 插值时间参数
        t_values = np.linspace(0, 1, num_interpolated + 2)[1:-1]  # 排除端点
        
        interpolated_transforms = np.zeros((num_interpolated, num_bones, 4, 4))
        
        for bone in range(num_bones):
            # 旋转插值 (球面线性插值)
            if method == 'slerp':
                interpolated_rots = self.slerp_quaternions(
                    start_rotations[bone], end_rotations[bone], t_values
                )
            else:
                # 线性插值后归一化
                interpolated_rots = []
                for t in t_values:
                    lerp_quat = (1-t) * start_rotations[bone] + t * end_rotations[bone]
                    lerp_quat = lerp_quat / np.linalg.norm(lerp_quat)
                    interpolated_rots.append(lerp_quat)
                interpolated_rots = np.array(interpolated_rots)
            
            # 平移插值 (线性插值)
            interpolated_trans = np.array([
                (1-t) * start_translations[bone] + t * end_translations[bone]
                for t in t_values
            ])
            
            # 构建变换矩阵
            for i, t in enumerate(t_values):
                transform = np.eye(4)
                
                # 旋转部分
                rotation_obj = Rotation.from_quat(interpolated_rots[i])
                transform[:3, :3] = rotation_obj.as_matrix()
                
                # 平移部分
                transform[:3, 3] = interpolated_trans[i]
                
                interpolated_transforms[i, bone] = transform
                
        return interpolated_transforms
        
    def slerp_quaternions(self, q1, q2, t_values):
        """球面线性插值四元数"""
        # 确保q1和q2在同一半球
        if np.dot(q1, q2) < 0:
            q2 = -q2
            
        interpolated = []
        for t in t_values:
            # SLERP公式
            dot_product = np.dot(q1, q2)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            if dot_product > 0.9995:
                # 如果四元数非常接近，使用线性插值
                result = (1-t) * q1 + t * q2
                result = result / np.linalg.norm(result)
            else:
                # 球面插值
                omega = np.arccos(abs(dot_product))
                sin_omega = np.sin(omega)
                
                coeff1 = np.sin((1-t) * omega) / sin_omega
                coeff2 = np.sin(t * omega) / sin_omega
                
                result = coeff1 * q1 + coeff2 * q2
                
            interpolated.append(result)
            
        return np.array(interpolated)
        
    def interpolate_sequence(self, start_frame, end_frame, num_interpolated=10, method='slerp'):
        """插值完整序列"""
        print(f"🎬 插值序列: Frame {start_frame} -> Frame {end_frame}")
        print(f"   插值帧数: {num_interpolated}")
        print(f"   插值方法: {method}")
        
        # 验证起始和结束帧
        print("🔍 验证起始和结束帧的变换质量...")
        start_error, _ = self.validate_transformation(start_frame)
        end_error, _ = self.validate_transformation(end_frame)
        
        if start_error > 0.1 or end_error > 0.1:
            print(f"⚠️  警告: 变换误差较大 (起始: {start_error:.3f}, 结束: {end_error:.3f})")
        
        # 1. 插值骨骼变换
        interpolated_bone_transforms = self.interpolate_bone_transforms(
            start_frame, end_frame, num_interpolated, method
        )
        
        # 2. 应用到网格
        interpolated_meshes = []
        for i in range(num_interpolated):
            print(f"   生成插值帧 {i+1}/{num_interpolated}")
            deformed_vertices = self.apply_bone_transforms_to_mesh(
                interpolated_bone_transforms[i]
            )
            interpolated_meshes.append(deformed_vertices)
            
        interpolated_meshes = np.array(interpolated_meshes)
        
        print(f"✅ 插值完成: {interpolated_meshes.shape}")
        
        return {
            'interpolated_meshes': interpolated_meshes,
            'interpolated_bone_transforms': interpolated_bone_transforms,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'method': method,
            'start_validation_error': start_error,
            'end_validation_error': end_error
        }
        
    def save_interpolation_results(self, results, output_dir, sequence_name="corrected_interpolation"):
        """保存修正后的插值结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存网格数据
        meshes_path = os.path.join(output_dir, f"{sequence_name}_meshes.npy")
        np.save(meshes_path, results['interpolated_meshes'])
        
        # 2. 保存骨骼变换
        transforms_path = os.path.join(output_dir, f"{sequence_name}_bone_transforms.npy")
        np.save(transforms_path, results['interpolated_bone_transforms'])
        
        # 3. 保存完整结果
        full_results = {
            **results,
            'corrected_rest_pose': self.rest_pose,
            'template_frame_idx': self.template_frame_idx,
            'triangles': self.triangles,
            'optimized_weights': self.optimized_weights,
            'joints': self.joints,
            'parents': self.parents
        }
        
        results_path = os.path.join(output_dir, f"{sequence_name}_complete_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(full_results, f)
            
        # 4. 保存验证OBJ文件
        print("💾 保存验证OBJ文件...")
        verification_dir = os.path.join(output_dir, "verification_objs")
        os.makedirs(verification_dir, exist_ok=True)
        
        # 保存起始帧、插值帧和结束帧进行对比
        frames_to_save = [
            (results['start_frame'], f"{sequence_name}_start_frame_{results['start_frame']:03d}.obj"),
            (0, f"{sequence_name}_interp_frame_000.obj"),
            (len(results['interpolated_meshes'])//2, f"{sequence_name}_interp_frame_mid.obj"),
            (len(results['interpolated_meshes'])-1, f"{sequence_name}_interp_frame_end.obj"),
            (results['end_frame'], f"{sequence_name}_end_frame_{results['end_frame']:03d}.obj")
        ]
        
        for frame_idx, filename in frames_to_save:
            mesh = o3d.geometry.TriangleMesh()
            
            if frame_idx == results['start_frame'] or frame_idx == results['end_frame']:
                # 原始帧
                mesh.vertices = o3d.utility.Vector3dVector(self.unified_meshes[frame_idx])
            else:
                # 插值帧
                mesh.vertices = o3d.utility.Vector3dVector(results['interpolated_meshes'][frame_idx])
                
            mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
            
            obj_path = os.path.join(verification_dir, filename)
            o3d.io.write_triangle_mesh(obj_path, mesh)
            
        print(f"✅ 修正插值结果已保存:")
        print(f"   网格数据: {meshes_path}")
        print(f"   骨骼变换: {transforms_path}")
        print(f"   完整结果: {results_path}")
        print(f"   验证文件: {verification_dir}")
        
        return results_path

def main():
    """测试修正后的插值系统"""
    # 输入路径
    unified_data_path = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\advanced_unified_results.pkl"
    optimized_weights_path = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembones_optimized\dembones_optimized_results.pkl"
    
    # 输出路径
    output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\corrected_interpolations"
    
    # 检查输入文件
    if not os.path.exists(unified_data_path):
        print(f"❌ 找不到统一数据文件: {unified_data_path}")
        return
    if not os.path.exists(optimized_weights_path):
        print(f"❌ 找不到优化权重文件: {optimized_weights_path}")
        return
        
    # 创建修正的插值器
    interpolator = CorrectedSkeletalInterpolator(unified_data_path, optimized_weights_path)
    
    print("\n🔧 测试修正后的插值系统...")
    
    # 验证几个关键帧的变换质量
    print("\n🔍 验证关键帧变换质量:")
    test_frames = [0, 30, 60, 90, 120, 156]
    for frame in test_frames:
        if frame < interpolator.unified_meshes.shape[0]:
            error, max_error = interpolator.validate_transformation(frame)
            print(f"   帧 {frame:3d}: 平均误差 {error:.6f}, 最大误差 {max_error:.6f}")
    
    # 测试插值
    print("\n📍 测试修正后的插值 (Frame 20 -> Frame 80)")
    result = interpolator.interpolate_sequence(
        start_frame=20, end_frame=80, num_interpolated=20, method='slerp'
    )
    
    results_path = interpolator.save_interpolation_results(
        result, output_dir, "test_corrected_interpolation"
    )
    
    print(f"\n🎉 修正插值测试完成!")
    print(f"📁 结果保存在: {results_path}")
    print(f"📊 验证误差: 起始帧 {result['start_validation_error']:.6f}, 结束帧 {result['end_validation_error']:.6f}")

if __name__ == "__main__":
    main()
