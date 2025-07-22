#!/usr/bin/env python3
"""
高级骨骼驱动插值系统
基于DemBones优化权重和统一拓扑的极高质量插值

特点:
1. 使用DemBones优化后的权重进行插值
2. 骨骼空间插值确保自然运动
3. 统一拓扑保证顶点一致性
4. 多种插值模式（线性、球面、样条）
"""

import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import open3d as o3d

class AdvancedSkeletalInterpolator:
    def __init__(self, optimized_data_path):
        """初始化高级骨骼插值器"""
        self.optimized_data_path = optimized_data_path
        self.load_optimized_data()
        
    def load_optimized_data(self):
        """加载DemBones优化数据"""
        print("🔄 加载DemBones优化数据...")
        with open(self.optimized_data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.optimized_weights = data['optimized_weights']  # (32140, 24)
        self.bone_transforms = data['bone_transforms']  # (157, 24, 4, 4)
        self.unified_meshes = data['unified_meshes']  # (157, 32140, 3)
        self.rest_pose = data['rest_pose']  # (32140, 3)
        self.triangles = data['triangles']  # (40000, 3)
        self.joints = data['joints']  # (24, 3)
        self.parents = data['parents']  # (24,)
        
        print(f"✅ 加载完成:")
        print(f"   优化权重: {self.optimized_weights.shape}")
        print(f"   骨骼变换: {self.bone_transforms.shape}")
        print(f"   统一网格: {self.unified_meshes.shape}")
        
        # 预计算骨骼旋转和平移
        self.bone_rotations, self.bone_translations = self.extract_bone_rt()
        
    def extract_bone_rt(self):
        """从骨骼变换矩阵提取旋转和平移"""
        print("🔧 提取骨骼旋转和平移数据...")
        
        num_frames, num_bones = self.bone_transforms.shape[:2]
        
        rotations = np.zeros((num_frames, num_bones, 4))  # 四元数
        translations = np.zeros((num_frames, num_bones, 3))
        
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
                    rotations[frame, bone] = quat
                except:
                    # 如果旋转矩阵不正确，使用单位四元数
                    rotations[frame, bone] = [0, 0, 0, 1]
                    
                translations[frame, bone] = t
                
        return rotations, translations
        
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
        
    def apply_bone_transforms_to_mesh(self, bone_transforms):
        """应用骨骼变换到网格"""
        num_vertices = self.rest_pose.shape[0]
        num_bones = bone_transforms.shape[0]
        
        deformed_vertices = np.zeros((num_vertices, 3))
        
        for vertex_idx in range(num_vertices):
            rest_pos = self.rest_pose[vertex_idx]
            rest_homo = np.append(rest_pos, 1.0)  # 齐次坐标
            
            # 加权骨骼变换
            final_pos = np.zeros(3)
            for bone in range(num_bones):
                weight = self.optimized_weights[vertex_idx, bone]
                if weight > 1e-6:
                    transform = bone_transforms[bone]
                    transformed = (transform @ rest_homo)[:3]
                    final_pos += weight * transformed
                    
            deformed_vertices[vertex_idx] = final_pos
            
        return deformed_vertices
        
    def interpolate_sequence(self, start_frame, end_frame, num_interpolated=10, method='slerp'):
        """插值完整序列"""
        print(f"🎬 插值序列: Frame {start_frame} -> Frame {end_frame}")
        print(f"   插值帧数: {num_interpolated}")
        print(f"   插值方法: {method}")
        
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
            'method': method
        }
        
    def interpolate_multiple_segments(self, frame_pairs, num_interpolated_per_segment=10, method='slerp'):
        """插值多个片段"""
        print(f"🎭 插值多个片段: {len(frame_pairs)}个片段")
        
        all_interpolated = []
        all_bone_transforms = []
        
        for i, (start_frame, end_frame) in enumerate(frame_pairs):
            print(f"\n📍 片段 {i+1}/{len(frame_pairs)}: {start_frame} -> {end_frame}")
            
            result = self.interpolate_sequence(
                start_frame, end_frame, num_interpolated_per_segment, method
            )
            
            all_interpolated.append(result['interpolated_meshes'])
            all_bone_transforms.append(result['interpolated_bone_transforms'])
            
        # 合并所有插值结果
        combined_meshes = np.concatenate(all_interpolated, axis=0)
        combined_transforms = np.concatenate(all_bone_transforms, axis=0)
        
        print(f"\n✅ 多片段插值完成:")
        print(f"   总插值帧数: {combined_meshes.shape[0]}")
        print(f"   网格形状: {combined_meshes.shape}")
        
        return {
            'interpolated_meshes': combined_meshes,
            'interpolated_bone_transforms': combined_transforms,
            'frame_pairs': frame_pairs,
            'method': method
        }
        
    def smooth_interpolation_with_spline(self, key_frames, num_interpolated_total=100):
        """使用样条曲线进行平滑插值"""
        print(f"🌊 样条曲线平滑插值: {len(key_frames)}个关键帧")
        
        if len(key_frames) < 2:
            raise ValueError("至少需要2个关键帧进行样条插值")
            
        # 准备关键帧数据
        key_bone_rotations = []
        key_bone_translations = []
        key_frame_indices = []
        
        for frame_idx in key_frames:
            key_bone_rotations.append(self.bone_rotations[frame_idx])
            key_bone_translations.append(self.bone_translations[frame_idx])
            key_frame_indices.append(frame_idx)
            
        key_bone_rotations = np.array(key_bone_rotations)  # (num_keys, 24, 4)
        key_bone_translations = np.array(key_bone_translations)  # (num_keys, 24, 3)
        
        # 创建插值时间轴
        t_key = np.array(key_frame_indices, dtype=float)
        t_interp = np.linspace(t_key[0], t_key[-1], num_interpolated_total)
        
        num_bones = key_bone_rotations.shape[1]
        interpolated_transforms = np.zeros((num_interpolated_total, num_bones, 4, 4))
        
        # 对每个骨骼进行样条插值
        for bone in range(num_bones):
            print(f"   处理骨骼 {bone+1}/{num_bones}")
            
            # 平移的样条插值
            trans_interpolator = interp1d(
                t_key, key_bone_translations[:, bone], 
                kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate'
            )
            interpolated_translations = trans_interpolator(t_interp)
            
            # 旋转的SLERP插值 (分段处理)
            interpolated_rotations = self.spline_interpolate_rotations(
                t_key, key_bone_rotations[:, bone], t_interp
            )
            
            # 构建变换矩阵
            for i in range(num_interpolated_total):
                transform = np.eye(4)
                
                # 旋转部分
                rotation_obj = Rotation.from_quat(interpolated_rotations[i])
                transform[:3, :3] = rotation_obj.as_matrix()
                
                # 平移部分
                transform[:3, 3] = interpolated_translations[i]
                
                interpolated_transforms[i, bone] = transform
                
        # 应用到网格
        print("🎨 应用样条插值到网格...")
        interpolated_meshes = []
        for i in range(num_interpolated_total):
            if i % 20 == 0:
                print(f"   生成帧 {i+1}/{num_interpolated_total}")
            deformed_vertices = self.apply_bone_transforms_to_mesh(
                interpolated_transforms[i]
            )
            interpolated_meshes.append(deformed_vertices)
            
        interpolated_meshes = np.array(interpolated_meshes)
        
        print(f"✅ 样条插值完成: {interpolated_meshes.shape}")
        
        return {
            'interpolated_meshes': interpolated_meshes,
            'interpolated_bone_transforms': interpolated_transforms,
            'key_frames': key_frames,
            'method': 'spline'
        }
        
    def spline_interpolate_rotations(self, t_key, key_rotations, t_interp):
        """对旋转进行分段SLERP插值"""
        interpolated = []
        
        for t in t_interp:
            # 找到t所在的区间
            if t <= t_key[0]:
                interpolated.append(key_rotations[0])
            elif t >= t_key[-1]:
                interpolated.append(key_rotations[-1])
            else:
                # 找到插值区间
                for i in range(len(t_key) - 1):
                    if t_key[i] <= t <= t_key[i + 1]:
                        # 区间内插值参数
                        local_t = (t - t_key[i]) / (t_key[i + 1] - t_key[i])
                        
                        # SLERP插值
                        q1 = key_rotations[i]
                        q2 = key_rotations[i + 1]
                        
                        # 确保在同一半球
                        if np.dot(q1, q2) < 0:
                            q2 = -q2
                            
                        # SLERP
                        dot_product = np.dot(q1, q2)
                        dot_product = np.clip(dot_product, -1.0, 1.0)
                        
                        if dot_product > 0.9995:
                            result = (1-local_t) * q1 + local_t * q2
                            result = result / np.linalg.norm(result)
                        else:
                            omega = np.arccos(abs(dot_product))
                            sin_omega = np.sin(omega)
                            
                            coeff1 = np.sin((1-local_t) * omega) / sin_omega
                            coeff2 = np.sin(local_t * omega) / sin_omega
                            
                            result = coeff1 * q1 + coeff2 * q2
                            
                        interpolated.append(result)
                        break
                        
        return np.array(interpolated)
        
    def save_interpolation_results(self, results, output_dir, sequence_name="interpolated"):
        """保存插值结果"""
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
            'rest_pose': self.rest_pose,
            'triangles': self.triangles,
            'optimized_weights': self.optimized_weights,
            'joints': self.joints,
            'parents': self.parents
        }
        
        results_path = os.path.join(output_dir, f"{sequence_name}_complete_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(full_results, f)
            
        # 4. 保存验证OBJ文件（前5帧）
        print("💾 保存验证OBJ文件...")
        verification_dir = os.path.join(output_dir, "verification_objs")
        os.makedirs(verification_dir, exist_ok=True)
        
        num_verification_frames = min(5, results['interpolated_meshes'].shape[0])
        for i in range(num_verification_frames):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(results['interpolated_meshes'][i])
            mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
            
            obj_path = os.path.join(verification_dir, f"{sequence_name}_frame_{i:03d}.obj")
            o3d.io.write_triangle_mesh(obj_path, mesh)
            
        print(f"✅ 插值结果已保存:")
        print(f"   网格数据: {meshes_path}")
        print(f"   骨骼变换: {transforms_path}")
        print(f"   完整结果: {results_path}")
        print(f"   验证文件: {verification_dir}")
        
        return results_path

def main():
    """演示高级骨骼插值功能"""
    # 输入路径
    optimized_data_path = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembones_optimized\dembones_optimized_results.pkl"
    
    # 输出路径
    output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\advanced_interpolations"
    
    # 检查输入文件
    if not os.path.exists(optimized_data_path):
        print(f"❌ 找不到优化数据文件: {optimized_data_path}")
        return
        
    # 创建插值器
    interpolator = AdvancedSkeletalInterpolator(optimized_data_path)
    
    print("\n🎬 开始高级骨骼驱动插值演示...")
    
    # 演示1: 简单两帧插值
    print("\n📍 演示1: 简单两帧插值 (Frame 10 -> Frame 30)")
    result1 = interpolator.interpolate_sequence(
        start_frame=10, end_frame=30, num_interpolated=15, method='slerp'
    )
    interpolator.save_interpolation_results(result1, output_dir, "demo1_simple")
    
    # 演示2: 多片段插值
    print("\n📍 演示2: 多片段插值")
    frame_pairs = [(5, 25), (50, 80), (100, 130)]
    result2 = interpolator.interpolate_multiple_segments(
        frame_pairs, num_interpolated_per_segment=8, method='slerp'
    )
    interpolator.save_interpolation_results(result2, output_dir, "demo2_multi_segment")
    
    # 演示3: 样条曲线平滑插值
    print("\n📍 演示3: 样条曲线平滑插值")
    key_frames = [0, 30, 60, 90, 120, 156]
    result3 = interpolator.smooth_interpolation_with_spline(
        key_frames, num_interpolated_total=80
    )
    interpolator.save_interpolation_results(result3, output_dir, "demo3_spline")
    
    print("\n🎉 所有插值演示完成!")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
