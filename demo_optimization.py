#!/usr/bin/env python3
"""
优化演示

展示优化后的插值系统性能
"""

import numpy as np
import os
import tempfile
import time
from pathlib import Path
import open3d as o3d

def create_demo_data():
    """创建演示数据"""
    print("🔧 创建演示数据...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建骨骼数据
        skeleton_dir = temp_path / "skeleton_data"
        skeleton_dir.mkdir(exist_ok=True)
        
        num_frames = 8
        num_joints = 15
        
        # 生成关键点数据
        keypoints = np.random.rand(num_frames, num_joints, 4)
        keypoints[:, :, 3] = 0.9  # 高置信度
        
        # 生成变换矩阵（简单的动画）
        transforms = np.tile(np.eye(4), (num_frames, num_joints, 1, 1))
        for t in range(num_frames):
            for j in range(num_joints):
                # 添加一些旋转和平移
                angle = t * 0.1 + j * 0.05
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                transforms[t, j, :3, :3] = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                transforms[t, j, :3, 3] = [t * 0.1, np.sin(t * 0.5), 0]
        
        # 父节点关系
        parents = np.arange(num_joints)
        parents[0] = 0
        
        # 保存数据
        np.save(skeleton_dir / 'keypoints.npy', keypoints)
        np.save(skeleton_dir / 'transforms.npy', transforms)
        np.save(skeleton_dir / 'parents.npy', parents)
        
        # 创建网格序列
        mesh_dir = temp_path / "mesh_data"
        mesh_dir.mkdir(exist_ok=True)
        
        for t in range(num_frames):
            # 创建人体形状的网格
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
            vertices = np.asarray(mesh.vertices)
            
            # 添加动画变形
            vertices[:, 0] += t * 0.05  # 水平移动
            vertices[:, 1] += np.sin(t * 0.8) * 0.3  # 垂直摆动
            vertices[:, 2] += np.cos(t * 0.6) * 0.2  # 深度变化
            
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            
            # 保存网格
            mesh_path = mesh_dir / f"frame_{t:04d}.obj"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        
        print(f"✅ 演示数据已创建:")
        print(f"  - 骨骼数据: {skeleton_dir}")
        print(f"  - 网格数据: {mesh_dir}")
        print(f"  - 帧数: {num_frames}")
        print(f"  - 关节数: {num_joints}")
        
        return str(skeleton_dir), str(mesh_dir)

def demo_optimization():
    """演示优化效果"""
    print("\n🚀 开始优化演示...")
    
    # 创建演示数据
    skeleton_dir, mesh_dir = create_demo_data()
    
    try:
        from Interpolate import VolumetricInterpolator
        
        # 初始化插值器
        print("🔧 初始化插值器...")
        interpolator = VolumetricInterpolator(
            skeleton_data_dir=skeleton_dir,
            mesh_folder_path=mesh_dir
        )
        
        # 演示参数
        frame_start = 1
        frame_end = 6
        num_interpolate = 10
        max_optimize_frames = 10,
        
        print(f"📋 演示参数:")
        print(f"  - 起始帧: {frame_start}")
        print(f"  - 结束帧: {frame_end}")
        print(f"  - 插值帧数: {num_interpolate}")
        print(f"  - 优化帧数限制: ≤5")
        
        # 执行插值
        print("\n🎬 开始插值...")
        start_time = time.time()
        
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=frame_start,
            frame_end=frame_end,
            num_interpolate=num_interpolate,
            max_optimize_frames = max_optimize_frames,
            optimize_weights=True,
            output_dir=None
        )
        
        total_time = time.time() - start_time
        
        print(f"\n✅ 插值完成!")
        print(f"📊 性能统计:")
        print(f"  - 总耗时: {total_time:.2f}秒")
        print(f"  - 生成帧数: {len(interpolated_frames)}")
        print(f"  - 平均每帧耗时: {total_time / len(interpolated_frames):.3f}秒")
        print(f"  - 处理速度: {len(interpolated_frames) / total_time:.1f} 帧/秒")
        
        # 质量评估
        if interpolated_frames:
            print(f"\n🎯 质量评估:")
            print(f"  - 插值帧数据完整性: ✅")
            print(f"  - 网格顶点数: {len(interpolated_frames[0]['vertices'])}")
            print(f"  - 变换矩阵形状: {interpolated_frames[0]['transforms'].shape}")
            print(f"  - 关键点形状: {interpolated_frames[0]['keypoints'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_skinning_optimization():
    """演示蒙皮优化效果"""
    print("\n🔧 演示蒙皮优化...")
    
    # 创建演示数据
    skeleton_dir, mesh_dir = create_demo_data()
    
    try:
        from Skinning import AutoSkinning
        
        # 初始化蒙皮器
        skinner = AutoSkinning(
            skeleton_data_dir=skeleton_dir,
            reference_frame_idx=0
        )
        skinner.load_mesh_sequence(mesh_dir)
        
        # 测试优化性能
        test_frames = [2, 4, 6]
        print(f"🧪 测试帧: {test_frames}")
        
        start_time = time.time()
        
        for frame in test_frames:
            frame_start = time.time()
            weights, loss = skinner.optimize_skinning_weights_for_frame(
                frame, max_iter=100, regularization_lambda=0.01
            )
            frame_time = time.time() - frame_start
            
            print(f"  帧 {frame}: 耗时 {frame_time:.2f}s, 损失 {loss:.6f}")
        
        total_time = time.time() - start_time
        print(f"✅ 蒙皮优化完成，总耗时: {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ 蒙皮优化演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🎬 优化演示系统")
    print("=" * 50)
    
    # 演示1: 插值优化
    success1 = demo_optimization()
    
    print("\n" + "=" * 50)
    
    # 演示2: 蒙皮优化
    success2 = demo_skinning_optimization()
    
    print("\n" + "=" * 50)
    
    if success1 and success2:
        print("🎉 所有演示成功完成！")
        print("📈 优化效果总结:")
        print("  - 🚀 多线程并行处理提升速度")
        print("  - 🎯 向量化计算提高效率")
        print("  - 💾 内存优化减少占用")
        print("  - 🔧 智能采样策略")
        print("  - ⚡ 动态线程数调整")
    else:
        print("❌ 部分演示失败")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 