#!/usr/bin/env python3
"""
快速LBS权重测试脚本

简单测试已优化的LBS权重在不同帧上的重建质量
"""

import numpy as np
import time
from pathlib import Path
from Skinning import InverseMeshCanonicalizer

def quick_lbs_test():
    """快速测试LBS权重质量"""
    
    print("🔍 快速LBS权重测试")
    print("=" * 50)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_path = "output/skinning_weights_fast.npz"
    reference_frame = 5
    
    # 初始化
    print("正在初始化...")
    canonicalizer = InverseMeshCanonicalizer(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=reference_frame
    )
    
    # 加载网格和权重
    canonicalizer.load_mesh_sequence(mesh_folder_path)
    
    if not canonicalizer.load_skinning_weights(weights_path):
        print("❌ 无法加载权重文件")
        return
    
    print(f"✅ 成功加载权重: {canonicalizer.skinning_weights.shape}")
    
    # 选择测试帧
    total_frames = len(canonicalizer.mesh_files)
    if total_frames <= 15:
        test_frames = list(range(total_frames))
    else:
        # 选择代表性帧
        test_frames = [0, 1, 2]  # 开始
        test_frames.extend([total_frames//4, total_frames//2, 3*total_frames//4])  # 中间
        test_frames.extend([total_frames-3, total_frames-2, total_frames-1])  # 结束
        # 移除参考帧，避免重复
        if reference_frame in test_frames:
            test_frames.remove(reference_frame)
        test_frames = sorted(test_frames)
    
    print(f"测试帧: {test_frames}")
    print(f"参考帧: {reference_frame}")
    
    # 测试每一帧
    print(f"\n🧪 开始测试 {len(test_frames)} 帧:")
    print("帧号   距离参考  平均误差    最大误差    RMSE      时间(s)")
    print("-" * 60)
    
    results = []
    total_start_time = time.time()
    
    for frame_idx in test_frames:
        if frame_idx >= total_frames:
            continue
            
        # 测试单帧
        frame_result = test_single_frame_reconstruction(canonicalizer, frame_idx)
        if frame_result:
            results.append(frame_result)
            
            # 输出结果
            distance = abs(frame_idx - reference_frame)
            print(f"{frame_idx:3d}    {distance:4d}     {frame_result['mean_error']:.6f}  "
                  f"{frame_result['max_error']:.6f}  {frame_result['rmse']:.6f}  "
                  f"{frame_result['time']:.3f}")
    
    total_time = time.time() - total_start_time
    
    # 计算总体统计
    if results:
        all_mean_errors = [r['mean_error'] for r in results]
        all_max_errors = [r['max_error'] for r in results]
        all_rmse = [r['rmse'] for r in results]
        all_times = [r['time'] for r in results]
        
        print("-" * 60)
        print(f"📊 总体统计 (基于 {len(results)} 帧):")
        print(f"平均重建误差: {np.mean(all_mean_errors):.6f} ± {np.std(all_mean_errors):.6f}")
        print(f"最大误差范围: [{np.min(all_max_errors):.6f}, {np.max(all_max_errors):.6f}]")
        print(f"平均RMSE: {np.mean(all_rmse):.6f}")
        print(f"平均处理时间: {np.mean(all_times):.3f}s")
        print(f"总测试时间: {total_time:.2f}s")
        print(f"估计帧率: {len(results)/np.sum(all_times):.1f} FPS")
        
        # 分析误差与距离的关系
        distances = [abs(r['frame_idx'] - reference_frame) for r in results]
        correlation = np.corrcoef(distances, all_mean_errors)[0, 1]
        print(f"误差与距离参考帧的相关性: {correlation:.3f}")
        
        # 质量评估
        avg_error = np.mean(all_mean_errors)
        if avg_error < 0.01:
            quality = "优秀"
        elif avg_error < 0.02:
            quality = "良好"
        elif avg_error < 0.05:
            quality = "一般"
        else:
            quality = "需要改进"
        
        print(f"\n🎯 质量评估: {quality} (平均误差: {avg_error:.6f})")
        
        # 保存简单结果
        save_simple_results(results, canonicalizer, "output/lbs_quick_test_results.txt")
        
    else:
        print("❌ 没有成功测试任何帧")

def test_single_frame_reconstruction(canonicalizer, frame_idx):
    """测试单帧重建"""
    try:
        import open3d as o3d
        
        # 加载目标网格
        target_mesh = o3d.io.read_triangle_mesh(str(canonicalizer.mesh_files[frame_idx]))
        target_vertices = np.asarray(target_mesh.vertices)
        
        # 归一化
        if frame_idx not in canonicalizer.frame_normalization_params:
            canonicalizer.frame_normalization_params[frame_idx] = \
                canonicalizer.compute_mesh_normalization_params(target_mesh)
        
        target_vertices_norm = canonicalizer.normalize_mesh_vertices(
            target_vertices, 
            canonicalizer.frame_normalization_params[frame_idx]
        )
        
        rest_vertices_norm = canonicalizer.normalize_mesh_vertices(
            canonicalizer.rest_pose_vertices, 
            canonicalizer.frame_normalization_params[canonicalizer.reference_frame_idx]
        )
        
        # 计算相对变换
        target_transforms = canonicalizer.transforms[frame_idx]
        rest_transforms = canonicalizer.transforms[canonicalizer.reference_frame_idx]
        
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(canonicalizer.num_joints):
            if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            else:
                relative_transforms[j] = np.eye(4)
        
        # LBS重建
        start_time = time.time()
        predicted_vertices = canonicalizer.apply_lbs_transform(
            rest_vertices_norm, 
            canonicalizer.skinning_weights, 
            relative_transforms
        )
        lbs_time = time.time() - start_time
        
        # 计算误差
        vertex_errors = np.linalg.norm(predicted_vertices - target_vertices_norm, axis=1)
        
        return {
            'frame_idx': frame_idx,
            'mean_error': float(np.mean(vertex_errors)),
            'max_error': float(np.max(vertex_errors)),
            'min_error': float(np.min(vertex_errors)),
            'std_error': float(np.std(vertex_errors)),
            'rmse': float(np.sqrt(np.mean(vertex_errors**2))),
            'time': lbs_time,
            'num_vertices': len(vertex_errors)
        }
        
    except Exception as e:
        print(f"❌ 测试帧 {frame_idx} 失败: {e}")
        return None

def save_simple_results(results, canonicalizer, output_path):
    """保存简单的测试结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("LBS权重快速测试结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"权重矩阵形状: {canonicalizer.skinning_weights.shape}\n")
        f.write(f"参考帧: {canonicalizer.reference_frame_idx}\n")
        f.write(f"测试帧数: {len(results)}\n\n")
        
        f.write("详细结果:\n")
        f.write("帧号\t距离参考\t平均误差\t最大误差\tRMSE\t\t时间(s)\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            distance = abs(result['frame_idx'] - canonicalizer.reference_frame_idx)
            f.write(f"{result['frame_idx']}\t{distance}\t\t{result['mean_error']:.6f}\t"
                   f"{result['max_error']:.6f}\t{result['rmse']:.6f}\t{result['time']:.3f}\n")
        
        # 总体统计
        all_mean_errors = [r['mean_error'] for r in results]
        all_times = [r['time'] for r in results]
        
        f.write("\n总体统计:\n")
        f.write(f"平均重建误差: {np.mean(all_mean_errors):.6f}\n")
        f.write(f"误差标准差: {np.std(all_mean_errors):.6f}\n")
        f.write(f"平均处理时间: {np.mean(all_times):.3f}s\n")
        f.write(f"估计帧率: {len(results)/np.sum(all_times):.1f} FPS\n")
    
    print(f"📄 结果已保存: {output_path}")

def test_specific_frames():
    """测试特定帧（用于调试）"""
    print("🔍 测试特定帧")
    
    # 配置
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_path = "output/skinning_weights_fast.npz"
    reference_frame = 5
    
    # 要测试的特定帧
    test_frames = [0, 10, 20, 30, 40]  # 修改为你想测试的帧
    
    # 初始化
    canonicalizer = InverseMeshCanonicalizer(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=reference_frame
    )
    
    canonicalizer.load_mesh_sequence(mesh_folder_path)
    
    if not canonicalizer.load_skinning_weights(weights_path):
        print("❌ 无法加载权重文件")
        return
    
    print(f"测试特定帧: {test_frames}")
    
    for frame_idx in test_frames:
        if frame_idx >= len(canonicalizer.mesh_files):
            print(f"⚠️  帧 {frame_idx} 超出范围")
            continue
            
        result = test_single_frame_reconstruction(canonicalizer, frame_idx)
        if result:
            print(f"帧 {frame_idx}: 平均误差={result['mean_error']:.6f}, "
                  f"最大误差={result['max_error']:.6f}, 时间={result['time']:.3f}s")

if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "specific":
        test_specific_frames()
    else:
        quick_lbs_test()
