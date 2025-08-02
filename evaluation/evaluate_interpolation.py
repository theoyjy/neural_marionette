#!/usr/bin/env python3
"""
插值评估脚本
实现几何误差、法向一致性、ARAP、时间平滑度、物理合法性检查等指标
"""

import os
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import time
import glob
from utils_mesh import *

def evaluate_single_pair(pair_dir, method_results_dir, gt_data=None, no_gt=False):
    """评估单个关键帧对的插值结果"""
    results = {}
    print(f"Evaluate single pair: {method_results_dir}")
    # 加载skeleton信息
    skeleton_path = pair_dir / "skeleton.json"
    if skeleton_path.exists():
        skeleton_info = load_skeleton_info(skeleton_path)
        parents = np.array(skeleton_info['parents'])
        gt_frame_indices = skeleton_info.get('gt_frame_indices', [])
    else:
        parents = None
        gt_frame_indices = []
    
    # 查找插值结果文件
    # 插值脚本生成的文件名格式是 frame_*_with_colors.obj
    interpolated_files = sorted(glob.glob(str(method_results_dir / "frame_*_with_colors.obj")))
    
    if not interpolated_files:
        # 也尝试查找其他可能的文件名格式
        interpolated_files = sorted(glob.glob(str(method_results_dir / "*.obj")))
        # 过滤掉起始帧和结束帧
        interpolated_files = [f for f in interpolated_files if "start_frame" not in f and "end_frame" not in f]
    
    if not interpolated_files:
        print(f"Warning: No interpolated files found in {method_results_dir}")
        return None
    
    # 加载插值结果
    interpolated_vertices = []
    interpolated_normals = []
    interpolated_meshes = []
    
    for obj_file in interpolated_files:
        vertices, faces, normals = load_mesh(obj_file)
        interpolated_vertices.append(vertices)
        interpolated_normals.append(normals)
        
        mesh = trimesh.load(obj_file)
        interpolated_meshes.append(mesh)
    
    # 3.1 几何误差（需真实中间帧）
    if not no_gt and gt_data is not None and len(gt_frame_indices) > 0:
        chamfer_distances = []
        normal_angles = []
        arap_errors = []
                
        # 使用索引从原始数据中获取真实帧
        for i, frame_idx in enumerate(gt_frame_indices):
            print(f"Processing gt frame index: {frame_idx}")
            if frame_idx < gt_data['vertices'].shape[0] and i < len(interpolated_vertices):
                gt_vertices = gt_data['vertices'][frame_idx]
                gt_normals = gt_data['normals'][frame_idx] if 'normals' in gt_data else None
                
                # Chamfer距离
                chamfer_dist = compute_chamfer_distance(gt_vertices, interpolated_vertices[i])
                chamfer_distances.append(chamfer_dist)
                
                # 法向一致性
                if gt_normals is not None:
                    angles = compute_normal_consistency(gt_normals, interpolated_normals[i])
                    normal_angles.append(np.mean(angles))
                
                # ARAP误差
                arap_error = compute_arap_error(gt_vertices, interpolated_vertices[i])
                arap_errors.append(arap_error)
       
        
        if chamfer_distances:
            results['mean_chamfer'] = np.mean(chamfer_distances)
            results['max_chamfer'] = np.max(chamfer_distances)
            results['std_chamfer'] = np.std(chamfer_distances)
        
        if normal_angles:
            results['mean_normal_angle'] = np.mean(normal_angles)
            results['max_normal_angle'] = np.max(normal_angles)
        
        if arap_errors:
            results['mean_arap_error'] = np.mean(arap_errors)
            results['max_arap_error'] = np.max(arap_errors)
    
    # 3.2 时间平滑度（jerk）
    if len(interpolated_vertices) >= 3:
        mean_jerk, max_jerk = compute_jerk(interpolated_vertices)
        results['mean_jerk'] = mean_jerk
        results['max_jerk'] = max_jerk
    
    # 3.3 物理/合法性检查
    # 骨长标准差
    if parents is not None and len(interpolated_vertices) > 0:
        # 这里需要从插值结果中提取关节位置
        # 简化处理：使用顶点作为关节的近似
        joints_sequence = interpolated_vertices  # 简化处理
        bone_length_sd = compute_bone_length_sd(joints_sequence, parents)
        results['bone_length_sd'] = bone_length_sd
    
    # 自碰撞计数
    self_intersection_counts = []
    for mesh in interpolated_meshes:
        count = compute_self_intersection_count(mesh)
        self_intersection_counts.append(count)
    
    if self_intersection_counts:
        results['mean_self_intersection_count'] = np.mean(self_intersection_counts)
        results['max_self_intersection_count'] = np.max(self_intersection_counts)
        results['total_self_intersection_count'] = np.sum(self_intersection_counts)
    
    # 脚部滑动
    foot_slide_pixels = compute_foot_slide(interpolated_vertices)
    results['foot_slide_pixels'] = foot_slide_pixels
    
    # 帧数统计
    results['num_frames'] = len(interpolated_vertices)
    
    # 5. 读取性能数据
    performance_data = load_performance_data(method_results_dir, pair_dir.name)
    if performance_data:
        results.update(performance_data)
    
    return results

def load_performance_data(method_results_dir, pair_name):
    """加载性能监控数据"""
    import glob
    
    # 查找性能数据文件
    performance_pattern = str(method_results_dir / f"performance_*_{pair_name}.json")
    performance_files = glob.glob(performance_pattern)
    
    if not performance_files:
        return None
    
    # 使用最新的性能文件（如果有多个）
    performance_file = sorted(performance_files)[-1]
    
    try:
        import json
        with open(performance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取汇总数据
        summary = data.get('summary', {})
        
        # 转换为评估结果格式
        performance_results = {}
        
        # 基本性能指标
        performance_results['interpolation_time_seconds'] = summary.get('total_time_seconds', 0)
        performance_results['monitoring_samples'] = summary.get('sample_count', 0)
        
        # CPU和内存指标
        if summary.get('psutil_available', False):
            cpu_stats = summary.get('cpu_usage_percent', {})
            mem_stats = summary.get('memory_usage_gb', {})
            
            performance_results['cpu_usage_avg_percent'] = cpu_stats.get('avg', 0)
            performance_results['cpu_usage_max_percent'] = cpu_stats.get('max', 0)
            performance_results['memory_usage_avg_gb'] = mem_stats.get('avg', 0)
            performance_results['memory_usage_max_gb'] = mem_stats.get('max', 0)
        
        # GPU指标
        if summary.get('gpu_available', False):
            gpu_stats = summary.get('gpu_usage_percent', {})
            gpu_mem_stats = summary.get('gpu_memory_gb', {})
            
            performance_results['gpu_usage_avg_percent'] = gpu_stats.get('avg', 0)
            performance_results['gpu_usage_max_percent'] = gpu_stats.get('max', 0)
            performance_results['gpu_memory_avg_gb'] = gpu_mem_stats.get('avg', 0)
            performance_results['gpu_memory_max_gb'] = gpu_mem_stats.get('max', 0)
        
        # 计算每帧平均处理时间
        num_frames = summary.get('sample_count', 1)
        if num_frames > 0:
            performance_results['time_per_frame_seconds'] = performance_results['interpolation_time_seconds'] / num_frames
        
        return performance_results
        
    except Exception as e:
        print(f"Warning: Failed to load performance data from {performance_file}: {e}")
        return None


def load_gt_data(hdf5_path, subject_id, sequence_id):
    """Load ground truth data for comparison"""
    try:
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            sidseq = f"{subject_id}_{sequence_id}"
            print(f"Loading DFAUST data for {sidseq}")
        
            if sidseq not in f:
                raise ValueError(f"Sequence {sidseq} not found in {hdf5_path}")

            # Read data in the same way as the official script
            # verts = f[sidseq].value.transpose([2, 0, 1])  # Old version h5py
            vertices = f[sidseq][:].transpose([2, 0, 1])
            faces = f['faces'][:]  # All sequences share the same faces
            
            # calculate normals
            normals = []
            for i in range(len(vertices)):
                normals.append(np.zeros_like(vertices[i]))

            return {
                'vertices': vertices,
                'faces': faces,
                'normals': normals
            }

    except Exception as e:
        print(f"Load GT data failed: {e}")
        return None

def evaluate_all_pairs(pairs_dir, results_dir, methods, gt_data=None, no_gt=False, database_name="dfaust", subject_id="50002", sequence_id="jumping_jacks"):
    """Evaluate all keyframe pairs"""
    all_results = []
    
    # 读取关键帧对索引
    import json
    pairs_index_file = Path(pairs_dir) / "pairs_index.json"
    if pairs_index_file.exists():
        with open(pairs_index_file, 'r') as f:
            pairs_index = json.load(f)
        pair_dirs = [Path(p) for p in pairs_index['pair_dirs']]
    else:
        # 如果没有索引文件，扫描目录
        pair_dirs = [d for d in Path(pairs_dir).iterdir() if d.is_dir() and d.name.startswith("pair_")]
        pair_dirs.sort()
    
    print(f"Start evaluating {len(pair_dirs)} keyframe pairs")
    
    # 先检查哪些pairs实际有插值结果
    available_pairs = []
    for pair_dir in pair_dirs:
        # 检查是否至少有一个方法的结果存在
        has_results = False
        for method in methods:
            # 从pairs_index.json获取k值
            pairs_index_file = pair_dir.parent / "pairs_index.json"
            k_value = 10  # 默认值
            if pairs_index_file.exists():
                try:
                    import json
                    with open(pairs_index_file, 'r') as f:
                        pairs_info = json.load(f)
                        k_value = pairs_info.get('k', 10)
                except:
                    pass
            
            evaluation_output_dir = Path("evaluation") / method / database_name / f"{subject_id}_{sequence_id}_k{k_value}" / pair_dir.name / "intp"
            if evaluation_output_dir.exists():
                # 检查是否有实际的插值结果文件
                obj_files = list(evaluation_output_dir.glob("*.obj"))
                if obj_files:
                    has_results = True
                    break
        
        if has_results:
            available_pairs.append(pair_dir)
    
    if not available_pairs:
        print("No interpolation results found. Make sure to run interpolation first.")
        return []
    
    print(f"Found interpolation results for {len(available_pairs)} pairs")
    
    for i, pair_dir in enumerate(available_pairs):
        print(f"Evaluating keyframe pair {i+1}/{len(available_pairs)}: {pair_dir.name}")
        
        for method in methods:
            # 查找插值结果目录  
            # 新的路径结构: evaluation/method/database_name/subjectid_sequenceid/intp/
            output_dir = Path("evaluation")
            interpolation_dir = None
            
            if output_dir.exists():
                # 从pairs_index.json获取k值
                pairs_index_file = pair_dir.parent / "pairs_index.json"
                k_value = 10  # 默认值
                if pairs_index_file.exists():
                    try:
                        import json
                        with open(pairs_index_file, 'r') as f:
                            pairs_info = json.load(f)
                            k_value = pairs_info.get('k', 10)
                    except:
                        pass
                
                # 新的路径结构，包含k值和pair-specific子目录
                evaluation_output_dir = Path("evaluation") / method / database_name / f"{subject_id}_{sequence_id}_k{k_value}" / pair_dir.name / "intp"
                print(f"Evaluation output directory: {evaluation_output_dir}")
                if evaluation_output_dir.exists():
                    # 查找所有可能的插值目录（因为start_frame和end_frame可能不同）
                    interpolation_method_dir = evaluation_output_dir
                    print(f"Interpolation method directory: {interpolation_method_dir}")
                    if interpolation_method_dir.exists():
                        # 查找所有形如 "X_Y_Z" 的子目录，也检查直接在intp目录下的文件
                        pattern_dirs = list(interpolation_method_dir.glob("*_*_*"))
                        if not pattern_dirs:
                            # 如果没有找到子目录，检查intp目录本身
                            pattern_dirs = [interpolation_method_dir]
                        print(f"Pattern directories: {pattern_dirs}")
                        for potential_dir in pattern_dirs:
                            # 检查是否有插值结果文件
                            obj_files = []
                            
                            if potential_dir.is_file() and potential_dir.suffix == '.obj':
                                # 如果pattern_dirs包含了直接的obj文件
                                obj_files = [potential_dir]
                                interpolation_dir = potential_dir.parent
                            else:
                                # 如果是目录，查找其中的obj文件
                                obj_files = list(potential_dir.glob("frame_*_with_colors.obj"))
                                if not obj_files:
                                    # 也尝试查找其他可能的文件名，包括interpolated_frame_*.obj
                                    obj_files = list(potential_dir.glob("interpolated_frame_*.obj"))
                                if not obj_files:
                                    obj_files = list(potential_dir.glob("*.obj"))
                                    obj_files = [f for f in obj_files if "start_frame" not in f.name and "end_frame" not in f.name]
                                
                                if obj_files:
                                    interpolation_dir = potential_dir
                            
                            if obj_files:
                                print(f"  {method}: Found {len(obj_files)} interpolated files (evaluation mode) in {interpolation_dir.name}")
                                break
                        
                        if interpolation_dir:
                            print(f"  {method}: Found interpolation directory: {interpolation_dir}")                
                
                if interpolation_dir is None:
                    print(f"  {method}: Result directory does not exist")
                    continue
            
            print(f"  {method}: {interpolation_dir}")
            
            results = evaluate_single_pair(pair_dir, interpolation_dir, gt_data, no_gt)
            
            if results:
                results['pair_id'] = pair_dir.name
                results['method'] = method
                all_results.append(results)
                print(f"  {method}: Completed")
            else:
                print(f"  {method}: Failed")
    
    return all_results

def save_results_to_csv(results, output_path):
    """保存结果到CSV文件"""
    if not results:
        print("No results to save")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return df

def generate_markdown_report(df, output_path):
    """生成Markdown报告"""
    if df.empty:
        print("No data to generate report")
        return
    
    report = []
    report.append("# Interpolation Evaluation Report\n")
    
    # 总体统计
    report.append("## Overall Statistics\n")
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        report.append(f"### {method} Method\n")
        
        # 计算平均值
        numeric_cols = method_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['pair_id']:
                mean_val = method_data[col].mean()
                std_val = method_data[col].std()
                report.append(f"- {col}: {mean_val:.6f} ± {std_val:.6f}\n")
        report.append("\n")
    
    # 方法对比
    if len(df['method'].unique()) > 1:
        report.append("## Method Comparison\n")
        
        baseline_data = df[df['method'] == 'baseline']
        dual_ref_data = df[df['method'] == 'dual_reference']
        
        if not baseline_data.empty and not dual_ref_data.empty:
            # 计算改进百分比
            comparison_cols = ['mean_chamfer', 'mean_jerk', 'mean_arap_error', 'bone_length_sd']
            
            for col in comparison_cols:
                if col in baseline_data.columns and col in dual_ref_data.columns:
                    baseline_mean = baseline_data[col].mean()
                    dual_ref_mean = dual_ref_data[col].mean()
                    
                    if baseline_mean > 0:
                        improvement = (baseline_mean - dual_ref_mean) / baseline_mean * 100
                        report.append(f"- {col}: {improvement:+.2f}% Improvement\n")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    print(f"Report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Interpolation evaluation script")
    parser.add_argument("--pairs_dir", type=str, 
                       default="evaluation/data/dfaust/keyframe_pairs", 
                       help="Keyframe pairs directory (如果指定具体路径，应包含database_name/subject_sequence子目录)")
    parser.add_argument("--results_dir", type=str, 
                       default="evaluation/results",
                       help="Interpolation results directory")
    parser.add_argument("--methods", nargs="+", 
                       default=["baseline", "dual_reference"],
                       help="Methods to evaluate")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/results",
                       help="Evaluation results output directory")
    parser.add_argument("--no_gt", action="store_true",
                       help="No GT mode, only calculate internal metrics")
    parser.add_argument("--gt_hdf5", type=str,
                       default="evaluation/data/dfaust/registrations_m.hdf5",
                       help="GT data HDF5 file path")
    parser.add_argument("--subject_id", type=str, default="50002",
                       help="DFAUST subject ID")
    parser.add_argument("--sequence_id", type=str, default="jumping_jacks",
                       help="DFAUST sequence ID")
    parser.add_argument("--database_name", type=str, default="dfaust",
                       help="数据库名称")
    
    args = parser.parse_args()
    
    # 从hdf5路径提取数据库名
    if not hasattr(args, 'database_name') or not args.database_name:
        hdf5_filename = Path(args.gt_hdf5).stem
        if "registrations_" in hdf5_filename:
            args.database_name = "dfaust"
        else:
            args.database_name = hdf5_filename.split("_")[0] if "_" in hdf5_filename else "unknown"
    
    # 加载GT数据
    gt_data = None
    if not args.no_gt:
        gt_data = load_gt_data(args.gt_hdf5, args.subject_id, args.sequence_id)
        if gt_data is None:
            print("Warning: Unable to load GT data, switching to no GT mode")
            args.no_gt = True
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"评估路径: evaluation/{args.database_name}/{args.subject_id}_{args.sequence_id}/")
    
    # 评估所有关键帧对
    start_time = time.time()
    results = evaluate_all_pairs(args.pairs_dir, args.results_dir, args.methods, gt_data, args.no_gt, args.database_name, args.subject_id, args.sequence_id)
    end_time = time.time()
    
    print(f"Evaluation completed, time taken: {end_time - start_time:.2f} seconds")
    
    if results:
        # 保存结果到CSV
        csv_path = output_dir / "results.csv"
        df = save_results_to_csv(results, csv_path)
        
        # 生成Markdown报告
        md_path = output_dir / "evaluation_report.md"
        generate_markdown_report(df, md_path)
        
        print(f"Evaluation completed! Results saved to: {output_dir}")
    else:
        print("No evaluation results")

if __name__ == "__main__":
    main() 