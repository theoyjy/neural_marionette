#!/usr/bin/env python3
"""
运行插值脚本
对每个关键帧对运行baseline和dual_reference方法
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
import time
from performance_monitor import PerformanceMonitor

def run_interpolation_for_pair(pair_info_path, method, output_dir, database_name=None, subject_id=None, sequence_id=None):
    """对单个关键帧对运行插值"""
    
    # 读取pair信息
    with open(pair_info_path, 'r') as f:
        pair_info = json.load(f)
    
    # 获取数据库目录（frame_xxx.obj文件所在目录）
    pair_info_path = Path(pair_info_path)
    # Frame文件在k值目录的父目录中（共享目录）
    database_dir = pair_info_path.parent.parent  # k15/ -> 50002_jumping_jacks/
    
    # 验证必要的frame文件是否存在
    start_idx = pair_info['start_idx']
    end_idx = pair_info['end_idx']
    gt_frame_indices = pair_info['gt_frame_indices']
    
    # 检查关键帧文件
    start_frame_path = database_dir / f"frame_{start_idx:03d}.obj"
    end_frame_path = database_dir / f"frame_{end_idx:03d}.obj"
    
    if not start_frame_path.exists() or not end_frame_path.exists():
        print(f"警告: 无法找到关键帧文件 {start_frame_path} 或 {end_frame_path}")
        return False
    
    # 检查优化帧文件
    missing_frames = []
    for frame_idx in gt_frame_indices:
        frame_path = database_dir / f"frame_{frame_idx:03d}.obj"
        if not frame_path.exists():
            missing_frames.append(frame_idx)
    
    if missing_frames:
        print(f"警告: 缺少优化帧文件: {missing_frames}")
        return False

    
    # 创建输出目录 - 使用新的路径结构
    if database_name and subject_id and sequence_id:
        # 从pair_info_path推断k值
        pair_info_path = Path(pair_info_path)
        k_dir = pair_info_path.parent  # k值目录
        k_dir_name = k_dir.name  # 例如 "k15"
        k_value = int(k_dir_name[1:]) if k_dir_name.startswith('k') else 10  # 提取k值
        
        # 获取pair文件名（不带扩展名）作为子目录名
        pair_name = pair_info_path.stem  # 例如 "pair_000"
        
        # 新的路径结构，包含k值和pair-specific子目录
        method_output_dir = Path(output_dir) / database_name / f"{subject_id}_{sequence_id}_k{k_value}" / pair_name
    else:
        # 保持原有结构作为fallback
        pair_name = pair_info_path.stem
        method_output_dir = Path(output_dir) / method / pair_name
    
    method_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算需要插值的帧数，为了节约时间，限制为最多5个中间帧
    # 原本应该是 k-1 个中间帧，现在改为 min(k-1, 5)
    k_frames = end_idx - start_idx - 1  # 实际的k-1个中间帧
    num_interpolate = min(k_frames, 5)  # 限制最多5个
    print(f"  检测到帧索引: start={start_idx}, end={end_idx}")
    print(f"  原本需要插值帧数: {k_frames}, 实际使用: {num_interpolate}")
    print(f"  优化帧索引: {gt_frame_indices}")
    
    # 运行插值命令 - 使用数据库目录路径和正确的帧索引
    python_exe = r"C:\Users\sky\miniconda3\envs\nmario\python.exe"
    cmd = [
        python_exe, "volumetric_interpolation_pipeline.py",
        str(database_dir),       # folder_path - 指向包含所有frame_xxx.obj文件的目录
        str(start_idx),          # start_frame index
        str(end_idx),            # end_frame index
        "--method", method,
        "--num_interpolate", str(num_interpolate),  # 生成k - 1个中间帧
        "--evaluation-mode",
        "--evaluation-output-dir", str(method_output_dir)
    ]
    
    print(f"运行插值: {' '.join(cmd)}")
    
    # 初始化性能监控器
    monitor = PerformanceMonitor(monitor_interval=0.5)
    monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        # 确保在项目根目录下运行
        project_root = Path(__file__).parent.parent  # evaluation/ -> neural_marionette/
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(project_root))
        end_time = time.time()
        
        # 停止性能监控
        monitor.stop_monitoring()
        
        # 保存性能数据
        performance_file = method_output_dir / f"performance_{method}_{pair_name}.json"
        performance_summary = monitor.save_performance_data(performance_file)
        
        if result.returncode == 0:
            print(f"插值成功: {method} - {pair_name} (耗时: {end_time - start_time:.2f}s)")
            print(f"性能数据已保存: {performance_file}")
            
            # 简化版性能汇总
            if performance_summary:
                cpu_avg = performance_summary.get('cpu_usage_percent', {}).get('avg', 0)
                mem_max = performance_summary.get('memory_usage_gb', {}).get('max', 0)
                gpu_avg = performance_summary.get('gpu_usage_percent', {}).get('avg', 0)
                gpu_mem_max = performance_summary.get('gpu_memory_gb', {}).get('max', 0)
                print(f"性能汇总: CPU平均{cpu_avg:.1f}%, 内存峰值{mem_max:.2f}GB, GPU平均{gpu_avg:.1f}%, GPU内存峰值{gpu_mem_max:.2f}GB")
            
            if result.stdout:
                print(f"标准输出: {result.stdout}")
            return True
        else:
            print(f"插值失败: {method} - {pair_name}")
            print(f"返回码: {result.returncode}")
            if result.stderr:
                print(f"错误输出: {result.stderr}")
            if result.stdout:
                print(f"标准输出: {result.stdout}")
            return False
    except subprocess.TimeoutExpired:
        monitor.stop_monitoring()
        # 即使超时也保存性能数据，有助于诊断性能问题
        performance_file = eval_interpolation_output / f"performance_{method}_{pair_name}_timeout.json"
        monitor.save_performance_data(performance_file)
        print(f"插值超时: {method} - {pair_name}")
        return False
    except Exception as e:
        monitor.stop_monitoring()
        # 异常情况也保存性能数据
        performance_file = eval_interpolation_output / f"performance_{method}_{pair_name}_error.json"
        monitor.save_performance_data(performance_file)
        print(f"插值异常: {method} - {pair_name} - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行插值评估")
    parser.add_argument("--pairs_dir", type=str, 
                       default="evaluation/data/dfaust/keyframe_pairs",
                       help="关键帧对目录 (如果指定具体路径，应包含database_name/subject_sequence子目录)")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/interpolation",
                       help="结果输出目录")
    parser.add_argument("--methods", nargs="+", 
                       default=["baseline", "dual_reference"],
                       help="要运行的插值方法")
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="最大处理的关键帧对数量")
    parser.add_argument("--database_name", type=str, default="dfaust",
                       help="数据库名称")
    parser.add_argument("--subject_id", type=str, default=None,
                       help="Subject ID")
    parser.add_argument("--sequence_id", type=str, default=None,
                       help="Sequence ID")
    parser.add_argument("--k", type=int, default=None,
                       help="指定k值（如果不指定，使用第一个找到的k值目录）")
    
    args = parser.parse_args()
    
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        print(f"错误: 关键帧对目录不存在 {pairs_dir}")
        return
    
    # 查找具体的database/subject_sequence路径和k值子目录
    k_dir = None
    
    if args.k:
        # 如果指定了k值，搜索包含该k值的目录
        k_pattern = f"k{args.k}"
        
        # 搜索可能的路径结构
        possible_paths = []
        
        # 1. 直接在pairs_dir下查找
        direct_k_dir = pairs_dir / k_pattern
        if direct_k_dir.exists():
            possible_paths.append(direct_k_dir)
        
        # 2. 在database/subject_sequence子目录下查找
        for db_dir in pairs_dir.iterdir():
            if db_dir.is_dir():
                for subject_dir in db_dir.iterdir():
                    if subject_dir.is_dir():
                        nested_k_dir = subject_dir / k_pattern
                        if nested_k_dir.exists():
                            possible_paths.append(nested_k_dir)
        
        if possible_paths:
            k_dir = possible_paths[0]  # 使用第一个找到的
            print(f"找到k{args.k}目录: {k_dir}")
        else:
            print(f"错误: 指定的k值目录不存在 k{args.k}")
            return
    else:
        # 查找所有k值子目录
        k_dirs = []
        
        # 在所有子目录中搜索k值目录
        for root_path in pairs_dir.rglob("k*"):
            if root_path.is_dir() and root_path.name.startswith('k') and root_path.name[1:].isdigit():
                k_dirs.append(root_path)
        if not k_dirs:
            print(f"错误: 在 {pairs_dir} 及其子目录中未找到k值子目录")
            return
        # 使用第一个找到的k值目录
        k_dir = k_dirs[0]
    
    print(f"使用k值目录: {k_dir}")
    
    # 读取关键帧对索引
    pairs_index_file = k_dir / "pairs_index.json"
    if pairs_index_file.exists():
        with open(pairs_index_file, 'r') as f:
            pairs_index = json.load(f)
        
        # 修正路径处理：如果路径是相对路径，转换为绝对路径
        pair_info_paths = []
        for p in pairs_index['pair_info_paths']:
            path = Path(p)
            if not path.is_absolute():
                # 如果是相对路径，相对于k_dir进行解析
                if path.name.startswith('pair_') and path.name.endswith('.json'):
                    # 简化：直接使用k_dir中的文件
                    path = k_dir / path.name
                else:
                    # 其他情况，可能需要相对于工作目录
                    path = Path.cwd() / path
            pair_info_paths.append(path)
        
        # 从pairs_index.json获取subject_id和sequence_id信息
        if not args.subject_id:
            args.subject_id = pairs_index.get('subject_id', '50002')
        if not args.sequence_id:
            args.sequence_id = pairs_index.get('sequence_id', 'jumping_jacks')
        print(f"从pairs_index.json获取信息: subject_id={args.subject_id}, sequence_id={args.sequence_id}")
    else:
        # 如果没有索引文件，扫描k_dir中的pair_xxx.json文件
        pair_info_paths = list(k_dir.glob("pair_*.json"))
        pair_info_paths.sort()
        
        # 使用默认值
        if not args.subject_id:
            args.subject_id = "50002"
        if not args.sequence_id:
            args.sequence_id = "jumping_jacks"
    
    if args.max_pairs:
        pair_info_paths = pair_info_paths[:args.max_pairs]
    
    print(f"找到 {len(pair_info_paths)} 个关键帧对")
    print(f"输出路径结构: evaluation/{args.database_name}/{args.subject_id}_{args.sequence_id}/")
    
    # 统计结果
    results = {method: {'success': 0, 'failed': 0} for method in args.methods}
    
    # 对每个关键帧对运行所有方法
    for i, pair_info_path in enumerate(pair_info_paths):
        print(f"\n处理关键帧对 {i+1}/{len(pair_info_paths)}: {pair_info_path.name}")
        
        for method in args.methods:
            success = run_interpolation_for_pair(
                pair_info_path, method, args.output_dir,
                database_name=args.database_name,
                subject_id=args.subject_id,
                sequence_id=args.sequence_id
            )
            if success:
                results[method]['success'] += 1
            else:
                results[method]['failed'] += 1
    
    # 输出统计结果
    print(f"\n插值完成统计:")
    for method in args.methods:
        total = results[method]['success'] + results[method]['failed']
        success_rate = results[method]['success'] / total * 100 if total > 0 else 0
        print(f"{method}: 成功 {results[method]['success']}/{total} ({success_rate:.1f}%)")

if __name__ == "__main__":
    main() 