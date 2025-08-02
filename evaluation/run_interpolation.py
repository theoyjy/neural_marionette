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

def run_interpolation_for_pair(pair_dir, method, output_dir, database_name=None, subject_id=None, sequence_id=None):
    """对单个关键帧对运行插值"""

    # get all frame_XXX.obj files
    frame_files = list(pair_dir.glob("frame_*.obj"))
    frame_files.sort(key=lambda x: x.name)
    if len(frame_files) < 2:
        print(f"警告: 无法找到足够的frame文件 {pair_dir}")
        return False

    
    # 创建输出目录 - 使用新的路径结构: evaluation/method/database_name/subjectid_sequenceid_k{k}/pair_xxx/
    if database_name and subject_id and sequence_id:
        # 从pairs_index.json获取k值和其他参数
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
        method_output_dir = Path(output_dir) / method / database_name / f"{subject_id}_{sequence_id}_k{k_value}" / pair_dir.name
    else:
        # 保持原有结构作为fallback
        method_output_dir = Path(output_dir) / method / pair_dir.name
    
    method_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为评估模式设置插值结果输出到evaluation results目录
    # 这样插值结果和最终的results.csv会在同一个individual_evaluations目录下
    eval_interpolation_output = method_output_dir / "intp"  # 改为intp
    eval_interpolation_output.mkdir(parents=True, exist_ok=True)
    
    # 将关键帧文件复制到输出目录
    # import shutil


    # # 复制所有frame_XXX.obj文件用于蒙皮优化
    # for frame_file in frame_files:
    #     shutil.copy2(frame_file, method_output_dir / frame_file.name)
    #     print(f"  复制优化帧: {frame_file.name}")

    
    start_idx = 0
    end_idx = len(frame_files) - 1 
    # 计算需要插值的帧数，应该等于GT中间帧数
    # 对于k个总帧数，中间帧数量是 k-1，即 end_idx - start_idx - 1
    num_interpolate = end_idx - start_idx - 1
    print(f"  检测到帧索引: start={start_idx}, end={end_idx}")
    
    # 运行插值命令 - 使用文件夹路径和正确的帧索引
    python_exe = r"C:\Users\sky\miniconda3\envs\nmario\python.exe"
    cmd = [
        python_exe, "volumetric_interpolation_pipeline.py",
        str(pair_dir),  # folder_path
        str(start_idx),          # start_frame index (通常是0)
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
        performance_file = eval_interpolation_output / f"performance_{method}_{pair_dir.name}.json"
        performance_summary = monitor.save_performance_data(performance_file)
        
        if result.returncode == 0:
            print(f"插值成功: {method} - {pair_dir.name} (耗时: {end_time - start_time:.2f}s)")
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
            print(f"插值失败: {method} - {pair_dir.name}")
            print(f"返回码: {result.returncode}")
            if result.stderr:
                print(f"错误输出: {result.stderr}")
            if result.stdout:
                print(f"标准输出: {result.stdout}")
            return False
    except subprocess.TimeoutExpired:
        monitor.stop_monitoring()
        # 即使超时也保存性能数据，有助于诊断性能问题
        performance_file = eval_interpolation_output / f"performance_{method}_{pair_dir.name}_timeout.json"
        monitor.save_performance_data(performance_file)
        print(f"插值超时: {method} - {pair_dir.name}")
        return False
    except Exception as e:
        monitor.stop_monitoring()
        # 异常情况也保存性能数据
        performance_file = eval_interpolation_output / f"performance_{method}_{pair_dir.name}_error.json"
        monitor.save_performance_data(performance_file)
        print(f"插值异常: {method} - {pair_dir.name} - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行插值评估")
    parser.add_argument("--pairs_dir", type=str, 
                       default="evaluation/data/dfaust/keyframe_pairs",
                       help="关键帧对目录 (如果指定具体路径，应包含database_name/subject_sequence子目录)")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/results",
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
    
    args = parser.parse_args()
    
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        print(f"错误: 关键帧对目录不存在 {pairs_dir}")
        return
    
    # 读取关键帧对索引
    pairs_index_file = pairs_dir / "pairs_index.json"
    if pairs_index_file.exists():
        with open(pairs_index_file, 'r') as f:
            pairs_index = json.load(f)
        pair_dirs = [Path(p) for p in pairs_index['pair_dirs']]
        
        # 从pairs_index.json获取subject_id和sequence_id信息
        if not args.subject_id:
            args.subject_id = pairs_index.get('subject_id', '50002')
        if not args.sequence_id:
            args.sequence_id = pairs_index.get('sequence_id', 'jumping_jacks')
        print(f"从pairs_index.json获取信息: subject_id={args.subject_id}, sequence_id={args.sequence_id}")
    else:
        # 如果没有索引文件，扫描目录
        pair_dirs = [d for d in pairs_dir.iterdir() if d.is_dir() and d.name.startswith("pair_")]
        pair_dirs.sort()
        
        # 使用默认值
        if not args.subject_id:
            args.subject_id = "50002"
        if not args.sequence_id:
            args.sequence_id = "jumping_jacks"
    
    if args.max_pairs:
        pair_dirs = pair_dirs[:args.max_pairs]
    
    print(f"找到 {len(pair_dirs)} 个关键帧对")
    print(f"输出路径结构: evaluation/{args.database_name}/{args.subject_id}_{args.sequence_id}/")
    
    # 统计结果
    results = {method: {'success': 0, 'failed': 0} for method in args.methods}
    
    # 对每个关键帧对运行所有方法
    for i, pair_dir in enumerate(pair_dirs):
        print(f"\n处理关键帧对 {i+1}/{len(pair_dirs)}: {pair_dir.name}")
        
        for method in args.methods:
            success = run_interpolation_for_pair(
                pair_dir, method, args.output_dir,
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