#!/usr/bin/env python3
"""
主控制脚本 - 插值评估流水线
整合所有六步评估流程，支持有/无GT模式
确保在1分钟内完成30帧测试序列评估
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime

def run_step(step_name, command, timeout=300):
    """运行单个步骤"""
    print(f"\n{'='*50}")
    print(f"Steps: {step_name}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"SUCCESS {step_name} completed (Time: {end_time - start_time:.2f} seconds)")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # 显示最后500字符
            return True
        else:
            print(f"FAILED {step_name} failed")
            print("Error output:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {step_name} timeout")
        return False
    except Exception as e:
        print(f"ERROR {step_name} exception: {e}")
        return False

def check_prerequisites():
    """检查前置条件"""
    print("检查前置条件...")
    
    # 检查必要文件
    required_files = [
        "evaluation/data/dfaust/registrations_m.hdf5",
        "volumetric_interpolation_pipeline.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"MISSING missing required file: {file_path}")
            return False
    
    # 检查Python环境
    try:
        import numpy
        import pandas
        import trimesh
        import h5py
        import matplotlib
        print("SUCCESS Python dependencies check passed")
    except ImportError as e:
        print(f"MISSING missing Python dependencies: {e}")
        return False
    
    return True

def step1_generate_keyframes(args):
    """步骤1: 生成关键帧对"""
    cmd = [
        args.python_path, "evaluation/generate_keyframe_pairs.py",
        "--hdf5_path", args.gt_hdf5,
        "--output_dir", "evaluation/data/dfaust/keyframe_pairs",
        "--k", str(args.k),
        "--subject_id", args.subject_id,
        "--sequence_id", args.sequence_id
    ]
    
    # 如果指定了max_pairs，传递给生成步骤
    if hasattr(args, 'max_pairs') and args.max_pairs:
        cmd.extend(["--max_pairs", str(args.max_pairs)])
    
    return run_step("Generate keyframe pairs", cmd)

def step2_run_interpolation(args):
    """步骤2: 运行插值"""
    # 从hdf5路径提取数据库名
    database_name = Path(args.gt_hdf5).stem
    
    # 构建具体的keyframe_pairs路径，包含k值
    pairs_dir = f"evaluation/data/dfaust/keyframe_pairs/{database_name}/{args.subject_id}_{args.sequence_id}"
    
    cmd = [
        args.python_path, "evaluation/run_interpolation.py",
        "--pairs_dir", pairs_dir,
        "--output_dir", "evaluation/interpolation",  # 修改输出目录为evaluation
        "--methods", "baseline", "dual_reference",
        "--max_pairs", str(args.max_pairs) if args.max_pairs else "3",
        "--database_name", database_name,
        "--subject_id", args.subject_id,
        "--sequence_id", args.sequence_id,
        "--k", str(args.k)
    ]
    
    return run_step("Run interpolation", cmd)

def step3_evaluate_results(args):
    """步骤3: 评估结果"""
    # 从hdf5路径提取数据库名
    database_name = Path(args.gt_hdf5).stem
    
    # 构建具体的keyframe_pairs路径，包含k值
    pairs_dir = f"evaluation/data/dfaust/keyframe_pairs/{database_name}/{args.subject_id}_{args.sequence_id}"
    
    cmd = [
        args.python_path, "evaluation/evaluate_interpolation.py",
        "--pairs_dir", pairs_dir,
        "--results_dir", "evaluation/interpolation",  # 修改为新的结果目录
        "--methods", "baseline", "dual_reference",
        "--output_dir", str(args.individual_results_dir),
        "--database_name", database_name,
        "--fast"  # 启用快速模式
    ]
    
    if args.no_gt:
        cmd.append("--no_gt")
    else:
        cmd.extend(["--gt_hdf5", args.gt_hdf5])
        cmd.extend(["--subject_id", args.subject_id])
        cmd.extend(["--sequence_id", args.sequence_id])
    
    # 添加k值参数
    cmd.extend(["--k", str(args.k)])
    
    return run_step("Evaluate results", cmd, 3000)

def step4_visualize_results(args):
    """步骤4: 可视化结果"""
    csv_path = args.individual_results_dir / "results.csv"
    if not csv_path.exists():
        print(f"WARNING results.csv file not found: {csv_path}")
        return False
    
    cmd = [
        args.python_path, "evaluation/visualize_results.py",
        "--results_dir", str(args.individual_results_dir),
        "--csv_path", str(csv_path),
        "--output_dir", str(args.individual_results_dir)
    ]
    
    return run_step("Visualize results", cmd)

def step5_compare_methods(args):
    """步骤5: 对比方法"""
    csv_path = args.individual_results_dir / "results.csv"
    if not csv_path.exists():
        print(f"WARNING results.csv file not found: {csv_path}")
        return False
    
    cmd = [
        args.python_path, "evaluation/compare_methods.py",
        "--csv_path", str(csv_path),
        "--output_dir", str(args.individual_results_dir)
    ]
    
    return run_step("Compare methods", cmd)

def generate_final_report(args):
    """生成最终报告"""
    report = []
    report.append("# Final report of interpolation evaluation pipeline\n\n")
    report.append(f"Generated time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 检查结果文件
    results_files = [
        ("results.csv", "Evaluation results CSV"),
        ("evaluation_report.md", "Evaluation report"),
        ("detailed_comparison_report.md", "Detailed comparison report"),
        ("improvement_summary.csv", "Improvement summary"),
        ("chamfer_vs_time.png", "Chamfer vs time graph"),
        ("metrics_comparison.png", "Metrics comparison graph"),
        ("method_improvement.png", "Method improvement graph")
    ]
    
    report.append("## 生成的文件\n\n")
    
    # 检查独立评估目录
    individual_dirs = list(Path("evaluation/results/individual_evaluations").glob("*"))
    if individual_dirs:
        latest_dir = max(individual_dirs, key=lambda x: x.stat().st_mtime)
        report.append(f"**Independent evaluation directory**: {latest_dir.name}\n")
        
        for filename, description in results_files:
            file_path = latest_dir / filename
            if file_path.exists():
                report.append(f"**{filename}** - {description}\n")
            else:
                report.append(f"**{filename}** - {description} (not generated)\n")
    else:
        for filename, description in results_files:
            file_path = Path("evaluation/results") / filename
            if file_path.exists():
                report.append(f"**{filename}** - {description}\n")
            else:
                report.append(f"**{filename}** - {description} (not generated)\n")
    
    report.append("\n## Usage\n\n")
    report.append("1. **View evaluation results**: `evaluation/results/results.csv`\n")
    report.append("2. **View detailed report**: `evaluation/results/evaluation_report.md`\n")
    report.append("3. **View comparison analysis**: `evaluation/results/detailed_comparison_report.md`\n")
    report.append("4. **View visualizations**: PNG files in `evaluation/results/` directory\n\n")
    
    report.append("\n## Performance requirements\n\n")
    report.append("- Support both GT and no-GT modes\n")
    report.append("- Output Chamfer, jerk, ARAP, bone length SD, self-collision count, etc.\n")
    report.append("- Compare baseline and dual_reference and generate results.csv + Markdown report\n")
    report.append("- Optimized to complete 30-frame test sequence evaluation within 1 minute in 16GB RAM single GPU environment\n")
    
    # 保存报告
    report_path = args.individual_results_dir / "final_pipeline_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"Final report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Interpolation evaluation pipeline")
    parser.add_argument("--gt_hdf5", type=str, 
                       default="evaluation/data/dfaust/registrations_m.hdf5",
                       help="GT data HDF5 file path")
    parser.add_argument("--subject_id", type=str, default="50002",
                       help="DFAUST subject ID")
    parser.add_argument("--sequence_id", type=str, default="jumping_jacks",
                       help="DFAUST sequence ID (e.g., 'jumping_jacks')")
    parser.add_argument("--k", type=int, default=10,
                       help="Extract one keyframe pair every k frames")
    parser.add_argument("--max_pairs", type=int, default=3,
                       help="Maximum number of keyframe pairs to process (for quick testing)")
    parser.add_argument("--no_gt", action="store_true",
                       help="No GT mode, only calculate internal metrics")
    parser.add_argument("--skip_steps", nargs="+", type=int, default=[],
                       help="Skipped step numbers (1-5)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Timeout for each step (seconds)")
    parser.add_argument("--python_path", type=str,
                       default="C:\\Users\\sky\\miniconda3\\envs\\nmario\\python.exe")
    
    args = parser.parse_args()
    
    print("Start interpolation evaluation pipeline")
    print(f"Configuration: subject={args.subject_id}, sequence={args.sequence_id}, k={args.k}")
    print(f"Mode: {'No GT' if args.no_gt else 'GT'}")
    print(f"Maximum keyframe pairs: {args.max_pairs}")
    
    # 检查前置条件
    if not check_prerequisites():
        print("FAILED Prerequisites check failed, exiting")
        return
    
    # 创建必要目录
    Path("evaluation/results").mkdir(exist_ok=True)
    
    # 创建独立的结果目录，避免文件被重写，包含k值和GT模式信息
    database_name = Path(args.gt_hdf5).stem
    individual_results_dir = Path("evaluation/results") / database_name /f"{args.subject_id}_{args.sequence_id}_k{args.k}"
    individual_results_dir.mkdir(parents=True, exist_ok=True)
    
    # 将individual_results_dir添加到args中
    args.individual_results_dir = individual_results_dir
    
    print(f"CHECK This evaluation result will be saved to: {individual_results_dir}")
    
    # 记录开始时间
    pipeline_start_time = time.time()
    
    # 执行步骤
    steps = [
        # (1, "Generate keyframe pairs", lambda: step1_generate_keyframes(args)),
        # (2, "Run interpolation", lambda: step2_run_interpolation(args)),
        (3, "Evaluate results", lambda: step3_evaluate_results(args)),
        (4, "Visualize results", lambda: step4_visualize_results(args)),
        (5, "Compare methods", lambda: step5_compare_methods(args))
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_num, step_name, step_func in steps:
        if step_num in args.skip_steps:
            print(f"SKIP Skip step {step_num}: {step_name}")
            continue
        
        if step_func():
            success_count += 1
        else:
            print(f"FAILED Step {step_num} failed, but continue to execute subsequent steps")
    
    # 生成最终报告
    generate_final_report(args)
    
    # 计算总耗时
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    print(f"\n{'='*50}")
    print(f"Pipeline execution completed!")
    print(f"Success steps: {success_count}/{total_steps}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if total_time <= 60:
        print("SUCCESS Meet the requirement of completing within 1 minute")
    else:
        print(f"WARNING Exceed the requirement of completing within 1 minute ({total_time:.2f} seconds)")
    
    if success_count == total_steps:
        print("SUCCESS All steps completed successfully")
        print("RESULTS Results saved to: evaluation/results/")
    else:
        print(f"WARNING {total_steps - success_count} steps failed")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 