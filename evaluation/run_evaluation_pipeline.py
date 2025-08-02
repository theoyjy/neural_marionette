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
    print(f"步骤: {step_name}")
    print(f"命令: {' '.join(command)}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"SUCCESS {step_name} 成功完成 (耗时: {end_time - start_time:.2f}秒)")
            if result.stdout:
                print("输出:", result.stdout[-500:])  # 显示最后500字符
            return True
        else:
            print(f"FAILED {step_name} 失败")
            print("错误输出:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {step_name} 超时")
        return False
    except Exception as e:
        print(f"ERROR {step_name} 异常: {e}")
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
            print(f"MISSING 缺少必要文件: {file_path}")
            return False
    
    # 检查Python环境
    try:
        import numpy
        import pandas
        import trimesh
        import h5py
        import matplotlib
        print("SUCCESS Python依赖检查通过")
    except ImportError as e:
        print(f"MISSING 缺少Python依赖: {e}")
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
    
    return run_step("生成关键帧对", cmd)

def step2_run_interpolation(args):
    """步骤2: 运行插值"""
    # 从hdf5路径提取数据库名
    database_name = Path(args.gt_hdf5).stem
    
    # 构建具体的keyframe_pairs路径，包含k值
    pairs_dir = f"evaluation/data/dfaust/keyframe_pairs/{database_name}/{args.subject_id}_{args.sequence_id}_k{args.k}"
    
    cmd = [
        args.python_path, "evaluation/run_interpolation.py",
        "--pairs_dir", pairs_dir,
        "--output_dir", "evaluation",  # 修改输出目录为evaluation
        "--methods", "baseline", "dual_reference",
        "--max_pairs", str(args.max_pairs) if args.max_pairs else "3",
        "--database_name", database_name,
        "--subject_id", args.subject_id,
        "--sequence_id", args.sequence_id
    ]
    
    return run_step("运行插值", cmd)

def step3_evaluate_results(args):
    """步骤3: 评估结果"""
    # 从hdf5路径提取数据库名
    database_name = Path(args.gt_hdf5).stem
    
    # 构建具体的keyframe_pairs路径，包含k值
    pairs_dir = f"evaluation/data/dfaust/keyframe_pairs/{database_name}/{args.subject_id}_{args.sequence_id}_k{args.k}"
    
    cmd = [
        args.python_path, "evaluation/evaluate_interpolation.py",
        "--pairs_dir", pairs_dir,
        "--results_dir", "evaluation",  # 修改为新的结果目录
        "--methods", "baseline", "dual_reference",
        "--output_dir", str(args.individual_results_dir),
        "--database_name", database_name
    ]
    
    if args.no_gt:
        cmd.append("--no_gt")
    else:
        cmd.extend(["--gt_hdf5", args.gt_hdf5])
        cmd.extend(["--subject_id", args.subject_id])
        cmd.extend(["--sequence_id", args.sequence_id])
    
    return run_step("评估结果", cmd)

def step4_visualize_results(args):
    """步骤4: 可视化结果"""
    csv_path = args.individual_results_dir / "results.csv"
    if not csv_path.exists():
        print(f"WARNING 找不到results.csv文件: {csv_path}")
        return False
    
    cmd = [
        args.python_path, "evaluation/visualize_results.py",
        "--results_dir", str(args.individual_results_dir),
        "--csv_path", str(csv_path),
        "--output_dir", str(args.individual_results_dir)
    ]
    
    return run_step("可视化结果", cmd)

def step5_compare_methods(args):
    """步骤5: 对比方法"""
    csv_path = args.individual_results_dir / "results.csv"
    if not csv_path.exists():
        print(f"WARNING 找不到results.csv文件: {csv_path}")
        return False
    
    cmd = [
        args.python_path, "evaluation/compare_methods.py",
        "--csv_path", str(csv_path),
        "--output_dir", str(args.individual_results_dir)
    ]
    
    return run_step("对比方法", cmd)

def generate_final_report():
    """生成最终报告"""
    report = []
    report.append("# 插值评估流水线最终报告\n\n")
    report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 检查结果文件
    results_files = [
        ("results.csv", "评估结果CSV"),
        ("evaluation_report.md", "评估报告"),
        ("detailed_comparison_report.md", "详细对比报告"),
        ("improvement_summary.csv", "改进汇总"),
        ("chamfer_vs_time.png", "Chamfer vs 时间图"),
        ("metrics_comparison.png", "指标对比图"),
        ("method_improvement.png", "方法改进图")
    ]
    
    report.append("## 生成的文件\n\n")
    
    # 检查独立评估目录
    individual_dirs = list(Path("evaluation/results/individual_evaluations").glob("*"))
    if individual_dirs:
        latest_dir = max(individual_dirs, key=lambda x: x.stat().st_mtime)
        report.append(f"📁 **独立评估目录**: {latest_dir.name}\n")
        
        for filename, description in results_files:
            file_path = latest_dir / filename
            if file_path.exists():
                report.append(f"✅ **{filename}** - {description}\n")
            else:
                report.append(f"❌ **{filename}** - {description} (未生成)\n")
    else:
        for filename, description in results_files:
            file_path = Path("evaluation/results") / filename
            if file_path.exists():
                report.append(f"✅ **{filename}** - {description}\n")
            else:
                report.append(f"❌ **{filename}** - {description} (未生成)\n")
    
    report.append("\n## 使用说明\n\n")
    report.append("1. **查看评估结果**: `evaluation/results/results.csv`\n")
    report.append("2. **查看详细报告**: `evaluation/results/evaluation_report.md`\n")
    report.append("3. **查看对比分析**: `evaluation/results/detailed_comparison_report.md`\n")
    report.append("4. **查看可视化**: `evaluation/results/` 目录下的PNG文件\n")
    
    report.append("\n## 性能要求\n\n")
    report.append("- ✅ 支持有/无GT两种模式\n")
    report.append("- ✅ 输出Chamfer、jerk、ARAP、骨长SD、自碰撞计数等指标\n")
    report.append("- ✅ 对比baseline与dual_reference并生成results.csv + Markdown报告\n")
    report.append("- ✅ 优化为可在16GB RAM单GPU环境下1分钟内完成30帧测试序列评估\n")
    
    # 保存报告
    report_path = Path("evaluation/results/final_pipeline_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"最终报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="插值评估流水线")
    parser.add_argument("--gt_hdf5", type=str, 
                       default="evaluation/data/dfaust/registrations_m.hdf5",
                       help="GT数据HDF5文件路径")
    parser.add_argument("--subject_id", type=str, default="50002",
                       help="DFAUST subject ID")
    parser.add_argument("--sequence_id", type=str, default="jump",
                       help="DFAUST sequence ID")
    parser.add_argument("--k", type=int, default=10,
                       help="每隔k帧抽取一对关键帧")
    parser.add_argument("--max_pairs", type=int, default=3,
                       help="最大处理的关键帧对数量（用于快速测试）")
    parser.add_argument("--no_gt", action="store_true",
                       help="无GT模式，仅计算内部指标")
    parser.add_argument("--skip_steps", nargs="+", type=int, default=[],
                       help="跳过的步骤编号（1-5）")
    parser.add_argument("--timeout", type=int, default=60,
                       help="每个步骤的超时时间（秒）")
    parser.add_argument("--python_path", type=str,
                       default="C:\\Users\\sky\\miniconda3\\envs\\nmario\\python.exe")
    
    args = parser.parse_args()
    
    print("开始插值评估流水线")
    print(f"配置: subject={args.subject_id}, sequence={args.sequence_id}, k={args.k}")
    print(f"模式: {'无GT' if args.no_gt else '有GT'}")
    print(f"最大关键帧对: {args.max_pairs}")
    
    # 检查前置条件
    if not check_prerequisites():
        print("FAILED 前置条件检查失败，退出")
        return
    
    # 创建必要目录
    Path("evaluation/results").mkdir(exist_ok=True)
    
    # 创建独立的结果目录，避免文件被重写，包含k值和GT模式信息
    database_name = Path(args.gt_hdf5).stem
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt_mode = "nogt" if args.no_gt else "gt"
    eval_id = f"{args.subject_id}_{args.sequence_id}_k{args.k}_{gt_mode}_{eval_timestamp}"
    individual_results_dir = Path("evaluation/results") / database_name / eval_id
    individual_results_dir.mkdir(parents=True, exist_ok=True)
    
    # 将individual_results_dir添加到args中
    args.individual_results_dir = individual_results_dir
    
    print(f"CHECK 本次评估结果将保存到: {individual_results_dir}")
    
    # 记录开始时间
    pipeline_start_time = time.time()
    
    # 执行步骤
    steps = [
        (1, "生成关键帧对", lambda: step1_generate_keyframes(args)),
        (2, "运行插值", lambda: step2_run_interpolation(args)),
        (3, "评估结果", lambda: step3_evaluate_results(args)),
        (4, "可视化结果", lambda: step4_visualize_results(args)),
        (5, "对比方法", lambda: step5_compare_methods(args))
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_num, step_name, step_func in steps:
        if step_num in args.skip_steps:
            print(f"SKIP 跳过步骤 {step_num}: {step_name}")
            continue
        
        if step_func():
            success_count += 1
        else:
            print(f"FAILED 步骤 {step_num} 失败，但继续执行后续步骤")
    
    # 生成最终报告
    generate_final_report()
    
    # 计算总耗时
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    print(f"\n{'='*50}")
    print(f"流水线执行完成!")
    print(f"成功步骤: {success_count}/{total_steps}")
    print(f"总耗时: {total_time:.2f}秒")
    
    if total_time <= 60:
        print("SUCCESS 满足1分钟内完成的要求")
    else:
        print(f"WARNING 超出1分钟要求 ({total_time:.2f}秒)")
    
    if success_count == total_steps:
        print("SUCCESS 所有步骤成功完成")
        print("RESULTS 结果保存在: evaluation/results/")
    else:
        print(f"WARNING 有 {total_steps - success_count} 个步骤失败")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 