#!/usr/bin/env python3
"""
结果可视化和汇总脚本
生成Chamfer vs 时间曲线图和其他可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
import glob

def plot_chamfer_vs_time(results_dir, output_dir):
    """绘制Chamfer距离 vs 时间曲线"""
    # 查找所有结果文件
    baseline_files = glob.glob(str(Path(results_dir) / "baseline" / "*" / "*.obj"))
    dual_ref_files = glob.glob(str(Path(results_dir) / "dual_reference" / "*" / "*.obj"))
    
    if not baseline_files or not dual_ref_files:
        print("未找到插值结果文件")
        return
    
    # 加载GT数据用于对比
    gt_data = None
    try:
        from utils_mesh import load_gt_data
        gt_data = load_gt_data("evaluation/data/registrations_m.hdf5", "50002", "jump")
    except:
        print("无法加载GT数据，跳过Chamfer vs 时间图")
        return
    
    # 计算每个方法的Chamfer距离
    methods_data = {}
    
    for method in ["baseline", "dual_reference"]:
        method_files = glob.glob(str(Path(results_dir) / method / "*" / "*.obj"))
        method_files.sort()
        
        chamfer_distances = []
        frame_indices = []
        
        for i, obj_file in enumerate(method_files):
            try:
                from utils_mesh import load_mesh, compute_chamfer_distance
                vertices, _, _ = load_mesh(obj_file)
                
                # 找到对应的GT帧
                # 这里需要根据文件名推断帧索引
                frame_idx = i  # 简化处理
                
                if frame_idx < len(gt_data['vertices']):
                    gt_vertices = gt_data['vertices'][frame_idx]
                    chamfer_dist = compute_chamfer_distance(gt_vertices, vertices)
                    chamfer_distances.append(chamfer_dist)
                    frame_indices.append(frame_idx)
            except Exception as e:
                print(f"处理文件 {obj_file} 时出错: {e}")
                continue
        
        methods_data[method] = {
            'frame_indices': frame_indices,
            'chamfer_distances': chamfer_distances
        }
    
    # 绘制曲线
    plt.figure(figsize=(12, 8))
    
    for method, data in methods_data.items():
        if data['frame_indices'] and data['chamfer_distances']:
            plt.plot(data['frame_indices'], data['chamfer_distances'], 
                    label=method, marker='o', linewidth=2, markersize=4)
    
    plt.xlabel('帧索引')
    plt.ylabel('Chamfer距离')
    plt.title('Chamfer距离 vs 时间')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    output_path = Path(output_dir) / "chamfer_vs_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chamfer vs 时间图已保存到: {output_path}")

def plot_metrics_comparison(csv_path, output_dir):
    """绘制指标对比图"""
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("没有数据可绘制")
        return
    
    # 选择要对比的指标
    metrics = ['mean_chamfer', 'mean_jerk', 'mean_arap_error', 'bone_length_sd']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("没有可用的指标进行对比")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 按方法分组
        method_data = []
        method_names = []
        
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            if metric in method_df.columns:
                method_data.append(method_df[metric].values)
                method_names.append(method)
        
        if method_data:
            # 绘制箱线图
            bp = ax.boxplot(method_data, labels=method_names, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            ax.set_title(f'{metric} 对比')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(output_dir) / "metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"指标对比图已保存到: {output_path}")

def plot_method_improvement(csv_path, output_dir):
    """绘制方法改进百分比图"""
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("没有数据可绘制")
        return
    
    # 计算改进百分比
    baseline_data = df[df['method'] == 'baseline']
    dual_ref_data = df[df['method'] == 'dual_reference']
    
    if baseline_data.empty or dual_ref_data.empty:
        print("缺少baseline或dual_reference数据")
        return
    
    # 选择要对比的指标
    metrics = ['mean_chamfer', 'mean_jerk', 'mean_arap_error', 'bone_length_sd']
    available_metrics = [m for m in metrics if m in df.columns]
    
    improvements = []
    metric_names = []
    
    for metric in available_metrics:
        if metric in baseline_data.columns and metric in dual_ref_data.columns:
            baseline_mean = baseline_data[metric].mean()
            dual_ref_mean = dual_ref_data[metric].mean()
            
            if baseline_mean > 0:
                improvement = (baseline_mean - dual_ref_mean) / baseline_mean * 100
                improvements.append(improvement)
                metric_names.append(metric)
    
    if not improvements:
        print("没有可计算的改进指标")
        return
    
    # 绘制改进百分比图
    plt.figure(figsize=(10, 6))
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = plt.bar(metric_names, improvements, color=colors, alpha=0.7)
    
    # 添加数值标签
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{improvement:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.xlabel('指标')
    plt.ylabel('改进百分比 (%)')
    plt.title('dual_reference vs baseline 改进百分比')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 保存图片
    output_path = Path(output_dir) / "method_improvement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"方法改进图已保存到: {output_path}")

def generate_summary_statistics(csv_path, output_dir):
    """生成汇总统计"""
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("没有数据生成统计")
        return
    
    summary = []
    summary.append("# 评估结果汇总统计\n\n")
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        summary.append(f"## {method} 方法\n\n")
        
        # 数值列统计
        numeric_cols = method_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['pair_id']:
                mean_val = method_data[col].mean()
                std_val = method_data[col].std()
                min_val = method_data[col].min()
                max_val = method_data[col].max()
                
                summary.append(f"- **{col}**:\n")
                summary.append(f"  - 平均值: {mean_val:.6f}\n")
                summary.append(f"  - 标准差: {std_val:.6f}\n")
                summary.append(f"  - 最小值: {min_val:.6f}\n")
                summary.append(f"  - 最大值: {max_val:.6f}\n\n")
        
        summary.append("\n")
    
    # 方法对比
    if len(df['method'].unique()) > 1:
        summary.append("## 方法对比\n\n")
        
        baseline_data = df[df['method'] == 'baseline']
        dual_ref_data = df[df['method'] == 'dual_reference']
        
        if not baseline_data.empty and not dual_ref_data.empty:
            comparison_cols = ['mean_chamfer', 'mean_jerk', 'mean_arap_error', 'bone_length_sd']
            
            summary.append("| 指标 | baseline | dual_reference | 改进百分比 |\n")
            summary.append("|------|----------|----------------|------------|\n")
            
            for col in comparison_cols:
                if col in baseline_data.columns and col in dual_ref_data.columns:
                    baseline_mean = baseline_data[col].mean()
                    dual_ref_mean = dual_ref_data[col].mean()
                    
                    if baseline_mean > 0:
                        improvement = (baseline_mean - dual_ref_mean) / baseline_mean * 100
                        summary.append(f"| {col} | {baseline_mean:.6f} | {dual_ref_mean:.6f} | {improvement:+.2f}% |\n")
                    else:
                        summary.append(f"| {col} | {baseline_mean:.6f} | {dual_ref_mean:.6f} | N/A |\n")
    
    # 保存汇总
    output_path = Path(output_dir) / "summary_statistics.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(summary)
    
    print(f"汇总统计已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="结果可视化和汇总")
    parser.add_argument("--results_dir", type=str, 
                       default="evaluation/results",
                       help="插值结果目录")
    parser.add_argument("--csv_path", type=str, 
                       default="evaluation/results/results.csv",
                       help="评估结果CSV文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/results",
                       help="可视化输出目录")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 生成各种可视化
    print("生成Chamfer vs 时间图...")
    plot_chamfer_vs_time(args.results_dir, output_dir)
    
    print("生成指标对比图...")
    plot_metrics_comparison(args.csv_path, output_dir)
    
    print("生成方法改进图...")
    plot_method_improvement(args.csv_path, output_dir)
    
    print("生成汇总统计...")
    generate_summary_statistics(args.csv_path, output_dir)
    
    print("可视化完成！")

if __name__ == "__main__":
    main() 