#!/usr/bin/env python3
"""
方法对比和结论自动化脚本
计算改进百分比并生成详细报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

def load_results(csv_path):
    """加载评估结果"""
    if not Path(csv_path).exists():
        print(f"错误: 结果文件不存在 {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def calculate_improvements(df):
    """计算改进百分比"""
    if df is None or df.empty:
        return None
    
    baseline_data = df[df['method'] == 'baseline']
    dual_ref_data = df[df['method'] == 'dual_reference']
    
    if baseline_data.empty or dual_ref_data.empty:
        print("错误: 缺少baseline或dual_reference数据")
        return None
    
    improvements = {}
    
    # 选择要对比的指标
    comparison_metrics = [
        'mean_chamfer', 'max_chamfer', 'std_chamfer',
        'mean_jerk', 'max_jerk',
        'mean_arap_error', 'max_arap_error',
        'mean_normal_angle', 'max_normal_angle',
        'bone_length_sd',
        'mean_self_intersection_count', 'max_self_intersection_count',
        'foot_slide_pixels'
    ]
    
    for metric in comparison_metrics:
        if metric in baseline_data.columns and metric in dual_ref_data.columns:
            baseline_mean = baseline_data[metric].mean()
            dual_ref_mean = dual_ref_data[metric].mean()
            
            if baseline_mean > 0:
                improvement_pct = (baseline_mean - dual_ref_mean) / baseline_mean * 100
                improvements[metric] = {
                    'baseline_mean': baseline_mean,
                    'dual_ref_mean': dual_ref_mean,
                    'improvement_pct': improvement_pct,
                    'improvement_abs': baseline_mean - dual_ref_mean
                }
    
    return improvements

def generate_detailed_report(df, improvements, output_path):
    """生成详细报告"""
    report = []
    report.append("# 插值方法对比详细报告\n\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 总体统计
    report.append("## 总体统计\n\n")
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        report.append(f"### {method} 方法\n\n")
        
        # 基本统计
        report.append(f"- 测试样本数: {len(method_data)}\n")
        
        # 数值指标统计
        numeric_cols = method_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['pair_id']:
                mean_val = method_data[col].mean()
                std_val = method_data[col].std()
                min_val = method_data[col].min()
                max_val = method_data[col].max()
                
                report.append(f"- **{col}**:\n")
                report.append(f"  - 平均值: {mean_val:.6f}\n")
                report.append(f"  - 标准差: {std_val:.6f}\n")
                report.append(f"  - 最小值: {min_val:.6f}\n")
                report.append(f"  - 最大值: {max_val:.6f}\n\n")
        
        report.append("\n")
    
    # 改进分析
    if improvements:
        report.append("## 改进分析\n\n")
        
        # 按改进程度排序
        sorted_improvements = sorted(improvements.items(), 
                                   key=lambda x: abs(x[1]['improvement_pct']), 
                                   reverse=True)
        
        report.append("### 改进百分比排名\n\n")
        report.append("| 指标 | baseline | dual_reference | 改进百分比 | 绝对改进 |\n")
        report.append("|------|----------|----------------|------------|----------|\n")
        
        for metric, data in sorted_improvements:
            baseline_val = data['baseline_mean']
            dual_ref_val = data['dual_ref_mean']
            improvement_pct = data['improvement_pct']
            improvement_abs = data['improvement_abs']
            
            report.append(f"| {metric} | {baseline_val:.6f} | {dual_ref_val:.6f} | {improvement_pct:+.2f}% | {improvement_abs:+.6f} |\n")
        
        report.append("\n")
        
        # 分类分析
        report.append("### 分类分析\n\n")
        
        # 几何质量指标
        geometry_metrics = ['mean_chamfer', 'max_chamfer', 'mean_arap_error', 'mean_normal_angle']
        geometry_improvements = {k: v for k, v in improvements.items() if k in geometry_metrics}
        
        if geometry_improvements:
            report.append("#### 几何质量指标\n\n")
            avg_geometry_improvement = np.mean([v['improvement_pct'] for v in geometry_improvements.values()])
            report.append(f"平均改进: {avg_geometry_improvement:+.2f}%\n\n")
        
        # 时间平滑度指标
        smoothness_metrics = ['mean_jerk', 'max_jerk']
        smoothness_improvements = {k: v for k, v in improvements.items() if k in smoothness_metrics}
        
        if smoothness_improvements:
            report.append("#### 时间平滑度指标\n\n")
            avg_smoothness_improvement = np.mean([v['improvement_pct'] for v in smoothness_improvements.values()])
            report.append(f"平均改进: {avg_smoothness_improvement:+.2f}%\n\n")
        
        # 物理合法性指标
        physics_metrics = ['bone_length_sd', 'mean_self_intersection_count', 'foot_slide_pixels']
        physics_improvements = {k: v for k, v in improvements.items() if k in physics_metrics}
        
        if physics_improvements:
            report.append("#### 物理合法性指标\n\n")
            avg_physics_improvement = np.mean([v['improvement_pct'] for v in physics_improvements.values()])
            report.append(f"平均改进: {avg_physics_improvement:+.2f}%\n\n")
    
    # 结论和建议
    report.append("## 结论和建议\n\n")
    
    if improvements:
        # 计算总体改进
        all_improvements = [v['improvement_pct'] for v in improvements.values()]
        positive_improvements = [imp for imp in all_improvements if imp > 0]
        negative_improvements = [imp for imp in all_improvements if imp < 0]
        
        report.append(f"- 总体指标数量: {len(all_improvements)}\n")
        report.append(f"- 改进指标数量: {len(positive_improvements)}\n")
        report.append(f"- 退化指标数量: {len(negative_improvements)}\n")
        report.append(f"- 平均改进: {np.mean(all_improvements):+.2f}%\n\n")
        
        if positive_improvements:
            report.append("### 主要改进\n\n")
            # 找出改进最大的3个指标
            top_improvements = sorted(improvements.items(), 
                                    key=lambda x: x[1]['improvement_pct'], 
                                    reverse=True)[:3]
            
            for metric, data in top_improvements:
                if data['improvement_pct'] > 0:
                    report.append(f"- **{metric}**: {data['improvement_pct']:+.2f}% 改进\n")
        
        if negative_improvements:
            report.append("\n### 需要注意的问题\n\n")
            # 找出退化最大的3个指标
            worst_degradations = sorted(improvements.items(), 
                                      key=lambda x: x[1]['improvement_pct'])[:3]
            
            for metric, data in worst_degradations:
                if data['improvement_pct'] < 0:
                    report.append(f"- **{metric}**: {data['improvement_pct']:+.2f}% 退化\n")
        
        # 总体评价
        avg_improvement = np.mean(all_improvements)
        if avg_improvement > 5:
            report.append("\n### 总体评价\n\n")
            report.append("✅ **dual_reference方法显著优于baseline方法**\n\n")
            report.append("建议采用dual_reference方法进行插值。\n")
        elif avg_improvement > 0:
            report.append("\n### 总体评价\n\n")
            report.append("✅ **dual_reference方法略优于baseline方法**\n\n")
            report.append("可以考虑采用dual_reference方法，但需要进一步优化。\n")
        else:
            report.append("\n### 总体评价\n\n")
            report.append("⚠️ **dual_reference方法需要进一步改进**\n\n")
            report.append("建议继续优化dual_reference方法或考虑其他改进方向。\n")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"详细报告已保存到: {output_path}")

def generate_summary_table(improvements, output_path):
    """生成汇总表格"""
    if not improvements:
        return
    
    # 创建DataFrame
    data = []
    for metric, imp_data in improvements.items():
        data.append({
            'metric': metric,
            'baseline_mean': imp_data['baseline_mean'],
            'dual_ref_mean': imp_data['dual_ref_mean'],
            'improvement_pct': imp_data['improvement_pct'],
            'improvement_abs': imp_data['improvement_abs']
        })
    
    df_summary = pd.DataFrame(data)
    df_summary = df_summary.sort_values('improvement_pct', ascending=False)
    
    # 保存为CSV
    df_summary.to_csv(output_path, index=False)
    print(f"汇总表格已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="方法对比和结论自动化")
    parser.add_argument("--csv_path", type=str, 
                       default="evaluation/results/results.csv",
                       help="评估结果CSV文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 加载结果
    df = load_results(args.csv_path)
    if df is None:
        return
    
    # 计算改进
    improvements = calculate_improvements(df)
    if improvements is None:
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 生成详细报告
    report_path = output_dir / "detailed_comparison_report.md"
    generate_detailed_report(df, improvements, report_path)
    
    # 生成汇总表格
    summary_path = output_dir / "improvement_summary.csv"
    generate_summary_table(improvements, summary_path)
    
    # 输出关键统计
    print("\n=== 关键统计 ===")
    all_improvements = [v['improvement_pct'] for v in improvements.values()]
    positive_count = len([imp for imp in all_improvements if imp > 0])
    negative_count = len([imp for imp in all_improvements if imp < 0])
    
    print(f"总指标数: {len(all_improvements)}")
    print(f"改进指标数: {positive_count}")
    print(f"退化指标数: {negative_count}")
    print(f"平均改进: {np.mean(all_improvements):+.2f}%")
    
    if improvements:
        best_metric = max(improvements.items(), key=lambda x: x[1]['improvement_pct'])
        worst_metric = min(improvements.items(), key=lambda x: x[1]['improvement_pct'])
        
        print(f"最佳改进: {best_metric[0]} ({best_metric[1]['improvement_pct']:+.2f}%)")
        print(f"最大退化: {worst_metric[0]} ({worst_metric[1]['improvement_pct']:+.2f}%)")
    
    print(f"\n对比分析完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main() 