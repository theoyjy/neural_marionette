#!/usr/bin/env python3
"""
DFAUST数据集批量评估脚本
自动运行多种参数组合的评估，充分测试registrations_m.hdf5和registrations_f.hdf5
"""

import os
import json
import subprocess
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

class BatchEvaluator:
    def __init__(self, python_path="python"):
        self.python_path = python_path
        self.results = []
        self.start_time = datetime.now()
        
    def load_sequences_info(self):
        """加载序列信息"""
        sequences_file = Path("evaluation/data/dfaust/all_sequences.json")
        if not sequences_file.exists():
            print("序列信息文件不存在，请先运行 check_dfaust_sequences.py")
            return None
        
        with open(sequences_file, 'r') as f:
            return json.load(f)
    
    def run_single_evaluation(self, hdf5_path, subject_id, sequence_id, k=10, max_pairs=2, no_gt=False):
        """运行单次评估"""
        print(f"\n{'='*60}")
        print(f"评估: {subject_id}_{sequence_id} (k={k}, max_pairs={max_pairs}, no_gt={no_gt})")
        print(f"{'='*60}")
        
        # 构造命令
        cmd = [
            self.python_path, "evaluation/run_evaluation_pipeline.py",
            "--gt_hdf5", hdf5_path,
            "--subject_id", subject_id,
            "--sequence_id", sequence_id,
            "--k", str(k),
            "--max_pairs", str(max_pairs),
            "--timeout", "300",  # 5分钟超时
            "--python_path", self.python_path  # 传递python路径给子进程
        ]
        
        if no_gt:
            cmd.append("--no_gt")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行评估
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            # 记录结果
            eval_result = {
                'subject_id': subject_id,
                'sequence_id': sequence_id,
                'k': k,
                'max_pairs': max_pairs,
                'no_gt': no_gt,
                'hdf5_path': hdf5_path,
                'success': success,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(eval_result)
            
            if success:
                print(f"✅ 评估成功完成 (耗时: {duration:.1f}秒)")
                
                # 尝试解析最终报告中的信息
                try:
                    database_name = Path(hdf5_path).stem
                    final_report = Path("evaluation/results") / database_name /f"{subject_id}_{sequence_id}_k{k} / final_pipeline_report.md"
                    if final_report.exists():
                        with open(final_report, 'r', encoding='utf-8') as f:
                            content = f.read()
                        eval_result['final_report'] = content
                except Exception as e:
                    print(f"无法读取最终报告: {e}")
                    
            else:
                print(f"评估失败 (返回码: {result.returncode}, 耗时: {duration:.1f}秒)")
                print(f"错误输出:\n{result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏰ 评估超时 (耗时: {duration:.1f}秒)")
            
            eval_result = {
                'subject_id': subject_id,
                'sequence_id': sequence_id,
                'k': k,
                'max_pairs': max_pairs,
                'no_gt': no_gt,
                'hdf5_path': hdf5_path,
                'success': False,
                'duration': duration,
                'returncode': 'timeout',
                'stdout': '',
                'stderr': 'Evaluation timeout',
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(eval_result)
            return False
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"💥 评估异常: {e}")
            
            eval_result = {
                'subject_id': subject_id,
                'sequence_id': sequence_id,
                'k': k,
                'max_pairs': max_pairs,
                'no_gt': no_gt,
                'hdf5_path': hdf5_path,
                'success': False,
                'duration': duration,
                'returncode': 'exception',
                'stdout': '',
                'stderr': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(eval_result)
            return False
    
    def run_comprehensive_evaluation(self, test_mode=False):
        """运行全面评估"""
        print("开始DFAUST数据集批量评估")
        print(f"测试模式: {'是' if test_mode else '否'}")
        print(f"Python路径: {self.python_path}")
        
        # 加载序列信息
        sequences_info = self.load_sequences_info()
        if not sequences_info:
            return
        
        # 定义测试参数组合
        if test_mode:
            # 测试模式：快速验证
            k_values = [10, 20]
            max_pairs = 1  # 固定值，用于快速测试
            # 只测试部分序列
            test_sequences = [
                ('male', '50002', 'jumping_jacks'),
                ('female', '50004', 'jumping_jacks'),
                ('male', '50007', 'running_on_spot'),
                ('female', '50020', 'running_on_spot')
            ]
        else:
            # 完整模式：全面测试
            k_values = [5, 10, 15, 20]
            max_pairs = 3  # 固定值，评估所有可用pairs（通常最多3个）
            test_sequences = []
            
            # 添加所有序列
            for gender in ['male', 'female']:
                for subject_id, sequences in sequences_info[gender].items():
                    for sequence_id in sequences:
                        test_sequences.append((gender, subject_id, sequence_id))
        
        print(f"计划测试 {len(test_sequences)} 个序列")
        print(f"参数组合: k={k_values}, max_pairs={max_pairs}")
        
        total_tests = len(test_sequences) * len(k_values)
        print(f"总测试数量: {total_tests}")
        
        # if not test_mode and total_tests > 100:
        #     confirm = input(f"将要运行 {total_tests} 个测试，这可能需要很长时间。继续吗？(y/N): ")
        #     if confirm.lower() != 'y':
        #         print("用户取消评估")
        #         return
        
        # 运行评估
        test_count = 0
        success_count = 0
        
        for gender, subject_id, sequence_id in test_sequences:
            # 确定HDF5文件路径
            if gender == 'male':
                hdf5_path = "evaluation/data/dfaust/registrations_m.hdf5"
            else:
                hdf5_path = "evaluation/data/dfaust/registrations_f.hdf5"
            
            for k in k_values:
                test_count += 1
                print(f"\n进度: {test_count}/{total_tests}")
                
                success = self.run_single_evaluation(
                    hdf5_path, subject_id, sequence_id, k, max_pairs, no_gt=False
                )
                
                if success:
                    success_count += 1
                
                # 清理output目录以节省空间
                self.cleanup_output_dir()
        
        print(f"\n{'='*60}")
        print(f"批量评估完成!")
        print(f"成功: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
        print(f"总耗时: {(datetime.now() - self.start_time).total_seconds():.1f}秒")
        print(f"{'='*60}")
        
        # 保存结果
        self.save_results()
        self.generate_summary_report()
    
    def cleanup_output_dir(self):
        """清理output目录以节省空间"""
        try:
            output_dir = Path("output")
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
        except Exception as e:
            print(f"⚠️ 清理output目录失败: {e}")
    
    def save_results(self):
        """保存评估结果"""
        # 保存详细结果到JSON
        results_dir = Path("evaluation/results/batch_evaluation")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式的详细结果
        json_file = results_dir / f"batch_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✅ 详细结果已保存到: {json_file}")
        
        # 保存CSV格式的汇总结果
        csv_file = results_dir / f"batch_summary_{timestamp}.csv"
        
        # 准备CSV数据
        csv_data = []
        for result in self.results:
            csv_row = {
                'timestamp': result['timestamp'],
                'subject_id': result['subject_id'],
                'sequence_id': result['sequence_id'],
                'k': result['k'],
                'max_pairs': result['max_pairs'],
                'no_gt': result['no_gt'],
                'hdf5_file': Path(result['hdf5_path']).name,
                'success': result['success'],
                'duration': result['duration'],
                'returncode': result['returncode']
            }
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"✅ 汇总结果已保存到: {csv_file}")
        
        return json_file, csv_file
    
    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.results:
            return
        
        results_dir = Path("evaluation/results/batch_evaluation")
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"batch_report_{timestamp}.md"
        
        report = []
        report.append(f"# DFAUST数据集批量评估报告\n\n")
        report.append(f"**评估时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**总耗时**: {(datetime.now() - self.start_time).total_seconds():.1f}秒\n")
        report.append(f"**总测试数**: {len(self.results)}\n\n")
        
        # 成功率统计
        success_count = sum(1 for r in self.results if r['success'])
        success_rate = success_count / len(self.results) * 100
        report.append(f"## 总体统计\n\n")
        report.append(f"- **成功**: {success_count}/{len(self.results)} ({success_rate:.1f}%)\n")
        report.append(f"- **失败**: {len(self.results) - success_count}/{len(self.results)} ({100-success_rate:.1f}%)\n")
        
        # 按HDF5文件统计
        male_results = [r for r in self.results if 'registrations_m.hdf5' in r['hdf5_path']]
        female_results = [r for r in self.results if 'registrations_f.hdf5' in r['hdf5_path']]
        
        if male_results:
            male_success = sum(1 for r in male_results if r['success'])
            male_rate = male_success / len(male_results) * 100
            report.append(f"- **男性数据**: {male_success}/{len(male_results)} ({male_rate:.1f}%)\n")
        
        if female_results:
            female_success = sum(1 for r in female_results if r['success'])
            female_rate = female_success / len(female_results) * 100
            report.append(f"- **女性数据**: {female_success}/{len(female_results)} ({female_rate:.1f}%)\n")
        
        # 按参数统计
        report.append(f"\n## 参数组合统计\n\n")
        
        # 按k值统计
        k_stats = {}
        for result in self.results:
            k = result['k']
            if k not in k_stats:
                k_stats[k] = {'total': 0, 'success': 0}
            k_stats[k]['total'] += 1
            if result['success']:
                k_stats[k]['success'] += 1
        
        report.append("### 按k值统计\n\n")
        for k in sorted(k_stats.keys()):
            stats = k_stats[k]
            rate = stats['success'] / stats['total'] * 100
            report.append(f"- **k={k}**: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
        
        # 按GT模式统计
        gt_stats = {}
        for result in self.results:
            gt_mode = "no_gt" if result['no_gt'] else "with_gt"
            if gt_mode not in gt_stats:
                gt_stats[gt_mode] = {'total': 0, 'success': 0}
            gt_stats[gt_mode]['total'] += 1
            if result['success']:
                gt_stats[gt_mode]['success'] += 1
        
        report.append("\n### 按GT模式统计\n\n")
        for mode, stats in gt_stats.items():
            rate = stats['success'] / stats['total'] * 100
            mode_name = "无GT模式" if mode == "no_gt" else "有GT模式"
            report.append(f"- **{mode_name}**: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
        
        # 失败案例分析
        failed_results = [r for r in self.results if not r['success']]
        if failed_results:
            report.append(f"\n## 失败案例分析\n\n")
            report.append(f"共有 {len(failed_results)} 个失败案例:\n\n")
            
            for i, result in enumerate(failed_results[:10]):  # 只显示前10个
                report.append(f"{i+1}. **{result['subject_id']}_{result['sequence_id']}** ")
                report.append(f"(k={result['k']}, max_pairs={result['max_pairs']}, no_gt={result['no_gt']})\n")
                report.append(f"   - 返回码: {result['returncode']}\n")
                report.append(f"   - 错误: {result['stderr'][:100]}...\n\n")
            
            if len(failed_results) > 10:
                report.append(f"... 还有 {len(failed_results) - 10} 个失败案例\n\n")
        
        # 性能统计
        successful_results = [r for r in self.results if r['success']]
        if successful_results:
            durations = [r['duration'] for r in successful_results]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            report.append(f"## 性能统计\n\n")
            report.append(f"- **平均耗时**: {avg_duration:.1f}秒\n")
            report.append(f"- **最大耗时**: {max_duration:.1f}秒\n")
            report.append(f"- **最小耗时**: {min_duration:.1f}秒\n")
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"SUCCESS 评估报告已保存到: {report_file}")
        
        # 生成汇总可视化
        self.generate_summary_visualizations(self.results, results_dir)

    def generate_summary_visualizations(self, results, output_dir):
        """生成批量评估的汇总可视化"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 准备数据
            successful_results = [r for r in results if r['success']]
            if not successful_results:
                print("WARNING 没有成功的评估结果，跳过可视化")
                return
            
            # 创建可视化目录
            viz_dir = Path(output_dir) / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. 成功率统计图
            self.plot_success_rates(results, viz_dir)
            
            # 2. 耗时分布图
            self.plot_duration_distribution(successful_results, viz_dir)
            
            # 3. 参数对比图
            self.plot_parameter_comparison(successful_results, viz_dir)
            
            print(f"SUCCESS 汇总可视化已保存到: {viz_dir}")
            
        except Exception as e:
            print(f"WARNING 生成可视化时出错: {e}")

    def plot_success_rates(self, results, output_dir):
        """绘制成功率统计图"""
        import matplotlib.pyplot as plt
        
        # 按HDF5文件统计
        m_results = [r for r in results if 'registrations_m.hdf5' in r.get('command', '')]
        f_results = [r for r in results if 'registrations_f.hdf5' in r.get('command', '')]
        
        m_success = len([r for r in m_results if r['success']])
        f_success = len([r for r in f_results if r['success']])
        
        categories = ['男性数据\n(registrations_m)', '女性数据\n(registrations_f)', '总计']
        success_counts = [m_success, f_success, len([r for r in results if r['success']])]
        total_counts = [len(m_results), len(f_results), len(results)]
        success_rates = [s/t*100 if t > 0 else 0 for s, t in zip(success_counts, total_counts)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 成功率柱状图
        bars = ax1.bar(categories, success_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('成功率 (%)')
        ax1.set_title('批量评估成功率统计')
        ax1.set_ylim(0, 105)
        
        # 添加数值标签
        for bar, rate, count, total in zip(bars, success_rates, success_counts, total_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%\n({count}/{total})', 
                    ha='center', va='bottom')
        
        # 成功/失败饼图
        success_total = len([r for r in results if r['success']])
        fail_total = len(results) - success_total
        
        if fail_total > 0:
            ax2.pie([success_total, fail_total], labels=['成功', '失败'], 
                   colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        else:
            ax2.pie([success_total], labels=['成功'], colors=['lightgreen'], autopct='%1.1f%%')
        
        ax2.set_title('总体成功/失败比例')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_duration_distribution(self, results, output_dir):
        """绘制耗时分布图"""
        import matplotlib.pyplot as plt
        
        durations = [r['duration'] for r in results if 'duration' in r]
        if not durations:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 耗时直方图
        ax1.hist(durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('耗时 (秒)')
        ax1.set_ylabel('频次')
        ax1.set_title('评估耗时分布')
        ax1.axvline(np.mean(durations), color='red', linestyle='--', label=f'平均值: {np.mean(durations):.1f}秒')
        ax1.legend()
        
        # 耗时箱线图
        ax2.boxplot(durations)
        ax2.set_ylabel('耗时 (秒)')
        ax2.set_title('耗时箱线图')
        ax2.set_xticklabels(['所有测试'])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_parameter_comparison(self, results, output_dir):
        """绘制参数对比图"""
        import matplotlib.pyplot as plt
        import re
        
        # 解析参数
        param_data = []
        for r in results:
            cmd = r.get('command', '')
            # 提取k值
            k_match = re.search(r'--k (\d+)', cmd)
            max_pairs_match = re.search(r'--max_pairs (\d+)', cmd)
            no_gt_match = '--no_gt' in cmd
            
            if k_match and max_pairs_match:
                param_data.append({
                    'k': int(k_match.group(1)),
                    'max_pairs': int(max_pairs_match.group(1)),
                    'no_gt': no_gt_match,
                    'duration': r['duration']
                })
        
        if not param_data:
            return
        
        df = pd.DataFrame(param_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # k值 vs 耗时
        k_groups = df.groupby('k')['duration'].mean()
        ax1.bar(k_groups.index, k_groups.values, color='lightblue')
        ax1.set_xlabel('k值')
        ax1.set_ylabel('平均耗时 (秒)')
        ax1.set_title('k值对评估耗时的影响')
        
        # max_pairs vs 耗时
        pairs_groups = df.groupby('max_pairs')['duration'].mean()
        ax2.bar(pairs_groups.index, pairs_groups.values, color='lightcoral')
        ax2.set_xlabel('max_pairs')
        ax2.set_ylabel('平均耗时 (秒)')
        ax2.set_title('max_pairs对评估耗时的影响')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="DFAUST数据集批量评估")
    parser.add_argument("--python_path", type=str, 
                       default="C:\\Users\\sky\\miniconda3\\envs\\nmario\\python.exe",
                       help="Python解释器路径")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，仅运行少量测试用于验证")
    parser.add_argument("--sequences", nargs="+", default=None,
                       help="指定要测试的序列 (格式: subject_id:sequence_id)")
    parser.add_argument("--k_values", nargs="+", type=int, default=[10, 20],
                       help="k值列表")
    parser.add_argument("--max_pairs", nargs="+", type=int, default=[1, 2],
                       help="max_pairs值列表")
    parser.add_argument("--no_gt_only", action="store_true",
                       help="仅运行无GT模式")
    
    args = parser.parse_args()
    
    evaluator = BatchEvaluator(args.python_path)
    
    if args.sequences:
        # 自定义序列评估
        print("🎯 运行自定义序列评估")
        # TODO: 实现自定义序列评估
        pass
    else:
        # 运行全面评估
        evaluator.run_comprehensive_evaluation(test_mode=args.test_mode)

if __name__ == "__main__":
    main()