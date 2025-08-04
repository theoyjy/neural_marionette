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
        print(f"Evaluation: {subject_id}_{sequence_id} (k={k}, max_pairs={max_pairs}, no_gt={no_gt})")
        print(f"{'='*60}")
        
        # 构造命令
        cmd = [
            self.python_path, "evaluation/run_evaluation_pipeline.py",
            "--gt_hdf5", hdf5_path,
            "--subject_id", subject_id,
            "--sequence_id", sequence_id,
            "--k", str(k),
            "--max_pairs", str(max_pairs),
            "--timeout", "300",  # 5 minutes timeout
            "--python_path", self.python_path  # Pass python path to subprocess
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
                print(f"Evaluation completed successfully (Time: {duration:.1f} seconds)")
                
                # 尝试解析最终报告中的信息
                try:
                    database_name = Path(hdf5_path).stem
                    final_report = Path("evaluation/results") / database_name /f"{subject_id}_{sequence_id}_k{k} / final_pipeline_report.md"
                    if final_report.exists():
                        with open(final_report, 'r', encoding='utf-8') as f:
                            content = f.read()
                        eval_result['final_report'] = content
                except Exception as e:
                    print(f"Cannot read final report: {e}")
                    
            else:
                print(f"Evaluation failed (Return code: {result.returncode}, Time: {duration:.1f} seconds)")
                print(f"Error output:\n{result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Evaluation timeout (Time: {duration:.1f} seconds)")
            
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
            print(f"Evaluation exception: {e}")
            
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
        print("Start DFAUST data set batch evaluation")
        print(f"Test mode: {'Yes' if test_mode else 'No'}")
        print(f"Python path: {self.python_path}")
        
        # Load sequence information
        sequences_info = self.load_sequences_info()
        if not sequences_info:
            return
        
        # Define test parameter combinations
        if test_mode:
            # Test mode: quick verification
            k_values = [10, 20]
            max_pairs = 1  # Fixed value, used for quick testing
            # Only test part of the sequences
            test_sequences = [
                ('male', '50002', 'jumping_jacks'),
                ('female', '50004', 'jumping_jacks'),
                ('male', '50007', 'running_on_spot'),
                ('female', '50020', 'running_on_spot')
            ]
        else:
            # Full mode: comprehensive testing
            k_values = [10, 20, 40, 80]
            max_pairs = 3  # Fixed value, evaluate all available pairs (usually at most 3)
            test_sequences = []
            
            # Add all sequences
            for gender in ['male', 'female']:
                for subject_id, sequences in sequences_info[gender].items():
                    for sequence_id in sequences:
                        test_sequences.append((gender, subject_id, sequence_id))
        
        print(f"Plan to test {len(test_sequences)} sequences")
        print(f"Parameter combinations: k={k_values}, max_pairs={max_pairs}")
        
        total_tests = len(test_sequences) * len(k_values)
        print(f"Total test number: {total_tests}")
        
        # if not test_mode and total_tests > 100:
        #     confirm = input(f"Will run {total_tests} tests, which may take a long time. Continue? (y/N): ")
        #     if confirm.lower() != 'y':
        #         print("User cancelled evaluation")
        #         return
        
        # Run evaluation
        test_count = 0
        success_count = 0
        
        for gender, subject_id, sequence_id in test_sequences:
            # Determine HDF5 file path
            if gender == 'male':
                hdf5_path = "evaluation/data/dfaust/registrations_m.hdf5"
            else:
                hdf5_path = "evaluation/data/dfaust/registrations_f.hdf5"
            
            for k in k_values:
                test_count += 1
                print(f"\nProgress: {test_count}/{total_tests}")
                
                success = self.run_single_evaluation(
                    hdf5_path, subject_id, sequence_id, k, max_pairs, no_gt=False
                )
                
                if success:
                    success_count += 1
                
                # Clean up output directory to save space
                self.cleanup_output_dir()
        
        print(f"\n{'='*60}")
        print(f"Batch evaluation completed!")
        print(f"Success: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
        print(f"Total time: {(datetime.now() - self.start_time).total_seconds():.1f} seconds")
        print(f"{'='*60}")
        
        # 保存结果
        self.save_results()
        self.generate_summary_report()
    
    def cleanup_output_dir(self):
        """Clean up output directory to save space while preserving skinning weights"""
        try:
            output_dir = Path("output")
            if output_dir.exists():
                import shutil
                # 遍历output目录下的每个pipeline目录
                for pipeline_dir in output_dir.iterdir():
                    if pipeline_dir.is_dir() and pipeline_dir.name.startswith("pipeline_"):
                        # 保留skinning_weights目录，清理其他目录
                        for item in pipeline_dir.iterdir():
                            if item.is_dir() and item.name != "skinning_weights":
                                print(f"Cleaning directory: {item}")
                                shutil.rmtree(item)
                            elif item.is_file():
                                print(f"Cleaning file: {item}")
                                item.unlink()
                print("Output directory cleaned while preserving skinning weights")
        except Exception as e:
            print(f"Clean up output directory failed: {e}")
    
    def save_results(self):
        """Save evaluation results"""
        # Save detailed results to JSON
        results_dir = Path("evaluation/results/batch_evaluation")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results in JSON format
        json_file = results_dir / f"batch_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to: {json_file}")
        
        # Save summary results in CSV format
        csv_file = results_dir / f"batch_summary_{timestamp}.csv"
        
        # Prepare CSV data
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
        print(f"Summary results saved to: {csv_file}")
        
        return json_file, csv_file
    
    def generate_summary_report(self):
        """Generate summary report"""
        if not self.results:
            return
        
        results_dir = Path("evaluation/results/batch_evaluation")
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"batch_report_{timestamp}.md"
        
        report = []
        report.append(f"# DFAUST data set batch evaluation report\n\n")
        report.append(f"**Evaluation time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Total time**: {(datetime.now() - self.start_time).total_seconds():.1f} seconds\n")
        report.append(f"**Total test number**: {len(self.results)}\n\n")
        
        # Success rate statistics
        success_count = sum(1 for r in self.results if r['success'])
        success_rate = success_count / len(self.results) * 100
        report.append(f"## Overall statistics\n\n")
        report.append(f"- **Success**: {success_count}/{len(self.results)} ({success_rate:.1f}%)\n")
        report.append(f"- **Failure**: {len(self.results) - success_count}/{len(self.results)} ({100-success_rate:.1f}%)\n")
        
        # Statistics by HDF5 file
        male_results = [r for r in self.results if 'registrations_m.hdf5' in r['hdf5_path']]
        female_results = [r for r in self.results if 'registrations_f.hdf5' in r['hdf5_path']]
        
        if male_results:
            male_success = sum(1 for r in male_results if r['success'])
            male_rate = male_success / len(male_results) * 100
            report.append(f"- **Male data**: {male_success}/{len(male_results)} ({male_rate:.1f}%)\n")
        
        if female_results:
            female_success = sum(1 for r in female_results if r['success'])
            female_rate = female_success / len(female_results) * 100
            report.append(f"- **Female data**: {female_success}/{len(female_results)} ({female_rate:.1f}%)\n")
        
        # Statistics by parameters
        report.append(f"\n## Parameter combination statistics\n\n")
        
        # Statistics by k value
        k_stats = {}
        for result in self.results:
            k = result['k']
            if k not in k_stats:
                k_stats[k] = {'total': 0, 'success': 0}
            k_stats[k]['total'] += 1
            if result['success']:
                k_stats[k]['success'] += 1
        
        report.append("### Statistics by k value\n\n")
        for k in sorted(k_stats.keys()):
            stats = k_stats[k]
            rate = stats['success'] / stats['total'] * 100
            report.append(f"- **k={k}**: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
        
        # Statistics by GT mode
        gt_stats = {}
        for result in self.results:
            gt_mode = "no_gt" if result['no_gt'] else "with_gt"
            if gt_mode not in gt_stats:
                gt_stats[gt_mode] = {'total': 0, 'success': 0}
            gt_stats[gt_mode]['total'] += 1
            if result['success']:
                gt_stats[gt_mode]['success'] += 1
        
        report.append("\n### Statistics by GT mode\n\n")
        for mode, stats in gt_stats.items():
            rate = stats['success'] / stats['total'] * 100
            mode_name = "No GT mode" if mode == "no_gt" else "With GT mode"
            report.append(f"- **{mode_name}**: {stats['success']}/{stats['total']} ({rate:.1f}%)\n")
        
        # Failure case analysis
        failed_results = [r for r in self.results if not r['success']]
        if failed_results:
            report.append(f"\n## Failure case analysis\n\n")
            report.append(f"There are {len(failed_results)} failure cases:\n\n")
            
            for i, result in enumerate(failed_results[:10]):  # Only show the first 10
                report.append(f"{i+1}. **{result['subject_id']}_{result['sequence_id']}** ")
                report.append(f"(k={result['k']}, max_pairs={result['max_pairs']}, no_gt={result['no_gt']})\n")
                report.append(f"   - Return code: {result['returncode']}\n")
                report.append(f"   - Error: {result['stderr'][:100]}...\n\n")
            
            if len(failed_results) > 10:
                report.append(f"... There are {len(failed_results) - 10} more failure cases\n\n")
        
        # Performance statistics
        successful_results = [r for r in self.results if r['success']]
        if successful_results:
            durations = [r['duration'] for r in successful_results]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            report.append(f"## Performance statistics\n\n")
            report.append(f"- **Average duration**: {avg_duration:.1f} seconds\n")
            report.append(f"- **Maximum duration**: {max_duration:.1f} seconds\n")
            report.append(f"- **Minimum duration**: {min_duration:.1f} seconds\n")
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"SUCCESS Evaluation report saved to: {report_file}")
        
        # Generate summary visualizations
        self.generate_summary_visualizations(self.results, results_dir)

    def generate_summary_visualizations(self, results, output_dir):
        """Generate summary visualizations for batch evaluation"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data
            successful_results = [r for r in results if r['success']]
            if not successful_results:
                print("WARNING No successful evaluation results, skip visualizations")
                return
            
            # Create visualization directory
            viz_dir = Path(output_dir) / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Success rate statistics
            self.plot_success_rates(results, viz_dir)
            
            # 2. Duration distribution
            self.plot_duration_distribution(successful_results, viz_dir)
            
            # 3. Parameter comparison
            self.plot_parameter_comparison(successful_results, viz_dir)
            
            print(f"SUCCESS Summary visualizations saved to: {viz_dir}")
            
        except Exception as e:
            print(f"WARNING Error generating visualizations: {e}")

    def plot_success_rates(self, results, output_dir):
        """Plot success rate statistics"""
        import matplotlib.pyplot as plt
        
        # Statistics by HDF5 file
        m_results = [r for r in results if 'registrations_m.hdf5' in r.get('command', '')]
        f_results = [r for r in results if 'registrations_f.hdf5' in r.get('command', '')]
        
        m_success = len([r for r in m_results if r['success']])
        f_success = len([r for r in f_results if r['success']])
        
        categories = ['Male data\n(registrations_m)', 'Female data\n(registrations_f)', 'Total']
        success_counts = [m_success, f_success, len([r for r in results if r['success']])]
        total_counts = [len(m_results), len(f_results), len(results)]
        success_rates = [s/t*100 if t > 0 else 0 for s, t in zip(success_counts, total_counts)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 成功率柱状图
        bars = ax1.bar(categories, success_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('Success rate (%)')
        ax1.set_title('Batch evaluation success rate statistics')
        ax1.set_ylim(0, 105)
        
        # Add value labels
        for bar, rate, count, total in zip(bars, success_rates, success_counts, total_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%\n({count}/{total})', 
                    ha='center', va='bottom')
        
        # Success/failure pie chart
        success_total = len([r for r in results if r['success']])
        fail_total = len(results) - success_total
        
        if fail_total > 0:
            ax2.pie([success_total, fail_total], labels=['Success', 'Failure'], 
                   colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        else:
            ax2.pie([success_total], labels=['Success'], colors=['lightgreen'], autopct='%1.1f%%')
        
        ax2.set_title('Overall success/failure ratio')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_duration_distribution(self, results, output_dir):
        """Plot duration distribution"""
        import matplotlib.pyplot as plt
        
        durations = [r['duration'] for r in results if 'duration' in r]
        if not durations:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Duration histogram
        ax1.hist(durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Evaluation duration distribution')
        ax1.axvline(np.mean(durations), color='red', linestyle='--', label=f'Average: {np.mean(durations):.1f} seconds')
        ax1.legend()
        
        # Duration box plot
        ax2.boxplot(durations)
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Duration box plot')
        ax2.set_xticklabels(['All tests'])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_parameter_comparison(self, results, output_dir):
        """Plot parameter comparison"""
        import matplotlib.pyplot as plt
        import re
        
        # Parse parameters
        param_data = []
        for r in results:
            cmd = r.get('command', '')
            # Extract k value
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
        
        # k value vs duration
        k_groups = df.groupby('k')['duration'].mean()
        ax1.bar(k_groups.index, k_groups.values, color='lightblue')
        ax1.set_xlabel('k value')
        ax1.set_ylabel('Average duration (seconds)')
        ax1.set_title('k value vs evaluation duration')
        
        # max_pairs vs duration
        pairs_groups = df.groupby('max_pairs')['duration'].mean()
        ax2.bar(pairs_groups.index, pairs_groups.values, color='lightcoral')
        ax2.set_xlabel('max_pairs')
        ax2.set_ylabel('Average duration (seconds)')
        ax2.set_title('max_pairs vs evaluation duration')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="DFAUST data set batch evaluation")
    parser.add_argument("--python_path", type=str, 
                       default="C:\\Users\\sky\\miniconda3\\envs\\nmario\\python.exe",
                       help="Python interpreter path")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode, only run a few tests for verification")
    parser.add_argument("--sequences", nargs="+", default=None,
                       help="Specify the sequences to test (format: subject_id:sequence_id)")
    parser.add_argument("--k_values", nargs="+", type=int, default=[10, 20],
                       help="k value list")
    parser.add_argument("--max_pairs", nargs="+", type=int, default=[1, 2],
                       help="max_pairs value list")
    parser.add_argument("--no_gt_only", action="store_true",
                       help="Only run no GT mode")
    
    args = parser.parse_args()
    
    evaluator = BatchEvaluator(args.python_path)
    
    if args.sequences:
        # Custom sequence evaluation
        print("🎯 Run custom sequence evaluation")
        # TODO: Implement custom sequence evaluation
        pass
    else:
        # 运行全面评估
        evaluator.run_comprehensive_evaluation(test_mode=args.test_mode)

if __name__ == "__main__":
    main()