import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def collect_results(results_dir):
    all_results = []
    for sequence_dir in os.listdir(results_dir):
        sequence_path = os.path.join(results_dir, sequence_dir)
        if os.path.isdir(sequence_path):
            parts = sequence_dir.rsplit('_k', 1)
            if len(parts) == 2:
                sequence_name, k_str = parts
                try:
                    k_value = int(k_str)
                except ValueError:
                    continue

                results_csv = os.path.join(sequence_path, 'results.csv')
                if os.path.exists(results_csv):
                    df = pd.read_csv(results_csv)
                    df['sequence'] = sequence_name
                    df['k'] = k_value
                    # Assuming num_frames is constant per csv
                    if 'num_frames' in df.columns:
                        df['num_frames'] = df['num_frames'].iloc[0]
                    all_results.append(df)
    
    if not all_results:
        return pd.DataFrame()
        
    return pd.concat(all_results, ignore_index=True)

def generate_comparison_plots(df, output_dir):
    if df.empty:
        print("DataFrame is empty, skipping plot generation.")
        return

    metrics = [
        'mean_chamfer', 'max_chamfer', 'mean_normal_angle', 
        'mean_arap_error', 'mean_jerk', 'bone_length_sd',
        'mean_self_intersection_count', 'foot_slide_pixels'
    ]
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plotting metrics vs. k
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df, x='k', y=metric, hue='method', style='sequence', markers=True, dashes=False, markersize=10)
            plt.title(f'{metric.replace("_", " ").title()} vs. Interval Length')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xlabel('Interval Length')
            plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_vs_interval_length.png'))
            plt.close()

    # Plotting metrics by method
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='method', y=metric)
            plt.title(f'{metric.replace("_", " ").title()} by Method')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xlabel('Method')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_by_method.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Summarize evaluation results.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing the evaluation results.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the overall results and plots.')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    overall_df = collect_results(args.results_dir)
    
    if not overall_df.empty:
        # Save the combined results to a CSV file
        overall_csv_path = os.path.join(args.output_dir, 'overall.csv')
        overall_df.to_csv(overall_csv_path, index=False)
        print(f"Overall results saved to {overall_csv_path}")

        # Generate and save comparison plots
        generate_comparison_plots(overall_df, args.output_dir)
        print(f"Comparison plots saved in {args.output_dir}")
    else:
        print("No results found to process.")

if __name__ == '__main__':
    main()
