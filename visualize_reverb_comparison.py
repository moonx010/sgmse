#!/usr/bin/env python3
"""
Visualize reverb comparison results.

Compare VB-DEMAND (additive noise) vs WSJ0-REVERB (dereverberation) models
on VOiCES Reverb test data.

Usage:
    python visualize_reverb_comparison.py \
        --vb_demand_results results/edinburgh_reverb/vb_demand/_results.csv \
        --wsj0_reverb_results results/edinburgh_reverb/wsj0_reverb/_results.csv \
        --output_dir plots/reverb_comparison
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path):
    """Load results CSV file."""
    if not os.path.exists(csv_path):
        print(f"WARNING: Results file not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def plot_comparison_bars(df_vb, df_reverb, output_dir):
    """Plot bar comparison of two models."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['pesq', 'estoi', 'si_sdr']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    # Calculate means and stds
    vb_means = [df_vb[m].mean() for m in metrics]
    vb_stds = [df_vb[m].std() for m in metrics]
    reverb_means = [df_reverb[m].mean() for m in metrics]
    reverb_stds = [df_reverb[m].std() for m in metrics]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(2)
    width = 0.6
    colors = ['#3498db', '#e74c3c']
    labels = ['VB-DEMAND\n(Additive Noise)', 'WSJ0-REVERB\n(Dereverberation)']

    for ax, metric, name, vb_m, vb_s, rev_m, rev_s in zip(
        axes, metrics, metric_names, vb_means, vb_stds, reverb_means, reverb_stds
    ):
        values = [vb_m, rev_m]
        errors = [vb_s, rev_s]

        bars = ax.bar(x, values, width, yerr=errors, color=colors,
                     edgecolor='black', linewidth=0.5, capsize=5)

        # Add value labels
        for bar, val, err in zip(bars, values, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight winner
        winner_idx = np.argmax(values)
        bars[winner_idx].set_edgecolor('green')
        bars[winner_idx].set_linewidth(3)

    plt.suptitle('VOiCES Reverb: Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_bars.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'model_comparison_bars.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_bars.png")
    plt.close()


def plot_distribution(df_vb, df_reverb, output_dir):
    """Plot distribution comparison (box plots)."""
    metrics = ['pesq', 'estoi', 'si_sdr']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric, name in zip(axes, metrics, metric_names):
        data = [df_vb[metric].values, df_reverb[metric].values]
        labels = ['VB-DEMAND', 'WSJ0-REVERB']

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(name)
        ax.set_title(f'{name} Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('VOiCES Reverb: Score Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'model_comparison_boxplot.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_boxplot.png")
    plt.close()


def plot_scatter(df_vb, df_reverb, output_dir):
    """Plot scatter comparison (file-by-file)."""
    metrics = ['pesq', 'estoi', 'si_sdr']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric, name in zip(axes, metrics, metric_names):
        # Match files by filename
        merged = pd.merge(df_vb[['filename', metric]], df_reverb[['filename', metric]],
                         on='filename', suffixes=('_vb', '_reverb'))

        x = merged[f'{metric}_vb']
        y = merged[f'{metric}_reverb']

        ax.scatter(x, y, alpha=0.5, s=20)

        # Add diagonal line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel(f'{name} (VB-DEMAND)')
        ax.set_ylabel(f'{name} (WSJ0-REVERB)')
        ax.set_title(f'{name}: File-by-File')
        ax.grid(True, alpha=0.3)

        # Count wins
        vb_wins = (x > y).sum()
        reverb_wins = (y > x).sum()
        ax.text(0.05, 0.95, f'VB wins: {vb_wins}\nReverb wins: {reverb_wins}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('VOiCES Reverb: File-by-File Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_scatter.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'model_comparison_scatter.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_scatter.png")
    plt.close()


def print_summary(df_vb, df_reverb):
    """Print summary statistics."""
    metrics = ['pesq', 'estoi', 'si_sdr']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR']

    print("\n" + "="*70)
    print("REVERB COMPARISON SUMMARY")
    print("="*70)
    print(f"\nDataset: VOiCES Reverb ({len(df_vb)} files)")
    print("\n" + "-"*70)
    print(f"{'Metric':<12} {'VB-DEMAND':<20} {'WSJ0-REVERB':<20} {'Winner':<10}")
    print("-"*70)

    for metric, name in zip(metrics, metric_names):
        vb_mean = df_vb[metric].mean()
        vb_std = df_vb[metric].std()
        rev_mean = df_reverb[metric].mean()
        rev_std = df_reverb[metric].std()

        winner = "VB-DEMAND" if vb_mean > rev_mean else "WSJ0-REVERB"
        if abs(vb_mean - rev_mean) < 0.01:
            winner = "TIE"

        print(f"{name:<12} {vb_mean:>7.3f} ± {vb_std:<7.3f}  {rev_mean:>7.3f} ± {rev_std:<7.3f}  {winner:<10}")

    print("-"*70)

    # Overall winner
    vb_wins = sum(1 for m in metrics if df_vb[m].mean() > df_reverb[m].mean())
    print(f"\nOverall: {'VB-DEMAND' if vb_wins >= 2 else 'WSJ0-REVERB'} wins {max(vb_wins, 3-vb_wins)}/3 metrics")


def main():
    parser = argparse.ArgumentParser(description="Visualize reverb comparison results")
    parser.add_argument("--vb_demand_results", type=str,
                       default="results/voices_reverb/vb_demand/_results.csv",
                       help="Path to VB-DEMAND results CSV")
    parser.add_argument("--wsj0_reverb_results", type=str,
                       default="results/voices_reverb/wsj0_reverb/_results.csv",
                       help="Path to WSJ0-REVERB results CSV")
    parser.add_argument("--output_dir", type=str, default="plots/reverb_comparison",
                       help="Output directory for plots")
    args = parser.parse_args()

    # Load results
    df_vb = load_results(args.vb_demand_results)
    df_reverb = load_results(args.wsj0_reverb_results)

    if df_vb is None or df_reverb is None:
        print("ERROR: Could not load results files. Run the comparison experiment first.")
        print("\nUsage:")
        print("  1. python run_reverb_comparison.py --config reverb_comparison_config.yaml")
        print("  2. python visualize_reverb_comparison.py")
        return

    # Print summary
    print_summary(df_vb, df_reverb)

    # Create plots
    print("\nGenerating plots...")
    plot_comparison_bars(df_vb, df_reverb, args.output_dir)
    plot_distribution(df_vb, df_reverb, args.output_dir)
    plot_scatter(df_vb, df_reverb, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
