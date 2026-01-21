#!/usr/bin/env python3
"""
Visualize evaluation results with plots and summary tables.

Usage:
    python visualize_results.py --results_dir evaluation_results --output_dir plots
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import numpy as np


def load_summary(results_dir):
    """Load the summary CSV file."""
    summary_files = glob(os.path.join(results_dir, "summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No summary file found in {results_dir}")

    # Use the most recent summary file
    summary_file = sorted(summary_files)[-1]
    print(f"Loading: {summary_file}")
    return pd.read_csv(summary_file)


def parse_dataset_name(name):
    """Parse dataset name into components."""
    # Format: {clean}_{noise}_{snr}dB
    # Examples: vb_tbus_0dB, libri_esc50_animals_5dB

    parts = name.rsplit('_', 1)  # Split from right to get SNR
    snr = parts[-1].replace('dB', '')

    remaining = parts[0]

    # Determine clean speech source
    if remaining.startswith('vb_'):
        clean_source = 'VoiceBank'
        noise_part = remaining[3:]  # Remove 'vb_'
    elif remaining.startswith('libri_'):
        clean_source = 'LibriSpeech'
        noise_part = remaining[6:]  # Remove 'libri_'
    else:
        clean_source = 'Unknown'
        noise_part = remaining

    # Determine noise source and type
    if noise_part.startswith('esc50_'):
        noise_source = 'ESC-50'
        noise_type = noise_part[6:].replace('_', ' ').title()
    else:
        noise_source = 'DEMAND'
        noise_type = noise_part.upper()

    return {
        'clean_source': clean_source,
        'noise_source': noise_source,
        'noise_type': noise_type,
        'snr': int(snr)
    }


def prepare_data(df):
    """Prepare data with parsed columns."""
    parsed = df['Dataset'].apply(parse_dataset_name)
    df['Clean Source'] = parsed.apply(lambda x: x['clean_source'])
    df['Noise Source'] = parsed.apply(lambda x: x['noise_source'])
    df['Noise Type'] = parsed.apply(lambda x: x['noise_type'])
    df['SNR (dB)'] = parsed.apply(lambda x: x['snr'])
    return df


def create_summary_table(df, output_dir):
    """Create summary tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Table 1: Mean metrics by Clean Source x Noise Source
    table1 = df.groupby(['Clean Source', 'Noise Source']).agg({
        'PESQ_mean': 'mean',
        'ESTOI_mean': 'mean',
        'SI-SDR_mean': 'mean'
    }).round(3)

    print("\n" + "="*80)
    print("TABLE 1: Mean Metrics by Clean Source x Noise Source")
    print("="*80)
    print(table1.to_string())
    table1.to_csv(os.path.join(output_dir, 'table_clean_noise_source.csv'))

    # Table 2: Mean metrics by SNR
    table2 = df.groupby('SNR (dB)').agg({
        'PESQ_mean': 'mean',
        'ESTOI_mean': 'mean',
        'SI-SDR_mean': 'mean'
    }).round(3)

    print("\n" + "="*80)
    print("TABLE 2: Mean Metrics by SNR")
    print("="*80)
    print(table2.to_string())
    table2.to_csv(os.path.join(output_dir, 'table_snr.csv'))

    # Table 3: Mean metrics by Noise Type
    table3 = df.groupby(['Noise Source', 'Noise Type']).agg({
        'PESQ_mean': 'mean',
        'ESTOI_mean': 'mean',
        'SI-SDR_mean': 'mean'
    }).round(3)

    print("\n" + "="*80)
    print("TABLE 3: Mean Metrics by Noise Type")
    print("="*80)
    print(table3.to_string())
    table3.to_csv(os.path.join(output_dir, 'table_noise_type.csv'))

    # Table 4: Full results table
    full_table = df[['Dataset', 'Clean Source', 'Noise Source', 'Noise Type',
                     'SNR (dB)', 'PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']].copy()
    full_table = full_table.sort_values(['Clean Source', 'Noise Source', 'Noise Type', 'SNR (dB)'])
    full_table.to_csv(os.path.join(output_dir, 'full_results_table.csv'), index=False)

    print("\n" + "="*80)
    print(f"Tables saved to {output_dir}/")
    print("="*80)

    return table1, table2, table3


def plot_metrics_by_snr(df, output_dir):
    """Plot metrics vs SNR for different conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    for ax, metric, name in zip(axes, metrics, metric_names):
        for (clean, noise), group in df.groupby(['Clean Source', 'Noise Source']):
            data = group.groupby('SNR (dB)')[metric].mean()
            label = f"{clean} + {noise}"
            marker = 'o' if clean == 'VoiceBank' else 's'
            linestyle = '-' if noise == 'DEMAND' else '--'
            ax.plot(data.index, data.values, marker=marker, linestyle=linestyle, label=label)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(name)
        ax.set_title(f'{name} vs SNR')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_snr.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_by_snr.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/metrics_by_snr.png")
    plt.close()


def plot_metrics_by_noise_type(df, output_dir):
    """Plot metrics by noise type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    # Average over SNR
    df_avg = df.groupby(['Noise Source', 'Noise Type']).agg({
        'PESQ_mean': 'mean',
        'ESTOI_mean': 'mean',
        'SI-SDR_mean': 'mean'
    }).reset_index()

    for ax, metric, name in zip(axes, metrics, metric_names):
        # DEMAND
        demand_data = df_avg[df_avg['Noise Source'] == 'DEMAND'].sort_values(metric, ascending=False)
        # ESC-50
        esc_data = df_avg[df_avg['Noise Source'] == 'ESC-50'].sort_values(metric, ascending=False)

        x = np.arange(max(len(demand_data), len(esc_data)))
        width = 0.35

        if len(demand_data) > 0:
            ax.bar(x[:len(demand_data)] - width/2, demand_data[metric], width, label='DEMAND', color='steelblue')
            ax.set_xticks(x[:len(demand_data)])
            ax.set_xticklabels(demand_data['Noise Type'], rotation=45, ha='right')

        if len(esc_data) > 0:
            ax.bar(x[:len(esc_data)] + width/2, esc_data[metric], width, label='ESC-50', color='coral')

        ax.set_ylabel(name)
        ax.set_title(f'{name} by Noise Type')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_noise_type.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_by_noise_type.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/metrics_by_noise_type.png")
    plt.close()


def plot_heatmap(df, output_dir):
    """Plot heatmap of PESQ for all conditions."""
    # Create pivot table
    df_pivot = df.pivot_table(
        values='PESQ_mean',
        index=['Clean Source', 'Noise Source', 'Noise Type'],
        columns='SNR (dB)',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(df_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                vmin=1.5, vmax=4.0, cbar_kws={'label': 'PESQ'})
    ax.set_title('PESQ Heatmap: All Conditions')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pesq_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pesq_heatmap.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/pesq_heatmap.png")
    plt.close()


def plot_domain_comparison(df, output_dir):
    """Plot comparison of in-domain vs out-of-domain."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Create domain labels
    df['Domain'] = df.apply(lambda x:
        'In-Domain' if (x['Clean Source'] == 'VoiceBank' and x['Noise Source'] == 'DEMAND')
        else 'Clean OOD' if (x['Clean Source'] == 'LibriSpeech' and x['Noise Source'] == 'DEMAND')
        else 'Noise OOD' if (x['Clean Source'] == 'VoiceBank' and x['Noise Source'] == 'ESC-50')
        else 'Both OOD', axis=1)

    metrics = ['PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']
    colors = {'In-Domain': 'green', 'Clean OOD': 'blue', 'Noise OOD': 'orange', 'Both OOD': 'red'}

    for ax, metric, name in zip(axes, metrics, metric_names):
        domain_means = df.groupby('Domain')[metric].mean()
        domain_stds = df.groupby('Domain')[metric].std()

        order = ['In-Domain', 'Clean OOD', 'Noise OOD', 'Both OOD']
        x = range(len(order))

        bars = ax.bar(x, [domain_means.get(d, 0) for d in order],
                     yerr=[domain_stds.get(d, 0) for d in order],
                     color=[colors[d] for d in order], capsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha='right')
        ax.set_ylabel(name)
        ax.set_title(f'{name} by Domain')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'domain_comparison.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/domain_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results_dir", type=str, default="evaluation_results",
                       help="Directory containing summary CSV")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Directory to save plots and tables")
    args = parser.parse_args()

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Load data
    df = load_summary(args.results_dir)
    df = prepare_data(df)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate tables
    create_summary_table(df, args.output_dir)

    # Generate plots
    print("\nGenerating plots...")
    plot_metrics_by_snr(df, args.output_dir)
    plot_metrics_by_noise_type(df, args.output_dir)
    plot_heatmap(df, args.output_dir)
    plot_domain_comparison(df, args.output_dir)

    print("\n" + "="*80)
    print("DONE! All plots and tables saved to:", args.output_dir)
    print("="*80)

    # Print quick summary
    print("\n### Quick Summary ###")
    print(f"Total datasets evaluated: {len(df)}")
    print(f"\nOverall PESQ: {df['PESQ_mean'].mean():.2f} ± {df['PESQ_mean'].std():.2f}")
    print(f"Overall ESTOI: {df['ESTOI_mean'].mean():.3f} ± {df['ESTOI_mean'].std():.3f}")
    print(f"Overall SI-SDR: {df['SI-SDR_mean'].mean():.1f} ± {df['SI-SDR_mean'].std():.1f} dB")


if __name__ == "__main__":
    main()
