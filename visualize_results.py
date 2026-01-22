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
    # Special case: edinburgh_reverb (no SNR, convolutive noise)

    # Handle reverb datasets (no SNR)
    if 'reverb' in name.lower() and 'dB' not in name:
        return {
            'clean_source': 'Edinburgh',
            'noise_source': 'Reverb',
            'noise_type': name.replace('_', ' ').title(),
            'snr': None,  # Reverb doesn't have SNR
            'degradation_type': 'convolutive'
        }

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
        'snr': int(snr),
        'degradation_type': 'additive'
    }


def prepare_data(df):
    """Prepare data with parsed columns."""
    parsed = df['Dataset'].apply(parse_dataset_name)
    df['Clean Source'] = parsed.apply(lambda x: x['clean_source'])
    df['Noise Source'] = parsed.apply(lambda x: x['noise_source'])
    df['Noise Type'] = parsed.apply(lambda x: x['noise_type'])
    df['SNR (dB)'] = parsed.apply(lambda x: x['snr'])
    df['Degradation Type'] = parsed.apply(lambda x: x.get('degradation_type', 'additive'))
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

    # Table 3: DEMAND noise types
    df_demand = df[df['Noise Source'] == 'DEMAND']
    if len(df_demand) > 0:
        table3_demand = df_demand.groupby('Noise Type').agg({
            'PESQ_mean': 'mean',
            'ESTOI_mean': 'mean',
            'SI-SDR_mean': 'mean'
        }).round(3).sort_values('PESQ_mean', ascending=False)

        print("\n" + "="*80)
        print("TABLE 3a: DEMAND Noise Types (in-domain)")
        print("="*80)
        print(table3_demand.to_string())
        table3_demand.to_csv(os.path.join(output_dir, 'table_demand_noise_types.csv'))

    # Table 4: ESC-50 noise categories
    df_esc = df[df['Noise Source'] == 'ESC-50']
    if len(df_esc) > 0:
        table3_esc = df_esc.groupby('Noise Type').agg({
            'PESQ_mean': 'mean',
            'ESTOI_mean': 'mean',
            'SI-SDR_mean': 'mean'
        }).round(3).sort_values('PESQ_mean', ascending=False)

        print("\n" + "="*80)
        print("TABLE 3b: ESC-50 Noise Categories (OOD)")
        print("="*80)
        print(table3_esc.to_string())
        table3_esc.to_csv(os.path.join(output_dir, 'table_esc50_noise_types.csv'))

    # Table 5: Full results table
    full_table = df[['Dataset', 'Clean Source', 'Noise Source', 'Noise Type',
                     'SNR (dB)', 'PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']].copy()
    full_table = full_table.sort_values(['Clean Source', 'Noise Source', 'Noise Type', 'SNR (dB)'])
    full_table.to_csv(os.path.join(output_dir, 'full_results_table.csv'), index=False)

    print("\n" + "="*80)
    print(f"Tables saved to {output_dir}/")
    print("="*80)

    return table1, table2


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


def plot_demand_noise_types(df, output_dir):
    """Plot metrics by DEMAND noise types."""
    df_demand = df[df['Noise Source'] == 'DEMAND']
    if len(df_demand) == 0:
        print("No DEMAND data found, skipping DEMAND plots")
        return

    metrics = ['PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    # 1. Bar chart averaged over SNR
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df_avg = df_demand.groupby('Noise Type').agg({
        'PESQ_mean': 'mean',
        'ESTOI_mean': 'mean',
        'SI-SDR_mean': 'mean'
    }).reset_index()

    for ax, metric, name in zip(axes, metrics, metric_names):
        data = df_avg.sort_values(metric, ascending=True)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(data)))
        ax.barh(data['Noise Type'], data[metric], color=colors)
        ax.set_xlabel(name)
        ax.set_title(f'{name} by DEMAND Noise Type')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('DEMAND Noise Types (in-domain)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_noise_types.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'demand_noise_types.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/demand_noise_types.png")
    plt.close()

    # 2. Line plot: SNR vs metrics for each noise type (separate lines for Clean Source)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    noise_types = df_demand['Noise Type'].unique()
    clean_sources = df_demand['Clean Source'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(noise_types)))
    linestyles = {'VoiceBank': '-', 'LibriSpeech': '--'}
    markers = {'VoiceBank': 'o', 'LibriSpeech': 's'}

    for ax, metric, name in zip(axes, metrics, metric_names):
        for noise_type, color in zip(noise_types, colors):
            for clean_src in clean_sources:
                data = df_demand[(df_demand['Noise Type'] == noise_type) &
                                (df_demand['Clean Source'] == clean_src)].sort_values('SNR (dB)')
                if len(data) > 0:
                    label = f"{noise_type} ({clean_src[:2]})"  # VB or Li
                    ax.plot(data['SNR (dB)'], data[metric],
                           marker=markers.get(clean_src, 'o'),
                           linestyle=linestyles.get(clean_src, '-'),
                           label=label, color=color, linewidth=1.5)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(name)
        ax.set_title(f'{name} vs SNR (DEMAND)')
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('DEMAND: Performance by SNR and Noise Type (solid: VoiceBank, dashed: LibriSpeech)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'demand_snr_by_noise.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'demand_snr_by_noise.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/demand_snr_by_noise.png")
    plt.close()


def plot_esc50_noise_types(df, output_dir):
    """Plot metrics by ESC-50 noise categories."""
    df_esc = df[df['Noise Source'] == 'ESC-50']
    if len(df_esc) == 0:
        print("No ESC-50 data found, skipping ESC-50 plots")
        return

    metrics = ['PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    # 1. Bar chart averaged over SNR
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df_avg = df_esc.groupby('Noise Type').agg({
        'PESQ_mean': 'mean',
        'ESTOI_mean': 'mean',
        'SI-SDR_mean': 'mean'
    }).reset_index()

    for ax, metric, name in zip(axes, metrics, metric_names):
        data = df_avg.sort_values(metric, ascending=True)
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(data)))
        ax.barh(data['Noise Type'], data[metric], color=colors)
        ax.set_xlabel(name)
        ax.set_title(f'{name} by ESC-50 Category')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('ESC-50 Noise Categories (OOD)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'esc50_noise_types.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'esc50_noise_types.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/esc50_noise_types.png")
    plt.close()

    # 2. Line plot: SNR vs metrics for each category (separate lines for Clean Source)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    categories = df_esc['Noise Type'].unique()
    clean_sources = df_esc['Clean Source'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    linestyles = {'VoiceBank': '-', 'LibriSpeech': '--'}
    markers = {'VoiceBank': 'o', 'LibriSpeech': 's'}

    for ax, metric, name in zip(axes, metrics, metric_names):
        for category, color in zip(categories, colors):
            for clean_src in clean_sources:
                data = df_esc[(df_esc['Noise Type'] == category) &
                             (df_esc['Clean Source'] == clean_src)].sort_values('SNR (dB)')
                if len(data) > 0:
                    label = f"{category} ({clean_src[:2]})"  # VB or Li
                    ax.plot(data['SNR (dB)'], data[metric],
                           marker=markers.get(clean_src, 'o'),
                           linestyle=linestyles.get(clean_src, '-'),
                           label=label, color=color, linewidth=1.5)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(name)
        ax.set_title(f'{name} vs SNR (ESC-50)')
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ESC-50: Performance by SNR and Noise Category (solid: VoiceBank, dashed: LibriSpeech)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'esc50_snr_by_category.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'esc50_snr_by_category.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/esc50_snr_by_category.png")
    plt.close()


def plot_noise_source_comparison(df, output_dir):
    """Compare overall performance between DEMAND and ESC-50."""
    metrics = ['PESQ_mean', 'ESTOI_mean', 'SI-SDR_mean']
    metric_names = ['PESQ', 'ESTOI', 'SI-SDR (dB)']

    # Average by Noise Source and SNR
    df_avg = df.groupby(['Noise Source', 'SNR (dB)']).agg({
        'PESQ_mean': ['mean', 'std'],
        'ESTOI_mean': ['mean', 'std'],
        'SI-SDR_mean': ['mean', 'std']
    }).reset_index()
    df_avg.columns = ['Noise Source', 'SNR (dB)',
                      'PESQ_mean', 'PESQ_std',
                      'ESTOI_mean', 'ESTOI_std',
                      'SI-SDR_mean', 'SI-SDR_std']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'DEMAND': 'steelblue', 'ESC-50': 'coral'}

    for ax, metric, name in zip(axes, metrics, metric_names):
        std_col = metric.replace('_mean', '_std')
        for source in ['DEMAND', 'ESC-50']:
            data = df_avg[df_avg['Noise Source'] == source].sort_values('SNR (dB)')
            if len(data) > 0:
                ax.errorbar(data['SNR (dB)'], data[metric],
                           yerr=data[std_col], marker='o', label=f'{source} (avg)',
                           color=colors[source], capsize=3, linewidth=2)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(name)
        ax.set_title(f'{name}: DEMAND vs ESC-50')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('In-domain (DEMAND) vs OOD (ESC-50) Noise Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_source_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'noise_source_comparison.pdf'), bbox_inches='tight')
    print(f"Saved: {output_dir}/noise_source_comparison.png")
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

    # Noise type analysis (separated by source)
    plot_demand_noise_types(df, args.output_dir)
    plot_esc50_noise_types(df, args.output_dir)
    plot_noise_source_comparison(df, args.output_dir)

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
