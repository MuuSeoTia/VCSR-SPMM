#!/usr/bin/env python3
"""
ISPASS-style publication-quality graphs for SpMM benchmark results.
Generates separate graphs for A100, H100, and H200.
VCSR methods positioned on the right-hand side for emphasis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ISPASS-inspired color palette (colorblind-friendly, professional)
COLORS = {
    'cusparse': '#4A90A4',      # Muted teal - vendor baseline
    'csr': '#8B5A8C',           # Muted purple - naive impl
    'vcsr_baseline': '#E8A838', # Warm gold - our baseline
    'vcsr_seg': '#D64541',      # Vibrant red - our optimized (highlight)
    'aspt': '#2C3E50',          # Dark slate - competitor
    'fastspmm': '#7F8C8D',      # Cool gray - competitor
}

ALGO_LABELS = {
    'cusparse': 'cuSPARSE',
    'csr': 'CSR (naive)',
    'vcsr_baseline': 'VCSR-Base',
    'vcsr_seg': 'VCSR-Seg',
    'aspt': 'ASpT',
    'fastspmm': 'FastSpMM',
}

# Order for display - VCSR methods on the RIGHT
ALGO_ORDER = ['cusparse', 'aspt', 'fastspmm', 'vcsr_baseline', 'vcsr_seg']


def load_data():
    """Load benchmark data from CSV files."""
    base = Path(__file__).parent.parent / 'reports'
    data = {}
    for gpu in ['a100', 'h100', 'h200']:
        csv_path = base / gpu / 'final_benchmark.csv'
        if csv_path.exists():
            data[gpu.upper()] = pd.read_csv(csv_path)
    return data


def create_gflops_comparison(df, gpu_name, output_dir):
    """
    Create grouped bar chart comparing GFLOP/s across algorithms.
    One chart per O value, grouped by matrix.
    """
    for O_val in [64, 128, 256]:
        subset = df[df['O'] == O_val].copy()
        if subset.empty:
            continue
        
        matrices = sorted(subset['matrix'].unique())
        n_matrices = len(matrices)
        n_algos = len(ALGO_ORDER)
        
        fig, ax = plt.subplots(figsize=(11, 4))
        
        x = np.arange(n_matrices)
        width = 0.14
        offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
        
        for i, algo in enumerate(ALGO_ORDER):
            algo_data = subset[subset['algo'] == algo]
            values = []
            for m in matrices:
                v = algo_data[algo_data['matrix'] == m]['gflops'].values
                values.append(v[0] if len(v) > 0 else 0)
            
            bars = ax.bar(x + offsets[i], values, width, 
                         label=ALGO_LABELS[algo], 
                         color=COLORS[algo],
                         edgecolor='white',
                         linewidth=0.3,
                         zorder=3)
            
            # Highlight our best result with thicker border
            if algo == 'vcsr_seg':
                for bar in bars:
                    bar.set_edgecolor('#8B0000')
                    bar.set_linewidth(0.8)
        
        ax.set_ylabel('Throughput (GFLOP/s)', fontweight='medium')
        ax.set_xlabel('Matrix', fontweight='medium')
        # GPU + O label in corner
        ax.text(0.02, 0.98, f'{gpu_name} | O={O_val}', transform=ax.transAxes, 
               fontsize=10, fontweight='bold', ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels([m[:11] + '..' if len(m) > 11 else m for m in matrices], 
                          rotation=30, ha='right', fontsize=8)
        
        ax.set_ylim(0, subset['gflops'].max() * 1.12)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                 ncol=5, frameon=True, fancybox=False, 
                 edgecolor='#CCCCCC', facecolor='white')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        out_path = output_dir / f'{gpu_name.lower()}_all_algo_O{O_val}.png'
        plt.savefig(out_path, format='png')
        plt.close()
        print(f"    {out_path.name}")


def create_speedup_vs_csr(all_data, output_dir):
    """
    Create speedup chart vs CSR (naive) baseline - excludes cuSPARSE.
    Shows speedup for all O values in separate subplots.
    X-axis: matrices, Y-axis: speedup
    """
    for gpu, df in all_data.items():
        O_values = sorted(df['O'].unique())
        n_O = len(O_values)
        
        # Layout based on number of O values
        if n_O <= 3:
            fig, axes = plt.subplots(1, n_O, figsize=(5*n_O, 4.5), sharey=False)
        else:
            fig, axes = plt.subplots(2, (n_O+1)//2, figsize=(6*((n_O+1)//2), 8), sharey=False)
        
        if n_O == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Algorithms to compare (exclude cuSPARSE and CSR itself) - VCSR on RHS
        compare_algos = [a for a in ['aspt', 'fastspmm', 'vcsr_baseline', 'vcsr_seg'] if a in df['algo'].values]
        
        for idx, O_val in enumerate(O_values):
            ax = axes[idx]
            subset = df[df['O'] == O_val].copy()
            
            # Get CSR baseline
            csr_base = subset[subset['algo'] == 'csr'][['matrix', 'ms']]
            csr_base = csr_base.rename(columns={'ms': 'base_ms'})
            
            matrices = sorted(subset['matrix'].unique())
            n_matrices = len(matrices)
            n_algos = len(compare_algos)
            
            x = np.arange(n_matrices)
            width = 0.18
            offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
            
            max_speedup = 1.0
            for i, algo in enumerate(compare_algos):
                algo_data = subset[subset['algo'] == algo].merge(csr_base, on='matrix')
                speedups = []
                for m in matrices:
                    row = algo_data[algo_data['matrix'] == m]
                    if len(row) > 0 and row['ms'].values[0] > 0:
                        sp = row['base_ms'].values[0] / row['ms'].values[0]  # CSR time / algo time
                        speedups.append(sp)
                        max_speedup = max(max_speedup, sp)
                    else:
                        speedups.append(0)
                
                bars = ax.bar(x + offsets[i], speedups, width,
                             label=ALGO_LABELS[algo] if idx == 0 else '',
                             color=COLORS[algo],
                             edgecolor='white',
                             linewidth=0.3,
                             zorder=3)
                
                if algo == 'vcsr_seg':
                    for bar in bars:
                        bar.set_edgecolor('#8B0000')
                        bar.set_linewidth(0.8)
            
            # X-axis: matrices
            ax.set_xlabel('')
            if idx == 0:
                ax.set_ylabel('Speedup vs CSR', fontweight='medium')
            ax.set_xticks(x)
            ax.set_xticklabels([m[:8] + '..' if len(m) > 8 else m for m in matrices],
                              rotation=45, ha='right', fontsize=7)
            
            # Add GPU + O label in corner
            ax.text(0.02, 0.98, f'{gpu} | O={O_val}', transform=ax.transAxes, 
                   fontsize=10, fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
            
            ax.set_ylim(0, max_speedup * 1.12)
            ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Remove empty subplots if odd number of O values
        for idx in range(len(O_values), len(axes)):
            fig.delaxes(axes[idx])
        
        # Shared legend at bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=4, frameon=True, fancybox=False,
                  edgecolor='#CCCCCC', facecolor='white')
        
        plt.tight_layout(rect=[0, 0.08, 1, 1.0])
        
        out_path = output_dir / f'{gpu.lower()}_speedup_vs_csr.png'
        plt.savefig(out_path, format='png')
        plt.close()
        print(f"    {out_path.name}")


def create_vcsr_vs_csr_diverging(all_data, output_dir):
    """
    Create grouped bar chart comparing ALL algorithms to CSR across matrices.
    X-axis: matrices, Y-axis: speedup
    Includes vcsr_baseline, vcsr_seg, aspt, fastspmm - all color coded.
    """
    for gpu, df in all_data.items():
        O_values = sorted(df['O'].unique())
        matrices = sorted(df['matrix'].unique())
        n_matrices = len(matrices)
        
        # Create one chart per O value
        for O_val in O_values:
            subset = df[df['O'] == O_val]
            
            fig, ax = plt.subplots(figsize=(12, 5))
            
            x = np.arange(n_matrices)
            # VCSR methods on RHS
            compare_algos = ['aspt', 'fastspmm', 'vcsr_baseline', 'vcsr_seg']
            n_algos = len(compare_algos)
            width = 0.18
            offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
            
            max_speedup = 1.0
            for i, algo in enumerate(compare_algos):
                speedups = []
                for matrix in matrices:
                    csr_ms = subset[(subset['matrix'] == matrix) & (subset['algo'] == 'csr')]['ms'].values
                    algo_ms = subset[(subset['matrix'] == matrix) & (subset['algo'] == algo)]['ms'].values
                    
                    if len(csr_ms) > 0 and len(algo_ms) > 0 and algo_ms[0] > 0:
                        speedup = csr_ms[0] / algo_ms[0]  # > 1 means algo is faster
                        speedups.append(speedup)
                        max_speedup = max(max_speedup, speedup)
                    else:
                        speedups.append(0)
                
                bars = ax.bar(x + offsets[i], speedups, width,
                             label=ALGO_LABELS[algo],
                             color=COLORS[algo],
                             edgecolor='white',
                             linewidth=0.3,
                             zorder=3)
                
                if algo == 'vcsr_seg':
                    for bar in bars:
                        bar.set_edgecolor('#8B0000')
                        bar.set_linewidth(0.8)
            
            # Add reference line at 1.0x (equal to CSR)
            ax.axhline(y=1.0, color='#8B5A8C', linestyle='--', linewidth=1.5, 
                      label='CSR (1.0×)', zorder=2, alpha=0.7)
            
            # X-axis: matrices
            ax.set_xticks(x)
            ax.set_xticklabels([m[:10] + '..' if len(m) > 10 else m for m in matrices],
                              rotation=35, ha='right', fontsize=8)
            
            # Y-axis: speedup
            ax.set_ylabel('Speedup vs CSR', fontweight='medium')
            ax.set_ylim(0, max_speedup * 1.12)
            
            # Add GPU + O label in corner
            ax.text(0.02, 0.98, f'{gpu} | O={O_val}', transform=ax.transAxes, 
                   fontsize=11, fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
            
            ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=5, frameon=True, edgecolor='#CCCCCC', fontsize=9)
            
            plt.tight_layout()
            out_path = output_dir / f'{gpu.lower()}_all_vs_csr_O{O_val}.png'
            plt.savefig(out_path, format='png')
            plt.close()
            print(f"    {out_path.name}")


def create_speedup_vs_csr_summary(all_data, output_dir):
    """
    Create summary bar chart showing geometric mean speedup vs CSR for each GPU.
    X-axis: GPU/algorithm, Y-axis: speedup
    Excludes cuSPARSE.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    gpus = list(all_data.keys())
    # VCSR methods on RHS
    compare_algos = ['aspt', 'fastspmm', 'vcsr_baseline', 'vcsr_seg']
    n_gpus = len(gpus)
    n_algos = len(compare_algos)
    
    x = np.arange(n_gpus)
    width = 0.18
    offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
    
    max_speedup = 1.0
    for i, algo in enumerate(compare_algos):
        if algo not in COLORS:
            continue
        speedups = []
        for gpu in gpus:
            df = all_data[gpu]
            csr_base = df[df['algo'] == 'csr'][['matrix', 'O', 'ms']].rename(columns={'ms': 'base_ms'})
            algo_data = df[df['algo'] == algo][['matrix', 'O', 'ms']]
            merged = pd.merge(algo_data, csr_base, on=['matrix', 'O'])
            merged['speedup'] = merged['base_ms'] / merged['ms']  # CSR time / algo time
            valid = merged['speedup'][merged['speedup'] > 0]
            if len(valid) > 0:
                geo_speedup = np.exp(np.mean(np.log(valid)))
                speedups.append(geo_speedup)
                max_speedup = max(max_speedup, geo_speedup)
            else:
                speedups.append(0)
        
        bars = ax.bar(x + offsets[i], speedups, width,
                     label=ALGO_LABELS.get(algo, algo),
                     color=COLORS[algo],
                     edgecolor='white',
                     linewidth=0.3,
                     zorder=3)
        
        if algo == 'vcsr_seg':
            for j, bar in enumerate(bars):
                bar.set_edgecolor('#8B0000')
                bar.set_linewidth(1.0)
                # Add value label on top
                height = bar.get_height()
                ax.annotate(f'{height:.1f}×',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color=COLORS['vcsr_seg'])
    
    # X-axis: GPUs
    ax.set_xlabel('')
    ax.set_xticks(x)
    ax.set_xticklabels(gpus, fontsize=11, fontweight='medium')
    
    # Y-axis: speedup
    ax.set_ylabel('Geometric Mean Speedup vs CSR', fontweight='medium')
    ax.set_ylim(0, max_speedup * 1.15)
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
             ncol=4, frameon=True, fancybox=False,
             edgecolor='#CCCCCC', facecolor='white')
    
    plt.tight_layout()
    out_path = output_dir / 'summary_speedup_vs_csr.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_vcsr_vs_competitors(all_data, output_dir):
    """
    Create comparison chart showing VCSR-baseline and VCSR-seg speedup over ASpT and FastSpMM.
    X-axis: matrices, Y-axis: speedup
    """
    for gpu, df in all_data.items():
        for O_val in sorted(df['O'].unique()):
            subset = df[df['O'] == O_val]
            matrices = sorted(subset['matrix'].unique())
            n_matrices = len(matrices)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            
            # Compare against ASpT and FastSpMM
            for ax_idx, competitor in enumerate(['aspt', 'fastspmm']):
                ax = axes[ax_idx]
                
                x = np.arange(n_matrices)
                vcsr_algos = ['vcsr_baseline', 'vcsr_seg']
                n_algos = len(vcsr_algos)
                width = 0.35
                offsets = [-width/2, width/2]
                
                max_speedup = 1.0
                for i, algo in enumerate(vcsr_algos):
                    speedups = []
                    for matrix in matrices:
                        comp_ms = subset[(subset['matrix'] == matrix) & (subset['algo'] == competitor)]['ms'].values
                        algo_ms = subset[(subset['matrix'] == matrix) & (subset['algo'] == algo)]['ms'].values
                        
                        if len(comp_ms) > 0 and len(algo_ms) > 0 and algo_ms[0] > 0:
                            speedup = comp_ms[0] / algo_ms[0]  # competitor time / VCSR time
                            speedups.append(speedup)
                            max_speedup = max(max_speedup, speedup)
                        else:
                            speedups.append(0)
                    
                    bars = ax.bar(x + offsets[i], speedups, width,
                                 label=ALGO_LABELS[algo],
                                 color=COLORS[algo],
                                 edgecolor='white',
                                 linewidth=0.3,
                                 zorder=3)
                    
                    if algo == 'vcsr_seg':
                        for bar in bars:
                            bar.set_edgecolor('#8B0000')
                            bar.set_linewidth(0.8)
                
                # Reference line at 1.0x
                ax.axhline(y=1.0, color=COLORS[competitor], linestyle='--', linewidth=1.5, 
                          label=f'{ALGO_LABELS[competitor]} (1.0×)', zorder=2, alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels([m[:9] + '..' if len(m) > 9 else m for m in matrices],
                                  rotation=40, ha='right', fontsize=8)
                
                if ax_idx == 0:
                    ax.set_ylabel('Speedup', fontweight='medium')
                
                # Label in corner: GPU + O + competitor
                ax.text(0.02, 0.98, f'{gpu} | O={O_val}', transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
                ax.text(0.98, 0.95, f'vs {ALGO_LABELS[competitor]}', transform=ax.transAxes, 
                       fontsize=11, fontweight='bold', ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
                
                ax.set_ylim(0, max_speedup * 1.12)
                ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                ax.legend(loc='upper left', frameon=True, edgecolor='#CCCCCC', fontsize=9)
            
            plt.tight_layout()
            out_path = output_dir / f'{gpu.lower()}_vcsr_vs_competitors_O{O_val}.png'
            plt.savefig(out_path, format='png')
            plt.close()
            print(f"    {out_path.name}")

def create_speedup_chart(df, gpu_name, output_dir):
    """
    Create speedup chart vs cuSPARSE - single figure with 3 subplots (O=64,128,256).
    VCSR bars appear on the RIGHT.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for idx, O_val in enumerate([64, 128, 256]):
        ax = axes[idx]
        subset = df[df['O'] == O_val].copy()
        if subset.empty:
            continue

        # cuSPARSE baseline
        cusparse_base = (
            subset[subset['algo'] == 'cusparse'][['matrix', 'gflops']]
            .rename(columns={'gflops': 'base_gf'})
        )

        matrices = sorted(subset['matrix'].unique())
        n_matrices = len(matrices)

        # IMPORTANT: VCSR ON THE RIGHT
        compare_algos = ['aspt', 'fastspmm', 'vcsr_baseline', 'vcsr_seg']
        n_algos = len(compare_algos)

        x = np.arange(n_matrices)
        width = 0.18
        offsets = np.linspace(
            -(n_algos - 1) * width / 2,
            (n_algos - 1) * width / 2,
            n_algos
        )

        max_speedup = 1.0
        for i, algo in enumerate(compare_algos):
            algo_data = subset[subset['algo'] == algo].merge(
                cusparse_base, on='matrix'
            )

            speedups = []
            for m in matrices:
                row = algo_data[algo_data['matrix'] == m]
                if len(row) > 0 and row['base_gf'].values[0] > 0:
                    sp = row['gflops'].values[0] / row['base_gf'].values[0]
                    speedups.append(sp)
                    max_speedup = max(max_speedup, sp)
                else:
                    speedups.append(0)

            bars = ax.bar(
                x + offsets[i],
                speedups,
                width,
                label=ALGO_LABELS[algo] if idx == 1 else '',
                color=COLORS[algo],
                edgecolor='white',
                linewidth=0.3,
                zorder=3
            )

            if algo == 'vcsr_seg':
                for bar in bars:
                    bar.set_edgecolor('#8B0000')
                    bar.set_linewidth(0.8)

        # cuSPARSE reference
        ax.axhline(
            y=1.0,
            color=COLORS['cusparse'],
            linestyle='--',
            linewidth=1.5,
            label='cuSPARSE (1.0×)' if idx == 1 else '',
            zorder=2,
            alpha=0.8
        )

        if idx == 0:
            ax.set_ylabel('Speedup vs cuSPARSE', fontweight='medium')

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m[:7] + '..' if len(m) > 7 else m for m in matrices],
            rotation=45,
            ha='right',
            fontsize=7
        )

        ax.text(
            0.98, 0.95, f'O={O_val}',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            ha='right',
            va='top',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      edgecolor='gray',
                      alpha=0.8)
        )

        ax.set_ylim(0, max_speedup * 1.15)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor='#CCCCCC'
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    out_path = output_dir / f'{gpu_name.lower()}_speedup_vs_cusparse.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")

def create_vcsr_vs_competitors_summary(all_data, output_dir):
    """
    Summary chart: geometric mean speedup of VCSR-baseline and VCSR-seg over ASpT and FastSpMM.
    X-axis: competitor (ASpT, FastSpMM), Y-axis: speedup, grouped by GPU
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    gpus = list(all_data.keys())
    competitors = ['aspt', 'fastspmm']
    vcsr_algos = ['vcsr_baseline', 'vcsr_seg']
    
    # Create grouped structure: for each competitor, show VCSR-base and VCSR-seg for each GPU
    n_groups = len(competitors)
    n_bars_per_group = len(gpus) * len(vcsr_algos)
    
    group_positions = np.arange(n_groups) * (n_bars_per_group + 1)
    
    gpu_colors = {'A100': ('#E8A838', '#D64541'), 'H100': ('#F4C76A', '#E86B67'), 'H200': ('#FCE08B', '#F08A87')}
    
    bar_idx = 0
    for g_idx, gpu in enumerate(gpus):
        df = all_data[gpu]
        for v_idx, vcsr_algo in enumerate(vcsr_algos):
            positions = []
            speedups = []
            for c_idx, comp in enumerate(competitors):
                # Calculate geometric mean speedup
                comp_data = df[df['algo'] == comp][['matrix', 'O', 'ms']].rename(columns={'ms': 'comp_ms'})
                vcsr_data = df[df['algo'] == vcsr_algo][['matrix', 'O', 'ms']]
                merged = pd.merge(vcsr_data, comp_data, on=['matrix', 'O'])
                merged['speedup'] = merged['comp_ms'] / merged['ms']
                valid = merged['speedup'][merged['speedup'] > 0]
                
                if len(valid) > 0:
                    geo = np.exp(np.mean(np.log(valid)))
                else:
                    geo = 0
                
                positions.append(group_positions[c_idx] + bar_idx)
                speedups.append(geo)
            
            color = gpu_colors.get(gpu, ('#888', '#444'))[v_idx]
            label = f'{gpu} {ALGO_LABELS[vcsr_algo]}'
            ax.bar(positions, speedups, 0.8, label=label, color=color, edgecolor='white', linewidth=0.3, zorder=3)
            
            # Add value labels
            for pos, sp in zip(positions, speedups):
                if sp > 0:
                    ax.annotate(f'{sp:.2f}×', xy=(pos, sp), xytext=(0, 2), 
                               textcoords='offset points', ha='center', va='bottom', 
                               fontsize=8, fontweight='bold')
            
            bar_idx += 1
        bar_idx += 0.5  # Gap between GPUs
    
    # Reference line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    
    # X-axis labels
    ax.set_xticks([g + (n_bars_per_group-1)/2 for g in group_positions])
    ax.set_xticklabels([ALGO_LABELS[c] for c in competitors], fontsize=11, fontweight='medium')
    
    ax.set_ylabel('Geometric Mean Speedup', fontweight='medium')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
             ncol=6, frameon=True, fancybox=False,
             edgecolor='#CCCCCC', facecolor='white', fontsize=8)
    
    plt.tight_layout()
    out_path = output_dir / 'summary_vcsr_vs_competitors.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_vcsr_seg_vs_base_vs_csr(all_data, output_dir):
    """
    Focused comparison: VCSR-Seg and VCSR-Base speedups vs CSR.
    X-axis: matrices, Y-axis: speedup vs CSR.
    """
    for gpu, df in all_data.items():
        O_values = sorted(df['O'].unique())
        matrices = sorted(df['matrix'].unique())
        n_matrices = len(matrices)

        for O_val in O_values:
            subset = df[df['O'] == O_val].copy()
            if subset.empty:
                continue

            csr_base = subset[subset['algo'] == 'csr'][['matrix', 'ms']].rename(columns={'ms': 'csr_ms'})

            fig, ax = plt.subplots(figsize=(12, 4.5))

            x = np.arange(n_matrices)
            vcsr_algos = ['vcsr_baseline', 'vcsr_seg']
            width = 0.32
            offsets = [-width / 2, width / 2]

            max_speedup = 1.0
            for i, algo in enumerate(vcsr_algos):
                algo_data = subset[subset['algo'] == algo][['matrix', 'ms']]
                merged = pd.merge(algo_data, csr_base, on='matrix', how='inner')

                speedups = []
                for m in matrices:
                    row = merged[merged['matrix'] == m]
                    if len(row) > 0 and row['ms'].values[0] > 0:
                        sp = row['csr_ms'].values[0] / row['ms'].values[0]
                        speedups.append(sp)
                        max_speedup = max(max_speedup, sp)
                    else:
                        speedups.append(0)

                bars = ax.bar(x + offsets[i], speedups, width,
                              label=ALGO_LABELS[algo],
                              color=COLORS[algo],
                              edgecolor='white',
                              linewidth=0.3,
                              zorder=3)

                if algo == 'vcsr_seg':
                    for bar in bars:
                        bar.set_edgecolor('#8B0000')
                        bar.set_linewidth(0.8)

            # Reference line at 1.0x (CSR baseline)
            ax.axhline(y=1.0, color=COLORS['csr'], linestyle='--', linewidth=1.5,
                       label='CSR (1.0×)', zorder=2, alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels([m[:10] + '..' if len(m) > 10 else m for m in matrices],
                               rotation=35, ha='right', fontsize=8)
            ax.set_ylabel('Speedup vs CSR', fontweight='medium')
            ax.set_ylim(0, max_speedup * 1.12)

            # GPU + O label in corner
            ax.text(0.02, 0.98, f'{gpu} | O={O_val}', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

            ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      ncol=3, frameon=True, edgecolor='#CCCCCC', fontsize=9)

            plt.tight_layout()
            out_path = output_dir / f'{gpu.lower()}_vcsr_seg_vs_base_vs_csr_O{O_val}.png'
            plt.savefig(out_path, format='png')
            plt.close()
            print(f"    {out_path.name}")


def main():
    print("=" * 60)
    print("  ISPASS-Style Publication Graph Generation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading benchmark data...")
    all_data = load_data()
    print(f"  Loaded: {list(all_data.keys())}")
    
    # Generate graphs for each GPU
    for gpu, df in all_data.items():
        print(f"\n{gpu} Graphs:")
        create_gflops_comparison(df, gpu, output_dir)
        create_speedup_chart(df, gpu, output_dir)
        # create_heatmap_speedup({gpu: df}, output_dir)
        # create_scaling_plot({gpu: df}, output_dir)
        # create_all_algo_comparison({gpu: df}, output_dir)
    
    # Generate summary graphs
    print("\nSummary Graphs:")
    create_geometric_mean_summary(all_data, output_dir)
    create_speedup_summary(all_data, output_dir)
    create_nnz_performance_scatter(all_data, output_dir)
    create_gpu_comparison(all_data, output_dir)
    
    # Generate speedup vs CSR graphs (no cuSPARSE)
    print("\nSpeedup vs CSR Graphs (excluding cuSPARSE):")
    for gpu, df in all_data.items():
        create_speedup_vs_csr({gpu: df}, output_dir)
        create_vcsr_vs_csr_diverging({gpu: df}, output_dir)
        create_speedup_vs_csr_summary(all_data, output_dir)
    
    # Generate VCSR vs competitors (ASpT, FastSpMM) graphs
    print("\nVCSR vs Competitors (ASpT, FastSpMM):")
    for gpu, df in all_data.items():
        create_vcsr_vs_competitors({gpu: df}, output_dir)
        create_vcsr_vs_competitors_summary(all_data, output_dir)

    # Focused VCSR-Seg vs VCSR-Base vs CSR graphs
    print("\nVCSR-Seg vs VCSR-Base vs CSR:")
    create_vcsr_seg_vs_base_vs_csr(all_data, output_dir)
    
    print("\n" + "=" * 60)
    print(f"  All graphs saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
        
    ax.set_ylim(0, subset['gflops'].max() * 1.12)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
        
    # Legend at bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
    ncol=5, frameon=True, fancybox=False, 
    edgecolor='#CCCCCC', facecolor='white')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    plt.tight_layout()
    out_path = output_dir / f'{gpu_name.lower()}_gflops_O{O_val}.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_speedup_chart(df, gpu_name, output_dir):
    """
    Create speedup chart vs cuSPARSE - single figure with 3 subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    
    for idx, O_val in enumerate([64, 128, 256]):
        ax = axes[idx]
        subset = df[df['O'] == O_val].copy()
        
        # Get cuSPARSE baseline
        cusparse_base = subset[subset['algo'] == 'cusparse'][['matrix', 'gflops']]
        cusparse_base = cusparse_base.rename(columns={'gflops': 'base_gf'})
        
        matrices = sorted(subset['matrix'].unique())
        n_matrices = len(matrices)
        
        # Only compare non-cusparse algorithms
        compare_algos = ['aspt', 'fastspmm', 'vcsr_baseline', 'vcsr_seg']
        n_algos = len(compare_algos)
        
        x = np.arange(n_matrices)
        width = 0.18
        offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
        
        max_speedup = 1.0
        for i, algo in enumerate(compare_algos):
            algo_data = subset[subset['algo'] == algo].merge(cusparse_base, on='matrix')
            speedups = []
            for m in matrices:
                row = algo_data[algo_data['matrix'] == m]
                if len(row) > 0 and row['base_gf'].values[0] > 0:
                    sp = row['gflops'].values[0] / row['base_gf'].values[0]
                    speedups.append(sp)
                    max_speedup = max(max_speedup, sp)
                else:
                    speedups.append(0)
            
            bars = ax.bar(x + offsets[i], speedups, width,
                         label=ALGO_LABELS[algo] if idx == 1 else '',
                         color=COLORS[algo],
                         edgecolor='white',
                         linewidth=0.3,
                         zorder=3)
            
            if algo == 'vcsr_seg':
                for bar in bars:
                    bar.set_edgecolor('#8B0000')
                    bar.set_linewidth(0.8)
        
        # Reference line at 1.0x (cuSPARSE baseline)
        ax.axhline(y=1.0, color=COLORS['cusparse'], linestyle='--', linewidth=1.5, 
                  label='cuSPARSE (1.0×)' if idx == 1 else '', zorder=2, alpha=0.8)
        
        ax.set_xlabel('Matrix', fontweight='medium')
        if idx == 0:
            ax.set_ylabel('Speedup vs cuSPARSE', fontweight='medium')
        # O label in corner instead of title
        ax.text(0.98, 0.95, f'O={O_val}', transform=ax.transAxes, 
               fontsize=10, fontweight='bold', ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels([m[:7] + '..' if len(m) > 7 else m for m in matrices],
                          rotation=45, ha='right', fontsize=7)
        
        ax.set_ylim(0, max_speedup * 1.15)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Shared legend below middle subplot
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=5, frameon=True, fancybox=False,
              edgecolor='#CCCCCC', facecolor='white')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    out_path = output_dir / f'{gpu_name.lower()}_speedup_vs_cusparse.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_geometric_mean_summary(all_data, output_dir):
    """
    Create summary bar chart showing geometric mean GFLOP/s across all matrices.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    gpus = list(all_data.keys())
    n_gpus = len(gpus)
    n_algos = len(ALGO_ORDER)
    
    x = np.arange(n_gpus)
    width = 0.14
    offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
    
    for i, algo in enumerate(ALGO_ORDER):
        geo_means = []
        for gpu in gpus:
            df = all_data[gpu]
            algo_data = df[df['algo'] == algo]['gflops']
            valid = algo_data[algo_data > 0]
            if len(valid) > 0:
                geo_mean = np.exp(np.mean(np.log(valid)))
                geo_means.append(geo_mean)
            else:
                geo_means.append(0)
        
        bars = ax.bar(x + offsets[i], geo_means, width,
                     label=ALGO_LABELS[algo],
                     color=COLORS[algo],
                     edgecolor='white',
                     linewidth=0.3,
                     zorder=3)
        
        if algo == 'vcsr_seg':
            for bar in bars:
                bar.set_edgecolor('#8B0000')
                bar.set_linewidth(1.0)
    
    ax.set_ylabel('Geometric Mean GFLOP/s', fontweight='medium')
    ax.set_xlabel('GPU Architecture', fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels(gpus, fontsize=11, fontweight='medium')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=5, frameon=True, fancybox=False,
             edgecolor='#CCCCCC', facecolor='white')
    
    plt.tight_layout()
    out_path = output_dir / 'summary_geomean_gflops.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_speedup_summary(all_data, output_dir):
    """
    Create summary showing geometric mean speedup vs cuSPARSE for each GPU.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    gpus = list(all_data.keys())
    compare_algos = [a for a in ALGO_ORDER if a != 'cusparse']
    n_gpus = len(gpus)
    n_algos = len(compare_algos)
    
    x = np.arange(n_gpus)
    width = 0.18
    offsets = np.linspace(-(n_algos-1)*width/2, (n_algos-1)*width/2, n_algos)
    
    for i, algo in enumerate(compare_algos):
        speedups = []
        for gpu in gpus:
            df = all_data[gpu]
            cusparse = df[df['algo'] == 'cusparse'][['matrix', 'O', 'gflops']].rename(columns={'gflops': 'base'})
            algo_data = df[df['algo'] == algo][['matrix', 'O', 'gflops']]
            merged = pd.merge(algo_data, cusparse, on=['matrix', 'O'])
            merged['speedup'] = merged['gflops'] / merged['base']
            valid = merged['speedup'][merged['speedup'] > 0]
            if len(valid) > 0:
                geo_speedup = np.exp(np.mean(np.log(valid)))
                speedups.append(geo_speedup)
            else:
                speedups.append(0)
        
        bars = ax.bar(x + offsets[i], speedups, width,
                     label=ALGO_LABELS[algo],
                     color=COLORS[algo],
                     edgecolor='white',
                     linewidth=0.3,
                     zorder=3)
        
        if algo == 'vcsr_seg':
            for j, bar in enumerate(bars):
                bar.set_edgecolor('#8B0000')
                bar.set_linewidth(1.0)
                # Add value label on top
                height = bar.get_height()
                ax.annotate(f'{height:.2f}×',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=COLORS['vcsr_seg'])
    
    # Reference line at 1.0x
    ax.axhline(y=1.0, color=COLORS['cusparse'], linestyle='--', linewidth=1.5, 
              label='cuSPARSE (1.0×)', zorder=2, alpha=0.8)
    
    ax.set_ylabel('Geometric Mean Speedup', fontweight='medium')
    ax.set_xlabel('GPU Architecture', fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels(gpus, fontsize=11, fontweight='medium')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=5, frameon=True, fancybox=False,
             edgecolor='#CCCCCC', facecolor='white')
    
    plt.tight_layout()
    out_path = output_dir / 'summary_speedup_vs_cusparse.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_heatmap_speedup(all_data, output_dir):
    """
    Create heatmap showing vcsr_seg speedup vs cuSPARSE for each matrix/O combination.
    """
    for gpu, df in all_data.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        
        matrices = sorted(df['matrix'].unique())
        O_values = [64, 128, 256]
        
        # Build speedup matrix
        speedup_matrix = np.zeros((len(matrices), len(O_values)))
        
        for i, matrix in enumerate(matrices):
            for j, O in enumerate(O_values):
                cusparse_gf = df[(df['matrix'] == matrix) & (df['O'] == O) & (df['algo'] == 'cusparse')]['gflops'].values
                vcsr_gf = df[(df['matrix'] == matrix) & (df['O'] == O) & (df['algo'] == 'vcsr_seg')]['gflops'].values
                if len(cusparse_gf) > 0 and len(vcsr_gf) > 0 and cusparse_gf[0] > 0:
                    speedup_matrix[i, j] = vcsr_gf[0] / cusparse_gf[0]
        
        # Create heatmap
        im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=10)
        
        ax.set_xticks(np.arange(len(O_values)))
        ax.set_yticks(np.arange(len(matrices)))
        ax.set_xticklabels([f'O={o}' for o in O_values])
        ax.set_yticklabels([m[:12] + '..' if len(m) > 12 else m for m in matrices], fontsize=8)
        
        # Add text annotations
        for i in range(len(matrices)):
            for j in range(len(O_values)):
                val = speedup_matrix[i, j]
                color = 'white' if val > 5 or val < 1 else 'black'
                ax.text(j, i, f'{val:.1f}×', ha='center', va='center', color=color, fontsize=8)
        
        # GPU label in corner
        ax.text(0.02, 1.02, f'{gpu} VCSR-Seg vs cuSPARSE', transform=ax.transAxes, 
               fontsize=10, fontweight='bold', ha='left', va='bottom')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Speedup', fontweight='medium')
        
        plt.tight_layout()
        out_path = output_dir / f'{gpu.lower()}_speedup_heatmap.png'
        plt.savefig(out_path, format='png')
        plt.close()
        print(f"    {out_path.name}")


def create_scaling_plot(all_data, output_dir):
    """
    Create O-width scaling plots showing how each algorithm scales with output width.
    """
    for gpu, df in all_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        # Pick representative matrices (small, medium, large)
        matrices = sorted(df['matrix'].unique())
        if len(matrices) >= 4:
            # Select diverse matrices by NNZ
            nnz_per_matrix = df.groupby('matrix')['nnz'].first().sort_values()
            selected = [
                nnz_per_matrix.index[0],  # smallest
                nnz_per_matrix.index[len(nnz_per_matrix)//3],  # medium-small
                nnz_per_matrix.index[2*len(nnz_per_matrix)//3],  # medium-large
                nnz_per_matrix.index[-1]  # largest
            ]
        else:
            selected = matrices[:4]
        
        O_values = [64, 128, 256]
        
        for idx, matrix in enumerate(selected):
            ax = axes[idx]
            matrix_data = df[df['matrix'] == matrix]
            
            for algo in ALGO_ORDER:
                algo_data = matrix_data[matrix_data['algo'] == algo]
                if algo_data.empty:
                    continue
                
                perf = []
                for O in O_values:
                    gf = algo_data[algo_data['O'] == O]['gflops'].values
                    perf.append(gf[0] if len(gf) > 0 else 0)
                
                ax.plot(O_values, perf, 'o-', label=ALGO_LABELS[algo], 
                       color=COLORS[algo], linewidth=1.8, markersize=6)
            
            nnz = matrix_data['nnz'].iloc[0] if len(matrix_data) > 0 else 0
            ax.set_title(f'{matrix}\n(nnz={nnz:,})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Output Width (O)')
            ax.set_ylabel('GFLOP/s')
            ax.set_xticks(O_values)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=5, frameon=True, edgecolor='#CCCCCC')
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.98])
        
        out_path = output_dir / f'{gpu.lower()}_scaling_O.png'
        plt.savefig(out_path, format='png')
        plt.close()
        print(f"    {out_path.name}")


def create_nnz_performance_scatter(all_data, output_dir):
    """
    Create scatter plot showing performance vs matrix NNZ for each algorithm.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    
    for idx, (gpu, df) in enumerate(all_data.items()):
        ax = axes[idx]
        
        # Only show O=128 for clarity
        subset = df[df['O'] == 128]
        
        for algo in ALGO_ORDER:
            algo_data = subset[subset['algo'] == algo]
            if algo_data.empty:
                continue
            
            ax.scatter(algo_data['nnz'] / 1e6, algo_data['gflops'], 
                      label=ALGO_LABELS[algo] if idx == 1 else '',
                      color=COLORS[algo], s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Matrix NNZ (millions)')
        if idx == 0:
            ax.set_ylabel('GFLOP/s (O=128)')
        ax.set_title(f'{gpu}', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=5, frameon=True, edgecolor='#CCCCCC')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    out_path = output_dir / 'summary_nnz_vs_gflops.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_gpu_comparison(all_data, output_dir):
    """
    Create cross-GPU comparison for VCSR-Seg performance.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    
    matrices = sorted(list(all_data.values())[0]['matrix'].unique())
    gpus = list(all_data.keys())
    n_matrices = len(matrices)
    n_gpus = len(gpus)
    
    # GPU colors
    gpu_colors = {'A100': '#1f77b4', 'H100': '#ff7f0e', 'H200': '#2ca02c'}
    
    x = np.arange(n_matrices)
    width = 0.25
    offsets = np.linspace(-(n_gpus-1)*width/2, (n_gpus-1)*width/2, n_gpus)
    
    for i, gpu in enumerate(gpus):
        df = all_data[gpu]
        # Use O=128 for comparison
        vcsr_data = df[(df['algo'] == 'vcsr_seg') & (df['O'] == 128)]
        
        values = []
        for m in matrices:
            v = vcsr_data[vcsr_data['matrix'] == m]['gflops'].values
            values.append(v[0] if len(v) > 0 else 0)
        
        ax.bar(x + offsets[i], values, width, label=gpu, 
               color=gpu_colors.get(gpu, '#333333'), edgecolor='white', linewidth=0.3)
    
    ax.set_ylabel('VCSR-Seg GFLOP/s (O=128)', fontweight='medium')
    ax.set_xlabel('Matrix', fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels([m[:10] + '..' if len(m) > 10 else m for m in matrices], 
                      rotation=30, ha='right', fontsize=8)
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper right', frameon=True, edgecolor='#CCCCCC')
    
    plt.tight_layout()
    out_path = output_dir / 'summary_gpu_comparison.png'
    plt.savefig(out_path, format='png')
    plt.close()
    print(f"    {out_path.name}")


def create_all_algo_comparison(all_data, output_dir):
    """
    Create comprehensive bar chart comparing all algorithms across all matrices (O=128).
    """
    for gpu, df in all_data.items():
        subset = df[df['O'] == 128]
        matrices = sorted(subset['matrix'].unique())

        fig, ax = plt.subplots(figsize=(12, 5))

        n_matrices = len(matrices)
        n_algos = len(ALGO_ORDER)

        x = np.arange(n_matrices)
        width = 0.14
        offsets = np.linspace(
            -(n_algos - 1) * width / 2,
            (n_algos - 1) * width / 2,
            n_algos
        )

        for i, algo in enumerate(ALGO_ORDER):
            algo_data = subset[subset['algo'] == algo]
            values = []
            for m in matrices:
                v = algo_data[algo_data['matrix'] == m]['gflops'].values
                values.append(v[0] if len(v) > 0 else 0)

            bars = ax.bar(
                x + offsets[i],
                values,
                width,
                label=ALGO_LABELS[algo],
                color=COLORS[algo],
                edgecolor='white',
                linewidth=0.3,
                zorder=3
            )

            # Emphasize VCSR-Seg
            if algo == 'vcsr_seg':
                for bar in bars:
                    bar.set_edgecolor('#8B0000')
                    bar.set_linewidth(0.8)

        # Axis labels
        ax.set_ylabel('Throughput (GFLOP/s)', fontweight='medium')
        ax.set_xlabel('Matrix', fontweight='medium')

        # GPU label
        ax.text(
            0.02, 0.98, f'{gpu} | O=128',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            ha='left',
            va='top',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='gray',
                alpha=0.8
            )
        )

        # X ticks
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m[:11] + '..' if len(m) > 11 else m for m in matrices],
            rotation=30,
            ha='right',
            fontsize=8
        )

        # Styling
        ax.set_ylim(0, subset['gflops'].max() * 1.12)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),
            ncol=5,
            frameon=True,
            fancybox=False,
            edgecolor='#CCCCCC',
            facecolor='white'
        )

        plt.tight_layout()
        out_path = output_dir / f'{gpu.lower()}_all_algo_O128.png'
        plt.savefig(out_path, format='png')
        plt.close()
        print(f"    {out_path.name}")
