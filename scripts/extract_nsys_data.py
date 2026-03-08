#!/usr/bin/env python3
"""
Extract Nsight Systems profiling data for LaTeX tables.
Certified metrics for publication-quality reporting.
"""

import sqlite3
import os
import csv
from pathlib import Path

# Matrix characteristics from SuiteSparse
MATRICES = {
    'ASIC_320k': {'M': 321821, 'N': 321821, 'NNZ': 2635364},
    'boyd2': {'M': 466316, 'N': 466316, 'NNZ': 890091},
    'cage15': {'M': 5154859, 'N': 5154859, 'NNZ': 99199551},
    'ca-GrQc': {'M': 5242, 'N': 5242, 'NNZ': 14496},
    'Flan_1565': {'M': 1564794, 'N': 1564794, 'NNZ': 59485419},
    'Hook_1498': {'M': 1498023, 'N': 1498023, 'NNZ': 31207734},
    'kron_g500-logn21': {'M': 2097152, 'N': 2097152, 'NNZ': 91042010},
    'neos3': {'M': 512209, 'N': 518832, 'NNZ': 2055024},
    'pltexpa': {'M': 26894, 'N': 70364, 'NNZ': 143059},
    'StocF-1465': {'M': 1465137, 'N': 1465137, 'NNZ': 11235263},
    'cant': {'M': 62451, 'N': 62451, 'NNZ': 4007383},
}

def extract_kernel_stats(sqlite_path):
    """Extract kernel execution statistics from nsys SQLite export."""
    if not os.path.exists(sqlite_path):
        return None
    
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    results = {}
    
    try:
        # Get kernel execution times - join with StringIds to get actual kernel names
        cur.execute("""
            SELECT 
                s.value as kernel_name,
                COUNT(*) as invocations,
                AVG(k.end - k.start) / 1e6 as avg_ms,
                MIN(k.end - k.start) / 1e6 as min_ms,
                MAX(k.end - k.start) / 1e6 as max_ms,
                SUM(k.end - k.start) / 1e6 as total_ms,
                AVG(k.registersPerThread) as avg_regs,
                AVG(k.gridX * k.gridY * k.gridZ) as avg_grid_size,
                AVG(k.blockX * k.blockY * k.blockZ) as avg_block_size,
                AVG(k.staticSharedMemory + k.dynamicSharedMemory) as avg_shared_mem
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            GROUP BY s.value
            ORDER BY total_ms DESC
        """)
        
        results['kernels'] = []
        for row in cur.fetchall():
            results['kernels'].append({
                'name': row[0],
                'invocations': row[1],
                'avg_ms': row[2],
                'min_ms': row[3],
                'max_ms': row[4],
                'total_ms': row[5],
                'avg_regs': row[6],
                'avg_grid_size': row[7],
                'avg_block_size': row[8],
                'avg_shared_mem': row[9]
            })
        
        # Get memory transfer stats
        cur.execute("""
            SELECT 
                copyKind,
                COUNT(*) as count,
                SUM(bytes) / 1e9 as total_gb,
                AVG(end - start) / 1e6 as avg_ms
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
        """)
        
        results['memcpy'] = []
        for row in cur.fetchall():
            results['memcpy'].append({
                'kind': row[0],
                'count': row[1],
                'total_gb': row[2],
                'avg_ms': row[3]
            })
        
        # Get GPU info
        cur.execute("SELECT * FROM TARGET_INFO_GPU LIMIT 1")
        gpu_info = cur.fetchone()
        if gpu_info:
            results['gpu'] = gpu_info
            
    except Exception as e:
        print(f"Error querying {sqlite_path}: {e}")
    
    conn.close()
    return results

def generate_matrix_table_latex():
    """Generate LaTeX table for matrix characteristics."""
    latex = r"""\begin{table}[t]
\centering
\caption{Characteristics of the SuiteSparse matrices used in our evaluation.}
\label{tab:matrix-characteristics}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccc}
\hline
\textbf{Matrix} & \textbf{Dimensions ($M \times N$)} & \textbf{NNZ} & \textbf{Avg. NNZ / Row} \\
\hline
"""
    
    for name, props in sorted(MATRICES.items()):
        M, N, NNZ = props['M'], props['N'], props['NNZ']
        avg_nnz = NNZ / M
        
        # Format with thousands separators
        M_fmt = f"{M:,}".replace(',', '{,}')
        N_fmt = f"{N:,}".replace(',', '{,}')
        NNZ_fmt = f"{NNZ:,}".replace(',', '{,}')
        
        latex += f"{name} & ${M_fmt} \\times {N_fmt}$ & {NNZ_fmt} & {avg_nnz:.2f} \\\\\n"
    
    latex += r"""\hline
\end{tabular}
}
\end{table}
"""
    return latex

def generate_kernel_profiling_latex(kernel_data):
    """Generate LaTeX table for kernel execution times."""
    latex = r"""\begin{table}[t]
\centering
\caption{VCSR-Seg kernel execution times (A100 GPU, Nsight Systems).}
\label{tab:kernel-profiling}
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
\textbf{Matrix} & \textbf{K} & \textbf{Invocations} & \textbf{Avg (ms)} & \textbf{Min (ms)} & \textbf{Max (ms)} & \textbf{Regs} & \textbf{SMEM (B)} \\
\midrule
"""
    
    for entry in kernel_data:
        latex += f"{entry['matrix']} & {entry['K']} & {entry['invocations']} & {entry['avg_ms']:.3f} & {entry['min_ms']:.3f} & {entry['max_ms']:.3f} & {entry.get('avg_regs', 0):.0f} & {entry.get('avg_shared_mem', 0):.0f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{Kernel execution time measured via NVIDIA Nsight Systems 2024.6.2. Times exclude host-side setup.\\Regs = registers per thread, SMEM = shared memory per block.}
\end{table}
"""
    return latex


def generate_kernel_comparison_latex(kernel_data):
    """Generate LaTeX table comparing VCSR-Baseline vs VCSR-Seg."""
    latex = r"""\begin{table}[t]
\centering
\caption{Kernel execution time comparison: VCSR-Baseline vs VCSR-Seg (A100 GPU).}
\label{tab:kernel-comparison}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Matrix} & \textbf{K} & \multicolumn{2}{c}{\textbf{VCSR-Baseline}} & \multicolumn{2}{c}{\textbf{VCSR-Seg}} & \textbf{Speedup} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & \textbf{Time (ms)} & \textbf{GFLOP/s} & \textbf{Time (ms)} & \textbf{GFLOP/s} & \\
\midrule
"""
    
    # Group by matrix and K
    grouped = {}
    for entry in kernel_data:
        key = (entry['matrix'], entry['K'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][entry['algo']] = entry
    
    for (matrix, K), algos in sorted(grouped.items()):
        baseline = algos.get('vcsr_baseline', {})
        seg = algos.get('vcsr_seg', {})
        
        if baseline and seg:
            baseline_ms = baseline.get('avg_ms', 0)
            seg_ms = seg.get('avg_ms', 0)
            speedup = baseline_ms / seg_ms if seg_ms > 0 else 0
            
            # Calculate GFLOP/s from matrix characteristics
            nnz = MATRICES.get(matrix, {}).get('NNZ', 0)
            gflops_baseline = (2 * nnz * K) / (baseline_ms * 1e6) if baseline_ms > 0 else 0
            gflops_seg = (2 * nnz * K) / (seg_ms * 1e6) if seg_ms > 0 else 0
            
            latex += f"{matrix} & {K} & {baseline_ms:.3f} & {gflops_baseline:.1f} & {seg_ms:.3f} & {gflops_seg:.1f} & {speedup:.2f}$\\times$ \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\vspace{1mm}
\footnotesize{Times measured via NVIDIA Nsight Systems 2024.6.2. Speedup = Baseline Time / Seg Time.}
\end{table}
"""
    return latex

def main():
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / 'reports' / 'a100'
    
    print("=" * 60)
    print("MATRIX CHARACTERISTICS TABLE")
    print("=" * 60)
    print(generate_matrix_table_latex())
    
    print("\n" + "=" * 60)
    print("NSYS KERNEL PROFILING DATA")
    print("=" * 60)
    
    # Find and process SQLite files
    kernel_data = []
    for sqlite_file in reports_dir.glob('*.sqlite'):
        print(f"\nProcessing: {sqlite_file.name}")
        
        # Parse filename: matrix_O{K}_algo.sqlite
        parts = sqlite_file.stem.split('_')
        if len(parts) >= 2:
            matrix = parts[0]
            K = int(parts[1].replace('O', '')) if parts[1].startswith('O') else 0
            algo = parts[2] if len(parts) > 2 else 'unknown'
        
        stats = extract_kernel_stats(str(sqlite_file))
        if stats and 'kernels' in stats:
            for kernel in stats['kernels']:
                kname = str(kernel['name'])[:50] if kernel['name'] else 'unknown'
                print(f"  Kernel: {kname}...")
                print(f"    Invocations: {kernel['invocations']}")
                print(f"    Avg: {kernel['avg_ms']:.3f} ms")
                print(f"    Min: {kernel['min_ms']:.3f} ms") 
                print(f"    Max: {kernel['max_ms']:.3f} ms")
                print(f"    Total: {kernel['total_ms']:.3f} ms")
                print(f"    Registers/Thread: {kernel['avg_regs']:.0f}")
                print(f"    Grid Size: {kernel['avg_grid_size']:.0f}")
                print(f"    Block Size: {kernel['avg_block_size']:.0f}")
                print(f"    Shared Mem: {kernel['avg_shared_mem']:.0f} bytes")
                
                kernel_data.append({
                    'matrix': matrix,
                    'K': K,
                    'algo': algo,
                    'kernel_name': kname,
                    'invocations': kernel['invocations'],
                    'avg_ms': kernel['avg_ms'],
                    'min_ms': kernel['min_ms'],
                    'max_ms': kernel['max_ms'],
                    'total_ms': kernel['total_ms'],
                    'avg_regs': kernel['avg_regs'],
                    'avg_grid_size': kernel['avg_grid_size'],
                    'avg_block_size': kernel['avg_block_size'],
                    'avg_shared_mem': kernel['avg_shared_mem']
                })
    
    # Save to CSV for further processing
    csv_path = project_root / 'reports' / 'nsys_kernel_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['matrix', 'K', 'algo', 'kernel_name', 
                                               'invocations', 'avg_ms', 'min_ms', 'max_ms', 'total_ms',
                                               'avg_regs', 'avg_grid_size', 'avg_block_size', 'avg_shared_mem'])
        writer.writeheader()
        writer.writerows(kernel_data)
    print(f"\nSaved kernel summary to: {csv_path}")
    
    if kernel_data:
        print("\n" + "=" * 60)
        print("LATEX KERNEL TABLE")
        print("=" * 60)
        print(generate_kernel_profiling_latex(kernel_data))

if __name__ == '__main__':
    main()

