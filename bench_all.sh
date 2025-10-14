#!/bin/bash
# Launch a batch job for every .mtx file under data/

set -euo pipefail

DATA_DIR="$HOME/projects/sparse-vs-dense-matmul/data"
JOB_SCRIPT="$HOME/projects/sparse-vs-dense-matmul/scripts/bench.sh"

# Find all .mtx files recursively
mapfile -t MATRICES < <(find "$DATA_DIR" -type f -name "*.mtx")

for mtx in "${MATRICES[@]}"; do
    job_name=$(basename "$mtx" .mtx)
    echo "[SUBMIT] $job_name"
    sbatch --job-name="spmm_${job_name}" \
           --export=MATRIX="$mtx" \
           "$JOB_SCRIPT"
done
