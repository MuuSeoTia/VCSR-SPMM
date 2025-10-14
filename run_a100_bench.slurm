#!/bin/bash
#SBATCH -p sharing
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH -J spmm_a100_bench
#SBATCH -o reports/a100/job_%j.out

set -euo pipefail

# Common parameters
SEGW="${SEGW:-4}"
BUNDLE="${BUNDLE:-32}"
TILEK="${TILEK:-128}"
REPEAT="${REPEAT:-10}"

DATA_DIR="$HOME/projects/sparse-vs-dense-matmul/data"
OUTDIR="reports/a100"
mkdir -p "$OUTDIR"

module load cuda cmake || true

# Build binary once
cmake -S . -B build_a100 -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=80
cmake --build build_a100 -j

BIN=./build_a100/spmm_vcsr

# Prepare timestamped CSV
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUTDIR/bench_${STAMP}.csv"
echo "gpu,mtx,O,algo,ms,gflop_s,segw,bundle,tileK,repeat" > "$CSV"

# Iterate over all .mtx files except ones with 'cant' in the path
mapfile -t MATRICES < <(find "$DATA_DIR" -type f -name "*.mtx" ! -path "*/cant/*")

for MATRIX in "${MATRICES[@]}"; do
  echo "[INFO] Using matrix: $MATRIX"
  ls -lh "$MATRIX"

  for O in 8 32 64 128 256 512; do
    for A in cusparse csr vcsr; do
      LOG="$OUTDIR/$(basename "$MATRIX" .mtx)_O${O}_${A}_segw${SEGW}_b${BUNDLE}_tk${TILEK}.txt"
      timeout 120s "$BIN" --mtx "$MATRIX" --O "$O" --segw "$SEGW" --bundle "$BUNDLE" --tileK "$TILEK" --repeat "$REPEAT" --algo "$A" \
        | tee "$LOG" | tail -n1 > /tmp/last.$$
      MS=$(awk '{for(i=1;i<=NF;i++) if($i=="ms") print $(i-1)}' /tmp/last.$$)
      GF=$(awk '{for(i=1;i<=NF;i++) if($i=="GFLOP/s") print $(i-1)}' /tmp/last.$$)
      if [[ -n "${MS:-}" && -n "${GF:-}" ]]; then
        echo "A100,$MATRIX,$O,$A,$MS,$GF,$SEGW,$BUNDLE,$TILEK,$REPEAT" >> "$CSV"
      else
        echo "[WARN] No summary for O=$O algo=$A; see $LOG"
      fi
    done
  done
done

rm -f /tmp/last.$$
echo "[OK] CSV: $CSV"

# Optional profiling (runs once on the last matrix)
if command -v nsys >/dev/null 2>&1; then
  nsys profile -o "$OUTDIR/last_O128_vcsr" \
    --trace=cuda,nvtx,osrt --sample=none \
    "$BIN" --mtx "$MATRIX" --O 128 --segw "$SEGW" --bundle "$BUNDLE" --tileK "$TILEK" --repeat 5 --algo vcsr
  nsys stats --report gpukernsum --format csv \
    "$OUTDIR/last_O128_vcsr.qdrep" \
    > "$OUTDIR/last_O128_vcsr_gpukernsum.csv"
fi
