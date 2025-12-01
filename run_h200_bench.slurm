#!/bin/bash
#SBATCH -p sharing
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH -J spmm_h200_bench
#SBATCH -o reports/h200/job_%j.out

set -euo pipefail

SEGW="${SEGW:-128}"
BUNDLE="${BUNDLE:-32}"
TILEK="${TILEK:-128}"
REPEAT="${REPEAT:-10}"

DATA_DIR="${DATA_DIR:-$HOME/projects/sparse-vs-dense-matmul/data}"
OUTDIR="reports/h200"
mkdir -p "$OUTDIR"

module load cuda cmake || true

cmake -S . -B build_h200 -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=90
cmake --build build_h200 -j
BIN=./build_h200/spmm_vcsr

STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUTDIR/bench_${STAMP}.csv"
echo "gpu,mtx,O,algo,ms,gflop_s,segw,bundle,tileK,repeat" > "$CSV"

mapfile -t MATRICES < <(
  find "$DATA_DIR" -type f -name "*.mtx" \
    ! -ipath "*/cant/*" ! -iname "*cant*" | sort
)

for MATRIX in "${MATRICES[@]}"; do
  echo "[INFO] Using matrix: $MATRIX"
  ls -lh "$MATRIX"

  for O in 64 128 256 512; do
    for A in cusparse csr vcsr ellr; do
      LOG="$OUTDIR/$(basename "$MATRIX" .mtx)_O${O}_${A}_segw${SEGW}_b${BUNDLE}_tk${TILEK}.txt"

      echo "[RUN] $A O=$O  (timeout 60s)"
      ( timeout 60s bash -c "
          $BIN --mtx \"$MATRIX\" --O \"$O\" \
               --segw \"$SEGW\" --bundle \"$BUNDLE\" \
               --tileK \"$TILEK\" --repeat \"$REPEAT\" \
               --algo \"$A\"
        " ) | tee "$LOG" | tail -n1 > /tmp/last.$$
      exitcode=${PIPESTATUS[0]}

      if [[ $exitcode -eq 124 ]]; then
        echo "[TIMEOUT] $MATRIX  O=$O  algo=$A stalled >60s → skipping" | tee -a "$OUTDIR/timeouts.log"
        continue
      elif [[ $exitcode -ne 0 ]]; then
        echo "[ERROR] $MATRIX  O=$O  algo=$A exited with code $exitcode" | tee -a "$OUTDIR/errors.log"
        continue
      fi

      MS=$(awk '{for(i=1;i<=NF;i++) if($i=="ms") print $(i-1)}' /tmp/last.$$)
      GF=$(awk '{for(i=1;i<=NF;i++) if($i=="GFLOP/s") print $(i-1)}' /tmp/last.$$)
      if [[ -n "${MS:-}" && -n "${GF:-}" ]]; then
        echo "H200,$MATRIX,$O,$A,$MS,$GF,$SEGW,$BUNDLE,$TILEK,$REPEAT" >> "$CSV"
      else
        echo "[WARN] No summary for $MATRIX O=$O algo=$A; see $LOG" | tee -a "$OUTDIR/warnings.log"
      fi
    done
  done
done

rm -f /tmp/last.$$
echo "[OK] CSV: $CSV"

if command -v nsys >/dev/null 2>&1; then
  nsys profile -o "$OUTDIR/last_O128_vcsr" \
    --trace=cuda,nvtx,osrt --sample=none \
    "$BIN" --mtx "$MATRIX" --O 128 --segw "$SEGW" --bundle "$BUNDLE" \
           --tileK "$TILEK" --repeat 5 --algo vcsr
  nsys stats --report gpukernsum --format csv \
    "$OUTDIR/last_O128_vcsr.qdrep" \
    > "$OUTDIR/last_O128_vcsr_gpukernsum.csv"
fi
