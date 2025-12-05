#!/bin/bash
#SBATCH -p sharing
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH -J spmm_a100_bench
#SBATCH -o reports/a100/job_%j.out

set -euo pipefail

########################
# Parameters
########################

SEGW="${SEGW:-128}"
BUNDLE="${BUNDLE:-32}"
TILEK="${TILEK:-128}"
REPEAT="${REPEAT:-10}"

GROUP="${GROUP:-0}"   # default group=0 if none provided
GROUP_SIZE="${GROUP_SIZE:-6}"  # number of matrices per group

DATA_DIR="$HOME/projects/sparse-vs-dense-matmul/data"
OUTDIR="reports/a100"
mkdir -p "$OUTDIR"

module load cuda cmake || true

########################
# Build
########################

cmake -S . -B build_a100 -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=80
cmake --build build_a100 -j
BIN=./build_a100/spmm_vcsr

########################
# Collect all matrices
########################

mapfile -t MATRICES < <(
  find "$DATA_DIR" -type f -name "*.mtx" \
  ! -ipath "*/cant/*" ! -iname "*cant*" | sort
)

NUM_MTX=${#MATRICES[@]}


START=$(( GROUP * GROUP_SIZE ))
END=$(( START + GROUP_SIZE ))

echo "[INFO] Running GROUP=$GROUP  (START=$START END=$END)"
echo "[INFO] Total matrices = $NUM_MTX"


STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$OUTDIR/bench_group${GROUP}_${STAMP}.csv"
echo "gpu,mtx,O,algo,ms,gflop_s,segw,bundle,tileK,repeat" > "$CSV"


for ((i=START; i<END && i<NUM_MTX; i++)); do
  MATRIX="${MATRICES[$i]}"
  echo "[INFO] Using matrix: $MATRIX"

  for O in 64 128 256 512; do
    for A in cusparse csr vcsr aspt fast; do

      LOG="$OUTDIR/$(basename "$MATRIX" .mtx)_G${GROUP}_O${O}_${A}.log"

      echo "[RUN] G${GROUP}: $A O=$O"

      ( timeout 55s bash -c "
            $BIN --mtx \"$MATRIX\" --O \"$O\" \
                 --segw \"$SEGW\" --bundle \"$BUNDLE\" \
                 --tileK \"$TILEK\" --repeat \"$REPEAT\" \
                 --algo \"$A\"
      " ) | tee "$LOG" | tail -n1 > /tmp/last.$$

      EXITCODE=${PIPESTATUS[0]}

      if [[ $EXITCODE -eq 124 ]]; then
        echo "[TIMEOUT] $MATRIX O=$O algo=$A" >> "$OUTDIR/timeouts.log"
        continue
      fi

      MS=$(awk '{for(i=1;i<=NF;i++) if($i=="ms") print $(i-1)}' /tmp/last.$$)
      GF=$(awk '{for(i=1;i<=NF;i++) if($i=="GFLOP/s") print $(i-1)}' /tmp/last.$$)

      if [[ -n "$MS" && -n "$GF" ]]; then
        echo "A100,$MATRIX,$O,$A,$MS,$GF,$SEGW,$BUNDLE,$TILEK,$REPEAT" >> "$CSV"
      fi

    done
  done
done

rm -f /tmp/last.$$
echo "[DONE] Results saved to: $CSV"
