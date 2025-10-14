#!/usr/bin/env bash
set -euo pipefail

BIN="$1"         # ./build_h100/spmm_vcsr or ./build_a100/spmm_vcsr
OUTCSV="$2"      # reports/h100/bench.csv or reports/a100/bench.csv
shift 2

# Remaining args are matrix files
MATS=("$@")

echo "gpu,sm,mtx,O,segw,bundle,tileK,algo,ms,gflops" > "$OUTCSV"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d ',')
SM=$("$BIN" --help 2>/dev/null | sed -n '1p' >/dev/null; echo "${BIN##*_}") # dummy insertion at runtime to extract sm from binary name

SEGW=${SEGW:-4}
BUNDLE=${BUNDLE:-32}
TILEK=${TILEK:-128}  #128 for H100, 64 for A100

for MTX in "${MATS[@]}"; do
  for O in 8 32 64 128 256 512; do
    for ALGO in cusparse csr vcsr; do
      LINE=$("$BIN" --mtx "$MTX" --O "$O" --segw "$SEGW" --bundle "$BUNDLE" --tileK "$TILEK" --repeat 10 --algo "$ALGO" \
             | tail -n1)
      MS=$(echo "$LINE" | awk '{print $(NF-3)}')        # grabs the number before "ms"
      GF=$(echo "$LINE" | awk '{print $(NF-1)}')        # grabs the number before "GFLOP/s"

      echo "$GPU,$SM,$MTX,$O,$SEGW,$BUNDLE,$TILEK,$ALGO,$MS,$GF" >> "$OUTCSV"
    done
  done
done
