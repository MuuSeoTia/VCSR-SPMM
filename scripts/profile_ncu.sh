#!/usr/bin/env bash
set -euo pipefail
APP="$1"; shift
ncu --set full --target-processes all \
  --metrics \
  sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  lts__t_sectors_srcunit_tex_op_read.sum,\
  lts__t_sectors_srcunit_tex_op_write.sum,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  "$APP" "$@"
