**VCSR-style SpMM** (sparse A \* dense B = dense C) on **A100/H100** :
- CSR baseline (naive GPU+)
- cuSPARSE SpMM (vendor baseline)
- **VCSR SpMM (segmented streaming, Case 2)**


Key ideas implemented:
- **Column segmentation** of A into width `segw` (default 4) to localize B row access per segment
- **Row semi-reordering** by nnz-per-segment and **bundle packing** (column-major by depth, bundle size = warp size 32 by default)
- **Streaming over A's columns with C stationary**; tiles of B (sgw × tileK) in shared memory, reused across depth iterations


### Build
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DSM_ARCH=80 .. # 80=A100, 90=H100
cmake --build . -j
```


### Run
```bash
./spmm_vcsr --mtx /path/to/matrix.mtx --O 128 --segw 4 --bundle 32 --tileK 64 --repeat 10 --algo all
```


Arguments:
- `--mtx <file>`: Matrix Market file (coordinate). If symmetric, ensure full matrix is present or pre-expand.
- `--O <cols>`: number of RHS columns.
- `--segw <w>`: column-segmentation width (suggest 4 or 8). Smaller improves B locality; too small increases groups.
- `--bundle <b>`: bundle size, usually 32 (warp). Maps 1 thread with 1 row within a group.
- `--tileK <t>`: C/B column tile for shared memory (e.g., 64/128). `smem = segw*tileK*sizeof(float)`
- `--repeat <r>`: repetitions for timing.
- `--algo <all|csr|cusparse|vcsr>`: which kernels i need to run


### Output shape
- Average time (ms) and **GFLOP/s = 2*nnz*O / time**.
- L2 norm of differences vs cuSPARSE for quick correctness sanity score based on sparsity benchmarking


### Profiling (Nsight Compute)
```bash
scripts/profile_ncu.sh ./spmm_vcsr --mtx ... --algo vcsr
```
The script collects dram throughput, global load/store sectors, sm efficiency, and L1/L2 hitrates.


### Notes
- Precision: FP32 NOTE: switch to TF32/BF16 with cuSPARSE easily and the custom kernel here remains FP32 for clarity.
- THIS IS FIRST ITERATION WITH BASELINE **VCSR**: extend to **VCSR-INTRLV** mapping, add **VCSR-MEM** packing soon