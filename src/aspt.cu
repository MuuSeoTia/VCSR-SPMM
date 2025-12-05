// aspt.cu
// ASpT-style CSR SpMM baseline using the "ssparse" warp-shuffle kernel,
// adapted to your SpMM harness (float, row-major B and C).
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstdio>
#include <cuda.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t err__ = (x);                                                  \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err__));                                     \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#endif

// ---- kernel config (matching the original code) ----
constexpr int MFACTOR      = 32;          // warp size used by the code
constexpr int LOG_MFACTOR  = 5;
constexpr int SBSIZE       = 128;         // threads per block
constexpr int SBF          = SBSIZE / 32; // rows per block in X (4)

// ASpT-style CSR SpMM kernel (single-precision)
//
// C(MxO) = A(MxN, CSR) * B(NxO)
// Layout:
//   csr_v : row pointers [0..M]
//   csr_e : column indices [0..nnz)
//   csr_ev: values [0..nnz)
//   B     : row-major, size N x O  (vin)
//   C     : row-major, size M x O  (vout)
//
// Assumes O is a multiple of 2*MFACTOR = 64 (true for your O=64,128,256,512).
__global__
void aspt_spmm_kernel(const int  M,
                      const int  sc,          // #columns (O)
                      const int* __restrict__ csr_v,
                      const int* __restrict__ csr_e,
                      const float* __restrict__ csr_ev,
                      const float* __restrict__ vin,
                      float* __restrict__ vout)
{
  // Each warp processes one row 'idx'
  int row_group = blockIdx.x * SBF;       // 4 rows per block in X
  int warp_id   = threadIdx.x >> 5;       // 0..3
  int idx       = row_group + warp_id;    // row index
  if (idx >= M) return;

  int lane   = threadIdx.x & (MFACTOR - 1); // 0..31
  // blockIdx.z tiles the column dimension (O) in chunks of 2*MFACTOR
  int offset  = (blockIdx.z << (LOG_MFACTOR + 1)) + lane;
  int offset2 = offset + MFACTOR;

  // Per-row accumulation in two 32-wide column segments
  float r  = 0.0f;
  float r2 = 0.0f;

  int loc1 = csr_v[idx];
  int loc2 = csr_v[idx + 1];

  int buf; 
  float buf2;
  int interm3 = loc1 + (((loc2 - loc1) >> 1) << 1); // even-aligned

  int jj = 0;
  int l;

  // Short rows: pre-load one (col,val) per lane
  if (loc2 - loc1 < 32) {
    if (loc1 + lane < loc2) {
      buf  = csr_e[loc1 + lane];
      buf2 = csr_ev[loc1 + lane];
    } else {
      buf  = 0;
      buf2 = 0.0f;
    }

    for (l = loc1; l < interm3; l += 2) {
      float v1 = __shfl_sync(0xffffffff, buf2, jj,     MFACTOR);
      float v2 = __shfl_sync(0xffffffff, buf2, jj + 1, MFACTOR);
      int   i1 = __shfl_sync(0xffffffff, buf,  jj,     MFACTOR) * sc;
      int   i2 = __shfl_sync(0xffffffff, buf,  jj + 1, MFACTOR) * sc;

      r  += v1 * vin[i1 + offset];
      r2 += v1 * vin[i1 + offset2];
      r  += v2 * vin[i2 + offset];
      r2 += v2 * vin[i2 + offset2];

      jj += 2;
    }

    if (interm3 < loc2) {
      float v1 = __shfl_sync(0xffffffff, buf2, jj, MFACTOR);
      int   i1 = __shfl_sync(0xffffffff, buf,  jj, MFACTOR) * sc;
      r  += v1 * vin[i1 + offset];
      r2 += v1 * vin[i1 + offset2];
    }
  }
  // Longer rows: ring-buffer style reuse of loaded (col,val)
  else {
    for (l = loc1; l < interm3; l += 2) {
      if (jj == 0) {
        int idx_e = l + lane;
        buf  = csr_e[idx_e];
        buf2 = csr_ev[idx_e];
      }

      float v1 = __shfl_sync(0xffffffff, buf2, jj,     MFACTOR);
      float v2 = __shfl_sync(0xffffffff, buf2, jj + 1, MFACTOR);
      int   i1 = __shfl_sync(0xffffffff, buf,  jj,     MFACTOR) * sc;
      int   i2 = __shfl_sync(0xffffffff, buf,  jj + 1, MFACTOR) * sc;

      r  += v1 * vin[i1 + offset];
      r2 += v1 * vin[i1 + offset2];
      r  += v2 * vin[i2 + offset];
      r2 += v2 * vin[i2 + offset2];

      jj = (jj + 2) & (MFACTOR - 1);
    }

    if (interm3 < loc2 && jj == 0) {
      int idx_e = l + lane;
      buf  = csr_e[idx_e];
      buf2 = csr_ev[idx_e];
    }

    if (interm3 < loc2) {
      float v1 = __shfl_sync(0xffffffff, buf2, jj, MFACTOR);
      int   i1 = __shfl_sync(0xffffffff, buf,  jj, MFACTOR) * sc;
      r  += v1 * vin[i1 + offset];
      r2 += v1 * vin[i1 + offset2];
    }
  }

  // Write result for this row + column tile
  vout[idx * sc + offset]  = r;
  vout[idx * sc + offset2] = r2;
}

// Host wrapper, matching your harness (driver-style stream)
//
// M: #rows of A (and C)
// d_rowptr: CSR rowptr (length M+1)
// d_col   : CSR col indices (length nnz)
// d_val   : CSR values (length nnz)
// dB      : B matrix, row-major, N x O
// O       : #columns of B/C (must be multiple of 64 for this kernel)
// dC      : C matrix, row-major, M x O
void run_aspt_spmm_gpu(const int   M,
                       const int*  d_rowptr,
                       const int*  d_col,
                       const float* d_val,
                       const float* dB,
                       int         O,
                       float*      dC,
                       CUstream    stream,   // <-- matches main.cpp
                       float&      ms)
{
  // Bridge driver stream -> runtime stream
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);

  if (O % (2 * MFACTOR) != 0) {
    fprintf(stderr,
            "[ASPT] O=%d is not a multiple of 64, this kernel expects O %% 64 == 0\n",
            O);
    std::exit(1);
  }

  const int rows_per_block = SBF; // 4 rows per block in X
  const int blocks_x       = (M + rows_per_block - 1) / rows_per_block;
  const int blocks_z       = (O + (2 * MFACTOR) - 1) / (2 * MFACTOR);

  dim3 block(SBSIZE, 1, 1);
  dim3 grid(blocks_x, 1, blocks_z);

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));

  CUDA_CHECK(cudaEventRecord(a, st));
  aspt_spmm_kernel<<<grid, block, 0, st>>>(
      M, O,
      d_rowptr, d_col, d_val,
      dB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(b, st));
  CUDA_CHECK(cudaEventSynchronize(b));
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
}
