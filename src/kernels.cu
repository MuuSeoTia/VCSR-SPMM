#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include "csr.hpp"
#include "vcsr.hpp"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__, cudaGetErrorString(err)); exit(1);} } while(0)
#endif

// ---------------- CSR SpMM (naive): each thread handles one row of A ----------------
__global__ void csr_spmm_kernel(const int M, const int N, const int O,
                                const int* __restrict__ rowptr,
                                const int* __restrict__ col,
                                const float* __restrict__ aval,
                                const float* __restrict__ B, int ldb,
                                float* __restrict__ C, int ldc){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row>=M) return;
  extern __shared__ float sm[]; // not used here
  // initialize output row accumulators in registers (stream over k)
  for(int k=0;k<O;++k){ C[row*ldc + k] = 0.f; }
  int s=rowptr[row], e=rowptr[row+1];
  for(int p=s;p<e;++p){
    int j = col[p]; float a = aval[p];
    const float* Bj = B + j*ldb;
    float* Crow = C + row*ldc;
    for(int k=0;k<O;++k){ Crow[k] += a * Bj[k]; }
  }
}

// VCSR SpMM (segmented streaming, B tiles in shared mem) 
// Grid: (num_groups, ceil_div(O, tileK)), Block: (bundle, 1)
__global__ void vcsr_spmm_kernel(
    const int O, const int tileK, const int bundle, const int segw,
    const int num_groups,
    const int* __restrict__ group_ptr,
    const int* __restrict__ group_depth,
    const int* __restrict__ group_seg_base,
    const int* __restrict__ group_rows,
    const int* __restrict__ lcol,
    const float* __restrict__ aval,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
  int g = blockIdx.x;
  if (g >= num_groups) return;

  const int k0   = blockIdx.y * tileK;
  const int lane = threadIdx.x;  // 0..bundle-1 (blockDim.x == bundle)

  // NOTE: Do NOT early-return before the barrier; compute a mask instead.
  const int row = (lane < bundle) ? group_rows[g * bundle + lane] : -1;
  const bool lane_active = (lane < bundle) && (row >= 0);

  const int base     = group_ptr[g];
  const int depth    = group_depth[g];
  const int seg_base = group_seg_base[g];

  extern __shared__ float smem[]; // size = segw * tileK

  // Cooperative load of B tile into shared memory.
  // Everyone participates; bounds-guard O so we never read past the last column.
  for (int srow = threadIdx.x; srow < segw; srow += blockDim.x) {
    const float* __restrict__ Bj = B + (seg_base + srow) * ldb + k0;
    float* __restrict__ S = smem + srow * tileK;
    #pragma unroll 1
    for (int kk = 0; kk < tileK; ++kk) {
      const int k = k0 + kk;
      // Guard O range; write 0 if out-of-range to keep consumers safe.
      S[kk] = (k < O) ? Bj[kk] : 0.0f;
    }
  }

  __syncthreads();  // all threads reach here (no early returns)

  // Depth-wise accumulation; mask inactive lanes; also guard lc range.
  #pragma unroll 1
  for (int d = 0; d < depth; ++d) {
    // Linear index in packed structure
    const int idx = base + d * bundle + lane;

    // Lanes without work simply skip compute but still iterate uniformly
    if (lane_active) {
      const int   lc = lcol[idx];
      const float a  = aval[idx];

      if (lc >= 0 && lc < segw) {
        float* __restrict__ Crow       = C + row * ldc + k0;
        const float* __restrict__ SB   = smem + lc * tileK;

        #pragma unroll 1
        for (int kk = 0; kk < tileK; ++kk) {
          const int k = k0 + kk;
          if (k < O) Crow[kk] += a * SB[kk];
        }
      }
    }
  }
}

// Host wrappers

void run_csr_spmm_gpu(const int M, const int* d_rowptr, const int* d_col, const float* d_val,
                        const float* dB, int O, float* dC, cudaStream_t st, float &ms){
  int threads = 256; int blocks = (M + threads - 1)/threads;
  size_t shmem=0;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st);
  csr_spmm_kernel<<<blocks, threads, shmem, st>>>(M, /*N=*/0, O,
    d_rowptr, d_col, d_val, dB, O, dC, O);
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(b, st); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b);
}

void run_vcsr_spmm_gpu(const VCSRSpMM &V, const float* dB, int O, int tileK, float* dC, cudaStream_t st, float &ms){
  
  const int num_groups = (int)V.group_depth.size();
  const dim3 grid(num_groups, (O + tileK - 1) / tileK);
  const dim3 block(V.bundle, 1);
  const size_t shmem = (size_t)V.segw * tileK * sizeof(float);

  // Allow large dynamic shared memory if needed
  int maxDynShmem = 0;
  cudaDeviceGetAttribute(&maxDynShmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if ((int)shmem <= maxDynShmem) {
    cudaFuncSetAttribute(vcsr_spmm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shmem);
  }

  // Copy packed arrays to device (use cudaMallocManaged for brevity)
  int *d_group_ptr,*d_group_depth,*d_group_seg_base,*d_group_rows,*d_lcol;
  float *d_val;
  CUDA_CHECK(cudaMallocAsync(&d_group_ptr, sizeof(int)*(num_groups+1), st));
  CUDA_CHECK(cudaMallocAsync(&d_group_depth, sizeof(int)*num_groups, st));
  CUDA_CHECK(cudaMallocAsync(&d_group_seg_base, sizeof(int)*num_groups, st));
  CUDA_CHECK(cudaMallocAsync(&d_group_rows, sizeof(int)*num_groups*V.bundle, st));
  CUDA_CHECK(cudaMallocAsync(&d_lcol, sizeof(int)*V.lcol.size(), st));
  CUDA_CHECK(cudaMallocAsync(&d_val, sizeof(float)*V.val.size(), st));

  CUDA_CHECK(cudaMemcpyAsync(d_group_ptr, V.group_ptr.data(), sizeof(int)*(num_groups+1), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d_group_depth, V.group_depth.data(), sizeof(int)*num_groups, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d_group_seg_base, V.group_seg_base.data(), sizeof(int)*num_groups, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d_group_rows, V.group_rows.data(), sizeof(int)*num_groups*V.bundle, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d_lcol, V.lcol.data(), sizeof(int)*V.lcol.size(), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(d_val, V.val.data(), sizeof(float)*V.val.size(), cudaMemcpyHostToDevice, st));

  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st);
  vcsr_spmm_kernel<<<grid, block, shmem, st>>>(
      O, tileK, V.bundle, V.segw, num_groups,
      d_group_ptr, d_group_depth, d_group_seg_base, d_group_rows,
      d_lcol, d_val,
      dB, O, dC, O);
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(b, st); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  cudaEventDestroy(a); cudaEventDestroy(b);

  cudaFreeAsync(d_group_ptr, st); cudaFreeAsync(d_group_depth, st);
  cudaFreeAsync(d_group_seg_base, st); cudaFreeAsync(d_group_rows, st);
  cudaFreeAsync(d_lcol, st); cudaFreeAsync(d_val, st);
}

// Simple cuSPARSE SpMM wrapper (CSR x Dense)
float run_cusparse_spmm(const int M, const int N, const int nnz,
                         const int* d_rowptr, const int* d_col, const float* d_val,
                         const float* dB, int O, float* dC){
  cusparseHandle_t h; cusparseCreate(&h);
  cudaStream_t st; cudaStreamCreate(&st); cusparseSetStream(h, st);

  // Descriptors
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  size_t bufferSize=0; void* dBuffer=nullptr;

  cusparseCreateCsr(&matA, M, N, nnz,
                    (void*)d_rowptr, (void*)d_col, (void*)d_val,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnMat(&matB, N, O, O, (void*)dB, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseCreateDnMat(&matC, M, O, O, (void*)dC, CUDA_R_32F, CUSPARSE_ORDER_ROW);

  float alpha=1.f, beta=0.f;
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseSpMM_bufferSize(h, opA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, matB, &beta, matC,
                          CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st);
  cusparseSpMM(h, opA, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, matA, matB, &beta, matC,
               CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
  cudaEventRecord(b, st); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b);

  cudaEventDestroy(a); cudaEventDestroy(b);
  cudaFree(dBuffer);
  cusparseDestroySpMat(matA); cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC);
  cusparseDestroy(h); cudaStreamDestroy(st);
  return ms;
}