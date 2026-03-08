// vcsr_naive.cu - Baseline VCSR SpMM (no segmentation)
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include "csr.hpp"

struct VCSRNaive {
    int M{0}, N{0};
    int bundle{32};
    std::vector<int> group_ptr;
    std::vector<int> group_depth;
    std::vector<int> group_rows;
    std::vector<int> col;
    std::vector<float> val;
};

VCSRNaive csr_to_vcsr_naive(const CSR &csr, int bundle) {
    VCSRNaive V;
    V.M = csr.M; V.N = csr.N; V.bundle = bundle;
    
    // Sort rows by nnz descending for load balancing
    std::vector<int> rows(csr.M);
    std::iota(rows.begin(), rows.end(), 0);
    std::sort(rows.begin(), rows.end(), [&](int a, int b) {
        return (csr.row_ptr[a+1] - csr.row_ptr[a]) > (csr.row_ptr[b+1] - csr.row_ptr[b]);
    });
    
    // Create groups of `bundle` rows
    for (size_t start = 0; start < rows.size(); start += bundle) {
        size_t end = std::min(start + (size_t)bundle, rows.size());
        int this_bundle = (int)(end - start);
        if (this_bundle == 0) break;
        
        // Find max depth (max nnz in this bundle)
        int depth = 0;
        for (size_t i = start; i < end; ++i) {
            int r = rows[i];
            depth = std::max(depth, csr.row_ptr[r+1] - csr.row_ptr[r]);
        }
        if (depth == 0) continue;
        
        int g_start = (int)V.val.size();
        V.group_ptr.push_back(g_start);
        V.group_depth.push_back(depth);
        
        // Store row IDs (pad with -1)
        for (int lane = 0; lane < bundle; ++lane) {
            int ridx = (int)start + lane;
            V.group_rows.push_back(ridx < (int)rows.size() ? rows[ridx] : -1);
        }
        
        // Pack by depth (column-major)
        for (int d = 0; d < depth; ++d) {
            for (int lane = 0; lane < bundle; ++lane) {
                int ridx = (int)start + lane;
                int row = (ridx < (int)rows.size()) ? rows[ridx] : -1;
                if (row < 0) {
                    V.col.push_back(-1);
                    V.val.push_back(0.f);
                } else {
                    int nnz = csr.row_ptr[row+1] - csr.row_ptr[row];
                    if (d < nnz) {
                        int p = csr.row_ptr[row] + d;
                        V.col.push_back(csr.col[p]);
                        V.val.push_back(csr.val[p]);
                    } else {
                        V.col.push_back(-1);
                        V.val.push_back(0.f);
                    }
                }
            }
        }
    }
    V.group_ptr.push_back((int)V.val.size());
    return V;
}

// Device data
static int* d_group_ptr = nullptr;
static int* d_group_depth = nullptr;
static int* d_group_rows = nullptr;
static int* d_col = nullptr;
static float* d_val = nullptr;

__global__ void vcsr_naive_kernel(
    int num_groups, int bundle, int O,
    const int* __restrict__ group_ptr,
    const int* __restrict__ group_depth,
    const int* __restrict__ group_rows,
    const int* __restrict__ col,
    const float* __restrict__ aval,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
    int gid = blockIdx.x;
    if (gid >= num_groups) return;
    
    int lane = threadIdx.x;
    if (lane >= bundle) return;
    
    int row = group_rows[gid * bundle + lane];
    if (row < 0) return;
    
    int base = group_ptr[gid];
    int depth = group_depth[gid];
    
    // Process all output columns
    for (int k = 0; k < O; ++k) {
        float acc = 0.0f;
        for (int d = 0; d < depth; ++d) {
            int idx = base + d * bundle + lane;
            int c = col[idx];
            if (c >= 0) {
                float a = aval[idx];
                float b = B[c * ldb + k];  // Random access to B
                acc += a * b;
            }
        }
        atomicAdd(&C[row * ldc + k], acc);
    }
}

void run_vcsr_naive_spmm_gpu(const VCSRNaive &V, const float* dB, int O,
                              float* dC, cudaStream_t st, float &ms) {
    int num_groups = (int)V.group_depth.size();
    if (num_groups == 0) { ms = 0; return; }
    
    // Upload if needed
    static bool uploaded = false;
    if (!uploaded) {
        cudaMalloc(&d_group_ptr, sizeof(int) * V.group_ptr.size());
        cudaMalloc(&d_group_depth, sizeof(int) * V.group_depth.size());
        cudaMalloc(&d_group_rows, sizeof(int) * V.group_rows.size());
        cudaMalloc(&d_col, sizeof(int) * V.col.size());
        cudaMalloc(&d_val, sizeof(float) * V.val.size());
        
        cudaMemcpy(d_group_ptr, V.group_ptr.data(), sizeof(int) * V.group_ptr.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_depth, V.group_depth.data(), sizeof(int) * V.group_depth.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_rows, V.group_rows.data(), sizeof(int) * V.group_rows.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, V.col.data(), sizeof(int) * V.col.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, V.val.data(), sizeof(float) * V.val.size(), cudaMemcpyHostToDevice);
        uploaded = true;
    }
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    
    dim3 grid(num_groups);
    dim3 block(V.bundle);
    
    cudaEventRecord(t0, st);
    vcsr_naive_kernel<<<grid, block, 0, st>>>(
        num_groups, V.bundle, O,
        d_group_ptr, d_group_depth, d_group_rows, d_col, d_val,
        dB, O, dC, O);
    cudaEventRecord(t1, st);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

void destroy_vcsr_naive_device(cudaStream_t) {
    if (d_group_ptr) cudaFree(d_group_ptr);
    if (d_group_depth) cudaFree(d_group_depth);
    if (d_group_rows) cudaFree(d_group_rows);
    if (d_col) cudaFree(d_col);
    if (d_val) cudaFree(d_val);
    d_group_ptr = d_group_depth = d_group_rows = d_col = nullptr;
    d_val = nullptr;
}


