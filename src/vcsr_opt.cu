// vcsr_opt.cu - VCSR-Augmented: vectorized loads + register tiling
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "csr.hpp"

struct VCSROpt {
    int M{0}, N{0};
    int segw{16};
    int bundle{32};
    std::vector<int> group_ptr;
    std::vector<int> group_depth;
    std::vector<int> group_seg_base;
    std::vector<int> group_rows;
    std::vector<int> lcol;
    std::vector<float> val;
    int num_segments{0};
};

VCSROpt csr_to_vcsr_opt(const CSR &csr, int segw, int bundle) {
    VCSROpt V;
    V.M = csr.M; V.N = csr.N; V.segw = segw; V.bundle = bundle;
    int S = (csr.N + segw - 1) / segw;
    V.num_segments = S;
    
    struct Entry { int lcol; float v; };
    std::vector<std::unordered_map<int, std::vector<Entry>>> seg_entries(S);
    
    for (int r = 0; r < csr.M; ++r) {
        for (int p = csr.row_ptr[r]; p < csr.row_ptr[r+1]; ++p) {
            int c = csr.col[p];
            float v = csr.val[p];
            int sid = c / segw;
            int lc = c % segw;
            seg_entries[sid][r].push_back({lc, v});
        }
    }
    
    for (int sid = 0; sid < S; ++sid) {
        for (auto &kv : seg_entries[sid]) {
            std::sort(kv.second.begin(), kv.second.end(),
                      [](const Entry &a, const Entry &b) { return a.lcol < b.lcol; });
        }
    }
    
    for (int sid = 0; sid < S; ++sid) {
        std::vector<int> active_rows;
        for (auto &kv : seg_entries[sid]) {
            if (!kv.second.empty()) active_rows.push_back(kv.first);
        }
        
        std::sort(active_rows.begin(), active_rows.end(), [&](int a, int b) {
            return seg_entries[sid][a].size() > seg_entries[sid][b].size();
        });
        
        for (size_t start = 0; start < active_rows.size(); start += bundle) {
            size_t end = std::min(start + (size_t)bundle, active_rows.size());
            
            int depth = 0;
            for (size_t i = start; i < end; ++i) {
                depth = std::max(depth, (int)seg_entries[sid][active_rows[i]].size());
            }
            if (depth == 0) continue;
            
            int g_start = (int)V.val.size();
            V.group_ptr.push_back(g_start);
            V.group_depth.push_back(depth);
            V.group_seg_base.push_back(sid * segw);
            
            for (int lane = 0; lane < bundle; ++lane) {
                int ridx = (int)start + lane;
                V.group_rows.push_back(ridx < (int)active_rows.size() ? active_rows[ridx] : -1);
            }
            
            for (int d = 0; d < depth; ++d) {
                for (int lane = 0; lane < bundle; ++lane) {
                    int ridx = (int)start + lane;
                    if (ridx < (int)active_rows.size()) {
                        int row = active_rows[ridx];
                        auto &vec = seg_entries[sid][row];
                        if (d < (int)vec.size()) {
                            V.lcol.push_back(vec[d].lcol);
                            V.val.push_back(vec[d].v);
                        } else {
                            V.lcol.push_back(-1);
                            V.val.push_back(0.f);
                        }
                    } else {
                        V.lcol.push_back(-1);
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
namespace vcsr_opt_dev {
    static int* d_group_ptr = nullptr;
    static int* d_group_depth = nullptr;
    static int* d_group_seg_base = nullptr;
    static int* d_group_rows = nullptr;
    static int* d_lcol = nullptr;
    static float* d_val = nullptr;
    static bool uploaded = false;
}

// VCSR-Aug kernel: vectorized float4 loads + register tiling
__global__ void vcsr_opt_kernel(
    int O, int tileK, int bundle, int segw, int N,
    int num_groups,
    const int* __restrict__ group_ptr,
    const int* __restrict__ group_depth,
    const int* __restrict__ group_seg_base,
    const int* __restrict__ group_rows,
    const int* __restrict__ lcol,
    const float* __restrict__ aval,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
    extern __shared__ float smem[];
    
    int gid = blockIdx.x;
    int k_tile = blockIdx.y;
    if (gid >= num_groups) return;
    
    int lane = threadIdx.x;
    int row = group_rows[gid * bundle + lane];
    
    int seg_base = group_seg_base[gid];
    int base = group_ptr[gid];
    int depth = group_depth[gid];
    int k0 = k_tile * tileK;
    int k_end = min(k0 + tileK, O);
    int actual_tileK = k_end - k0;
    
    // Vectorized load of B tile using float4
    bool aligned = ((k0 & 3) == 0) && ((actual_tileK & 3) == 0);
    
    for (int srow = threadIdx.x; srow < segw; srow += blockDim.x) {
        int global_row_B = seg_base + srow;
        float* smem_row = smem + srow * tileK;
        
        if (global_row_B < N) {
            const float* B_row = B + global_row_B * ldb + k0;
            
            if (aligned) {
                // Vectorized float4 load
                int num_vec = actual_tileK >> 2;
                for (int v = 0; v < num_vec; ++v) {
                    float4 val4 = reinterpret_cast<const float4*>(B_row)[v];
                    smem_row[v*4 + 0] = val4.x;
                    smem_row[v*4 + 1] = val4.y;
                    smem_row[v*4 + 2] = val4.z;
                    smem_row[v*4 + 3] = val4.w;
                }
            } else {
                // Scalar fallback
                for (int kk = 0; kk < actual_tileK; ++kk) {
                    smem_row[kk] = B_row[kk];
                }
            }
        } else {
            for (int kk = 0; kk < actual_tileK; ++kk) {
                smem_row[kk] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    if (row < 0) return;
    
    // Register tiling: 16 accumulators
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    float acc8 = 0, acc9 = 0, acc10 = 0, acc11 = 0;
    float acc12 = 0, acc13 = 0, acc14 = 0, acc15 = 0;
    
    // Process in chunks of 16
    for (int kk_base = 0; kk_base < actual_tileK; kk_base += 16) {
        int chunk = min(16, actual_tileK - kk_base);
        
        // Reset accumulators
        acc0 = acc1 = acc2 = acc3 = 0;
        acc4 = acc5 = acc6 = acc7 = 0;
        acc8 = acc9 = acc10 = acc11 = 0;
        acc12 = acc13 = acc14 = acc15 = 0;
        
        for (int d = 0; d < depth; ++d) {
            int idx = base + d * bundle + lane;
            int lc = lcol[idx];
            if (lc >= 0 && lc < segw) {
                float a = aval[idx];
                float* SB = smem + lc * tileK + kk_base;
                
                // Fully unrolled accumulation
                if (chunk > 0)  acc0  += a * SB[0];
                if (chunk > 1)  acc1  += a * SB[1];
                if (chunk > 2)  acc2  += a * SB[2];
                if (chunk > 3)  acc3  += a * SB[3];
                if (chunk > 4)  acc4  += a * SB[4];
                if (chunk > 5)  acc5  += a * SB[5];
                if (chunk > 6)  acc6  += a * SB[6];
                if (chunk > 7)  acc7  += a * SB[7];
                if (chunk > 8)  acc8  += a * SB[8];
                if (chunk > 9)  acc9  += a * SB[9];
                if (chunk > 10) acc10 += a * SB[10];
                if (chunk > 11) acc11 += a * SB[11];
                if (chunk > 12) acc12 += a * SB[12];
                if (chunk > 13) acc13 += a * SB[13];
                if (chunk > 14) acc14 += a * SB[14];
                if (chunk > 15) acc15 += a * SB[15];
            }
        }
        
        // Write results
        float* C_row = C + row * ldc + k0 + kk_base;
        if (chunk > 0)  atomicAdd(&C_row[0],  acc0);
        if (chunk > 1)  atomicAdd(&C_row[1],  acc1);
        if (chunk > 2)  atomicAdd(&C_row[2],  acc2);
        if (chunk > 3)  atomicAdd(&C_row[3],  acc3);
        if (chunk > 4)  atomicAdd(&C_row[4],  acc4);
        if (chunk > 5)  atomicAdd(&C_row[5],  acc5);
        if (chunk > 6)  atomicAdd(&C_row[6],  acc6);
        if (chunk > 7)  atomicAdd(&C_row[7],  acc7);
        if (chunk > 8)  atomicAdd(&C_row[8],  acc8);
        if (chunk > 9)  atomicAdd(&C_row[9],  acc9);
        if (chunk > 10) atomicAdd(&C_row[10], acc10);
        if (chunk > 11) atomicAdd(&C_row[11], acc11);
        if (chunk > 12) atomicAdd(&C_row[12], acc12);
        if (chunk > 13) atomicAdd(&C_row[13], acc13);
        if (chunk > 14) atomicAdd(&C_row[14], acc14);
        if (chunk > 15) atomicAdd(&C_row[15], acc15);
    }
}

void run_vcsr_opt_spmm_gpu(const VCSROpt &V, const float* dB, int O, int tileK,
                           float* dC, cudaStream_t st, float &ms) {
    using namespace vcsr_opt_dev;
    
    int num_groups = (int)V.group_depth.size();
    if (num_groups == 0) { ms = 0; return; }
    
    if (!uploaded) {
        cudaMalloc(&d_group_ptr, sizeof(int) * V.group_ptr.size());
        cudaMalloc(&d_group_depth, sizeof(int) * V.group_depth.size());
        cudaMalloc(&d_group_seg_base, sizeof(int) * V.group_seg_base.size());
        cudaMalloc(&d_group_rows, sizeof(int) * V.group_rows.size());
        cudaMalloc(&d_lcol, sizeof(int) * V.lcol.size());
        cudaMalloc(&d_val, sizeof(float) * V.val.size());
        
        cudaMemcpy(d_group_ptr, V.group_ptr.data(), sizeof(int) * V.group_ptr.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_depth, V.group_depth.data(), sizeof(int) * V.group_depth.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_seg_base, V.group_seg_base.data(), sizeof(int) * V.group_seg_base.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_group_rows, V.group_rows.data(), sizeof(int) * V.group_rows.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lcol, V.lcol.data(), sizeof(int) * V.lcol.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, V.val.data(), sizeof(float) * V.val.size(), cudaMemcpyHostToDevice);
        uploaded = true;
    }
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    
    int num_k_tiles = (O + tileK - 1) / tileK;
    dim3 grid(num_groups, num_k_tiles);
    dim3 block(V.bundle);
    size_t smem_bytes = V.segw * tileK * sizeof(float);
    
    cudaEventRecord(t0, st);
    vcsr_opt_kernel<<<grid, block, smem_bytes, st>>>(
        O, tileK, V.bundle, V.segw, V.N, num_groups,
        d_group_ptr, d_group_depth, d_group_seg_base, d_group_rows, d_lcol, d_val,
        dB, O, dC, O);
    cudaEventRecord(t1, st);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

void destroy_vcsr_opt_device(cudaStream_t) {
    using namespace vcsr_opt_dev;
    if (d_group_ptr) cudaFree(d_group_ptr);
    if (d_group_depth) cudaFree(d_group_depth);
    if (d_group_seg_base) cudaFree(d_group_seg_base);
    if (d_group_rows) cudaFree(d_group_rows);
    if (d_lcol) cudaFree(d_lcol);
    if (d_val) cudaFree(d_val);
    d_group_ptr = d_group_depth = d_group_seg_base = d_group_rows = d_lcol = nullptr;
    d_val = nullptr;
    uploaded = false;
}


