#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <getopt.h>
#include "mmio.hpp"
#include "fastspmm.hpp"
#include "csr.hpp"
#include "vcsr.hpp"

// Forward declarations from vcsr_naive.cu
struct VCSRNaive {
    int M{0}, N{0}, bundle{32};
    std::vector<int> group_ptr, group_depth, group_rows, col;
    std::vector<float> val;
};
VCSRNaive csr_to_vcsr_naive(const CSR &csr, int bundle);
void run_vcsr_naive_spmm_gpu(const VCSRNaive &V, const float* dB, int O, float* dC, cudaStream_t st, float &ms);
void destroy_vcsr_naive_device(cudaStream_t st);

// Forward declarations from vcsr_seg.cu
struct VCSRSeg {
    int M{0}, N{0}, segw{16}, bundle{32};
    std::vector<int> group_ptr, group_depth, group_seg_base, group_rows, lcol;
    std::vector<float> val;
    int num_segments{0};
    std::vector<int> segment_group_start;
};
VCSRSeg csr_to_vcsr_seg(const CSR &csr, int segw, int bundle);
void run_vcsr_seg_spmm_gpu(const VCSRSeg &V, const float* dB, int O, int tileK, float* dC, cudaStream_t st, float &ms, bool use_prefetch);
void destroy_vcsr_seg_device(cudaStream_t st);

// Forward declarations from vcsr_opt.cu
struct VCSROpt {
    int M{0}, N{0}, segw{16}, bundle{32};
    std::vector<int> group_ptr, group_depth, group_seg_base, group_rows, lcol;
    std::vector<float> val;
    int num_segments{0};
};
VCSROpt csr_to_vcsr_opt(const CSR &csr, int segw, int bundle);
void run_vcsr_opt_spmm_gpu(const VCSROpt &V, const float* dB, int O, int tileK, float* dC, cudaStream_t st, float &ms);
void destroy_vcsr_opt_device(cudaStream_t st);

// Forward declarations from vcsr_tc.cu
struct VCSRTC {
    int M{0}, N{0}, segw{16}, bundle{16};
    std::vector<int> group_ptr, group_depth, group_seg_base, group_rows, lcol;
    std::vector<float> val;
    int num_segments{0};
};
VCSRTC csr_to_vcsr_tc(const CSR &csr, int bundle);
void run_vcsr_tc_spmm_gpu(const VCSRTC &V, const float* dB, int O, float* dC, cudaStream_t st, float &ms);
void destroy_vcsr_tc_device(cudaStream_t st);

// Forward decls from kernels.cu
void run_csr_spmm_gpu(const int M, const int* d_rowptr, const int* d_col, const float* d_val,
                      const float* dB, int O, float* dC, cudaStream_t st, float &ms);
void run_vcsr_spmm_gpu(const VCSRSpMM &V, const float* dB, int O, int tileK, float* dC, cudaStream_t st, float &ms);
void run_aspt_spmm_gpu(const int M, const int* d_rowptr, const int* d_col, const float* d_val,
                       const float* dB, int O, float* dC, cudaStream_t st, float& ms);
void run_fastspmm_gpu(const FastCST &cst, const float* dB, int O, float* dC, cudaStream_t st, float &ms);
float run_cusparse_spmm(const int M, const int N, const int nnz,
                        const int* d_rowptr, const int* d_col, const float* d_val,
                        const float* dB, int O, float* dC);

static void usage() {
    printf("Usage: spmm_vcsr --mtx file.mtx --O 128 [options]\n");
    printf("Algorithms: cusparse, csr, vcsr_naive, vcsr_baseline, vcsr_seg, vcsr_tc, aspt, fast, all\n");
}

int main(int argc, char** argv) {
    std::string mtx; int O=128, segw=16, bundle=32, tileK=64, repeat=10; std::string algo="all";
  static struct option long_opts[] = {
        {"mtx", required_argument, 0, 'm'}, {"O", required_argument, 0, 'o'},
        {"segw", required_argument, 0, 's'}, {"bundle", required_argument, 0, 'b'},
        {"tileK", required_argument, 0, 't'}, {"repeat", required_argument, 0, 'r'},
        {"algo", required_argument, 0, 'a'}, {0,0,0,0}
  };
    int copt, idx;
    while ((copt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch(copt) {
            case 'm': mtx = optarg; break; case 'o': O = atoi(optarg); break;
            case 's': segw = atoi(optarg); break; case 'b': bundle = atoi(optarg); break;
            case 't': tileK = atoi(optarg); break; case 'r': repeat = atoi(optarg); break;
            case 'a': algo = optarg; break; default: usage(); return 1;
    }
  }
    if (mtx.empty()) { usage(); return 1; }

    printf("=== VCSR-SpMM Benchmark ===\n");
    printf("Loading: %s\n", mtx.c_str());
  COO coo = read_matrix_market_coo(mtx);
  CSR csr = coo_to_csr(coo);
  printf("Matrix: %d x %d, nnz=%d\n", csr.M, csr.N, csr.nnz);
    printf("Params: O=%d, segw=%d, bundle=%d, tileK=%d\n\n", O, segw, bundle, tileK);

    float *dB=nullptr, *dC=nullptr;
    size_t bytesB = (size_t)csr.N * O * sizeof(float), bytesC = (size_t)csr.M * O * sizeof(float);
  cudaMalloc(&dB, bytesB); cudaMalloc(&dC, bytesC);
  int *d_rowptr=nullptr, *d_col=nullptr; float *d_val=nullptr;
    cudaMalloc(&d_rowptr, sizeof(int)*(csr.M+1)); cudaMalloc(&d_col, sizeof(int)*csr.nnz); cudaMalloc(&d_val, sizeof(float)*csr.nnz);
  cudaMemcpy(d_rowptr, csr.row_ptr.data(), sizeof(int)*(csr.M+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col, csr.col.data(), sizeof(int)*csr.nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, csr.val.data(), sizeof(float)*csr.nnz, cudaMemcpyHostToDevice);
  std::vector<float> hB(csr.N * O);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (float)((i % 13) - 6) / 7.f;
  cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);
  cudaStream_t st; cudaStreamCreate(&st);
    auto gflops = [&](double ms) { return (2.0 * csr.nnz * O) / (ms * 1e6); };

    FastCST fast_cst; bool fast_built = false;
    if (algo == "all" || algo == "fast" || algo == "fastspmm") { fast_cst = build_cst_from_csr(csr); upload_cst_to_device(fast_cst); fast_built = true; }

    printf("%-16s %10s %12s %s\n", "Kernel", "Time (ms)", "GFLOP/s", "Notes");
    printf("%-16s %10s %12s %s\n", "----------------", "----------", "------------", "-----");

  if (algo == "all" || algo == "cusparse") {
    float ms_sum = 0.0f;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); ms_sum += run_cusparse_spmm(csr.M, csr.N, csr.nnz, d_rowptr, d_col, d_val, dB, O, dC); }
        printf("cuSPARSE         %10.3f %12.2f (vendor)\n", ms_sum/repeat, gflops(ms_sum/repeat));
}
    if (algo == "all" || algo == "csr") {
        float ms_sum = 0.0f, ms;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_csr_spmm_gpu(csr.M, d_rowptr, d_col, d_val, dB, O, dC, st, ms); ms_sum += ms; }
        printf("CSR (naive)      %10.3f %12.2f\n", ms_sum/repeat, gflops(ms_sum/repeat));
    }
    if (algo == "all" || algo == "vcsr_naive") {
        VCSRNaive V = csr_to_vcsr_naive(csr, bundle); float ms_sum = 0.0f, ms;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_vcsr_naive_spmm_gpu(V, dB, O, dC, st, ms); ms_sum += ms; }
        printf("VCSR-Naive       %10.3f %12.2f [groups=%zu]\n", ms_sum/repeat, gflops(ms_sum/repeat), V.group_depth.size());
        destroy_vcsr_naive_device(st);
    }
    if (algo == "all" || algo == "vcsr_baseline") {
        VCSRSeg V = csr_to_vcsr_seg(csr, segw, bundle); float ms_sum = 0.0f, ms;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_vcsr_seg_spmm_gpu(V, dB, O, tileK, dC, st, ms, true); ms_sum += ms; }
        printf("VCSR-Baseline    %10.3f %12.2f [segw=%d]\n", ms_sum/repeat, gflops(ms_sum/repeat), segw);
        destroy_vcsr_seg_device(st);
    }
    if (algo == "all" || algo == "vcsr_seg") {
        VCSROpt V = csr_to_vcsr_opt(csr, segw, bundle); float ms_sum = 0.0f, ms;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_vcsr_opt_spmm_gpu(V, dB, O, tileK, dC, st, ms); ms_sum += ms; }
        printf("VCSR-Seg         %10.3f %12.2f [segw=%d]\n", ms_sum/repeat, gflops(ms_sum/repeat), segw);
        destroy_vcsr_opt_device(st);
  }
    if (algo == "all" || algo == "vcsr_tc") {
        VCSRTC V = csr_to_vcsr_tc(csr, 16); float ms_sum = 0.0f, ms;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_vcsr_tc_spmm_gpu(V, dB, O, dC, st, ms); ms_sum += ms; }
        printf("VCSR-TC          %10.3f %12.2f [TC]\n", ms_sum/repeat, gflops(ms_sum/repeat));
        destroy_vcsr_tc_device(st);
    }
  if (algo == "all" || algo == "aspt") {
        if (O % 64 != 0) { printf("ASpT             %10s %12s (O%%64!=0)\n", "N/A", "N/A"); }
        else { float ms_sum = 0.0f, ms;
            for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_aspt_spmm_gpu(csr.M, d_rowptr, d_col, d_val, dB, O, dC, st, ms); ms_sum += ms; }
            printf("ASpT             %10.3f %12.2f\n", ms_sum/repeat, gflops(ms_sum/repeat)); }
    }
  if (algo == "all" || algo == "fast" || algo == "fastspmm") {
        float ms_sum = 0.0f, ms;
        for (int it = 0; it < repeat; ++it) { cudaMemset(dC, 0, bytesC); run_fastspmm_gpu(fast_cst, dB, O, dC, st, ms); ms_sum += ms; }
        printf("FastSpMM         %10.3f %12.2f [TC]\n", ms_sum/repeat, gflops(ms_sum/repeat));
  }
    printf("\n");
    cudaStreamDestroy(st); cudaFree(dB); cudaFree(dC); cudaFree(d_rowptr); cudaFree(d_col); cudaFree(d_val);
    if (fast_built) destroy_cst(fast_cst);
  return 0;
}
