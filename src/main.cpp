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

// Forward decls from kernels.cu
void run_csr_spmm_gpu(const int M, const int* d_rowptr, const int* d_col, const float* d_val,
                      const float* dB, int O, float* dC, cudaStream_t st, float &ms);
void run_vcsr_spmm_gpu(const VCSRSpMM &V, const float* dB, int O, int tileK, float* dC, cudaStream_t st, float &ms);
void run_aspt_spmm_gpu(const int   M,
                       const int*  d_rowptr,
                       const int*  d_col,
                       const float* d_val,
                       const float* dB,
                       int         O,
                       float*      dC,
                       cudaStream_t st,
                       float&      ms);
void run_fastspmm_gpu(const FastCST &cst,
                      const float* dB, int O,
                      float* dC, cudaStream_t st, float &ms);
float run_cusparse_spmm(const int M, const int N, const int nnz,
                        const int* d_rowptr, const int* d_col, const float* d_val,
                        const float* dB, int O, float* dC);


static void usage(){
  printf("Usage: spmm_vcsr --mtx file.mtx --O 128 [--segw 4] [--bundle 32] [--tileK 64] [--repeat 10] [--algo all|csr|cusparse|vcsr|aspt|fast]\n");
}

int main(int argc, char** argv){
  std::string mtx; int O=128, segw=4, bundle=32, tileK=64, repeat=10; std::string algo="all";
  static struct option long_opts[] = {
    {"mtx", required_argument, 0, 'm'},
    {"O", required_argument, 0, 'o'},
    {"segw", required_argument, 0, 's'},
    {"bundle", required_argument, 0, 'b'},
    {"tileK", required_argument, 0, 't'},
    {"repeat", required_argument, 0, 'r'},
    {"algo", required_argument, 0, 'a'},
    {0,0,0,0}
  };
  int copt; int idx;
  while((copt=getopt_long(argc, argv, "", long_opts, &idx))!=-1){
    switch(copt){
      case 'm': mtx=optarg; break;
      case 'o': O=atoi(optarg); break;
      case 's': segw=atoi(optarg); break;
      case 'b': bundle=atoi(optarg); break;
      case 't': tileK=atoi(optarg); break;
      case 'r': repeat=atoi(optarg); break;
      case 'a': algo=optarg; break;
      default: usage(); return 1;
    }
  }
  if(mtx.empty()){ usage(); return 1; }

  printf("Loading MTX: %s\n", mtx.c_str());
  COO coo = read_matrix_market_coo(mtx);
  CSR csr = coo_to_csr(coo);
  printf("Matrix: %d x %d, nnz=%d\n", csr.M, csr.N, csr.nnz);

  // Device allocations
  float *dB=nullptr, *dC=nullptr; size_t bytesB=(size_t)csr.N * O * sizeof(float), bytesC=(size_t)csr.M * O * sizeof(float);
  cudaMalloc(&dB, bytesB); cudaMalloc(&dC, bytesC);
  // Copy CSR to device
  int *d_rowptr=nullptr, *d_col=nullptr; float *d_val=nullptr;
  cudaMalloc(&d_rowptr, sizeof(int)*(csr.M+1));
  cudaMalloc(&d_col, sizeof(int)*csr.nnz);
  cudaMalloc(&d_val, sizeof(float)*csr.nnz);
  cudaMemcpy(d_rowptr, csr.row_ptr.data(), sizeof(int)*(csr.M+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col, csr.col.data(), sizeof(int)*csr.nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, csr.val.data(), sizeof(float)*csr.nnz, cudaMemcpyHostToDevice);

  std::vector<float> hB(csr.N * O);
  for(size_t i=0;i<hB.size();++i) hB[i] = (float)((i%13)-6)/7.f; // deterministic-ish
  cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, bytesC);

  cudaStream_t st; cudaStreamCreate(&st);

  auto gflops = [&](double ms){ return (2.0 * csr.nnz * O) / (ms*1e6); };

   FastCST fast_cst;
  if (algo == "all" || algo == "fast") {
    fast_cst = build_cst_from_csr(csr);
    upload_cst_to_device(fast_cst);
}
  if (algo == "all" || algo == "cusparse") {
    float ms_sum = 0.0f;

    for (int it = 0; it < repeat; ++it) {
        cudaMemset(dC, 0, bytesC);
        ms_sum += run_cusparse_spmm(
            csr.M, csr.N, csr.nnz,
            d_rowptr, d_col, d_val,
            dB, O, dC
        );
    }

    double msavg = ms_sum / repeat;
    printf("cuSPARSE:   %8.3f ms  %8.2f GFLOP/s\n",
           msavg, gflops(msavg));
}


  if(algo=="all" || algo=="csr"){
    float ms_sum=0, ms; 
    
    for(int it=0; it<repeat; ++it){ 
        cudaMemset(dC, 0, bytesC); run_csr_spmm_gpu(csr.M, d_rowptr, d_col, d_val, dB, O, dC, st, ms); ms_sum += ms; 
    }

    double msavg = ms_sum / repeat; printf("CSR (naive):%8.3f ms  %8.2f GFLOP/s\n", msavg, gflops(msavg));
  }

  if(algo=="all" || algo=="vcsr"){
    VCSRSpMM V = csr_to_vcsr_spmm(csr, segw, bundle);
    float ms_sum=0, ms; 
    for(int it=0; it<repeat; ++it){ 
        cudaMemset(dC, 0, bytesC); run_vcsr_spmm_gpu(V, dB, O, tileK, dC, st, ms); ms_sum += ms; }
        
    double msavg = ms_sum / repeat; printf("VCSR (seg): %8.3f ms  %8.2f GFLOP/s  [segw=%d,bundle=%d,tileK=%d,groups=%zu]\n",
      msavg, gflops(msavg), segw, bundle, tileK, V.group_depth.size());
  }


if (algo == "all" || algo == "fast") {
    float ms_sum = 0.0f, ms;

    for (int it = 0; it < repeat; ++it) {
        cudaMemset(dC, 0, bytesC);
        run_fastspmm_gpu(fast_cst, dB, O, dC, st, ms);
        ms_sum += ms;
    }

    double msavg = ms_sum / repeat;
    printf("FastSpMM:   %8.3f ms  %8.2f GFLOP/s\n",
           msavg, gflops(msavg));
}

  if (algo == "all" || algo == "aspt") {
    float ms_sum = 0.0f, ms;

    for (int it = 0; it < repeat; ++it) {
        cudaMemset(dC, 0, bytesC);
        run_aspt_spmm_gpu(
            csr.M,
            d_rowptr, d_col, d_val,
            dB, O, dC,
            st, ms
        );
        ms_sum += ms;
    }

    double msavg = ms_sum / repeat;
    printf("ASpT:       %8.3f ms  %8.2f GFLOP/s\n",
           msavg, gflops(msavg));
}


  cudaStreamDestroy(st);
  cudaFree(dB); cudaFree(dC);
  cudaFree(d_rowptr); cudaFree(d_col); cudaFree(d_val);
  if (algo == "all" || algo == "fast") {
    destroy_cst(fast_cst);
  }

  return 0;

}