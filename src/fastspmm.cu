#include "fastspmm.hpp"
#include <unordered_map>
#include <algorithm>
#include <mma.h>

using namespace nvcuda;
struct FastCSTDevice {
    int M, N;
    int num_row_windows;
    int num_tcb;
    int row_window_size;
    int tcb_width;

    const int      *rowWindowOffset;
    const int      *tcbColOffset;
    const uint16_t *tcbRowOffset;
    const uint8_t  *tcbColIndex;
    const __half   *tcbValue;
    const int      *tcb2B;
};

__global__
void fastspmm_kernel(FastCSTDevice cdev,
                     const float * __restrict__ B, int O,
                     float * __restrict__ C)
{
    const int RWS  = FASTSPMM_ROW_WINDOW;
    const int TCBW = FASTSPMM_TCB_WIDTH;

    int roww_id = blockIdx.x;
    int tileN   = blockIdx.y;  

    if (roww_id >= cdev.num_row_windows) return;

    int row_base  = roww_id * RWS;
    int col_base  = tileN * 16;

    if (col_base >= O) return;

    int first_tcb = cdev.rowWindowOffset[roww_id];
    int last_tcb  = cdev.rowWindowOffset[roww_id + 1];

    // WMMA accumulator
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Shared memory tiles
    __shared__ __half As[16 * 16]; // row-major 16x16
    __shared__ __half Bs[16 * 16]; // row-major 16x16

    int lane = threadIdx.x;

    for (int tcb_id = first_tcb; tcb_id < last_tcb; ++tcb_id) {
        // Zero shared-memory tiles
        for (int i = lane; i < 16 * 16; i += blockDim.x) {
            As[i] = __float2half(0.0f);
            Bs[i] = __float2half(0.0f);
        }
        __syncthreads();

        int nnz_start = cdev.tcbColOffset[tcb_id];
        int nnz_end   = cdev.tcbColOffset[tcb_id + 1];
        int baseIndex = nnz_start;

        const uint16_t *rowOffBase = cdev.tcbRowOffset + tcb_id * (RWS + 1);
        const uint8_t  *colIdxBase = cdev.tcbColIndex + baseIndex;
        const __half   *valBase    = cdev.tcbValue    + baseIndex;

        // Decompress A into As (16x16)
        for (int lr = 0; lr < RWS; ++lr) {
            int start = rowOffBase[lr]     - baseIndex;
            int end   = rowOffBase[lr + 1] - baseIndex;

            for (int idx = start + lane; idx < end; idx += blockDim.x) {
                uint8_t lc = colIdxBase[idx];
                __half v   = valBase[idx];
                if (lr < 16 && lc < 16) {
                    As[lr * 16 + lc] = v;
                }
            }
        }

        // Gather B rows for this TCB into Bs
        const int *tcb2B = cdev.tcb2B + tcb_id * TCBW;

        for (int k = lane; k < TCBW; k += blockDim.x) {
            int global_row_B = tcb2B[k];
            if (global_row_B < 0) continue;

            int b_row_off = global_row_B * O + col_base;
            for (int n = 0; n < 16 && (col_base + n) < O; ++n) {
                float bv = B[b_row_off + n];
                Bs[k * 16 + n] = __float2half(bv);
            }
        }

        __syncthreads();

        // Load WMMA fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, As, 16);
        wmma::load_matrix_sync(b_frag, Bs, 16);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store C tile. Handle last row window < 16 rows carefully.
    __shared__ float Csub[16 * 16];

    if (lane < 32) {
        wmma::store_matrix_sync(Csub, c_frag, 16, wmma::mem_row_major);
    }
    __syncthreads();

    int max_rows = min(RWS, cdev.M - row_base);
    for (int i = lane; i < max_rows * 16 && (col_base + (i % 16)) < O; i += blockDim.x) {
        int lr = i / 16;
        int lc = i % 16;
        int gr = row_base + lr;
        int gc = col_base + lc;
        C[gr * O + gc] = Csub[lr * 16 + lc];
    }
}

FastCST build_cst_from_csr(const CSR &csr)
{
    FastCST cst;
    cst.M = csr.M;
    cst.N = csr.N;

    const int RWS = FASTSPMM_ROW_WINDOW;
    const int TCBW = FASTSPMM_TCB_WIDTH;

    cst.num_row_windows = (csr.M + RWS - 1) / RWS;
    cst.h_rowWindowOffset.assign(cst.num_row_windows + 1, 0);

    std::vector<int>       rowWindowOffset;
    std::vector<int>       tcbColOffset;
    std::vector<uint16_t>  tcbRowOffset;
    std::vector<uint8_t>   tcbColIndex;
    std::vector<__half>    tcbValue;
    std::vector<int>       tcb2B;

    rowWindowOffset.reserve(cst.num_row_windows + 1);
    tcbColOffset.push_back(0); 

    int global_tcb_id = 0;
    int global_nnz_cst = 0;

    rowWindowOffset.push_back(0);

    for (int w = 0; w < cst.num_row_windows; ++w) {
        int row_begin = w * RWS;
        int row_end   = std::min(csr.M, row_begin + RWS);

        std::vector<int> cols;
        for (int r = row_begin; r < row_end; ++r) {
            for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
                cols.push_back(csr.col[p]);
            }
        }
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

        if (cols.empty()) {
            rowWindowOffset.push_back(global_tcb_id);
            continue;
        }

        std::unordered_map<int,int> col2cond;
        col2cond.reserve(cols.size());
        for (int i = 0; i < (int)cols.size(); ++i) {
            col2cond[cols[i]] = i;
        }

        int num_tcb_window = (int)((cols.size() + TCBW - 1) / TCBW);

        std::vector<std::vector<std::vector<std::pair<uint8_t,float>>>> buckets(
            num_tcb_window,
            std::vector<std::vector<std::pair<uint8_t,float>>>(RWS)
        );

        // 2. Distribute nnz into TCB buckets
        for (int r = row_begin; r < row_end; ++r) {
            int local_row = r - row_begin;
            for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p) {
                int col = csr.col[p];
                auto it = col2cond.find(col);
                if (it == col2cond.end()) continue;
                int cond = it->second;
                int tcb_id_local = cond / TCBW;
                int local_col = cond % TCBW;
                float v = csr.val[p];
                buckets[tcb_id_local][local_row].push_back(
                    {(uint8_t)local_col, v}
                );
            }
        }

        for (int tlocal = 0; tlocal < num_tcb_window; ++tlocal) {
            int tcb_id = global_tcb_id + tlocal;

            int row_off_base_index = (tcb_id) * (RWS + 1);
            if ((int)tcbRowOffset.size() < row_off_base_index + (RWS + 1)) {
                tcbRowOffset.resize(row_off_base_index + (RWS + 1));
            }

            int tcb_nnz_start = global_nnz_cst;

            // For each row
            for (int lr = 0; lr < RWS; ++lr) {
                tcbRowOffset[row_off_base_index + lr] = (uint16_t)global_nnz_cst;

                auto &row_elems = buckets[tlocal][lr];
                for (auto &e : row_elems) {
                    tcbColIndex.push_back(e.first);
                    tcbValue.push_back(__float2half(e.second));
                    ++global_nnz_cst;
                }
            }
            tcbRowOffset[row_off_base_index + RWS] = (uint16_t)global_nnz_cst;

            // TCBColOffset entry for this TCB
            tcbColOffset.push_back(global_nnz_cst);

            // TCB2B: map local 0..15 -> original global column
            for (int lc = 0; lc < TCBW; ++lc) {
                int cond_idx = tlocal * TCBW + lc;
                int orig_col = (cond_idx < (int)cols.size()) ? cols[cond_idx] : -1;
                tcb2B.push_back(orig_col);
            }
        }

        global_tcb_id += num_tcb_window;
        rowWindowOffset.push_back(global_tcb_id);
    }

    cst.num_tcb = global_tcb_id;

    // Move into cst
    cst.h_rowWindowOffset = std::move(rowWindowOffset);
    cst.h_tcbColOffset    = std::move(tcbColOffset);
    cst.h_tcbRowOffset    = std::move(tcbRowOffset);
    cst.h_tcbColIndex     = std::move(tcbColIndex);
    cst.h_tcbValue        = std::move(tcbValue);
    cst.h_tcb2B           = std::move(tcb2B);

    return cst;
}

void upload_cst_to_device(FastCST &cst)
{
    const int RWS = FASTSPMM_ROW_WINDOW;
    size_t num_roww   = cst.h_rowWindowOffset.size();
    size_t num_tcb    = cst.h_tcbColOffset.size(); 
    size_t num_rowoff = cst.h_tcbRowOffset.size();
    size_t num_colidx = cst.h_tcbColIndex.size();
    size_t num_val    = cst.h_tcbValue.size();
    size_t num_tcb2b  = cst.h_tcb2B.size();

    cudaMalloc(&cst.d_rowWindowOffset, num_roww   * sizeof(int));
    cudaMalloc(&cst.d_tcbColOffset,    num_tcb    * sizeof(int));
    cudaMalloc(&cst.d_tcbRowOffset,    num_rowoff * sizeof(uint16_t));
    cudaMalloc(&cst.d_tcbColIndex,     num_colidx * sizeof(uint8_t));
    cudaMalloc(&cst.d_tcbValue,        num_val    * sizeof(__half));
    cudaMalloc(&cst.d_tcb2B,           num_tcb2b  * sizeof(int));

    cudaMemcpy(cst.d_rowWindowOffset, cst.h_rowWindowOffset.data(),
               num_roww * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cst.d_tcbColOffset, cst.h_tcbColOffset.data(),
               num_tcb * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cst.d_tcbRowOffset, cst.h_tcbRowOffset.data(),
               num_rowoff * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cst.d_tcbColIndex, cst.h_tcbColIndex.data(),
               num_colidx * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cst.d_tcbValue, cst.h_tcbValue.data(),
               num_val * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(cst.d_tcb2B, cst.h_tcb2B.data(),
               num_tcb2b * sizeof(int), cudaMemcpyHostToDevice);
}

void run_fastspmm_gpu(const FastCST &cst,
                      const float *dB, int O,
                      float *dC,
                      cudaStream_t st,
                      float &ms)
{
    int num_tiles_N = (O + 15) / 16;

    FastCSTDevice cdev;
    cdev.M = cst.M;
    cdev.N = cst.N;
    cdev.num_row_windows = cst.num_row_windows;
    cdev.num_tcb = cst.num_tcb;
    cdev.row_window_size = FASTSPMM_ROW_WINDOW;
    cdev.tcb_width = FASTSPMM_TCB_WIDTH;
    cdev.rowWindowOffset = cst.d_rowWindowOffset;
    cdev.tcbColOffset    = cst.d_tcbColOffset;
    cdev.tcbRowOffset    = cst.d_tcbRowOffset;
    cdev.tcbColIndex     = cst.d_tcbColIndex;
    cdev.tcbValue        = cst.d_tcbValue;
    cdev.tcb2B           = cst.d_tcb2B;

    dim3 block(32, 1, 1); // 1 warp
    dim3 grid(cst.num_row_windows, num_tiles_N, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, st);
    fastspmm_kernel<<<grid, block, 0, st>>>(cdev, dB, O, dC);
    cudaEventRecord(stop, st);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void destroy_cst(FastCST &cst)
{
    cudaFree(cst.d_rowWindowOffset);
    cudaFree(cst.d_tcbColOffset);
    cudaFree(cst.d_tcbRowOffset);
    cudaFree(cst.d_tcbColIndex);
    cudaFree(cst.d_tcbValue);
    cudaFree(cst.d_tcb2B);

    cst.d_rowWindowOffset = nullptr;
    cst.d_tcbColOffset    = nullptr;
    cst.d_tcbRowOffset    = nullptr;
    cst.d_tcbColIndex     = nullptr;
    cst.d_tcbValue        = nullptr;
    cst.d_tcb2B           = nullptr;
}
