#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>   // WMMA

using namespace nvcuda;

__global__ void wmma_test_kernel(float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // 16x16x16 accumulator fragment in float
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 1.0f);

    // Just write one element back so we can see something happened
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = acc_frag.x[0];
        printf("WMMA fragment created on device, first element = %f\n", acc_frag.x[0]);
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Compiled for architecture < 7.0, WMMA not available\n");
    }
#endif
}

int main() {
    // Sanity check device
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);

    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    wmma_test_kernel<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host readback: %f\n", h_out);

    cudaFree(d_out);
    return 0;
}
