#include <iostream>

constexpr int BLOCKSIZE=32;

/**
* Performs single precision GEMM: C = alpha * (A@B) + beta * C
 * with warp-level global VRAM memory access coalescing, ensuring
 * threads within the same warp access contiguous memory addresses.
 *
 * @param A  (float): MxK input matrix
 * @param B  (float): KxN input matrix
 * @param C  (float): MxN output matrix
 */
__global__ void warp_coallesced_gemm(int M, int K, int N,
                     float alpha, const float *A, const float *B,
                     float beta, float *C) {
    const unsigned int m = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const unsigned int n = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (m < M && n < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[m * K + k] * B[k * N + n]; // A[m][k] * B[k][n]
        }
        C[m * N + n] = alpha * tmp + beta * C[m * N + n];
    }
}


void step_2() {
    constexpr int M = 4096, K = 4096, N = 4096;
    float *A, *B, *C;
    cudaMallocManaged(&A, sizeof(float) * M * K);
    cudaMallocManaged(&B, sizeof(float) * K * N);
    cudaMallocManaged(&C, sizeof(float) * M * N);
    for (int i = 0; i < M * K; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 2.0;
    }

    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    dim3 gridDim((M+BLOCKSIZE)/BLOCKSIZE, (N+BLOCKSIZE)/BLOCKSIZE);

    warp_coallesced_gemm<<<gridDim, blockDim>>>(M, K, N, 1.0, A, B, 0.0, C);
    cudaDeviceSynchronize();
    for (int i = 0; i <10; ++i) {
        std::cout << C[i] << " ";
    }
}