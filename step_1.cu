// NAIVE KERNEL


#include <iostream>

/*
 * Performs single precision GEMM:  C = alpha * (A@B) + beta * C
 * A (float): MxK 2D matrix
 * B (float): KxN 2D matrix
 * C (float): MxN 2D matrix
*/
__global__ void naive_gemm(int M, int K, int N,
                     float alpha, const float *A, const float *B,
                     float beta, float *C) {
    // this thread will compute C[i][j]
    const unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;
    // 'if' condition when M or N aren't multiples of 32
    if (m < M && n < N) {
        // compute tmp as <Row(A, m), Col(B, n)>
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[m * K + k] * B[k * N + n]; // A[m][k] * B[k][n]
        }
        C[m * N + n] = alpha * tmp + beta * C[m * N + n];
    }
}




void step_1() {
    float *A, *B, *C;
    constexpr int M = 4096, N = 4096, K = 4096;
    cudaMallocManaged(&A, sizeof(float) * M * K);
    cudaMallocManaged(&B, sizeof(float) * K * N);
    cudaMallocManaged(&C, sizeof(float) * M * N);
    for (int i = 0; i < M * K; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 2.0;
    }
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((M + 31) / 32, (N + 31) / 32, 1);
    naive_gemm<<<gridDim, blockDim>>>(M, K, N, 1, A, B, 0, C);
    cudaDeviceSynchronize();
    for (int i = 0; i <10; ++i) {
        std::cout << C[i] << " ";
    }
}