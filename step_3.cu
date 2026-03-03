#include <iostream>

constexpr int BLOCKSIZE = 32;

/**
 * Tiled SGEMM: C = alpha * (A @ B) + beta * C
 *
 * Each block computes a (BLOCKSIZE x BLOCKSIZE) tile of C by iterating over
 * K in chunks, staging tiles of A and B in shared memory to reduce global
 * memory traffic.
 *
 * @tparam BLOCKSIZE  Tile edge length; BLOCKSIZE^2 must be <= 1024.
 * @param M, N, K     Matrix dimensions (must be multiples of BLOCKSIZE).
 * @param alpha, beta BLAS scalars.
 * @param A           Row-major [M x K], device memory.
 * @param B           Row-major [K x N], device memory.
 * @param C           Row-major [M x N], device memory, read and written.
 *
 * Launch: grid(ceil(M/BLOCKSIZE), ceil(N/BLOCKSIZE)), block(BLOCKSIZE^2).
 */

__global__ void tiled_shared_gemm(int M, int K, int N,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    // the output block that we want to compute in this threadblock
    const unsigned int cRow = blockIdx.x;
    const unsigned int cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & cos that we're accessing in this thread
    const unsigned int threadRow = threadIdx.x / BLOCKSIZE;
    const unsigned int threadCol = threadIdx.x % BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // load elements from global memory into shared memory
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // dot product on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // sync threads to avoid next block being loaded before all threads are done
        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}


void step_3() {
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
    dim3 gridDim((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE);

    tiled_shared_gemm<<<gridDim, blockDim>>>(M, N, K, 1.0, A, B, 0.0, C);
    cudaDeviceSynchronize();
    for (int i = 0; i <10; ++i) {
        std::cout << C[i] << " ";
    }
}