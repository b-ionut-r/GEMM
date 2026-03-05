//
// Created by Bujor Ionut Raul on 05.03.2026.
//

#include <iostream>

constexpr int BM=64, BK=8, BN=64, TM=8, TN=8;

// 2D blocktiling
// Task: multiply matrices A and B, store the result in C
// A: MxK
// B: KxN
// C: MxN
// Over A, we slide a tile of shape BMxBK, and over B we slide a tile of shape BKxBN
// Each block of threads will compute a BMxBN tile of C, and each thread will compute a 2D TMxTN
// tile within it.
// So, each block will work for a 64x64 tile of C, by employing 64x64/(8*8) = 64 threads
// The kernel is meant to be launched like this:
// work_2d_gemm<<<dim3(CEIL(M/BM), CEIL(N/BN)), BM*BN/(TM*TN)>>>(M, K, N, alpha, A, B, beta, C);


__global__ void work_2d_gemm(int M, int K, int N,
                             float alpha, const float *A, const float *B,
                             float beta, float *C) {
    // blockIdx.x will be blockRow, and blockIdx.y will be blockCol in C
    // First, let's properly set the starting positions using pointer arithmetic
    B += blockIdx.y * BN; // blockRow = 0, blockCol = blockIdx.y => row = 0, col = blockIdx.y * BN
    C += blockIdx.x * BM * N + blockIdx.y * BN; // blockRow = blockIdx.x, blockCol = blockIdx.y => row = blockIdx.x * BM, col = blockIdx.y * BN

    // Inner indexes of thread's tiles (TMxTN) within their block's C tile (BMxBN)
    const unsigned int threadRow = threadIdx.x / (BN/TN);
    const unsigned int threadCol = threadIdx.x % (BN/TN);

    // Initialize SMEM space to be used by all threads inside this block
    // to compute together the BMxBN tile of C
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];


    // Allocate register thread-level cache
    float threadResults[TM*TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};


    // Loop over cols of A, rows of B to accumulate dot product
    for (int blckIdx=0; blckIdx < K; blckIdx += BK) {
        // Load SMEM As and Bs, carefully, to ensure coallescing
        for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
            As[idx] = A[idx / BK * K + idx % BK];
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
            Bs[idx] = B[idx / BN * N + idx % BN];
        }
        __syncthreads();

        // Now that it's loaded, we have a TMxBK tile in SMEM from A, and a BKxTN tile in SMEM from B,
        // so we can compute the TMxTN tile of C for this iteration
        // we'll use outer product of Col(A, dotIdx) and Row(B, dotIdx) to update the TMxTN tile of C in registers
        // because we know that A @ B = Σ (Col(A,k) ⊗ Row(B,k))
        for (int dotIdx=0; dotIdx < BK; ++dotIdx) {
            // Load Col(A, dotIdx) and Row(B, dotIdx) into registers
            for (int i=0; i<TM; ++i)
                regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
            for (int j=0; j<TN; ++j)
                regB[j] = Bs[(dotIdx) * BN + threadCol * TN + j];
            for (int i=0; i<TM; ++i) {
                for(int j=0; j<TN; ++j) {
                    threadResults[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
        // sync threads here before next step of the loop to ensure all work is done
        // before updating current As and Bs in SMEM
        __syncthreads();

        // Advance tiles
        A += BK; // jump BK cols in A along this block row
        B += BK * N; // jump BK rows in A along this block column
    }

    // Fill in results in C from threadResults, applying alpha and beta
    for (int i=0; i<TM; ++i) {
        for (int j=0; j<TN; ++j) {
            // C's index will be ith row jth col inside the threadTileRowC block row and the threadTileColC block col
            // TM * threadTileRowC + i gives the row index inside the block, and TN * threadTileColC + j gives the col index inside the block
            C[(TM * threadRow + i) * N + (TN * threadCol + j)] = alpha * threadResults[i * TN + j]
                                       + beta * C[(TM * threadRow + i) * N + (TN * threadCol + j)];
        }
    }
}



void step_5() {
    const int M = 4096, N = 4096, K= 4096;
    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));
    for (int i = 0; i < M * K; ++i) {
        A[i] = 1.0;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 2.0;
    }
    dim3 blockDim(BM * BN / (TM * TN));
    dim3 gridDim((M + BM - 1) / BM, (N + BN -1)/ BN);
    work_2d_gemm<<<gridDim, blockDim>>>(M, K, N, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
}