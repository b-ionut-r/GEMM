// 1D Blocktiling: each thread computes a 1D column of results in C
// => less memory loads & store instructions => less memory overhead => faster kernel


#include <iostream>

constexpr int BM = 64, BK = 8, BN = 64, TM = 8;

__global__ void work_1d_gemm(int M, int K, int N,
                             float alpha, const float *A, const float *B,
                             float beta, float *C) {

    // A block is responsible for fully computing a BMxBN tile of C
    // For this, it employs BMxBN/TM threads, each of which computes a column of length TM in C

    // First, let's properly set the starting positions using pointer arithmetic
    A += BM * blockIdx.x * K;  // row = BM * blockIdx.x, col = 0
    B += BN * blockIdx.y;  // row = 0, col = BN * blockIdx.y
    C += BM * blockIdx.x * N + BN * blockIdx.y;  // row = BM * blockIdx.x, col = BN * blockIdx.y

    // COMPUTE indices — maps BMxBN/TM threads onto the BM×BN output tile
    // Using % and / we ensure threads in the same warp access contiguous memory addresses in A and B
    // so that GMEM transactions can be coalesced
    // LOAD indices — maps BMxBN/TM threads onto the BM×BK As tile, and BK×BN Bs tile respectively
    const unsigned int innerRowA = threadIdx.x / BK;  // 0..63
    const unsigned int innerColA = threadIdx.x % BK;  // 0..7
    const unsigned int innerRowB = threadIdx.x / BN;  // 0..7
    const unsigned int innerColB = threadIdx.x % BN;  // 0..63
    const unsigned int &threadRow = innerRowB;
    const unsigned int &threadCol = innerColB;

    // Employ shared memory to cache sliding windows
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // Loop through the K dimension in chunks of BK
    for (int slideIdx = 0; slideIdx < K; slideIdx += BK) {

        // Each thread loads in SMEM
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        // Sync to ensure everything is loaded before computations
        __syncthreads();

        // DotIdx loop
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float bTmp = Bs[dotIdx * BN + threadCol]; // Bs: row = dotIdx, col = threadCol
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * bTmp;
            }
        }
        // Advance original pointers into the next chunk
        A += BK;
        B += BK * N;

        // Sync to ensure all threads are done with the current loop step
        __syncthreads();
    }
    // Accumulate here
    for (unsigned int resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta  * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}




void step_4() {
    int M = 4096, K = 4096, N = 4096;
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
    dim3 BDim(BM * BN / TM);
    dim3 GDim((M + BM - 1) / BM, (N + BN - 1) / BN);
    work_1d_gemm<<<GDim, BDim>>>(M, K, N, 1.0, A, B, 0.0, C);
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
}

