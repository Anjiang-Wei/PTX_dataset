
#include <cuda_runtime.h>
#define M 4096
#define N 4096
#define K 4096
#define ALPHA 1.0f
#define BETA 0.0f
// opt.cu
// Optimized SGEMM for A100 (sm_80) with 128x128x8 tiling.
// Must match the declaration: extern __global__ void sgemm_optimized(...)
// Uses 1D grid and 1D block with 512 threads per block.

__global__
void sgemm_optimized(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 8;
    constexpr int BLK_X  = 32;  // logical width
    constexpr int BLK_Y  = 16;  // logical height (32*16 = 512 threads)
    constexpr int RM     = TILE_M / BLK_Y;  // 8
    constexpr int RN     = TILE_N / BLK_X;  // 4

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // 1D thread index -> 2D logical position
    int tid = threadIdx.x;
    int tx = tid % BLK_X;
    int ty = tid / BLK_X;

    // 1D block index -> 2D tile position
    int num_tiles_x = (N + TILE_N - 1) / TILE_N;
    int tile_x = blockIdx.x % num_tiles_x;
    int tile_y = blockIdx.x / num_tiles_x;
    int block_m = tile_y * TILE_M;
    int block_n = tile_x * TILE_N;

    int row0 = block_m + ty * RM;
    int col0 = block_n + tx * RN;

    float acc[RM][RN] = {0};

    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int THR = BLK_X * BLK_Y;  // 512 total threads

    for (int t = 0; t < num_tiles; ++t) {
        int aCol = t * TILE_K;
        for (int idx = tid; idx < TILE_M * TILE_K; idx += THR) {
            int i = idx / TILE_K, k = idx % TILE_K;
            int gRow = block_m + i, gCol = aCol + k;
            As[i][k] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        int bRow = t * TILE_K;
        for (int idx = tid; idx < TILE_K * TILE_N; idx += THR) {
            int k = idx / TILE_N, j = idx % TILE_N;
            int gRow = bRow + k, gCol = block_n + j;
            Bs[k][j] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float aFrag[RM], bFrag[RN];
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                aFrag[i] = As[ty * RM + i][kk];
            #pragma unroll
            for (int j = 0; j < RN; ++j)
                bFrag[j] = Bs[kk][tx * RN + j];
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                for (int j = 0; j < RN; ++j)
                    acc[i][j] += aFrag[i] * bFrag[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        int r = row0 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            int c = col0 + j;
            if (c >= N) continue;
            float val = ALPHA * acc[i][j];
            if (BETA != 0.0f) val += BETA * C[r * N + c];
            C[r * N + c] = val;
        }
    }
}
