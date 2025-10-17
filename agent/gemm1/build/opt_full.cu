
#include <cuda_runtime.h>
#define M 2048
#define N 2048
#define K 2048
#define ALPHA 1.0f
#define BETA 0.0f
const int BLOCKSIZE = 32; // used only by baseline kernel
// opt.cu
// Optimized SGEMM using advanced register blocking for A100 (sm_80)
// Uses 128x128 tiles with 32x8 threads, each thread computing 16x4 outputs

__global__
void sgemm_optimized(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C) {
    // Configuration
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 16;
    constexpr int BLK_X = 32;   // Threads in X (N) dimension
    constexpr int BLK_Y = 8;    // Threads in Y (M) dimension
    constexpr int REG_M = 16;   // Each thread computes 16 rows
    constexpr int REG_N = 4;    // Each thread computes 4 cols

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_M][TILE_K + 4];
    __shared__ float Bs[TILE_K][TILE_N + 4];

    int tx = threadIdx.x;  // 0-31
    int ty = threadIdx.y;  // 0-7
    int tid = ty * BLK_X + tx;

    // Block tile position
    int blockM = blockIdx.y * TILE_M;
    int blockN = blockIdx.x * TILE_N;

    // Thread's output position within the tile
    int threadM = ty * REG_M;     // 0, 16, 32, ..., 112 (8 values)
    int threadN = tx * REG_N;     // 0, 4, 8, ..., 124 (32 values)

    // Register accumulation array
    float acc[REG_M][REG_N];
    #pragma unroll
    for (int i = 0; i < REG_M; ++i)
        #pragma unroll
        for (int j = 0; j < REG_N; ++j)
            acc[i][j] = 0.0f;

    int numTiles = (K + TILE_K - 1) / TILE_K;

    // Main loop over K dimension
    for (int t = 0; t < numTiles; ++t) {
        int kStart = t * TILE_K;

        // Load A tile cooperatively
        // 128x16 = 2048 elements, 256 threads
        for (int i = tid; i < TILE_M * TILE_K; i += BLK_X * BLK_Y) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gRow = blockM + row;
            int gCol = kStart + col;
            As[row][col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        // Load B tile cooperatively
        // 16x128 = 2048 elements, 256 threads
        for (int i = tid; i < TILE_K * TILE_N; i += BLK_X * BLK_Y) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gRow = kStart + row;
            int gCol = blockN + col;
            Bs[row][col] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // Compute on the tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_reg[REG_M];
            float b_reg[REG_N];

            // Load A values into registers
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                a_reg[i] = As[threadM + i][k];
            }

            // Load B values into registers
            #pragma unroll
            for (int j = 0; j < REG_N; ++j) {
                b_reg[j] = Bs[k][threadN + j];
            }

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_N; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < REG_M; ++i) {
        int row = blockM + threadM + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < REG_N; ++j) {
            int col = blockN + threadN + j;
            if (col >= N) continue;

            float val = ALPHA * acc[i][j];
            if (BETA != 0.0f) {
                val += BETA * C[row * N + col];
            }
            C[row * N + col] = val;
        }
    }
}
