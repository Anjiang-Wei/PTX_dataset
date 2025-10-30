// opt.cu
// Optimized SGEMM for A100 (sm_80) using Tensor Cores via WMMA API.
// Uses 1D grid and 1D blocks with 64x64 tile size and 16x16x16 WMMA fragments.

#include <mma.h>
using namespace nvcuda;

__global__
void sgemm_optimized(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C) {
    // WMMA dimensions: M=16, N=16, K=16
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Tile dimensions: 64x64 output to fit shared memory
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 16;  // Match WMMA_K for efficient Tensor Core usage

    // Each block has 4x4 = 16 warps (32 threads/warp * 16 warps = 512 threads)
    constexpr int WARPS_N = 4;

    // Shared memory for tiles (use half precision for Tensor Cores)
    // As: 64x16 x 2 bytes = 2048 bytes
    // Bs: 16x64 x 2 bytes = 2048 bytes
    // C_smem: 64x64 x 4 bytes = 16384 bytes
    // Total: 20480 bytes < 49152 bytes limit
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N];
    __shared__ float C_smem[TILE_M][TILE_N];

    // Thread and warp indices (1D thread indexing)
    int tid = threadIdx.x;
    int warpId = tid / 32;

    // Determine which WMMA tile this warp handles
    int warp_m = warpId / WARPS_N;  // 0-3
    int warp_n = warpId % WARPS_N;  // 0-3

    // 1D block index -> 2D tile position
    int num_tiles_x = (N + TILE_N - 1) / TILE_N;
    int tile_x = blockIdx.x % num_tiles_x;
    int tile_y = blockIdx.x / num_tiles_x;
    int block_m = tile_y * TILE_M;
    int block_n = tile_x * TILE_N;

    // Accumulators: each warp handles one 16x16 output tile
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Main loop over K dimension
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        int tile_k = t * TILE_K;

        // Load A tile: TILE_M x TILE_K
        // Cooperative loading by all threads
        int num_elements = TILE_M * TILE_K;
        for (int idx = tid; idx < num_elements; idx += 512) {
            int i = idx / TILE_K;
            int k = idx % TILE_K;
            int global_i = block_m + i;
            int global_k = tile_k + k;
            float val = (global_i < M && global_k < K) ? A[global_i * K + global_k] : 0.0f;
            As[i][k] = __float2half(val);
        }

        // Load B tile: TILE_K x TILE_N
        num_elements = TILE_K * TILE_N;
        for (int idx = tid; idx < num_elements; idx += 512) {
            int k = idx / TILE_N;
            int j = idx % TILE_N;
            int global_k = tile_k + k;
            int global_j = block_n + j;
            float val = (global_k < K && global_j < N) ? B[global_k * N + global_j] : 0.0f;
            Bs[k][j] = __float2half(val);
        }

        __syncthreads();

        // WMMA computation
        // Each warp computes one 16x16 output tile using 16x16x16 WMMA
        int warp_row = warp_m * WMMA_M;
        int warp_col = warp_n * WMMA_N;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        // Load fragments from shared memory and compute
        wmma::load_matrix_sync(a_frag, &As[warp_row][0], TILE_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_col], TILE_N);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // Store results back to global memory with ALPHA/BETA scaling
    int warp_row = warp_m * WMMA_M;
    int warp_col = warp_n * WMMA_N;

    // Store each warp's accumulator to shared memory
    wmma::store_matrix_sync(&C_smem[warp_row][warp_col], acc, TILE_N, wmma::mem_row_major);
    __syncthreads();

    // Cooperatively write to global memory with ALPHA/BETA scaling
    for (int idx = tid; idx < TILE_M * TILE_N; idx += 512) {
        int i = idx / TILE_N;
        int j = idx % TILE_N;
        int global_i = block_m + i;
        int global_j = block_n + j;

        if (global_i < M && global_j < N) {
            float val = ALPHA * C_smem[i][j];
            if (BETA != 0.0f) {
                val += BETA * C[global_i * N + global_j];
            }
            C[global_i * N + global_j] = val;
        }
    }
}
