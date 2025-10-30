// Baseline SGEMM kernel using 1D grid and 1D blocks
// Same launch config as optimized (512 threads, 128x128 tiles) but simpler implementation
// No shared memory, no register blocking - just direct global memory access
__global__ void sgemm_global_mem_coalesce(const float *A, const float *B, float *C) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int BLK_X = 32;   // Logical threads in X dimension
    constexpr int BLK_Y = 16;   // Logical threads in Y dimension (32*16 = 512 threads)
    constexpr int REG_M = 8;    // Each thread computes 8 rows
    constexpr int REG_N = 4;    // Each thread computes 4 cols

    // 1D thread index -> 2D logical position
    int tid = threadIdx.x;
    int tx = tid % BLK_X;
    int ty = tid / BLK_X;

    // 1D block index -> 2D tile position
    int num_tiles_x = (N + TILE_N - 1) / TILE_N;
    int tile_x = blockIdx.x % num_tiles_x;
    int tile_y = blockIdx.x / num_tiles_x;
    int blockM = tile_y * TILE_M;
    int blockN = tile_x * TILE_N;

    // Thread's output position within the tile
    int threadM = ty * REG_M;
    int threadN = tx * REG_N;

    // Compute results directly from global memory
    for (int i = 0; i < REG_M; ++i) {
        int row = blockM + threadM + i;
        if (row >= M) continue;

        for (int j = 0; j < REG_N; ++j) {
            int col = blockN + threadN + j;
            if (col >= N) continue;

            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = ALPHA * sum + BETA * C[row * N + col];
        }
    }
}
