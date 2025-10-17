
#include <cuda_runtime.h>
#define M 2048
#define N 2048
#define K 2048
#define ALPHA 1.0f
#define BETA 0.0f
const int BLOCKSIZE = 32; // used only by baseline kernel
__global__ void sgemm_global_mem_coalesce(const float *A, const float *B, float *C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = ALPHA * tmp + BETA * C[cRow * N + cCol];
    }
}
