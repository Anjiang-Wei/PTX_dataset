//pass
//--gridDim=1 --blockDim=1024
// SOURCE: https://github.com/openmm/openmm/commit/65caf22b4413262c1fb76d420c9bc71caeaa3088
#define DATA_TYPE int
#define KEY_TYPE int
__device__ KEY_TYPE MAX_KEY;
__device__ KEY_TYPE MIN_KEY;
__device__ KEY_TYPE SORT_KEY;
#define getValue(value) SORT_KEY

/**
 * Calculate the minimum and maximum value in the array to be sorted.  This kernel
 * is executed as a single work group.
 */
__global__ void computeRange(const DATA_TYPE* __restrict__ data, unsigned int length, KEY_TYPE* __restrict__ range) {
    extern __shared__ KEY_TYPE rangeBuffer[];
    KEY_TYPE minimum = MAX_KEY;
    KEY_TYPE maximum = MIN_KEY;

    // Each thread calculates the range of a subset of values.

    for (unsigned int index = threadIdx.x; index < length; index += blockDim.x) {
        KEY_TYPE value = getValue(data[index]);
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }

    // Now reduce them.

    rangeBuffer[threadIdx.x] = minimum;
    __syncthreads();
    for (unsigned int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdx.x+step < blockDim.x && threadIdx.x%(2*step) == 0)
            rangeBuffer[threadIdx.x] = min(rangeBuffer[threadIdx.x], rangeBuffer[threadIdx.x+step]);
        __syncthreads();
    }
    minimum = rangeBuffer[0];
    //__syncthreads(); // <--- BUG
    rangeBuffer[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdx.x+step < blockDim.x && threadIdx.x%(2*step) == 0)
            rangeBuffer[threadIdx.x] = max(rangeBuffer[threadIdx.x], rangeBuffer[threadIdx.x+step]);
        __syncthreads();
    }
    maximum = rangeBuffer[0];
    if (threadIdx.x == 0) {
        range[0] = minimum;
        range[1] = maximum;
    }
}