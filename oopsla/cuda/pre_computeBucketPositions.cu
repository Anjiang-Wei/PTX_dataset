//pass
//--blockDim=1024 --gridDim=1
// SOURCE: https://github.com/openmm/openmm/commit/9abaa587caf25b801b231fdd65bb58d9d662cead?diff=split?diff=split

/**
 * Sum the bucket sizes to compute the start position of each bucket.  This kernel
 * is executed as a single work group.
 */
__global__ void computeBucketPositions(unsigned int numBuckets, unsigned int* __restrict__ bucketOffset) {
    extern __shared__ unsigned int posBuffer[];
    unsigned int globalOffset = 0;
    for (unsigned int startBucket = 0; startBucket < numBuckets; startBucket += blockDim.x) {
        // Load the bucket sizes into local memory.

        unsigned int globalIndex = startBucket+threadIdx.x;
        //__syncthreads(); // <- comment to obtain data-race
        posBuffer[threadIdx.x] = (globalIndex < numBuckets ? bucketOffset[globalIndex] : 0);
        __syncthreads();

        // Perform a parallel prefix sum.

        for (unsigned int step = 1; step < blockDim.x; step *= 2) {
            unsigned int add = (threadIdx.x >= step ? posBuffer[threadIdx.x-step] : 0);
            __syncthreads();
            posBuffer[threadIdx.x] += add;
            __syncthreads();
        }

        // Write the results back to global memory.

        if (globalIndex < numBuckets)
            bucketOffset[globalIndex] = posBuffer[threadIdx.x]+globalOffset;
        globalOffset += posBuffer[blockDim.x-1];
    }
}
