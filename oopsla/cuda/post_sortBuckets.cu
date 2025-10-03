//pass
//--blockDim=1024 --gridDim=1024
// SOURCE: https://github.com/openmm/openmm/commit/9abaa587caf25b801b231fdd65bb58d9d662cead?diff=split?diff=split
#define DATA_TYPE int
#define MAX_VALUE 2147483647
#define KEY_TYPE double
__device__  double getValue(int arg);
/**
 * Sort the data in each bucket.
 */
__global__ void sortBuckets(DATA_TYPE* __restrict__ data, const DATA_TYPE* __restrict__ buckets, unsigned int numBuckets, const unsigned int* __restrict__ bucketOffset) {
    extern __shared__ DATA_TYPE dataBuffer[];
    for (unsigned int index = blockIdx.x; index < numBuckets; index += gridDim.x) {
        unsigned int startIndex = (index == 0 ? 0 : bucketOffset[index-1]);
        unsigned int endIndex = bucketOffset[index];
        unsigned int length = endIndex-startIndex;
        if (length <= blockDim.x) {
            // Load the data into local memory.

            if (threadIdx.x < length)
                dataBuffer[threadIdx.x] = buckets[startIndex+threadIdx.x];
            else
                dataBuffer[threadIdx.x] = MAX_VALUE;
            __syncthreads();

            // Perform a bitonic sort in local memory.

            for (unsigned int k = 2; k <= blockDim.x; k *= 2) {
                for (unsigned int j = k/2; j > 0; j /= 2) {
                    int ixj = threadIdx.x^j;
                    if (ixj > threadIdx.x) {
                        DATA_TYPE value1 = dataBuffer[threadIdx.x];
                        DATA_TYPE value2 = dataBuffer[ixj];
                        KEY_TYPE lowKey = ((threadIdx.x&k) == 0 ? getValue(value1) : getValue(value2));
                        KEY_TYPE highKey = ((threadIdx.x&k) == 0 ? getValue(value2) : getValue(value1));
                        if (lowKey > highKey) {
                            dataBuffer[threadIdx.x] = value2;
                            dataBuffer[ixj] = value1;
                        }
                    }
                    __syncthreads();
                }
            }

            // Write the data to the sorted array.

            if (threadIdx.x < length)
                data[startIndex+threadIdx.x] = dataBuffer[threadIdx.x];
            __syncthreads(); // <--- comment to trigger BUG
        }
        else {
            // Copy the bucket data over to the output array.

            for (unsigned int i = threadIdx.x; i < length; i += blockDim.x)
                data[startIndex+i] = buckets[startIndex+i];
#ifndef GPUVERIFY
#ifndef FAIAL
            __threadfence_block();
#endif
#endif
            __syncthreads();

            // Perform a bitonic sort in global memory.

            for (unsigned int k = 2; k < 2*length; k *= 2) {
                for (unsigned int j = k/2; j > 0; j /= 2) {
                    for (unsigned int i = threadIdx.x; i < length; i += blockDim.x) {
                        int ixj = i^j;
                        if (ixj > i && ixj < length) {
                            DATA_TYPE value1 = data[startIndex+i];
                            DATA_TYPE value2 = data[startIndex+ixj];
                            bool ascending = ((i&k) == 0);
                            for (unsigned int mask = k*2; mask < 2*length; mask *= 2)
                                ascending = ((i&mask) == 0 ? !ascending : ascending);
                            KEY_TYPE lowKey  = (ascending ? getValue(value1) : getValue(value2));
                            KEY_TYPE highKey = (ascending ? getValue(value2) : getValue(value1));
                            if (lowKey > highKey) {
                                data[startIndex+i] = value2;
                                data[startIndex+ixj] = value1;
                            }
                        }
                    }
#ifndef GPUVERIFY
#ifndef FAIAL
                    __threadfence_block();
#endif
#endif
                    __syncthreads();
                }
            }
        }
    }
}
