//pass
//--blockDim=1024 --gridDim=1
// SOURCE: https://github.com/openmm/openmm/commit/de666e305b61a1cff0bfc7d7f51c23de5d13ff43#diff-61d236e0ffbb53c3b2b72982769e03a93b3b668ba5323d514a51c2465acc58c0
#define SYNC_WARPS __syncthreads();

//inline __device__ int3 trimTo3(int4 v);
inline __device__ float3 trimTo3(float4 v) {
    return make_float3(v.x, v.y, v.z);
}
/**
 * Sum a value over all threads.
 */
__device__ float reduceValue(float value, volatile float* temp) {
    const int thread = threadIdx.x;
    __syncthreads(); // <-- uncomment to trigger BUG
    temp[thread] = value;
    __syncthreads();
    for (uint step = 1; step < 32; step *= 2) {
        if (thread+step < blockDim.x && thread%(2*step) == 0)
            temp[thread] = temp[thread] + temp[thread+step];
        SYNC_WARPS
    }
    for (uint step = 32; step < blockDim.x; step *= 2) {
        if (thread+step < blockDim.x && thread%(2*step) == 0)
            temp[thread] = temp[thread] + temp[thread+step];
        __syncthreads();
    }
    return temp[0];
}

/**
 * Perform the first step of computing the RMSD.  This is executed as a single work group.
 */
extern "C" __global__ void computeRMSDPart1(int numParticles, const float4* __restrict__ posq, const float4* __restrict__ referencePos,
         const int* __restrict__ particles, float* buffer) {
    extern __shared__ volatile float temp[];

    // Compute the center of the particle positions.
    
    float3 center = make_float3(0);
    for (int i = threadIdx.x; i < numParticles; i += blockDim.x)
        center += trimTo3(posq[particles[i]]);
    center.x = reduceValue(center.x, temp)/numParticles;
    center.y = reduceValue(center.y, temp)/numParticles;
    center.z = reduceValue(center.z, temp)/numParticles;
    // Compute the correlation matrix.
    
    float R[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    float sum = 0;
    for (int i = threadIdx.x; i < numParticles; i += blockDim.x) {
        int index = particles[i];
        float3 pos = trimTo3(posq[index]) - center;
        float3 refPos = trimTo3(referencePos[index]);
        R[0][0] += pos.x*refPos.x;
        R[0][1] += pos.x*refPos.y;
        R[0][2] += pos.x*refPos.z;
        R[1][0] += pos.y*refPos.x;
        R[1][1] += pos.y*refPos.y;
        R[1][2] += pos.y*refPos.z;
        R[2][0] += pos.z*refPos.x;
        R[2][1] += pos.z*refPos.y;
        R[2][2] += pos.z*refPos.z;
        sum += dot(pos, pos);
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = reduceValue(R[i][j], temp);
    sum = reduceValue(sum, temp);

    // Copy everything into the output buffer to send back to the host.
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                buffer[3*i+j] = R[i][j];
        buffer[9] = sum;
        buffer[10] = center.x;
        buffer[11] = center.y;
        buffer[12] = center.z;
    }
    return;
}
