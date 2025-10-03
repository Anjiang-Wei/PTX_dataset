#pragma once
#include <cuda_runtime.h>

// --- float3 arithmetic operators ---

__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
	    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
	    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3& operator+=(float3 &a, const float3 &b) {
	    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}

__host__ __device__ inline float dot(const float3 &a, const float3 &b) {
	    return a.x*b.x + a.y*b.y + a.z*b.z;
}

