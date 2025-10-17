
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define M 2048
#define N 2048
#define K 2048
#define ALPHA 1.0f
#define BETA 0.0f

// Launch config for optimized kernel (must match opt.cu)
#define OPT_TILE_M 128
#define OPT_TILE_N 128
#define OPT_THREADS_X 32
#define OPT_THREADS_Y 8


    extern __global__ void sgemm_global_mem_coalesce(const float*,const float*,float*);
    extern __global__ void sgemm_optimized(const float*,const float*,float*);
    

#define CUDA_CHECK(x) do { cudaError_t e=x; if(e!=cudaSuccess){   fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// Baseline launcher: 1D block of BLOCKSIZE*BLOCKSIZE threads
template <typename F>
void launch_baseline(F kernel, const char* name, float* dA, float* dB, float* dC, int BLOCKSIZE) {
  dim3 grid((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE);
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  printf("Launching %s with grid=(%d,%d) block=(%d)\n",
         name, grid.x, grid.y, block.x);
  kernel<<<grid, block>>>(dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Optimized launcher: 32x8 threads with 16x4 register blocking, 128x128 tiles
template <typename F>
void launch_optimized(F kernel, const char* name, float* dA, float* dB, float* dC) {
  dim3 grid((N + OPT_TILE_N - 1) / OPT_TILE_N,
            (M + OPT_TILE_M - 1) / OPT_TILE_M);
  dim3 block(OPT_THREADS_X, OPT_THREADS_Y);
  printf("Launching %s with grid=(%d,%d) block=(%d,%d) tile=(%d,%d)\n",
         name, grid.x, grid.y, OPT_THREADS_X, OPT_THREADS_Y, OPT_TILE_M, OPT_TILE_N);
  kernel<<<grid, block>>>(dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(){
  printf("Comparing kernels on sm_80\n");
  size_t szA=M*K*sizeof(float), szB=K*N*sizeof(float), szC=M*N*sizeof(float);
  float *hA=(float*)malloc(szA),*hB=(float*)malloc(szB);
  for(int i=0;i<M*K;++i)hA[i]=(float)rand()/RAND_MAX;
  for(int i=0;i<K*N;++i)hB[i]=(float)rand()/RAND_MAX;

  float *dA,*dB,*dC1,*dC2;
  CUDA_CHECK(cudaMalloc(&dA,szA)); CUDA_CHECK(cudaMalloc(&dB,szB));
  CUDA_CHECK(cudaMalloc(&dC1,szC)); CUDA_CHECK(cudaMalloc(&dC2,szC));
  CUDA_CHECK(cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC1,0,szC)); CUDA_CHECK(cudaMemset(dC2,0,szC));

  // Launch baseline (BLOCKSIZE*BLOCKSIZE threads)
  launch_baseline(sgemm_global_mem_coalesce, "baseline", dA, dB, dC1, 32);

  // Launch optimized (32x8 threads with 16x4 register blocking, 128x128 tiles)
  launch_optimized(sgemm_optimized, "optimized", dA, dB, dC2);

  float *hC1=(float*)malloc(szC),*hC2=(float*)malloc(szC);
  CUDA_CHECK(cudaMemcpy(hC1,dC1,szC,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hC2,dC2,szC,cudaMemcpyDeviceToHost));

  // Relative L2 difference
  double diff=0,ref=0;
  for(int i=0;i<M*N;++i){double a=hC1[i],b=hC2[i];diff+=(a-b)*(a-b);ref+=a*a;}
  diff=sqrt(diff/(ref+1e-12));
  printf("Relative L2 diff = %.6e\n",diff);
  printf(diff<1e-4?"Equivalent.\n":"Different!\n");

  free(hA);free(hB);free(hC1);free(hC2);
  CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC1));CUDA_CHECK(cudaFree(dC2));
  return 0;
}
