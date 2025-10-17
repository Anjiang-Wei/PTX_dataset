#!/usr/bin/env python3
import os, subprocess, sys, textwrap, shutil

# --- Configuration ---
M, N, K = 1024, 1024, 1024
ALPHA, BETA = 1.0, 0.0
BLOCKSIZE = 32  # used by baseline kernel only (hard-coded into base_full.cu)
SM_ARCH = "sm_80"

# Launch params for the optimized kernel (must match opt.cu hyperparams)
OPT_TILE_M = 64
OPT_TILE_N = 64
OPT_THREADS_X = 32
OPT_THREADS_Y = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SRC_FILE = os.path.join(SCRIPT_DIR, "base.cu")
OPT_SRC_FILE  = os.path.join(SCRIPT_DIR, "opt.cu")
OUTDIR = os.path.join(SCRIPT_DIR, "build")

BASE_WRAPPER = os.path.join(OUTDIR, "base_full.cu")
OPT_WRAPPER  = os.path.join(OUTDIR, "opt_full.cu")
BASE_PTX = os.path.join(OUTDIR, "base.ptx")
OPT_PTX  = os.path.join(OUTDIR, "opt.ptx")
EXE = os.path.join(OUTDIR, "compare_kernels")


HARNESS_TEMPLATE = """
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define M {M}
#define N {N}
#define K {K}
#define ALPHA {ALPHA}f
#define BETA {BETA}f

// Launch config for optimized kernel (must match opt.cu)
#define OPT_TILE_M {OPT_TILE_M}
#define OPT_TILE_N {OPT_TILE_N}
#define OPT_THREADS_X {OPT_THREADS_X}
#define OPT_THREADS_Y {OPT_THREADS_Y}

{KERNEL_DECLS}

#define CUDA_CHECK(x) do {{ cudaError_t e=x; if(e!=cudaSuccess){{ \
  fprintf(stderr,"CUDA error %s:%d: %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}} }}while(0)

// Baseline launcher: 1D block of BLOCKSIZE*BLOCKSIZE threads
template <typename F>
void launch_baseline(F kernel, const char* name, float* dA, float* dB, float* dC, int BLOCKSIZE) {{
  dim3 grid((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE);
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  printf("Launching %s with grid=(%d,%d) block=(%d)\\n",
         name, grid.x, grid.y, block.x);
  kernel<<<grid, block>>>(dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
}}

// Optimized launcher: 2D block (OPT_THREADS_X, OPT_THREADS_Y), 128x128 tiles
template <typename F>
void launch_optimized(F kernel, const char* name, float* dA, float* dB, float* dC) {{
  dim3 grid((N + OPT_TILE_N - 1) / OPT_TILE_N,
            (M + OPT_TILE_M - 1) / OPT_TILE_M);
  dim3 block(OPT_THREADS_X, OPT_THREADS_Y);
  printf("Launching %s with grid=(%d,%d) block=(%d,%d) tile=(%d,%d)\\n",
         name, grid.x, grid.y, block.x, block.y, OPT_TILE_M, OPT_TILE_N);
  kernel<<<grid, block>>>(dA, dB, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
}}

int main(){{
  printf("Comparing kernels on {SM_ARCH}\\n");
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
  launch_baseline(sgemm_global_mem_coalesce, "baseline", dA, dB, dC1, {BLOCKSIZE});

  // Launch optimized (32x8 threads, 128x128 tiles)
  launch_optimized(sgemm_optimized, "optimized", dA, dB, dC2);

  float *hC1=(float*)malloc(szC),*hC2=(float*)malloc(szC);
  CUDA_CHECK(cudaMemcpy(hC1,dC1,szC,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hC2,dC2,szC,cudaMemcpyDeviceToHost));

  // Relative L2 difference
  double diff=0,ref=0;
  for(int i=0;i<M*N;++i){{double a=hC1[i],b=hC2[i];diff+=(a-b)*(a-b);ref+=a*a;}}
  diff=sqrt(diff/(ref+1e-12));
  printf("Relative L2 diff = %.6e\\n",diff);
  printf(diff<1e-4?"Equivalent.\\n":"Different!\\n");

  free(hA);free(hB);free(hC1);free(hC2);
  CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC1));CUDA_CHECK(cudaFree(dC2));
  return 0;
}}
"""

def wrap_kernel(src, out):
    with open(src) as f: body = f.read().strip()
    header = f"""
    #include <cuda_runtime.h>
    #define M {M}
    #define N {N}
    #define K {K}
    #define ALPHA {ALPHA}f
    #define BETA {BETA}f
    const int BLOCKSIZE = {BLOCKSIZE}; // used only by baseline kernel
    """
    with open(out,"w") as f: f.write(textwrap.dedent(header)+body+"\n")

def write_harness():
    os.makedirs(OUTDIR,exist_ok=True)
    decls = """
    extern __global__ void sgemm_global_mem_coalesce(const float*,const float*,float*);
    extern __global__ void sgemm_optimized(const float*,const float*,float*);
    """
    with open(os.path.join(OUTDIR,"harness.cu"),"w") as f:
        f.write(HARNESS_TEMPLATE.format(
            M=M, N=N, K=K, ALPHA=ALPHA, BETA=BETA,
            BLOCKSIZE=BLOCKSIZE, SM_ARCH=SM_ARCH, KERNEL_DECLS=decls,
            OPT_TILE_M=OPT_TILE_M, OPT_TILE_N=OPT_TILE_N,
            OPT_THREADS_X=OPT_THREADS_X, OPT_THREADS_Y=OPT_THREADS_Y))

def nvcc_compile():
    subprocess.check_call(["nvcc","-arch="+SM_ARCH,"-O3","-ptx",BASE_WRAPPER,"-o",BASE_PTX])
    subprocess.check_call(["nvcc","-arch="+SM_ARCH,"-O3","-ptx",OPT_WRAPPER,"-o",OPT_PTX])
    subprocess.check_call([
        "nvcc","-arch="+SM_ARCH,"-O3",
        os.path.join(OUTDIR,"harness.cu"),BASE_WRAPPER,OPT_WRAPPER,
        "-o",EXE])

def main():
    if not shutil.which("nvcc"):
        sys.exit("nvcc not found in PATH.")
    os.makedirs(OUTDIR,exist_ok=True)
    wrap_kernel(BASE_SRC_FILE,BASE_WRAPPER)
    wrap_kernel(OPT_SRC_FILE,OPT_WRAPPER)
    write_harness()
    nvcc_compile()
    print("\n=== Running binary ===")
    subprocess.check_call([EXE])

if __name__=="__main__":
    main()
