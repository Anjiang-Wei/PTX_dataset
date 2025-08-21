# PTX_dataset

## Mirage
#### Example Equivalent CUDA code
https://github.com/Anjiang-Wei/mirage_ptx/tree/evaluation/dataset

#### Example Equivalent PTX code (all equivalent)
https://github.com/Anjiang-Wei/mirage_ptx/tree/evaluation/dataset_ptx

They are all equivalent, based on different schedules explored by superoptimization.

#### Generation method
First generate CUDA code based on the saved schedule for GQA kernel according to the AE readme https://github.com/mirage-project/mirage/blob/evaluation/ae.md
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gqa_bs1.json
```
Then lower to PTX with this script https://github.com/Anjiang-Wei/mirage_ptx/blob/evaluation/gen_ptx.py

## Cutlass

### Example Equivalent Pair of PTX
[GEMM 1](https://github.com/Anjiang-Wei/cutlass_ptx/tree/main/build/tools/library/generated_ptx/gemm/80/bf16_s16816gemm_bf16/cutlass_tensorop_bf16_s16816gemm_bf16_128x128_64x3_nt_align2.ptx)
[GEMM 2](https://github.com/Anjiang-Wei/cutlass_ptx/tree/main/build/tools/library/generated_ptx/gemm/80/bf16_s16816gemm_bf16/cutlass_tensorop_bf16_s16816gemm_bf16_128x128_64x3_nt_align4.ptx)

#### Example Generated CUDA code
https://github.com/Anjiang-Wei/cutlass_ptx/tree/main/build/tools/library/generated/gemm/80

but some may be MM with transpose, need to take a look at the filenames

#### Example PTX code
https://github.com/Anjiang-Wei/cutlass_ptx/tree/main/build/tools/library/generated_ptx/gemm/80

#### Generation method
When building Cutlass profiler, a lot of template will be instantiated with different parameters. During runtime, Cutlass profiler can thus search for many equivalent versions to find the best configuration https://github.com/Anjiang-Wei/cutlass_ptx/blob/main/media/docs/cpp/profiler.md
```
mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80" -DCUTLASS_LIBRARY_KERNELS=*gemm*  -DCUTLASS_UNITY_BUILD_ENABLED=ON
make cutlass_profiler -j
```
During compilation, the `.cu` files are saved in `build/tools/library/generated/gemm`. Then I create a script to compile those `.cu` files into PTX.

Usage:
```
cd build
./generate_ptx.py -j 20 --arch 80 -v
```

## Triton

### Example Pairs of equivalent PTX
[matmul0](https://github.com/Anjiang-Wei/PTX_dataset/blob/main/triton_ptx/gated_mlp/0/matmul_kernel.ptx)
[matmul1](https://github.com/Anjiang-Wei/PTX_dataset/blob/main/triton_ptx/gated_mlp/1/matmul_kernel.ptx) ([All matmuls](https://github.com/Anjiang-Wei/PTX_dataset/blob/main/triton_ptx/gated_mlp/))

[gqa0](https://github.com/Anjiang-Wei/PTX_dataset/blob/main/triton_ptx/gqa/0/_attn_fwd.ptx)
[gqa1](https://github.com/Anjiang-Wei/PTX_dataset/blob/main/triton_ptx/gqa/1/_attn_fwd.ptx) ([All GQAs](https://github.com/Anjiang-Wei/PTX_dataset/tree/main/triton_ptx/gqa))

To generate them, use auto-tuning from Triton:
```
python3 gated_mlp.py
python3 gqa.py
```

The helper function is `triton_ptx_dump.py`

## TVM
