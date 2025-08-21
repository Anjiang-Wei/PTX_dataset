# PTX_dataset

## Mirage
#### Example Equivalent CUDA code
https://github.com/Anjiang-Wei/mirage_ptx/tree/evaluation/dataset

#### Example Equivalent PTX code
https://github.com/Anjiang-Wei/mirage_ptx/tree/evaluation/dataset_ptx

They are all equivalent, based on different schedules explored by superoptimization.

#### Generation method
First generate CUDA code based on the saved schedule for GQA kernel according to the AE readme https://github.com/mirage-project/mirage/blob/evaluation/ae.md
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gqa_bs1.json
```
Then lower to PTX with this script https://github.com/Anjiang-Wei/mirage_ptx/blob/evaluation/gen_ptx.py

## Cutlass

#### Generation method
Cutlass profiler can generate equivalent CUDA code while searching for the best configuration https://github.com/Anjiang-Wei/cutlass_ptx/blob/main/media/docs/cpp/profiler.md
```
mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80" -DCUTLASS_LIBRARY_KERNELS=*gemm*  -DCUTLASS_UNITY_BUILD_ENABLED=ON
make cutlass_profiler -j
```


## Triton

How to generate PTX from Triton: https://github.com/triton-lang/triton/issues/2166

## TVM
