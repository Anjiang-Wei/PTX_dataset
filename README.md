# PTX_dataset

## Mirage
#### Example Equivalent CUDA code:
https://github.com/Anjiang-Wei/mirage_ptx/tree/evaluation/dataset

#### Example Equivalent PTX code:
https://github.com/Anjiang-Wei/mirage_ptx/tree/evaluation/dataset_ptx


#### Generation method:
First generate CUDA code based on the saved schedule for GQA kernel according to the AE readme https://github.com/mirage-project/mirage/blob/evaluation/ae.md
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gqa_bs1.json
```
Then lower to PTX with this script https://github.com/Anjiang-Wei/mirage_ptx/blob/evaluation/gen_ptx.py
