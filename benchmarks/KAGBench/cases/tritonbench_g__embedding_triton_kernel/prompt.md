# embedding

Implement `embedding_triton_kernel.py` with the following public API:

```python
def embedding(input_ids, weight: torch.Tensor, vob_start_id, vob_end_id, out: torch.Tensor):
    ...
```

`input_ids` is a one-dimensional CUDA integer tensor. `weight` is a two-dimensional CUDA floating-point tensor. `out` is a CUDA floating-point tensor with shape `(input_ids.numel(), weight.shape[1])` and the same dtype as `weight`.

Expected behavior:

- Fill `out` in place, one row per element of `input_ids`.
- For each position `i`, read `token_id = input_ids[i]`.
- If `vob_start_id <= token_id < vob_end_id`, write `weight[token_id - vob_start_id]` into `out[i]`.
- Otherwise, write zeros into `out[i]`.
- The return value is ignored by the tests; correctness is determined by the final contents of `out`.
- Do not modify `input_ids` or `weight`.