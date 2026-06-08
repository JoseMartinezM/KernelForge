# cosine_compute

Implement `cosine_compute.py` with the following public API:

```python
def cos(A):
    ...
```

`cos` receives a CUDA `torch.Tensor` `A` and returns a new tensor containing the cosine of every element of `A`.

Requirements:

- The output must have the same shape as `A`.
- The output must have the same dtype as `A`.
- The input tensor must not be modified.
- The function should support ordinary contiguous tensors with arbitrary dimensional shapes.
- The cosine computation should be elementwise over all values in the input tensor.
