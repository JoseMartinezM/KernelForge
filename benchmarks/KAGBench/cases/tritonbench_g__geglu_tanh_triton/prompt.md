# GEGLU tanh activation

Implement the public GEGLU forward and backward APIs in `geglu_tanh_triton.py`.

```python
def geglu_forward(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...

def geglu_backward(a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Inputs used by the tests are contiguous two-dimensional CUDA tensors with shape `(batch_size, 128)` and dtype `torch.float32`.

`geglu_forward(a, b)` returns `(a_out, b_out, c)`:

- `a_out` and `b_out` must contain the original values of `a` and `b` for the tested shapes.
- The input tensors passed to `geglu_forward` must not be modified in place.
- `c` has the same shape, dtype, and device as `a` and `b` and is computed elementwise as:

```python
gelu_tanh(a) * b
```

where:

```python
gelu_tanh(a) = 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a**3)))
```

`geglu_backward(a, b, dc)` returns `(da, db)`, the gradients of `c` with respect to `a` and `b` for upstream gradient `dc`. The returned tensors must have the same shape, dtype, and device as the inputs. Implement the analytic derivative of the formula above. The backward API may overwrite the tensors passed as `a` and `b` before returning the gradients, so callers that need the original values should pass clones.

Test cases include negative values and multiple batch sizes, including batch size 1.
