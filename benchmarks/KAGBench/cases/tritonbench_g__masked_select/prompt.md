# masked_select

Implement `masked_select.py` with the following public API:

```python
def masked_select(inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ...
```

`masked_select` receives an input tensor `inp` and a boolean tensor `mask`. The shapes of `inp` and `mask` must follow normal PyTorch broadcasting rules.

Expected behavior:

- Broadcast `inp` and `mask` as PyTorch would for `torch.masked_select`.
- Return a one-dimensional tensor containing the broadcasted `inp` elements where the broadcasted `mask` is `True`.
- Preserve the same element order as `torch.masked_select(inp, mask)`.
- The output dtype and device must match `inp`.
- The output length is the number of `True` entries in the broadcasted mask, so the result may be empty.
- Do not modify `inp` or `mask` in place.
