# Task

Implement `sigmoid_adaptive_avg_pool2d(input, output_size)`.

The function accepts a 4D PyTorch tensor in NCHW layout and an adaptive average pooling output size. It returns the result of applying 2D adaptive average pooling followed by an elementwise sigmoid.

Behavior:
1. Compute `torch.nn.functional.adaptive_avg_pool2d(input, output_size)`.
2. Return `torch.sigmoid` of the pooled tensor.

`output_size` may be an integer or a pair of integers such as `(2, 2)` or `(4, 4)`. Preserve the input dtype and device behavior of the underlying PyTorch operations.
