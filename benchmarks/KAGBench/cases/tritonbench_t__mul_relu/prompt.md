Implement `mul_relu(input, other, inplace=False, out=None)`.

The function multiplies `input` by `other`, where `other` may be a tensor or a Python number, then applies the Rectified Linear Unit activation to the product. The returned tensor should contain zero wherever the product is negative and the product value wherever it is non-negative.

Support same-shape tensor operands, scalar `other` values, and normal PyTorch broadcasting for tensor operands. The `inplace` argument controls whether the activation step is requested in-place on the intermediate product; it must not modify `input`. The `out` argument is part of the public signature but is not required to alter the result semantics.
