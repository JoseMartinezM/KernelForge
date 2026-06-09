import torch.nn.functional as F


def softmax_mul(input, other, dim, dtype=None, out=None):
    result = F.softmax(input, dim=dim, dtype=dtype) * other
    if out is not None:
        out.copy_(result)
        return out
    return result
