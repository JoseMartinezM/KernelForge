import torch.nn.functional as F


def add_gelu(input, other, alpha=1, approximate='none', out=None):
    result = F.gelu(input + alpha * other, approximate=approximate)
    if out is not None:
        out.copy_(result)
        return out
    return result
