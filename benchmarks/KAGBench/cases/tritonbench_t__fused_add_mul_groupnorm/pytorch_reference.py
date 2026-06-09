import torch
import torch.nn.functional as F


def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-05, *, out=None):
    z = torch.add(input1, input2)
    m = torch.mul(z, input2)
    result = F.group_norm(m, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
    if out is not None:
        out.copy_(result)
        return out
    return result
