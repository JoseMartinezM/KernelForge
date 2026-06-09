import torch
import torch.nn.functional as F


def mul_relu(input, other, inplace=False, out=None):
    result = torch.mul(input, other)
    return F.relu(result, inplace=inplace)
