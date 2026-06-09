import torch
import torch.nn.functional as F


def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    output = torch.matmul(input, weight.T)
    if bias is not None:
        output = output + bias
    return F.log_softmax(output, dim=dim, dtype=dtype)
