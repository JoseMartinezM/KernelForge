import torch.nn.functional as F


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    linear_out = F.linear(input, weight, bias)
    return F.softplus(linear_out, beta=beta, threshold=threshold)
