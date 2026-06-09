import torch


def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    z = torch.mv(input, vec)
    s = torch.sigmoid(z)
    y = torch.sub(s, other, alpha=alpha)
    if out is not None:
        out.copy_(y)
        return out
    return y
