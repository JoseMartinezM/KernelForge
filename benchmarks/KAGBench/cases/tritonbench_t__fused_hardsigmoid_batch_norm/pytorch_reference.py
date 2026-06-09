import torch.nn.functional as F


def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05, inplace=False):
    normalized_x = F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
    return F.hardsigmoid(normalized_x, inplace=inplace)
