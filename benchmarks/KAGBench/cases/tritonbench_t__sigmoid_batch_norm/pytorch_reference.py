import torch
import torch.nn.functional as F


def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    normalized_input = F.batch_norm(input, running_mean, running_var, weight, bias, training=training, momentum=momentum, eps=eps)
    return torch.sigmoid(normalized_input)
