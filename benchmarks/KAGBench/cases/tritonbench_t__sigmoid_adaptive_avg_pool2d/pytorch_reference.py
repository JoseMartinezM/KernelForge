from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_adaptive_avg_pool2d(
    input: Tensor, output_size: Union[int, Tuple[int, int]]
) -> Tensor:
    pooled_output = F.adaptive_avg_pool2d(input, output_size)
    return torch.sigmoid(pooled_output)
