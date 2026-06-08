import torch


_SQRT_2_OVER_PI = 0.7978845608028654
_GELU_TANH_COEFF = 0.044715


def _gelu_tanh(a: torch.Tensor) -> torch.Tensor:
    z = _SQRT_2_OVER_PI * (a + _GELU_TANH_COEFF * a * a * a)
    return 0.5 * a * (1.0 + torch.tanh(z))


def _gelu_tanh_grad(a: torch.Tensor) -> torch.Tensor:
    z = _SQRT_2_OVER_PI * (a + _GELU_TANH_COEFF * a * a * a)
    tanh_z = torch.tanh(z)
    dz_da = _SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_TANH_COEFF * a * a)
    return 0.5 * (1.0 + tanh_z) + 0.5 * a * (1.0 - tanh_z * tanh_z) * dz_da


def geglu_forward(a: torch.Tensor, b: torch.Tensor):
    c = _gelu_tanh(a) * b
    return a, b, c


def geglu_backward(a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor):
    gelu_a = _gelu_tanh(a)
    da = dc * b * _gelu_tanh_grad(a)
    db = dc * gelu_a
    return da, db
