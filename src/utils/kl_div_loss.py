import torch
from typing import Literal


def kl_div_loss(mean: torch.Tensor, logvar: torch.Tensor, reduction: Literal["sum", "mean"] = "mean") -> torch.Tensor:
    """compute loss `Kullback-leibler divergence` from Normal Distribution.
    `mean` and `logvar` must be same shape, and their first dim is treated
    as `batch_size`. `reduction` works along batch axis.

    If reduction is unknow option, raises ValueError.
    """
    bsz = mean.size(0)
    logvar = logvar.view(bsz, -1)
    mean = mean.view(bsz, -1)
    L = -0.5 * (1.0 + logvar - mean * mean - torch.exp(logvar)).sum(dim=-1)

    match reduction:
        case "sum":
            return L.sum()
        case "mean":
            return L.mean()
        case _:
            raise ValueError(f"Unsupported option of reduction: '{reduction}'")
