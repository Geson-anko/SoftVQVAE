import torch
import torch.nn as nn

from .softvq import SoftVectorQuantizing, softmax_with_temperature


class DotProductSoftVQ(SoftVectorQuantizing):
    r"""Dot product calculation of SoftVQ.
    `q_dist` is computed by dot product, not by measuring distance.
    This layer is created for variation of softvq layer of Straight SoftVQ VAE.
    So, `detach_distribution_grad` argument of :math:`forward` is default `False`.

    This class have following methods and attributes.
    Attributes:
    - :attr:`num_quantizing` - CodeBook size (Quantizing vector num.)
    - :attr:`quantizing_dim` - The vector dimension of CodeBook.
    - :attr:`temperature` - Softmax temperature.
    - :attr:`_weight` - CodeBook (Quantizing) weight.

    Methods:
    - :meth:`__init__` - Constructor.
    - :meth:`forward` - Forward process of Soft Vector Quantizing.
    - :meth:`compute_quantizing_distribution` - Compute quantizing distribution.
    """

    def __init__(
        self,
        num_quantizing: int,
        quantizing_dim: int,
        temperature: float = 1,
        *,
        mean: float = 0,
        std: float = 0.1,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__(num_quantizing, quantizing_dim, temperature, mean=mean, std=std, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, detach_distribution_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, detach_distribution_grad)

    def compute_quantizing_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantizing distribution.
        Args:
            x (torch.Tensor): Input tensor dimension must be 2.

        Returns:
            q_distribution (torch.Tensor): Quantizing distribution.

        Shape:
            x: (batch, quantizing_dim)
        """
        dot = torch.matmul(x, self._weight.T) / (self.quantizing_dim**0.5)
        return softmax_with_temperature(dot, self.temperature, dim=-1)
