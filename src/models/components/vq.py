import torch
import torch.nn as nn
import torch.nn.functional as F

from .softvq import SoftVectorQuantizing


class VectorQuantizing(SoftVectorQuantizing):
    r"""Vector Quantizing Layer class. Gradient is computed by straight through estimator.

    This class have following methods and attributes.
    Attributes:
    - :attr:`num_quantizing` - CodeBook size (Quantizing vector num.)
    - :attr:`quantizing_dim` - The vector dimension of quantizing.

    Methods:
    - :meth:`__init__` - Constructor.
    - :meth:`forward` - Forward process of Soft Vector Quantizing.
    - :meth:`compute_quantizing_distribution` - Compute quantizing distribution.
    """

    def __init__(
        self,
        num_quantizing: int,
        quantizing_dim: int,
        *args,
        mean: float = 0.0,
        std: float = 0.1,
        dtype=None,
        device=None,
        **kwds,
    ) -> None:
        """
        Args:
            num_quantizing (int): The number of quantizing vectors.
            quantizing_dim (int): The dimension number of quantizing vectors.
            mean (float): The mean value of weight when initializing.
            std (float): The standard deviation value of weight when initializing.
            dtype (Any): Data type.
            device (Any): Device.
        """
        super().__init__(num_quantizing, quantizing_dim, 1e-8, mean=mean, std=std, device=device, dtype=dtype)

        self.proj = nn.Linear(quantizing_dim, num_quantizing, bias=False, device=device, dtype=dtype)

    def compute_quantizing_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantizing distribution.
        Args:
            x (torch.Tensor): Input tensor dimension must be 2.

        Returns:
            q_distribution (torch.Tensor): Quantizing distribution.
        """
        prob = torch.softmax(self.proj(x), dim=-1)
        q_dist = (
            F.one_hot(torch.argmax(prob, dim=-1), num_classes=self.num_quantizing) + prob - prob.detach()
        )  # Straight through estimator.
        return q_dist

    def forward(self, x: torch.Tensor, detach_distribution_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, detach_distribution_grad)
