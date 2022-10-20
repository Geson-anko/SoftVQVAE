import torch
import torch.nn as nn


def softmax_with_temperature(tensor: torch.Tensor, temperature: float, dim: int, dtype=None) -> torch.Tensor:
    """Computing softmax with temperature scaling.
    Args:
        tensor (torch.Tensor): Input tensor.
        temperature (float): Scaling value.
        dim (int): Softmax dimension.
        dtype (Any): Output data type.

    Returns:
        output (torch.Tensor): Output tensor.

    Raises:
        ZeroDivisionError: It was raised when temperature is zero.
    """

    return torch.softmax(tensor / temperature, dim, dtype=dtype)


class SoftVectorQuantizing(nn.Module):
    r"""Soft Vector Quantizing Layer class.

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
        temperature: float = 1.0,
        *,
        mean: float = 0.0,
        std: float = 1.0,
        dtype=None,
        device=None,
    ) -> None:
        """Construct this layer and set CodeBook (Quantizing) weight. CodeBook weight is
        initialized by normal distribution with `mean` and `std`.

        Args:
            num_quantizing (int): The number of quantizing vectors.
            quantizing_dim (int): The dimension number of quantizing vectors.
            temperature (float): Temperature of softmax.
            mean (float): The mean value of weight when initializing.
            std (float): The standard deviation value of weight when initializing.
            dtype (Any): Data type.
            device (Any): Device.
        """
        super().__init__()

        self.num_quantizing = num_quantizing
        self.quantizing_dim = quantizing_dim
        self.temperature = temperature

        self._weight = nn.Parameter(
            torch.empty((num_quantizing, quantizing_dim), dtype=dtype, device=device),
            requires_grad=True,
        )
        nn.init.normal_(self._weight, mean=mean, std=std)

    def forward(self, x: torch.Tensor, detach_distributioin_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process of Soft Vector Quantizing.
        Args:
            x (torch.Tensor): The last dimension of input tensor must be `self.quantizing_dim`.

        Returns:
            quantizied (torch.Tensor): Soft vector quantized input tensor.
            q_distribution (torch.Tensor): Attention map of quantizing vector.
        """

        input_shape = x.shape
        h = x.view(-1, self.quantizing_dim)

        q_distribution = self.compute_quantizing_distribution(h)
        if detach_distributioin_grad:
            q_distribution = q_distribution.detach()
        quantized = torch.matmul(q_distribution, self._weight)

        return (
            quantized.view(input_shape),
            q_distribution.view((*input_shape[:-1], self.num_quantizing)),
        )

    def compute_quantizing_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantizing distribution.
        Args:
            x (torch.Tensor): Input tensor dimension must be 2.

        Returns:
            q_distribution (torch.Tensor): Quantizing distribution.
        """
        delta = self._weight.unsqueeze(0) - x.unsqueeze(1)  # Broadcasting to (B, Q, E)
        distance = -torch.mean(delta * delta, dim=-1)
        return softmax_with_temperature(distance, self.temperature, dim=-1)
