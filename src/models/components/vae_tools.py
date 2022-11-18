import copy

import torch
import torch.nn as nn
from torch import Tensor


def sample(mean: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(logvar / 2)
    return mean + torch.randn_like(std) * std


class MeanAndLogvar(nn.Module):
    r"""This class has 2 output heads for vae encoder mean and logvar."""

    def __init__(self, layer: nn.Module):
        r"""
        Argument `layer` becomes :attr:`mean_layer`, and deep copied it becomes :attr:`logvar_layer`.

        Args:
            layer: base layer.
        """
        super().__init__()
        self.mean_layer = layer
        self.logvar_layer = copy.deepcopy(layer)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""
        Args:
            x (Tensor): Input data.

        Returns:
            mean (Tensor): Output of :attr:`mean_layer(x)`
            logvar (Tensor): Output of :attr:`logvar_layer(x)`
        """
        return self.mean_layer(x), self.logvar_layer(x)


class VariationalAutoEncoder(nn.Module):
    """The frame class of Variational AutoEncoder."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """
        Args:
            encoder (nn.Module): Encoder instance of VAE. The forward pass
                must return `mean` and `logvar` Tensor.
            decoder (nn.Module): Decoder instance of VAE.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input for encoder.
        Returns:
            x_hat (Tensor): Reconstructed input data.
            mean (Tensor): `mean` for gaussian distribution.
            logvar (Tensor): `logvar` for gaussian distribution.
        """

        mean, logvar = self.encoder(x)
        z = sample(mean, logvar)
        x_hat = self.decoder(z)

        return x_hat, mean, logvar
