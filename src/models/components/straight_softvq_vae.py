from typing import Tuple

import torch.nn as nn
from torch import Tensor

from .dense_image_vae import sample


class StraightSoftVQVAE(nn.Module):
    """Straight SoftVQ VAE class.
    This model data flow is,

        X - [Encoder] -> mean, logvar - [sample] -> z
            - [SoftVQ] -> quantized - [Decoder] -> X_hat

    So, don't need to compute quantizing loss.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, softvq: nn.Module) -> None:
        super().__init__()
        """Constructor.
        Args:
            encoder (nn.Module): This forward pass must return `mean` and `logvar`.
            decoder (nn.Module): decoder of vae.
            softvq (nn.Module): The args of this forward pass must have `detach_distribution_grad` option.
        """

        self.encoder = encoder
        self.decoder = decoder
        self.softvq = softvq

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass of Straight SoftVQ VAE
        Args:
            x (Tensor): Input data of vae.

        Returns:
            x_hat (Tensor): Reconstructed input data/
            mean (Tensor): Mean tensor of encoder output of vae
            logvar (Tensor): log variant of encoder output of vae
            q_dist (Tensor): Quantizing distribution (Attention map).

        Shape:
            x: (batch, *)
            x_hat: same shape of x
            mean: (*, latent_size)
            logvar: sames shape of mean.
            q_dist: (*, num_quantizing)
        """

        mean, logvar = self.encoder(x)
        z = sample(mean, logvar)
        quantized, q_dist = self.softvq(z, detach_distribution_grad=False)
        x_hat = self.decoder(quantized).view(x.shape)
        return x_hat, mean, logvar, quantized, q_dist
