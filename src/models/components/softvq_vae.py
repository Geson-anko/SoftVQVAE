r"""
ここではSoftVQ VAEを定義します。要件を満たす任意のVAEモジュールを使用可能です。

Defines SoftVQ VAE here. We can use any VAE that meets the requirements.
"""

import torch.nn as nn
from torch import Tensor


class SoftVQVAE(nn.Module):
    """Pure SoftVQ VAE class. The SoftVQ layer and VAE of this class satisfies following
    requirements.
    VAE:
        - Input is data batch.
        - Return value of `forward` pass is (x_hat, mean, logvar).

    SoftVQ:
        - Input is latent space data.
        - Return value of `forward` pass is (quantized, q_distribution)
    """

    def __init__(self, vae: nn.Module, softvq: nn.Module) -> None:
        super().__init__()

        self.softvq = softvq
        self.vae = vae

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Forward pass of SoftVQ VAE.
        Args:
            x (Tensor): Input data of vae.

        Returns:
            x_hat (Tensor): Reconstructed input data.
            mean (Tensor): Mean tensor of encoder output of vae
            logvar (Tensor): log variant of encoder output of vae
            quantized (Tensor): Soft quantized latent space (mean).
            q_dist (Tensor): Quantizing distributioin (Attention map).
        """

        x_hat, mean, logvar = self.vae(x)
        quantized, q_dist = self.softvq(mean.detach())
        return x_hat, mean, logvar, quantized, q_dist
