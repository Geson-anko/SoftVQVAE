import torch
import torch.nn as nn
from torch import Tensor

from .vae_tools import sample


class Encoder(nn.Module):
    """Simple 3 layer vae encoder module."""

    def __init__(
        self,
        input_size: int,
        lin1_size: int,
        lin2_size: int,
        latent_size: int,
    ) -> None:
        super().__init__()

        self.main_layers = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(lin2_size, latent_size)
        self.logvar_layer = nn.Linear(lin2_size, latent_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns mean and logvar."""

        x = x.view(x.size(0), -1)
        h = self.main_layers(x)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)

        return mean, logvar


class Decoder(nn.Module):
    """Simple 3 layer vae decoder module."""

    def __init__(
        self,
        latent_size: int,
        lin1_size: int,
        lin2_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.main_layers = nn.Sequential(
            nn.Linear(latent_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        return self.main_layers(x)


class DenseImageVAEWithBN(nn.Module):
    """Variational AutoEncoder."""

    def __init__(
        self,
        input_size: int = 784,
        encoder_lin1_size: int = 256,
        encoder_lin2_size: int = 64,
        latent_size: int = 16,
        decoder_lin1_size: int = 64,
        decoder_lin2_size: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_size,
            encoder_lin1_size,
            encoder_lin2_size,
            latent_size,
        )

        self.decoder = Decoder(
            latent_size,
            decoder_lin1_size,
            decoder_lin2_size,
            input_size,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns reconstructed data, mean and logvar value."""
        h = x.view(x.size(0), -1)
        mean, logvar = self.encoder.forward(h)
        z = sample(mean, logvar)
        x_hat = self.decoder.forward(z).view(x.shape)

        return x_hat, mean, logvar


if __name__ == "__main__":
    _ = Encoder(784, 256, 64, 16)
    _ = Decoder(16, 64, 256, 784)
    _ = DenseImageVAEWithBN()
