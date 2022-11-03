import torch

from src.models.components.straight_softvq_vae import StraightSoftVQVAE
from src.models.components.dense_image_vae import Encoder, Decoder
from src.models.components.softvq import SoftVectorQuantizing

batch_size = 16
input_size = 784
lin1sz = 256
lin2sz = 64
latent_size = 16
num_quantizing = 10


def test__init__():
    cls = StraightSoftVQVAE

    encoder = Encoder(input_size, lin1sz, lin2sz, latent_size)
    decoder = Decoder(latent_size, lin2sz, lin1sz, input_size)
    softvq = SoftVectorQuantizing(num_quantizing, latent_size)

    ssvqvae = cls(encoder, decoder, softvq)

    assert ssvqvae.encoder is encoder
    assert ssvqvae.decoder is decoder
    assert ssvqvae.softvq is softvq


def test_forward():
    cls = StraightSoftVQVAE

    encoder = Encoder(input_size, lin1sz, lin2sz, latent_size)
    decoder = Decoder(latent_size, lin2sz, lin1sz, input_size)
    softvq = SoftVectorQuantizing(num_quantizing, latent_size)

    ssvqvae = cls(encoder, decoder, softvq)
    dummy = torch.randn(batch_size, input_size)

    outputs = ssvqvae.forward(dummy)

    assert len(outputs) == 5
    x_hat, mean, logvar, quantized, q_dist = outputs
    assert isinstance(x_hat, torch.Tensor)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
    assert isinstance(quantized, torch.Tensor)
    assert isinstance(q_dist, torch.Tensor)
    assert x_hat.shape == (batch_size, input_size)
    assert mean.shape == (batch_size, latent_size)
    assert logvar.shape == (batch_size, latent_size)
    assert quantized.shape == (batch_size, latent_size)
    assert q_dist.shape == (batch_size, num_quantizing)
