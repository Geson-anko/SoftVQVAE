import torch

from src.models.components import dense_image_vae as mod

width, height, channels = 28, 28, 1
input_size = width * height * channels
encoder_lin1_size = 256
encoder_lin2_size = 64
latent_size = 16
decoder_lin1_size = 64
decoder_lin2_size = 256
output_size = input_size


def test_Encoder():
    encoder = mod.Encoder(input_size, encoder_lin1_size, encoder_lin2_size, latent_size)

    dummy = torch.randn(1, input_size)
    out = encoder.forward(dummy)
    mean, logvar = out
    assert len(out) == 2
    assert isinstance(out, tuple)
    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], torch.Tensor)

    assert mean.shape == (1, latent_size)
    assert logvar.shape == (1, latent_size)


def test_Decoder():
    decoder = mod.Decoder(
        latent_size,
        decoder_lin1_size,
        decoder_lin2_size,
        input_size,
    )

    dummy = torch.randn(1, latent_size)
    out = decoder.forward(dummy)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, input_size)


def test_DenseImageVAE():
    vae = mod.DenseImageVAE(
        input_size,
        encoder_lin1_size,
        encoder_lin2_size,
        latent_size,
        decoder_lin1_size,
        decoder_lin2_size,
    )

    dummy = torch.randn(1, channels, width, height)
    out = vae.forward(dummy)
    assert len(out) == 3
    x_hat, mean, logvar = out
    assert isinstance(x_hat, torch.Tensor)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
    assert x_hat.shape == dummy.shape
    assert mean.shape == (1, latent_size)
    assert logvar.shape == (1, latent_size)
