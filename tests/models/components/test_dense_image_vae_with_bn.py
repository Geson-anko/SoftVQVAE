import torch

from src.models.components import dense_image_vae_with_bn as mod

batch_size = 8
width, height, channels = 32, 32, 1
input_size = width * height * channels
encoder_lin1_size = 512
encoder_lin2_size = 128
latent_size = 32
decoder_lin1_size = 128
decoder_lin2_size = 512
output_size = input_size


def test_Encoder():
    encoder = mod.Encoder(input_size, encoder_lin1_size, encoder_lin2_size, latent_size)

    dummy = torch.randn(batch_size, input_size)
    out = encoder.forward(dummy)
    mean, logvar = out
    assert len(out) == 2
    assert isinstance(out, tuple)
    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], torch.Tensor)

    assert mean.shape == (batch_size, latent_size)
    assert logvar.shape == (batch_size, latent_size)


def test_Decoder():
    decoder = mod.Decoder(
        latent_size,
        decoder_lin1_size,
        decoder_lin2_size,
        input_size,
    )

    dummy = torch.randn(batch_size, latent_size)
    out = decoder.forward(dummy)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, input_size)


def test_DenseImageVAEWithBN():
    vae = mod.DenseImageVAEWithBN(
        input_size,
        encoder_lin1_size,
        encoder_lin2_size,
        latent_size,
        decoder_lin1_size,
        decoder_lin2_size,
    )

    dummy = torch.randn(batch_size, channels, width, height)
    out = vae.forward(dummy)
    assert len(out) == 3
    x_hat, mean, logvar = out
    assert isinstance(x_hat, torch.Tensor)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
    assert x_hat.shape == dummy.shape
    assert mean.shape == (batch_size, latent_size)
    assert logvar.shape == (batch_size, latent_size)
