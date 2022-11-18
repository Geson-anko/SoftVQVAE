import torch
import torch.nn as nn

from src.models.components import vae_tools as mod
from src.models.components.dense_image_vae import Decoder, Encoder


def test_MeanAndLogvar():
    cls = mod.MeanAndLogvar

    linear = nn.Linear(10, 20)
    inst = cls(linear)

    assert inst.mean_layer is linear
    assert isinstance(inst.logvar_layer, nn.Linear)

    linear = nn.Linear(10, 20)
    inst = cls(linear)
    data = torch.randn(10)
    out = inst.forward(data)
    out1, out2 = out
    assert out1.shape == out2.shape
    assert out1.dtype == out2.dtype


def test_sample():
    size = 1_000_000
    error = 0.1
    f = mod.sample

    mean = torch.tensor([1.0, 2.0], dtype=torch.float)
    expand_mean = mean.view(2, 1).repeat((1, size))
    std = torch.tensor([1.5, 3.0], dtype=torch.float)
    expand_std = std.view(2, 1).repeat((1, size))

    out = f(expand_mean, torch.log(expand_std * expand_std))
    out_mean = out.mean(dim=1)
    out_std = out.std(dim=1)
    assert torch.all((out_mean - mean).abs() < error)
    assert torch.all((out_std - std).abs() < error)


def test_VariationalAutoEncoder():
    cls = mod.VariationalAutoEncoder

    encoder = Encoder(784, 256, 64, 16)
    decoder = Decoder(16, 64, 256, 784)

    vae = cls(encoder, decoder)

    assert vae.encoder is encoder
    assert vae.decoder is decoder

    data = torch.randn(2, 784)
    x_hat, mean, logvar = vae.forward(data)
    assert x_hat.shape == data.shape
    assert mean.shape == (2, 16)
    assert logvar.shape == mean.shape
