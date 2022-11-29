import torch

from src.models.components import dot_product_softvq as mod

batch_size = 16
latent_size = 32
input_size = 784
num_quantizing = 8


def test_compute_quantizing_distribution():
    softvq = mod.DotProductSoftVQ(num_quantizing, latent_size)

    x = torch.randn(batch_size, latent_size)
    result = softvq.compute_quantizing_distribution(x)
    assert result.shape == (batch_size, num_quantizing)
    assert (1.0 - result.sum(-1)).abs().mean().item() < 1e-6
    assert result.requires_grad


def test_forward():
    softvq = mod.DotProductSoftVQ(num_quantizing, latent_size)

    x = torch.randn(batch_size, latent_size)
    q, p = softvq.forward(x)  # default `detach_distribution_grad=False`
    assert q.shape == (batch_size, latent_size)
    assert p.shape == (batch_size, num_quantizing)
    assert p.requires_grad
    assert q.requires_grad

    q, p = softvq.forward(x, True)
    assert not p.requires_grad
    assert q.requires_grad
