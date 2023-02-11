import torch

from src.models.components import vq as mod

batch_size = 16
latent_size = 32
input_size = 784
num_quantizing = 8


def test_compute_quantizing_distribution():
    vq = mod.VectorQuantizing(num_quantizing, latent_size)

    x = torch.randn(batch_size, latent_size)
    result = vq.compute_quantizing_distribution(x)

    assert result.shape == (batch_size, num_quantizing)
    torch.testing.assert_close(torch.max(result, dim=-1).values, torch.ones(batch_size))
    torch.testing.assert_close(torch.sum(result, dim=-1), torch.ones(batch_size))
    assert result.requires_grad is True


def test_forward():
    vq = mod.VectorQuantizing(num_quantizing, latent_size)

    x = torch.randn(batch_size, latent_size)
    q, p = vq.forward(x)  # default `detach_distribution_grad=False`
    assert q.shape == (batch_size, latent_size)
    assert p.shape == (batch_size, num_quantizing)
    assert p.requires_grad
    assert q.requires_grad

    q, p = vq.forward(x, True)
    assert not p.requires_grad
    assert q.requires_grad
