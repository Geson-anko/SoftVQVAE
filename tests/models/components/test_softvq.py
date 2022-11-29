import torch

from src.models.components import softvq


def test_softmax_with_temperature():
    f = softvq.softmax_with_temperature

    x = torch.ones(4)
    assert (torch.tensor([0.25, 0.25, 0.25, 0.25]) - f(x, 1.0, 0)).abs().mean().item() < 1e-8

    x = torch.ones(1, 4)
    assert (torch.tensor([[1.0, 1.0, 1.0, 1.0]]) - f(x, 1.0, 0)).abs().mean().item() < 1e-8

    x = torch.tensor([2.0, 1.0])
    t = 0.5
    a = torch.tensor([0.8807970285415649, 0.11920291185379028])
    assert (f(x, t, 0) - a).abs().mean().item() < 1e-8

    x = x.type(torch.float)
    tgt_t = torch.float64
    assert f(x, 1.0, 0, dtype=tgt_t).dtype is tgt_t


def test__init__():
    cls = softvq.SoftVectorQuantizing

    dflt = cls(16, 8)
    assert dflt.num_quantizing == 16
    assert dflt.quantizing_dim == 8
    assert dflt.temperature == 1.0
    assert tuple(dflt._weight.shape) == (16, 8)
    assert dflt._weight.requires_grad

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    opt = cls(10, 32, 2.0, mean=1, std=3, dtype=torch.float64, device=device)
    assert opt.num_quantizing == 10
    assert opt.quantizing_dim == 32
    assert opt.temperature == 2.0
    assert opt._weight.dtype is torch.float64
    assert opt._weight.device == torch.device(device)


def test_compute_quantizing_distribution():
    cls = softvq.SoftVectorQuantizing

    nq, qd = 16, 8
    batch = 4
    dflt = cls(nq, qd)

    x = torch.randn(batch, qd)
    result = dflt.compute_quantizing_distribution(x)
    assert tuple(result.shape) == (batch, nq)
    assert (1.0 - result.sum(-1)).abs().mean().item() < 1e-6
    assert result.requires_grad


def test_forward():
    cls = softvq.SoftVectorQuantizing

    nq, qd = 16, 8
    batch = 4
    dflt = cls(nq, qd)

    x = torch.randn(batch, qd)
    q, p = dflt.forward(x)
    assert tuple(q.shape) == (batch, qd)
    assert tuple(p.shape) == (batch, nq)
    assert not p.requires_grad

    q, p = dflt.forward(x, False)
    assert p.requires_grad


def test_quantize_from_q_dist():
    cls = softvq.SoftVectorQuantizing
    nq, qd = 16, 8
    dflt = cls(nq, qd)

    q_dist = torch.ones(nq) / nq
    out = dflt.quantize_from_q_dist(q_dist)

    assert torch.nn.functional.mse_loss(out, dflt._weight.mean(0)).item() < 1e-8
