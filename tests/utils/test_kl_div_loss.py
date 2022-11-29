import torch

from src.utils import kl_div_loss as mod


def test_kl_div_loss():
    f = mod.kl_div_loss

    mean, logvar = torch.zeros(10,), torch.zeros(
        10,
    )
    assert f(mean, logvar) == 0.0

    mean, logvar = torch.ones(10, 2), 2 * torch.ones(10, 2)
    mean_ans = torch.e**2 - 2
    sum_ans = 10 * (torch.e**2 - 2)
    assert (f(mean, logvar, reduction="mean") - mean_ans) < 1e-4
    assert (f(mean, logvar, reduction="sum") - sum_ans) < 1e-4

    try:
        f(mean, logvar, "aaaa")
        assert False, "Exception is not thrown!"
    except ValueError:
        pass
