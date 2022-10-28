import torch

from src.models.components import softvq_vae as mod
from src.models.components.dense_image_vae import DenseImageVAE
from src.models.components.softvq import SoftVectorQuantizing


def test__init__():
    latent_size = 16
    num_quantizing = 10
    vae = DenseImageVAE(latent_size=latent_size)
    softvq = SoftVectorQuantizing(num_quantizing, latent_size)

    softvq_vae = mod.SoftVQVAE(vae, softvq)

    assert softvq_vae.softvq is softvq
    assert softvq_vae.vae is vae


def test_forward():
    batch_size = 16
    latent_size = 32
    input_size = 784
    num_quantizing = 8
    vae = DenseImageVAE(input_size, latent_size=latent_size)
    softvq = SoftVectorQuantizing(num_quantizing, latent_size)

    softvq_vae = mod.SoftVQVAE(vae, softvq)
    dummy = torch.randn(batch_size, input_size)

    outputs = softvq_vae.forward(dummy)

    assert len(outputs) == 5
    x_hat, mean, logvar, quantized, q_dist = outputs
    assert isinstance(x_hat, torch.Tensor)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
    assert isinstance(quantized, torch.Tensor)
    assert isinstance(q_dist, torch.Tensor)
