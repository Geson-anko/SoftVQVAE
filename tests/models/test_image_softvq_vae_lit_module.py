import hydra
import omegaconf
import pyrootutils
import pytest

from src.models.components.dense_image_vae_with_bn import (
    Decoder,
    DenseImageVAEWithBN,
    Encoder,
)
from src.models.components.softvq import SoftVectorQuantizing
from src.models.components.softvq_vae import SoftVQVAE
from src.models.image_softvq_vae_lit_module import ImageSoftVQVAELitModule
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "image_simple_softvq_vae.yaml")


def test_instantiate():
    instance = hydra.utils.instantiate(cfg)

    assert isinstance(instance, ImageSoftVQVAELitModule)
    assert isinstance(instance.softvq_vae, SoftVQVAE)
    assert isinstance(instance.softvq_vae.vae, DenseImageVAEWithBN)
    assert isinstance(instance.softvq_vae.vae.encoder, Encoder)
    assert isinstance(instance.softvq_vae.vae.decoder, Decoder)
    assert isinstance(instance.softvq_vae.softvq, SoftVectorQuantizing)


@pytest.mark.slow
def test_training_fast_dev_run():

    sh_command = [
        "src/train.py",
        "model=image_simple_softvq_vae",
        "datamodule=mnist_only_image",
        "datamodule.batch_size=128",
        "callbacks=none",
        "logger=tensorboard",
        'tags="[SoftVQVAE, test]"',
        "test=False",
        "trainer.max_epochs=1",
        "trainer.accelerator=cpu",
        "+trainer.fast_dev_run=True",
        "+trainer.log_every_n_steps=1",
    ]

    run_sh_command(sh_command)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_training_fast_dev_run_gpu():

    sh_command = [
        "src/train.py",
        "model=image_simple_softvq_vae",
        "datamodule=mnist_only_image",
        "datamodule.batch_size=128",
        "callbacks=none",
        "logger=tensorboard",
        'tags="[SoftVQVAE, test]"',
        "test=False",
        "trainer.max_epochs=1",
        "trainer.accelerator=gpu",
        "+trainer.fast_dev_run=True",
        "+trainer.log_every_n_steps=1",
    ]

    run_sh_command(sh_command)


@pytest.mark.slow
def test_training():
    sh_command = [
        "src/train.py",
        "++task_name=debug_training",
        "model=image_simple_softvq_vae",
        "datamodule=mnist_only_image",
        "datamodule.batch_size=128",
        "callbacks=none",
        "logger=tensorboard",
        'tags="[SoftVQVAE, test]"',
        "test=False",
        "trainer.max_epochs=1",
        "trainer.accelerator=cpu",
        "+trainer.log_every_n_steps=1",
        "+trainer.max_steps=1",
    ]

    run_sh_command(sh_command)
