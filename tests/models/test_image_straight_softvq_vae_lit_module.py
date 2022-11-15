import hydra
import omegaconf
import pyrootutils
import pytest

from src.models.components.dense_image_vae_with_bn import Decoder, Encoder
from src.models.components.softvq import SoftVectorQuantizing
from src.models.components.straight_softvq_vae import StraightSoftVQVAE
from src.models.image_straight_softvq_vae_lit_module import (
    ImageStraightSoftVQVAELitModule,
)
from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "image_straight_softvq_vae.yaml")


def test_instantiate():
    instance = hydra.utils.instantiate(cfg)

    assert isinstance(instance, ImageStraightSoftVQVAELitModule)
    assert isinstance(instance.straight_softvq_vae, StraightSoftVQVAE)
    assert isinstance(instance.straight_softvq_vae.encoder, Encoder)
    assert isinstance(instance.straight_softvq_vae.decoder, Decoder)
    assert isinstance(instance.straight_softvq_vae.softvq, SoftVectorQuantizing)


@pytest.mark.slow
def test_training_fast_dev_run():

    sh_command = [
        "src/train.py",
        "model=image_straight_softvq_vae",
        "datamodule=mnist_only_image",
        "datamodule.batch_size=128",
        "callbacks=none",
        "logger=tensorboard",
        'tags="[StraightSoftVQVAE, test]"',
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
        "model=image_straight_softvq_vae",
        "datamodule=mnist_only_image",
        "datamodule.batch_size=128",
        "callbacks=none",
        "logger=tensorboard",
        'tags="[StraightSoftVQVAE, test]"',
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
        "model=image_straight_softvq_vae",
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
