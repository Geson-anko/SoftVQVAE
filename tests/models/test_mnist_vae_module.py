import hydra
import omegaconf
import pyrootutils

from src.models import mnist_vae_module as mod

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist_vae.yaml")


def test_instantiate():
    _ = hydra.utils.instantiate(cfg)
