import hydra
import omegaconf
import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "image_straight_dot_product_softvq_vae_with_bn.yaml")


def test_instantiate():
    _ = hydra.utils.instantiate(cfg)
