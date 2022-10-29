from pathlib import Path

import hydra
import omegaconf
import pyrootutils
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.datamodules.mnist_only_image import MNISTOnlyImage

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist_only_image.yaml")
cfg.data_dir = str(root / "data")


def test_instantiate():
    instance = hydra.utils.instantiate(cfg)
    assert isinstance(instance, MNISTOnlyImage)


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_only_image(batch_size: int):
    data_dir = "data/"
    dm = MNISTOnlyImage(data_dir=data_dir, batch_size=batch_size)

    assert dm.data_train is None
    assert Path(data_dir, "MNISTDatasetOnlyImage").exists()
    assert Path(data_dir, "MNISTDatasetOnlyImage", "raw").exists()

    dm.setup()
    assert isinstance(dm.data_train, Dataset)
    assert isinstance(dm.train_dataloader(), DataLoader)

    assert len(dm.data_train) == 70000

    batch = next(iter(dm.train_dataloader()))
    assert len(batch) == batch_size
    assert isinstance(batch, torch.Tensor)
    assert batch.dtype is torch.float32
