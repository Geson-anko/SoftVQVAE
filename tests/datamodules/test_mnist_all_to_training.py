from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.datamodules.mnist_all_to_training import MNISTAllToTraining


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_all_to_training(batch_size: int):
    data_dir = "data/"

    dm = MNISTAllToTraining(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert dm.data_train is None
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert isinstance(dm.data_train, Dataset)
    assert isinstance(dm.train_dataloader(), DataLoader)

    assert len(dm.data_train) == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
