import torch
from torchvision import transforms

from src.datamodules.components.mnist_dataset_only_image import MNISTDatasetOnlyImage

root = "./data"


def test_MNISTDatasetOnlyImage():
    dset = MNISTDatasetOnlyImage(root, download=True, transform=transforms.ToTensor())

    # test 10 times.
    for i in range(10):
        data = dset[i]
        assert isinstance(dset[i], torch.Tensor)
        assert data.shape == (1, 28, 28)
