from typing import Any

from torchvision.datasets import MNIST


class MNISTDatasetOnlyImage(MNIST):
    """Extract only images from MNIST dataset."""

    def __getitem__(self, index: int) -> Any:
        return super().__getitem__(index)[0]
