from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms

from .components.mnist_dataset_only_image import MNISTDatasetOnlyImage


class MNISTOnlyImage(LightningDataModule):
    """This data module loads all MNIST dataset only images."""

    def __init__(
        self, data_dir: str = "data/", batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.data_train: Optional[Dataset] = None

    def prepare_data(self) -> None:
        MNISTDatasetOnlyImage(self.hparams.data_dir, train=True, download=True)
        MNISTDatasetOnlyImage(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data."""

        if self.data_train is None:
            trainset = MNISTDatasetOnlyImage(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNISTDatasetOnlyImage(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])

            self.data_train = dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
