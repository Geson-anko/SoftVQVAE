from typing import Any, List, Sequence

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from torchvision.utils import make_grid

from ..utils.kl_div_loss import kl_div_loss


class MNISTLitVAEModule(LightningModule):
    """Image VAE for MNIST dataset."""

    logger: TensorBoardLogger

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        r_loss_scaler: float = 1.0,
        kl_div_loss_scaler: float = 1.0,
        log_image_row_col: Sequence[int] = (8, 8),
        log_latent_space_sample_num: int = 128,
    ) -> None:
        """Construction.
        Args:
            - net (Module): Must return reconstructed image, mean, logvar
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        self.criterion = torch.nn.MSELoss()

        self.train_loss_inter_epoch = MeanMetric()
        self.train_rec_loss_inter_epoch = MeanMetric()
        self.train_kl_div_loss_inter_epoch = MeanMetric()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        self.train_loss_inter_epoch.reset()
        self.train_rec_loss_inter_epoch.reset()
        self.train_kl_div_loss_inter_epoch.reset()
        return super().on_epoch_start()

    def step(self, batch: Any):
        x, y = batch
        x_hat, mean, logvar = self.forward(x)

        rec_l = self.criterion(x, x_hat)
        kl_l = kl_div_loss(mean, logvar)

        loss = rec_l * self.hparams.r_loss_scaler + kl_l * self.hparams.kl_div_loss_scaler
        return loss, rec_l, kl_l

    def training_step(self, batch: Any, batch_idx: int):
        loss, rec_l, kl_l = self.step(batch)

        # update and log metrics
        self.train_loss_inter_epoch(loss)
        self.train_rec_loss_inter_epoch(rec_l)
        self.train_kl_div_loss_inter_epoch(kl_l)

        self.log("training_step/loss", loss, prog_bar=True)
        self.log("training_step/reconstruction_loss", rec_l, prog_bar=True)
        self.log("training_step/kl_div_loss", kl_l, prog_bar=True)

        return loss

    @torch.no_grad()
    def log_reconstructed_images(self):
        """Log to tensorboard logger."""
        size = self.hparams.log_image_row_col
        row, col = size
        max_image_num = int(row * col)
        reconstructed_images = []
        remain_num = max_image_num
        for batch in self.trainer.datamodule.train_dataloader():
            x, y = batch
            x_hat, _, _ = self.forward(x.to(self.device))
            x_hat = x_hat.cpu()
            if x_hat.size(0) < remain_num:
                reconstructed_images.append(x_hat)
                remain_num -= x_hat.size(0)
            else:
                reconstructed_images.append(x_hat[:remain_num])
                break

        reconstructed_images = torch.cat(reconstructed_images, dim=0)

        grid_rec_image = make_grid(reconstructed_images, row, padding=2)

        tb_logger: SummaryWriter = self.logger.experiment
        tb_logger.add_image("reconstructed_images", grid_rec_image, self.global_step)

    @torch.no_grad()
    def log_latent_space_distribution(self):
        """Log distribution histogram of latent space to logger."""
        sample_size = self.hparams.log_latent_space_sample_num
        mean_samples, std_samples = [], []
        remain_num = sample_size
        for batch in self.trainer.datamodule.train_dataloader():
            x, y = batch
            _, mean, logvar = self.forward(x.to(self.device))
            mean = mean.cpu()
            std = torch.exp(0.5 * logvar).cpu()
            if mean.size(0) < remain_num:
                mean_samples.append(mean)
                std_samples.append(std)
                remain_num -= mean.size(0)

        mean_samples = torch.cat(mean_samples).flatten()
        std_samples = torch.cat(std_samples).flatten()

        tb_logger: SummaryWriter = self.logger.experiment
        tb_logger.add_histogram("latent_space/mean_distribution", mean_samples, self.global_step)
        tb_logger.add_histogram("latent_space/std_distribution", std_samples, self.global_step)

    def on_train_epoch_end(self) -> None:

        self.log("inter_epoch/loss", self.train_loss_inter_epoch)
        self.log("inter_epoch/reconstruction_loss", self.train_rec_loss_inter_epoch)
        self.log("inter_epoch/kl_div_loss", self.train_kl_div_loss_inter_epoch)
        self.log_reconstructed_images()
        self.log_latent_space_distribution()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist_vae.yaml")
    _ = hydra.utils.instantiate(cfg)
