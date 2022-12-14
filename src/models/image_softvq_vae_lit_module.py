from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric

from ..utils.figure_tools import make_grid_of_irrq_prob_figures
from ..utils.kl_div_loss import kl_div_loss


class ImageSoftVQVAELitModule(pl.LightningModule):
    """SoftVQ VAE frame module class. The forward pass of SoftVQ VAE module must return following
    values.

    - x_hat (Tensor): Reconstructed input data.
    - mean (Tensor): Mean tensor of encoder output of vae
    - logvar (Tensor): log variant of encoder output of vae
    - quantized (Tensor): Soft quantized latent space (mean).
    - q_dist (Tensor): Quantizing distributioin (Attention map).
    """

    logger: TensorBoardLogger

    def __init__(
        self,
        softvq_vae: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        r_loss_scaler: float = 1.0,
        kl_div_loss_scaler: float = 1.0,
        log_irrq_prob_row_col: Sequence[int] = (3, 3),
        log_latent_space_sample_num: int = 128,
        log_code_book_usage_sample_num: int = 128,
        log_random_sample_rrq_imgs_row_col: Sequence[int] = (3, 3),
    ) -> None:
        super().__init__()

        assert hasattr(softvq_vae, "softvq")
        assert hasattr(softvq_vae, "vae")
        assert hasattr(softvq_vae.vae, "encoder")
        assert hasattr(softvq_vae.vae, "decoder")

        self.save_hyperparameters(logger=False, ignore=["softvq_vae"])
        self.softvq_vae = softvq_vae

        self.loss_inter_epoch = MeanMetric()
        self.r_loss_inter_epoch = MeanMetric()
        self.kl_div_loss_inter_epoch = MeanMetric()
        self.q_loss_inter_epoch = MeanMetric()
        self.rq_loss_inter_epoch = MeanMetric()
        self.delta_r_rq_inter_epoch = MeanMetric()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        return self.softvq_vae(x)

    def on_train_start(self) -> None:
        self.loss_inter_epoch.reset()
        self.r_loss_inter_epoch.reset()
        self.kl_div_loss_inter_epoch.reset()
        self.q_loss_inter_epoch.reset()
        self.rq_loss_inter_epoch.reset()
        self.delta_r_rq_inter_epoch.reset()

    def step(self, batch: Any):
        x = batch
        x_hat, mean, logvar, quantized, q_dist = self.forward(x)
        x_rq = self.softvq_vae.vae.decoder(quantized).view(x.shape)

        r_l = F.mse_loss(x, x_hat)
        kl_l = kl_div_loss(mean, logvar)
        q_l = F.mse_loss(quantized, mean.detach())
        rq_l = F.mse_loss(x, x_rq.detach())

        loss = r_l * self.hparams.r_loss_scaler + kl_l * self.hparams.kl_div_loss_scaler + q_l

        return loss, r_l, kl_l, q_l, rq_l

    def training_step(self, batch: Any, batch_idx: int):
        loss, r_l, kl_l, q_l, rq_l = self.step(batch)

        delta_r_rq = (r_l - rq_l).abs()

        # update and log metrics
        self.loss_inter_epoch(loss)
        self.r_loss_inter_epoch(r_l)
        self.kl_div_loss_inter_epoch(kl_l)
        self.q_loss_inter_epoch(q_l)
        self.rq_loss_inter_epoch(rq_l)
        self.delta_r_rq_inter_epoch(delta_r_rq)

        header = "training_step/{0}"
        self.log(header.format("loss"), loss, prog_bar=True)
        self.log(header.format("reconstruction_loss"), r_l, prog_bar=True)
        self.log(header.format("kl_div_loss"), kl_l, prog_bar=True)
        self.log(header.format("quantizing_loss"), q_l, prog_bar=True)
        self.log(header.format("reconstructed_quantizing_loss"), rq_l, prog_bar=True)
        self.log(header.format("delta_r_rq"), delta_r_rq, prog_bar=True)

        return loss

    @torch.no_grad()
    def log_grid_of_irrq_prob_figure(self):
        """log the figure to tensor board."""
        size = self.hparams.log_irrq_prob_row_col
        row, col = size
        max_image_num = int(row * col)
        in_imgs = []
        rec_imgs = []
        rec_q_imgs = []
        probs = []

        remain_num = max_image_num
        for batch in self.trainer.datamodule.train_dataloader():
            x = batch
            x_hat, _, _, quantized, q_dist = self.forward(x.to(self.device))
            x_rq = self.softvq_vae.vae.decoder(quantized).view(x_hat.shape)
            x, x_hat, x_rq = x.cpu(), x_hat.cpu(), x_rq.cpu()
            q_dist = q_dist.view(q_dist.size(0), -1, q_dist.size(-1)).cpu()

            if x_hat.size(0) < remain_num:
                in_imgs.append(x)
                rec_imgs.append(x_hat)
                rec_q_imgs.append(x_rq)
                probs.append(q_dist)
            else:
                in_imgs.append(x[:remain_num])
                rec_imgs.append(x_hat[:remain_num])
                rec_q_imgs.append(x_rq[:remain_num])
                probs.append(q_dist[:remain_num])
                break

        psz = (0, 2, 3, 1)
        in_imgs = torch.cat(in_imgs, dim=0).permute(psz).numpy()
        rec_imgs = torch.cat(rec_imgs, dim=0).permute(psz).numpy()
        rec_q_imgs = torch.cat(rec_q_imgs, dim=0).permute(psz).numpy()
        probs = torch.cat(probs, dim=0).numpy()

        fontdict = {"fontsize": 5}
        imshow_settings = {"vmin": 0.0, "vmax": 1.0}

        figure = make_grid_of_irrq_prob_figures(
            row,
            col,
            in_imgs,
            rec_imgs,
            rec_q_imgs,
            probs,
            label_fontdict=fontdict,
            imshow_settings=imshow_settings,
            base_fig_size=(3.2, 2.4),
        )

        tb_logger: SummaryWriter = self.logger.experiment
        tb_logger.add_figure("i-r-rq-prob-images", figure, self.global_step)

    @torch.no_grad()
    def log_random_sample_rrq_imgs(self):
        """log reconstruction and reconstructed quantizing images from random sampled latent
        space."""
        row, col = self.hparams.log_random_sample_rrq_imgs_row_col
        max_image_num = int(row * col)
        batch = next(iter(self.trainer.datamodule.train_dataloader()))
        batch_size = len(batch)
        x = batch[:2].to(self.device)
        _, mean, _, _, _ = self.forward(x)
        latent_space_shape = mean.shape[1:]
        data_shape = x.shape[1:]

        probs = []
        rec_imgs = []
        rec_q_imgs = []
        for i in range(0, max_image_num, batch_size):
            if i + batch_size <= max_image_num:
                bsz = batch_size
            else:
                bsz = max_image_num % batch_size
            sampled_latent_data = torch.randn(bsz, *latent_space_shape, device=self.device)
            quantized, q_dist = self.softvq_vae.softvq(sampled_latent_data)
            x_rq = self.softvq_vae.vae.decoder(quantized).cpu().view(bsz, *data_shape)
            x_hat = self.softvq_vae.vae.decoder(sampled_latent_data).cpu().view(bsz, *data_shape)
            q_dist = q_dist.view(q_dist.size(0), -1, q_dist.size(-1)).cpu()

            probs.append(q_dist)
            rec_imgs.append(x_hat)
            rec_q_imgs.append(x_rq)

        psz = (0, 2, 3, 1)
        rec_imgs = torch.cat(rec_imgs, dim=0).permute(psz).numpy()
        rec_q_imgs = torch.cat(rec_q_imgs, dim=0).permute(psz).numpy()
        probs = torch.cat(probs, dim=0).numpy()

        in_imgs = np.zeros_like(rec_imgs)

        fontdict = {"fontsize": 5}
        imshow_settings = {"vmin": 0.0, "vmax": 1.0}

        figure = make_grid_of_irrq_prob_figures(
            row,
            col,
            in_imgs,
            rec_imgs,
            rec_q_imgs,
            probs,
            label_fontdict=fontdict,
            imshow_settings=imshow_settings,
            base_fig_size=(3.2, 2.4),
        )

        tb_logger: SummaryWriter = self.logger.experiment
        tb_logger.add_figure("random-sample-r-rq-prob-images", figure, self.global_step)

    @torch.no_grad()
    def log_latent_space_distribution(self):
        """Log distribution histogram of latent space to logger."""
        sample_size = self.hparams.log_latent_space_sample_num
        mean_samples, std_samples, quantized_samples = [], [], []
        remain_num = sample_size
        for batch in self.trainer.datamodule.train_dataloader():
            x = batch
            _, mean, logvar, quantized, _ = self.forward(x.to(self.device))
            mean = mean.cpu()
            std = torch.exp(0.5 * logvar).cpu()
            if mean.size(0) < remain_num:
                mean_samples.append(mean)
                std_samples.append(std)
                quantized_samples.append(quantized)
                remain_num -= mean.size(0)
            else:
                mean_samples.append(mean[:remain_num])
                std_samples.append(std[:remain_num])
                quantized_samples.append(quantized[:remain_num])
                break

        mean_samples = torch.cat(mean_samples).flatten()
        std_samples = torch.cat(std_samples).flatten()
        quantized_samples = torch.cat(quantized_samples).flatten()

        tb_logger: SummaryWriter = self.logger.experiment
        tb_logger.add_histogram("latent_space/mean_distribution", mean_samples, self.global_step)
        tb_logger.add_histogram("latent_space/std_distribution", std_samples, self.global_step)
        tb_logger.add_histogram("latent_space/quantized_distribution", quantized_samples, self.global_step)

    @torch.no_grad()
    def log_codebook_average_usage(self):
        """log mean of quantizing distribution as codebook average usage."""
        sample_size = self.hparams.log_code_book_usage_sample_num

        sum_q_dist: Optional[torch.Tensor] = None
        remain_num = sample_size
        length = 0
        roop_break = False
        for batch in self.trainer.datamodule.train_dataloader():
            x = batch
            _, mean, _ = self.softvq_vae.vae(x.to(self.device))
            q_dists: torch.Tensor = self.softvq_vae.softvq(mean)[1]
            if q_dists.size(0) >= remain_num:
                q_dists = q_dists[:remain_num]
                roop_break = True

            q_dists_flat = q_dists.view(-1, q_dists.size(-1))
            length += q_dists_flat.size(0)
            qsum = q_dists_flat.sum(0).cpu().float()

            if sum_q_dist is None:
                sum_q_dist = qsum
            else:
                sum_q_dist += qsum

            if roop_break:
                break

        avg_q_dist = sum_q_dist.numpy() / length
        avg_usage = np.mean(avg_q_dist)

        fig = plt.figure(figsize=(6.4, 2.4))
        ax = fig.add_subplot()
        ax.bar(range(len(avg_q_dist)), avg_q_dist)
        ax.set_ylabel("Vector index")
        ax.set_title("Codebook average usage per vector")

        tb_logger: SummaryWriter = self.logger.experiment
        tb_logger.add_figure("codebook_average_usage/per_vector", fig, self.global_step)
        self.log("codebook_average_usage/usage", avg_usage, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        header = "inter_epoch/{0}"
        self.log(header.format("loss"), self.loss_inter_epoch, on_epoch=True)
        self.log(header.format("reconstruction_loss"), self.r_loss_inter_epoch, on_epoch=True)
        self.log(header.format("kl_div_loss"), self.kl_div_loss_inter_epoch, on_epoch=True)
        self.log(header.format("quantizing_losss"), self.q_loss_inter_epoch, on_epoch=True)
        self.log(header.format("reconstructed_quantizing_loss"), self.rq_loss_inter_epoch, on_epoch=True)
        self.log(header.format("delta_r_rq"), self.delta_r_rq_inter_epoch, on_epoch=True)

        self.log_latent_space_distribution()
        self.log_grid_of_irrq_prob_figure()
        self.log_codebook_average_usage()
        self.log_random_sample_rrq_imgs()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
