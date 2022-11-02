#!/bin/bash

python src/train.py \
    datamodule=mnist_only_image \
    model=image_simple_softvq_vae \
    callbacks=for_mnist_vae \
    logger=tensorboard \
    tags="[SoftVQ VAE, training]" \
    test=False \
    task_name="mnist-softvq-vae" \
    trainer.max_epochs=100 \
    trainer.accelerator=cpu \
    model.r_loss_scaler=10000 \
    +trainer.log_every_n_steps=5
