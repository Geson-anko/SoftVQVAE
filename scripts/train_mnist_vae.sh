#!/bin/bash

python src/train.py \
    datamodule=mnist_all_to_training \
    model=mnist_vae \
    callbacks=for_mnist_vae \
    logger=tensorboard \
    tags="[SoftVQ, mnist_vae]" \
    test=False \
    task_name="mnist_vae" \
    trainer.max_epochs=30 \
    model.r_loss_scaler=10000
