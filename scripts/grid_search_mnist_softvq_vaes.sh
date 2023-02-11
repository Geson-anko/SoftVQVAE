#!/bin/bash

python src/train.py -m \
    hparams_search=mnist_gird_search_softvq_vae \
    experiment=mnist_softvq_vae \
    trainer.accelerator=gpu \
    task_name=mnist-softvq-vae-grid-search

python src/train.py -m \
    hparams_search=mnist_gird_search_softvq_vae \
    experiment=mnist_softvq_vae \
    model=image_dot_product_softvq_vae \
    tags="[SoftVQ VAE, image_simple_softvq_vae, mnist, dot product softvq]" \
    trainer.accelerator=gpu \
    task_name=mnist-softvq-vae-grid-search


python src/train.py -m \
    hparams_search=mnist_gird_search_straight_softvq_vae \
    experiment=mnist_straight_dot_product_softvq_vae_with_bn \
    trainer.accelerator=gpu \
    task_name=mnist-straight-softvq-vae-grid-search

python src/train.py -m \
    hparams_search=mnist_gird_search_straight_softvq_vae \
    experiment=mnist_straight_softvq_vae \
    trainer.accelerator=gpu \
    task_name=mnist-straight-softvq-vae-grid-search


python src/train.py -m \
    hparams_search=mnist_gird_search_straight_softvq_vae \
    experiment=mnist_straight_dot_product_softvq_vae_with_bn \
    trainer.accelerator=gpu \
    task_name=mnist-straight-softvq-vae-grid-search \
    model.kl_div_loss_scaler=0.01

python src/train.py -m \
    hparams_search=mnist_gird_search_straight_softvq_vae \
    experiment=mnist_straight_softvq_vae \
    trainer.accelerator=gpu \
    task_name=mnist-straight-softvq-vae-grid-search \
    model.kl_div_loss_scaler=0.01
