#!/bin/bash


python src/train.py -m \
    hparams_search=mnist_softvq_vae_optuna \
    experiment=mnist_softvq_vae \
    model=image_dot_product_softvq_vae \
    tags="[SoftVQ VAE, image_simple_softvq_vae, mnist, dot product softvq]" \
    trainer.accelerator=gpu \
