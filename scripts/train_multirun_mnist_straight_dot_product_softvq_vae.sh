#!/bin/bash

python src/train.py -m \
    hparams_search=mnist_straight_softvq_vae_optuna \
    experiment=mnist_straight_dot_product_softvq_vae_with_bn \
    trainer.accelerator=gpu
