#!/bin/bash

python src/train.py -m \
    hparams_search=mnist_straight_softvq_vae_optuna \
    experiment=mnist_straight_softvq_vae \
    trainer.accelerator=gpu
