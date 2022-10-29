"""This module defines functions to draw matplotlib figure, for viewing 
experiment results.
このモジュールでは、実験の経過をmatplotlibを用いて描画するための関数を定義します。
"""
from typing import Optional

import matplotlib.figure as figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def create_irrq_prob_figure(
    input_image: np.ndarray,
    reconstruction_image: np.ndarray,
    reconstructed_quantizing_image: np.ndarray,
    probability_distribution: np.ndarray,
    figure: Optional[figure.FigureBase] = None,
    label_fontdict: Optional[dict] = None,
    imshow_settings: Optional[dict] = None,
) -> figure.FigureBase:
    """Create figure for logging image softvq vae.
    `irrq` is short name of `i`nput_image, `r`econstruction_image,
    `r`econstructed_`q`uantizing_image.

    Args:
        input_image (np.ndarray): Input of vae.
        reconstruction_image (np.ndarray): Output of vae.
        quantized_reconstruction_image (np.ndarray): Reconstructed quantized image.
        probability_distribution (np.ndarray): output `q_dist` of softvq.
        figure (Optional[figure.FigureBase]): Ploting figure.
        fontdict (Optional[dict]): The font settings of label.
        imshow_settings (Optional[dict]): If None, generate settings, `{vmin: 0.0, vmax: 1.0}`.

    Shape:
        input_image: (width, height, channels)
        reconstruction_image: (width, height, channels)
        quantized_reconstruction_image: (width, height, channels)
        probability_distribution: (num_quantizing, )

    Returns:
        output_figure (figure.FigureBase): If input figure is provided, returns same object.
    """
    if figure is None:
        figure = plt.figure()
    if imshow_settings is None:
        imshow_settings = {
            "vmin": 0.0,
            "vmax": 1.0,
        }

    gs = gridspec.GridSpec(2, 3)

    ax0 = figure.add_subplot(gs[0, 0])
    ax0.imshow(input_image, **imshow_settings)
    ax0.set_title("Input Image", fontdict=label_fontdict)

    ax1 = figure.add_subplot(gs[0, 1])
    ax1.imshow(reconstruction_image, **imshow_settings)
    ax1.set_title("Reconstruction Image", fontdict=label_fontdict)

    ax2 = figure.add_subplot(gs[0, 2])
    ax2.imshow(reconstructed_quantizing_image, **imshow_settings)
    ax2.set_title("Reconstructed\nQuantizing Image", fontdict=label_fontdict)

    for ax in [ax0, ax1, ax2]:
        ax.set_axis_off()

    pax = figure.add_subplot(gs[1, :])
    pax.bar(range(len(probability_distribution)), probability_distribution)
    pax.set_ylim(0, 1)
    pax.set_title("probability distribution", fontdict=label_fontdict)
    pax.set_xlabel("vector index", fontdict=label_fontdict)

    return figure
