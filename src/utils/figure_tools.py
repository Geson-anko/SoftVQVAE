"""This module defines functions to draw matplotlib figure, for viewing experiment results.

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
    """Create figure for logging image softvq vae. `irrq` is short name of `i`nput_image,
    `r`econstruction_image, `r`econstructed_`q`uantizing_image.

    Args:
        input_image (np.ndarray): Input of vae.
        reconstruction_image (np.ndarray): Output of vae.
        quantized_reconstruction_image (np.ndarray): Reconstructed quantized image.
        probability_distribution (np.ndarray): output `q_dist` of softvq. If this has multiple distributions,
            the one of the distributions is selected randomly.
        figure (Optional[figure.FigureBase]): Plotting figure.
        fontdict (Optional[dict]): The font settings of label.
        imshow_settings (Optional[dict]): If None, generate settings, `{vmin: 0.0, vmax: 1.0}`.

    Shape:
        input_image: (width, height, channels)
        reconstruction_image: (width, height, channels)
        quantized_reconstruction_image: (width, height, channels)
        probability_distribution: (num_quantizing, ) | (num_dists, num_quantizing)

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
    ax2.set_title("Reconstructed\nQuantization Image", fontdict=label_fontdict)

    for ax in [ax0, ax1, ax2]:
        ax.set_axis_off()

    # probability distribution
    title = "probability distribution"
    if probability_distribution.ndim == 2:
        length = len(probability_distribution)
        idx = np.random.randint(length)
        probability_distribution = probability_distribution[idx]
        title = f"{title} (vector {idx} of 0...{length-1})"

    pax = figure.add_subplot(gs[1, :])
    pax.bar(range(len(probability_distribution)), probability_distribution)
    # pax.set_ylim(0, 1)
    pax.set_title(title, fontdict=label_fontdict)
    pax.set_xlabel("vector index", fontdict=label_fontdict)
    pax.set_ylabel("prob (max 1.0)")

    return figure


def make_grid_of_irrq_prob_figures(
    row: int,
    col: int,
    input_images: np.ndarray,
    reconstruction_images: np.ndarray,
    reconstructed_quantizing_images: np.ndarray,
    probability_distributioins: np.ndarray,
    base_fig_size: tuple[int, int] = (6.4, 4.8),
    label_fontdict: Optional[dict] = None,
    imshow_settings: Optional[dict] = None,
) -> figure.Figure:
    """Make grid of figures for logging output of softvq vae. If the number of images is smaller
    than `row x col`, empty subfigures are shown in tail of the figure.

    Arg:
        row (int): Row size of grid.
        col (int): Column size of grid.
        input_images (np.ndarray): Input of vae.
        reconstruction_images (np.ndarray): Output of vae.
        quantized_reconstruction_images (np.ndarray): Reconstructed quantized image.
        probability_distributions (np.ndarray): output `q_dist` of softvq.
        base_fig_size (tuple[int, int]): Base size of figure. Expanded by `row` and `col`.
        fontdict (Optional[dict]): See `create_irrq_prob_figure` docs.
        imshow_settings (Optional[dict]): See `create_irrq_prob_figure` docs.

    Shape:
        input_images: (num, width, height, channels)
        reconstruction_images: (num, width, height, channels)
        quantized_reconstruction_image: (num, width, height, channels)
        probability_distribution: (num, num_quantizing) | (num, num_dists, num_quantizing)

    Returns:
        output_figure (figure.Figure)
    """

    figsize = (base_fig_size[0] * col, base_fig_size[1] * row)
    num_images = len(input_images)

    root_fig = plt.figure(figsize=figsize)

    subfigs = root_fig.subfigures(row, col, False)
    index = 0
    for r in range(row):
        for c in range(col):
            if index < num_images:
                fig: figure.SubFigure = subfigs[r][c]

                in_img = input_images[index]
                rec_img = reconstruction_images[index]
                rec_q_img = reconstructed_quantizing_images[index]
                prob = probability_distributioins[index]
                fig = create_irrq_prob_figure(in_img, rec_img, rec_q_img, prob, fig, label_fontdict, imshow_settings)
            else:
                break

            index += 1

        if not (index < num_images):
            break

    return root_fig
