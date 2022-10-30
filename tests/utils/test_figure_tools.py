import os

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.utils import figure_tools as mod

width, height, c = 32, 32, 3
num_quantizing = 32
num_probs = 10

result_dir = "data/test_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


@pytest.mark.parametrize("prob_dist_dim", [1, 2])
def test_create_irrq_prob_figure(prob_dist_dim: int):
    f = mod.create_irrq_prob_figure
    input_image = np.random.rand(width, height, c)
    rec_image = np.random.rand(width, height, c)
    qr_image = np.random.rand(width, height, c)
    if prob_dist_dim == 1:
        prob = np.random.rand(num_quantizing)
    elif prob_dist_dim == 2:
        prob = np.random.rand(num_probs, num_quantizing)

    fig_out = f(input_image, rec_image, qr_image, prob)
    assert isinstance(fig_out, figure.FigureBase)

    fig = plt.figure()
    fontdict = {"fontsize": 10}
    imshow_settings = {"vmin": -1.0, "vmax": 1.0}
    fig_out = f(input_image, rec_image, qr_image, prob, fig, fontdict, imshow_settings)

    assert fig is fig_out

    fig.savefig(os.path.join(result_dir, f"{__name__}.test_create_irrq_prob_figure.dim{prob_dist_dim}.png"))

    plt.close()


@pytest.mark.parametrize("image_num", [6, 4])
def test_make_grid_of_irrq_prob_figures(image_num: int):
    f = mod.make_grid_of_irrq_prob_figures

    row, col = 2, 3
    in_imgs = np.random.rand(image_num, width, height, c)
    rec_imgs = np.random.rand(image_num, width, height, c)
    rq_imgs = np.random.rand(image_num, width, height, c)
    probs = np.random.rand(image_num, num_quantizing)
    base_fig_size = (6.4, 4.8)
    fontdict = {"fontsize": 10}
    imshow_settings = {"vmin": 0.0, "vmax": 1.0}

    fig = f(row, col, in_imgs, rec_imgs, rq_imgs, probs, base_fig_size, fontdict, imshow_settings)

    assert isinstance(fig, figure.Figure)

    fig.savefig(os.path.join(result_dir, f"{__name__}.test_make_grid_of_irrq_prob_figures.{image_num}.png"))
    plt.close()
