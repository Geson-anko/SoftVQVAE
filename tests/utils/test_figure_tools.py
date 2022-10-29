import os

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_tools as mod

width, height, c = 32, 32, 3
num_quantizing = 32

result_dir = "data/test_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def test_create_irrq_prob_figure():
    f = mod.create_irrq_prob_figure
    input_image = np.random.rand(width, height, c)
    rec_image = np.random.rand(width, height, c)
    qr_image = np.random.rand(width, height, c)
    prob = np.random.rand(num_quantizing)

    fig_out = f(input_image, rec_image, qr_image, prob)
    assert isinstance(fig_out, figure.FigureBase)

    fig = plt.figure()
    fontdict = {"fontsize": 10}
    imshow_settings = {"vmin": -1.0, "vmax": 1.0}
    fig_out = f(input_image, rec_image, qr_image, prob, fig, fontdict, imshow_settings)

    assert fig is fig_out

    fig.savefig(os.path.join(result_dir, f"{__name__}.test_create_irrq_prob_figure.png"))

    plt.close()
