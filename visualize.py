"""
Author: Yikang Liao
Email: yliao1@andrew.cmu.edu
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import ImageGrid
#from sklearn_theano.feature_extraction import fetch_overfeat_weights_and_biases




def make_visual(layer_weights):
    """ layer_weight is 784*100
    """
    max_scale = layer_weights.max(axis=-1)[..., np.newaxis] # 1*100
    min_scale = layer_weights.min(axis=-1)[..., np.newaxis] # 1*100
    
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype('uint8')


def make_mosaic(layer_weights):
    # Dirty hack (TM)
    lw_shape = layer_weights.shape
    lw = make_visual(layer_weights).reshape(10, 10, *lw_shape[1:])
    lw = lw.transpose(0, 2, 1, 3)
    lw = lw.reshape(10 * lw_shape[-2], 10 * lw_shape[-1])
    return lw


def plot_filters(layer_weights, title=None, show=False):
    mosaic = make_mosaic(layer_weights)
    plt.imshow(mosaic, interpolation='nearest', cmap='gray')
    ax = plt.gca()
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def plot_filters(layer_weights,show=False):
    plt.clf()
    fig = plt.figure(1, (32., 32.))
    plt.gray()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                 axes_pad=0.05,  # pad between axes in inch.
                 share_all=True,
                 )

    for i in range(100):
        grid[i].imshow(layer_weights[i], interpolation='nearest', cmap='gray')  # The AxesGrid object work as a list of axes.


