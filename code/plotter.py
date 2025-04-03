# Project:  CensorshipDissent
# Filename: plotter.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

from opttaunormal import *

import matplotlib.pyplot as plt
import os.path as osp


def opt_tau_dissentnormal(N=100, num_samples=1, seed=None, num_cores=1, anno=''):
    """
    Plot the authority's mean optimal dissent threshold as a function of
    punishment severity and individual boldness when individuals' desires to
    dissent are known and normally distributed.

    :param N: an int number of individuals
    :param num_samples: an int number of samples to generate
    :param seed: an int seed for random number generation
    :param num_cores: an int number of processors to parallelize over
    :param anno: a string filename annotation
    """
    # Load authority's optimal dissent thresholds and average over samples.
    ss, beta, mus, sigmas, opt_taus = opttaunormal(N, num_samples, seed, num_cores)
    opt_taus = np.mean(opt_taus, axis=-1)

    # Set up the figure.
    fig, ax = plt.subplots(1, len(ss), figsize=(1.5*len(ss), 4.5), dpi=300, \
                           facecolor='white', sharex=True, sharey=True, \
                           constrained_layout=True)

    # Plot each heatmap.
    decfmt = np.vectorize(lambda x: '{:.2f}'.format(x))
    xticks = np.arange(0, len(sigmas), 4)
    yticks = np.arange(0, len(mus), 5)
    for i, s in enumerate(ss):
        im = ax[i].imshow(opt_taus[i], cmap='Reds_r', vmin=0, vmax=1)
        ax[i].set(title=r'$s/\beta = ${}'.format(s / beta), \
                  xticks=xticks, xticklabels=decfmt(sigmas[xticks]), \
                  yticks=yticks, yticklabels=decfmt(mus[yticks]))

    # Add major axes labels.
    fig.supxlabel(r'Dissent Desires Stddev., $\sigma$')
    fig.supylabel(r'Mean Dissent Desire, $\mu$')

    # Add a colorbar for all subplots.
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.ax.set_ylabel(r'Opt. Dissent Threshold, $\tau^*$', rotation=-90, \
                       va='bottom')

    # Crop and save.
    fig.savefig(osp.join('figs', 'opt_tau_dissentnormal' + anno + '.png'))
