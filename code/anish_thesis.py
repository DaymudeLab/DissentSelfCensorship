# Project:  CensorshipDissent
# Filename: anish_thesis.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
anish_thesis: Network simulation experiment for Anish Nahar's honors thesis.
"""

from engine import engine
from helper import dump_np, load_np

import argparse
from itertools import repeat
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from tqdm.contrib.concurrent import process_map


def anish_thesis_worker(idx, T, N, R, pis, psis, mu_deltas, beta, nus, tau,
                        seeds):
    """
    Parallel worker function for anish_thesis.

    :param idx: a tuple of int indices specifying the parameters for this run
    :param T: an int number of trials to run for this set of parameters
    :param N: an int number of individuals
    :param R: an int number of rounds to run per trial
    :param pis: a list of string punishment functions
    :param psis: a list of the authority's float punishment severities (> 0)
    :param mu_deltas: a list of individuals' float mean desired dissents (in [0,1])
    :param beta: the individuals' float boldness (> 0)
    :param nu: a list of the authority's float surveillances (in [0,1])
    :param tau: the authority's float tolerance (in [0,1])
    :param seeds: a list of int seeds for random number generation

    :returns: a tuple (idx, deltafs, actfs):
        - idx: the tuple of int indices given as input
        - deltafs: a TxN array of final desired dissents for each trial/individual
        - actfs: a TxN array of final actions for each trial/individual
    """
    # Unpack parameters.
    pi = pis[idx[0]]
    psi = psis[idx[1]]
    mu_delta = mu_deltas[idx[2]]
    nu = nus[idx[3]]

    deltafs, actfs = np.zeros((T, N)), np.zeros((T, N))

    for t in range(T):
        rng = np.random.default_rng(seeds[t])
        deltas = np.minimum(np.maximum(rng.normal(mu_delta, 0.1, N), 0), 1)
        _, delta_hist, _, act_hist = engine(
            N=N, R=R, rule='d2a', w=0.5, deg0=5, ptri=0.25, deltas=deltas,
            betas=np.repeat(beta, N), nu=nu, pi=pi, tau=tau, sigma_tau=0.05,
            psi=psi, sigma_psi=0.05, seed=seeds[t])
        deltafs[t], actfs[t] = delta_hist[:,R-1], act_hist[:,R-1]

    return (idx, deltafs, actfs)


def plot_heatmap(ax, results, mu_deltas, nus):
    """
    Plot a heatmap of mean final desired dissents and actions as a function of
    initial mean desired dissent and surveillance.

    :param ax: a matplotlib.Axes object to plot on
    :param results: a (#mu_deltas)x(#nus)xTxN array of final desired dissents or
                    actions for each trial/individual
    :param mu_deltas: a list of individuals' float mean initial desired dissents
                      (in [0,1])
    :param nus: a list of the authority's float surveillances (in [0,1])
    """
    # Compute the mean final desired dissent across trials and individuals for
    # each initial mean desired dissent and surveillance.
    means = np.zeros((len(mu_deltas), len(nus)))
    for idx in np.ndindex(len(mu_deltas), len(nus)):
        means[idx[::-1]] = results[idx].mean()

    # Plot the changes as colors.
    im = ax.pcolormesh(mu_deltas, nus, means, cmap='YlOrRd', shading='auto',
                       vmin=0, vmax=1)

    # Set axes information that is common across plots.
    ax.set(xlim=[min(mu_deltas), max(mu_deltas)], ylim=[0, 1])
    ax.set_box_aspect(1)

    return im


def anish_thesis(seed=73462379, num_cores=1):
    """
    An experiment investigating individuals' convergence mean desired dissents
    and actions across a variety of parameter sweeps.

    :param seed: an int seed for random number generation
    :param num_cores: an int number of processors to parallelize over
    """
    # Numbers of trials, individuals, and rounds.
    T = 25
    N = 500
    R = 100

    # Sweep parameters.
    pis = ['constant', 'linear']
    psis = [0.5, 1, 1.5]
    mu_deltas = np.linspace(0.1, 0.9, 50)
    nus = np.linspace(0, 1, 50)

    # Fixed parameters.
    beta = 1
    tau = 0.25

    # Try to load the results files. If they don't exist, compute them and write
    # them to file for next time.
    deltas_fname = osp.join('results', 'anish_thesis_deltas.npy')
    acts_fname = osp.join('results', 'anish_thesis_acts.npy')
    try:
        deltaf_all = load_np(deltas_fname)
        actf_all = load_np(acts_fname)
    except FileNotFoundError:
        # Set up array to hold all of the data.
        deltaf_all = np.zeros((len(pis), len(psis), len(mu_deltas), len(nus), T,
                               N), dtype=np.float32)
        actf_all = np.copy(deltaf_all)

        # Set up random seeds for each trial.
        seeds = np.random.default_rng(seed).integers(0, 2**32, size=T)

        # Parallelize the experiment sweep and trials.
        idxs = list(np.ndindex(len(pis), len(psis), len(mu_deltas), len(nus)))
        p = process_map(anish_thesis_worker, idxs, repeat(T), repeat(N),
                        repeat(R), repeat(pis), repeat(psis), repeat(mu_deltas),
                        repeat(beta), repeat(nus), repeat(tau), repeat(seeds),
                        max_workers=num_cores)
        for idx, deltafs, actfs in p:
            deltaf_all[idx] = deltafs
            actf_all[idx] = actfs

        # Save results for next time.
        dump_np(deltas_fname, deltaf_all)
        dump_np(acts_fname, actf_all)

    # Plot the mean final desired dissent and action heatmaps for each severity-
    # boldness case and punishment function.
    labels = [r'(A) $\psi < \beta_i$',
              r'(B) $\psi = \beta_i$',
              r'(C) $\psi > \beta_i$']
    for i, results in enumerate([deltaf_all, actf_all]):
        fig, ax = plt.subplots(2, 3, figsize=(10, 5.5), dpi=300, sharex='col',
                               sharey='row', facecolor='white',
                               layout='constrained')
        for j, label in enumerate(labels):
            im = plot_heatmap(ax[0, j], results[0, j], mu_deltas, nus)
            im = plot_heatmap(ax[1, j], results[1, j], mu_deltas, nus)
            ax[0, j].set_title(label, weight='bold')
            ax[1, j].set_xticks([min(mu_deltas), tau, max(mu_deltas)])
            ax[1, j].set_xticklabels([f'{min(mu_deltas)}', r'$\tau$',
                                    f'{max(mu_deltas)}'])
        ax[0, 0].set(ylabel=r'Constant Punishment $\pi$')
        ax[1, 0].set(ylabel=r'Linear Punishment $\pi$')
        fig.supxlabel(r'Mean Initial Desired Dissent $\mu_\delta$')
        fig.supylabel(r'Surveillance $\nu$')
        cbar = fig.colorbar(im, ax=ax.ravel().tolist())
        if i == 0:
            cbar.set_label('Mean Final Desired Dissent')
            fig.savefig(osp.join('..', 'figs', 'anish_thesis_deltas.png'))
        else:
            cbar.set_label('Mean Final Action')
            fig.savefig(osp.join('..', 'figs', 'anish_thesis_acts.png'))


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-R', '--rand_seed', type=int, default=73462379,
                        help='Seed for random number generation')
    parser.add_argument('-P', '--num_cores', type=int, default=1,
                        help='Number of cores to parallelize over')
    args = parser.parse_args()

    # Run selected experiments.
    anish_thesis(args.rand_seed, args.num_cores)
