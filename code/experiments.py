# Project:  CensorshipDissent
# Filename: experiments.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
experiments: Network simulation experiments for the Censorship-Dissent model.
"""

from engine import engine
from helper import dump_np

import argparse
from itertools import repeat
import numpy as np
import os.path as osp
from tqdm.contrib.concurrent import process_map


def anish_thesis_worker(idx, T, N, R, pis, psis, mu_deltas, nus, seeds):
    """
    Parallel worker function for anish_thesis.

    :param idx: a tuple of int indices specifying the parameters for this run
    :param T: an int number of trials to run for this set of parameters
    :param N: an int number of individuals
    :param R: an int number of rounds to run per trial
    :param pis: a list of string punishment functions
    :param psis: a list of the authority's float punishment severities (> 0)
    :param mu_deltas: a list of individuals' float mean desired dissents (in [0,1])
    :param nu: a list of the authority's float surveillances (in [0,1])
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
        _, delta_hist, act_hist = engine(
            N=N, R=R, rule='d2a', w=0.5, deg=3, ptri=0.25, deltas=deltas,
            betas=np.repeat(1, N), nu=nu, pi=pi, tau=0.25, sigma_tau=0.05,
            psi=psi, sigma_psi=0.05, seed=seeds[t])
        deltafs[t], actfs[t] = delta_hist[:,R-1], act_hist[:,R-1]

    return (idx, deltafs, actfs)


def anish_thesis(seed=None, num_cores=1):
    """
    An experiment investigating individuals' convergence mean desired dissents
    and actions across a variety of parameter sweeps.

    :param seed: an int seed for random number generation
    :param num_cores: an int number of processors to parallelize over
    """
    # Numbers of trials, individuals, and rounds.
    T = 10
    N = 250
    R = 100

    # Sweep parameters.
    pis = ['constant', 'linear']
    psis = [0.5, 1, 1.5]
    mu_deltas = np.linspace(0.1, 0.9, 25)
    nus = np.linspace(0, 1, 25)

    # Set up array to hold all of the data.
    deltaf_all = np.zeros((len(pis), len(psis), len(mu_deltas), len(nus), T, N),
                          dtype=np.float32)
    actf_all = np.copy(deltaf_all)

    # Set up random seeds for each trial.
    seeds = np.random.default_rng(seed).integers(0, 2**32, size=T)

    # Parallelize the experiment sweep and trials.
    idxs = list(np.ndindex(len(pis), len(psis), len(mu_deltas), len(nus)))
    p = process_map(anish_thesis_worker, idxs, repeat(T), repeat(N), repeat(R),
                    repeat(pis), repeat(psis), repeat(mu_deltas), repeat(nus),
                    repeat(seeds), max_workers=num_cores)
    for idx, deltafs, actfs in p:
        deltaf_all[idx] = deltafs
        actf_all[idx] = actfs

    # Save results.
    dump_np(osp.join('results', 'anish_thesis_deltas.npy'), deltaf_all)
    dump_np(osp.join('results', 'anish_thesis_acts.npy'), actf_all)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-E', '--exps', type=str, nargs='+', required=True, \
                        help='IDs of experiments to run')
    parser.add_argument('-R', '--rand_seed', type=int, default=None, \
                        help='Seed for random number generation')
    parser.add_argument('-P', '--num_cores', type=int, default=1, \
                        help='Number of cores to parallelize over')
    args = parser.parse_args()

    # Run selected experiments.
    exps = {'anish_thesis': anish_thesis}
    for id in args.exps:
        exps[id](args.rand_seed, args.num_cores)
