# Project:  CensorshipDissent
# Filename: experiments.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
experiments: TODO
"""

from engine import engine
from helper import dump_np

import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def anish_thesis(seed=None, num_cores=1):
    """
    TODO
    """
    # Fixed parameters.
    # T = 10
    T = 2
    # N = 250
    N = 25
    R = 100
    rule = 'd2a'
    w = 0.5
    # deg = 5
    deg = 3
    ptri = 0.25
    betas = np.repeat(1, N)
    tau = 0.25
    sigma_tau = 0.05
    sigma_psi = 0.05

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

    # Perform the experiment.
    for i, j, k, l in tqdm(np.ndindex(len(pis), len(psis), len(mu_deltas), len(nus))):
        pi, psi, mu_delta, nu = pis[i], psis[j], mu_deltas[k], nus[l]
        for t in range(T):
            rng = np.random.default_rng(seeds[t])
            deltas = np.minimum(np.maximum(rng.normal(mu_delta, 0.1, N), 0), 1)
            _, delta_hist, act_hist = engine(N, R, rule, w, deg, ptri, deltas,
                                             betas, nu, pi, tau, sigma_tau, psi, sigma_psi, seeds[t])
            deltaf_all[i, j, k, l, t] = delta_hist[:,R-1]
            actf_all[i, j, k, l, t] = act_hist[:,R-1]

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
