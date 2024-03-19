# Project:  CensorshipDissent
# Filename: opttaunormal.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
Computationally evaluate the authority's optimal dissent threshold when
individuals' desires to dissent are normally distributed
"""

from helper import *

import argparse
from itertools import product, repeat
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def opttaunormal_trial(N, s, beta, mu, sigma, seed=None):
    """
    Generates normally distributed dissent desires with the given mean and
    standard deviation (truncated to [0,1]) and computes the authority's
    corresponding optimal dissent threshold (tau).

    :param N: an int number of individuals
    :param s: a float authority's punishment severity (> 0)
    :param beta: a float boldness constant (> 0)
    :param mu: a float mean individual desire to dissent
    :param sigma: a float standard deviation of individuals' desires to dissent
    :param seed: an int seed for random number generation
    :returns: a float authority's optimal dissent threshold
    """

    # Sample individuals' dissent desires via rejection sampling.
    rng = np.random.default_rng(seed)
    d = []
    while len(d) < N:
        vals = rng.normal(mu, sigma, size=1000)
        d += list(vals[(vals >= 0) & (vals <= 1)])
    d = np.array(d[:N])

    # Generate critical values of tau for the authority's utility.
    taus = set(list(d) + list(np.maximum(np.zeros(N), d - s/beta)))
    taus = np.array(list(taus))

    # Define a single individual's contribution to the authority's utility.
    def auth_utility_i(d_i, tau):
        if tau < d_i - s/beta:
            return -d_i - s
        elif d_i - s/beta <= tau and tau < d_i:
            return -tau
        else:  # tau >= d_i
            return -d_i
    vauth_utility_i = np.vectorize(auth_utility_i)

    # Evaluate the optimal tau for this distribution.
    auth_utilities = np.array([np.sum(vauth_utility_i(d, tau)) for tau in taus])
    opt_tau = taus[np.argmax(auth_utilities)]

    return opt_tau


def opttaunormal_worker(params, beta, N, num_samples, seed):
    """
    Evaluates the authority's optimal dissent threshold when individuals'
    desires to dissent are normally distributed for a given set of parameters.

    :param params: a tuple including
        :param s: a float authority's punishment severity (> 0)
        :param mu: a float mean individual desire to dissent
        :param sigma: a float stddev of individuals' desires to dissent
    :param beta: a float boldness constant (> 0)
    :param N: an int number of individuals
    :param num_samples: an int number of samples to generate
    :param seed: an int seed for random number generation
    :returns: a tuple keying the authority's optimal dissent thresholds over
              the sampled trials
    """
    s, mu, sigma = params
    seeds = np.random.default_rng(seed).integers(0, 2**32, size=num_samples)
    opt_taus = [opttaunormal_trial(N, s, beta, mu, sigma, seed) for seed in seeds]

    return (s, mu, sigma, opt_taus)


def opttaunormal(N=100, num_samples=1, seed=None, num_cores=1):
    """
    Evaluates the authority's optimal dissent threshold when individuals'
    desires to dissent are normally distributed over a range of parameters.

    :param N: an int number of individuals
    :param num_samples: an int number of samples to generate
    :param seed: an int seed for random number generation
    :param num_cores: an int number of processors to parallelize over
    :returns: the parameters used and the authority's optimal dissent thresholds
    """
    # Set up experiment parameter ranges.
    ss = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    beta = 1
    mus = np.arange(-0.25, 1.251, 0.05)
    sigmas = np.arange(0.1, 0.51, 0.05)

    fname = osp.join('results', 'opttaunormal_N' + str(N) + '_S' + \
                     str(num_samples) + '_R' + str(seed) + '.npy')
    try:  # Try to load the pre-computed optimal tau results from file.
        opt_taus = load_np(fname)
    except FileNotFoundError:  # If they don't exist, compute and store them.
        tqdm.write('Evaluating optimal authority thresholds...')
        p = process_map(opttaunormal_worker, product(ss, mus, sigmas), \
                        repeat(beta), repeat(N), repeat(num_samples), \
                        repeat(seed), max_workers=num_cores)
        opt_taus = np.array([x[3] for x in sorted(p, key=lambda x: (x[:3]))])\
                     .reshape(len(ss), len(mus), len(sigmas), num_samples)
        dump_np(fname, opt_taus)

    return ss, beta, mus, sigmas, opt_taus


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-N', '--num_individuals', type=int, default=100, \
                        help='Number of individuals in the population')
    parser.add_argument('-S', '--num_samples', type=int, default=1, \
                        help='Number of samples to generate')
    parser.add_argument('-R', '--rand_seed', type=int, default=None, \
                        help='Seed for random number generation')
    parser.add_argument('-P', '--num_cores', type=int, default=1, \
                        help='Number of cores to parallelize over')
    args = parser.parse_args()

    opttaunormal(args.num_individuals, args.num_samples, args.rand_seed, \
                 args.num_cores)
