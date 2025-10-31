# Project:  CensorshipDissent
# Filename: hillclimbing.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
hillclimbing: An adaptive authority experiment using random hill climbing.
"""

from opt_action import opt_actions

import argparse
from cmcrameri import cm
from helper import dump_np, load_np
from itertools import product, repeat
from math import expm1, isclose
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

#bayesian optimization imports
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from skopt.space import Real



def texponential(rng, bound, scale, size):
    """
    Generates `size` samples from a truncated exponential distribution with
    mean `scale` and range [0, bound]. The underlying sampling method uses
    rejection sampling from a non-truncated exponential distribution whose mean
    is approximated to yield the desired truncated distribution's mean.

    :param rng: a numpy.random.Generator instance for random number generation
    :param bound: a float upper bound of the truncated exponential distribution
    :param scale: a float mean of the truncated exponential distribution; must
    be in (0, bound / 2)
    :param size: an int number of samples to generate
    :returns: a 1xsize array of float random samples
    """
    # Use bisection search to obtain the mean of an exponential distribution
    # that, when rejection sampled to [0, bound], effectively samples from the
    # truncated exponential distribution on [0, bound] with the desired mean.
    # Specifically, the mean X of the desired truncated exponential and the
    # mean Y of the corresponding non-truncated exponential are related as:
    # X = Y - bound / (exp(bound / Y) - 1). This method approximates Y given X.
    lower, upper = scale, 2*scale
    while upper - bound / expm1(bound / upper) < scale:
        lower = upper
        upper *= 2
    while True:
        mid = (upper + lower) / 2
        approx_mean = mid - bound / expm1(bound / mid)
        if isclose(scale, approx_mean):
            break
        elif approx_mean < scale:
            lower = mid
        else:  # approx_mean >= scale
            upper = mid

    # Generate the desired number of samples using rejection sampling.
    samples = np.array([])
    while len(samples) < size:
        x = rng.exponential(scale=mid, size=size)
        samples = np.append(samples, x[np.where(x <= bound)])

    return samples[:size]

def run_bayesian_optimization(N, R, delta, beta, pi, alpha, seed):
    """
    :param N: int number of individuals
    :param R_: int number of calls (budget) for gp_minimize
    :param delta: float mean population desired dissent
    :param beta: float mean population boldness
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: float authority's adamancy
    :param seed: int seed for this trial's random number generator
    """

    # Set up random number generation.
    rng = np.random.default_rng(seed)

    # Initialize the population's desired dissents according to an exponential
    # distribution truncated to [0, 1] with the given mean.
    deltas = texponential(rng, bound=1, scale=delta, size=N)

    # Initialize the population's boldness constants according to an
    # exponential distribution with the given mean.
    betas = rng.exponential(scale=beta, size=N)

    dimensions = [
        Real(0.0, 1.0, name='tau'),
        Real(1e-9, 20.0, name='psi'), # Using 20 as upper bound for now
        Real(0.0, 1.0, name='nu')
    ]

    def eval_authority_costs(params):
        tau = params[0]
        psi = params[1] 
        nu = params[2]  

        acts = opt_actions(deltas, betas, nu, pi, tau, psi)
        cond = (acts > tau) & (rng.random(N) < (nu + (1 - nu) * acts))

        if pi == 'uniform':
            punish = cond * psi
        elif pi == 'proportional':
            punish = cond * psi * (acts - tau)
        
        pol_cost = acts.sum()
        pun_cost = punish.sum()

        # authority's adamancy-weighted cost
        total_cost = alpha * pol_cost + pun_cost

        return (total_cost, pol_cost, pun_cost)
    
    # gp_minimize needs a function that returns the single it is trying to minimize (total cost/negative utility)
    def objective_func(params):
        total_cost, _, _ = eval_authority_costs(params)
        return total_cost

    n_warmup = max(10, int(R * 0.2)) # 20% warmup

    res = gp_minimize(func=objective_func, dimensions=dimensions, n_calls=R, 
                      n_initial_points=n_warmup, noise="gaussian", random_state=seed)
    
    best_params = res.x
    best_total_cost, best_pol_cost, best_pun_cost = eval_authority_costs(best_params)

    return (best_total_cost, best_pol_cost, best_pun_cost, best_params, res)


def bo_sweep_worker(idx, db, N, R_calls, pi, alpha, seeds):
    """
    Worker function handling the repeated BO trials for a single setting of
    (delta, beta). This replaces the old sweep_worker

    :param idx: tuple (i, j) index for this (delta, beta) pair
    :param db: tuple (delta, beta) values
    :param N: int number of individuals
    :param R_calls: int number of calls (budget) for gp_minimize
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: float authority's adamancy
    :param seeds: a 1xT array of int seeds, one for each trial
    :returns: (idx, params_mean, params_std, costs_mean, costs_std)
    """
    # Set up worker-specific results arrays
    w_params = np.zeros((len(seeds), 3))
    w_costs = np.zeros(len(seeds))       
    w_pol_costs = np.zeros(len(seeds))   
    w_pun_costs = np.zeros(len(seeds))

    # Get the (delta, beta) for this worker
    delta, beta = db

    # Run the specified number of trials for this parameter setting
    for t in range(len(seeds)):
        # Call your new function for each trial
        best_total_cost, best_pol_cost, best_pun_cost, best_params_list, _ = run_bayesian_optimization(
            N, R_calls, delta, beta, pi, alpha, seeds[t]
        )
        
        # Store the single best result for this trial
        w_costs[t] = best_total_cost
        w_pol_costs[t] = best_pol_cost
        w_pun_costs[t] = best_pun_cost
        w_params[t, :] = best_params_list

    # Return the index + means/standard deviations across all trials
    return (idx, 
            w_params.mean(axis=0), w_params.std(axis=0),
            w_costs.mean(axis=0), w_costs.std(axis=0),
            w_pol_costs.mean(axis=0), w_pol_costs.std(axis=0),
            w_pun_costs.mean(axis=0), w_pun_costs.std(axis=0))


def bo_sweep(N, R, pi, alpha, seed, granularity, trials, threads):   
    """
    Varying the population's mean desired dissent and boldness, and running a
    full Bayesian Optimization for each to find the authority's optimal costs and params

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param seed: an int seed for random number generation
    :param granularity: an int number of delta and beta values to sweep over
    :param trials: an int number of trials to run per parameter setting
    :param threads: an int number of threads to parallelize over
    """

    # Set up the independent variables.
    deltas = np.linspace(0.005, 0.495, granularity)
    betas = np.linspace(0.1, 10, granularity)

    params_res = np.zeros((granularity, granularity, trials, 3)) # tau, psi, nu
    costs_res = np.zeros((granularity, granularity, trials)) # total_cost

    # Set up random seeds for the trials
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32, size=trials)

    # Set up results containers
    # We store mean/std of final params and costs
    params = np.zeros((granularity, granularity, 2, 3))
    costs = np.zeros((granularity, granularity, 2))
    pol_costs = np.zeros((granularity, granularity, 2))
    pun_costs = np.zeros((granularity, granularity, 2))

    # Run the experiment with the specified number of parallel threads
    idxs = list(product(range(granularity), range(granularity)))
    dbs = list(product(deltas, betas))

    p = process_map(bo_sweep_worker, idxs, dbs, repeat(N), repeat(R),
                    repeat(pi), repeat(alpha), repeat(seeds),
                    max_workers=threads, chunksize=1)
    
    # Collect and store results
    for (i, j), w_params_mean, w_params_std, \
                w_costs_mean, w_costs_std, \
                w_pol_costs_mean, w_pol_costs_std, \
                w_pun_costs_mean, w_pun_costs_std in p:
        
        params[i, j, 0] = w_params_mean
        params[i, j, 1] = w_params_std
        costs[i, j, 0] = w_costs_mean
        costs[i, j, 1] = w_costs_std
        pol_costs[i, j, 0] = w_pol_costs_mean
        pol_costs[i, j, 1] = w_pol_costs_std
        pun_costs[i, j, 0] = w_pun_costs_mean
        pun_costs[i, j, 1] = w_pun_costs_std

    # Dump all results to file
    resultsdir = osp.join('..', 'results', f'BO_sweep_N{N}_R{R}_{pi}_S{seed}')
    dump_np(osp.join(resultsdir, 'deltas.npy'), deltas)
    dump_np(osp.join(resultsdir, 'betas.npy'), betas)
    dump_np(osp.join(resultsdir, 'params.npy'), params)
    dump_np(osp.join(resultsdir, 'costs.npy'), costs)
    dump_np(osp.join(resultsdir, 'pol_costs.npy'), pol_costs)
    dump_np(osp.join(resultsdir, 'pun_costs.npy'), pun_costs)


def plot_sweep(N, R, pi, alpha, seed):
    """
    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation
    """
    # Load results means from file.
    resultsdir = osp.join('..', 'results', f'BO_sweep_N{N}_R{R}_{pi}_S{seed}')
    deltas = load_np(osp.join(resultsdir, 'deltas.npy'))
    betas = load_np(osp.join(resultsdir, 'betas.npy'))

    params = load_np(osp.join(resultsdir, 'params.npy'))[:, :, 0]
    pol_costs = alpha * load_np(osp.join(resultsdir, 'pol_costs.npy'))[:, :, 0]
    pun_costs = load_np(osp.join(resultsdir, 'pun_costs.npy'))[:, :, 0]
    total_costs = pol_costs + pun_costs

    # Set up the figure.
    fig = plt.figure(figsize=(6.8, 8), dpi=300, facecolor='w',
                     layout='constrained')
    gs = fig.add_gridspec(3, 2)
    axes_bd = [fig.add_subplot(gs[i[::-1]])
               for i in product(range(2), range(3))]

    # Plot average final parameter values and final costs.
    data = [pol_costs, pun_costs, total_costs,
            params[:, :, 0], # tau
            params[:, :, 1], # psi
            params[:, :, 2]] # nu
    
    lims = [(0, None), (0, None), (0, None), (0, 1), (0, None), (0, 1)]
    cmaps = [cm.devon_r, cm.bilbao_r, cm.batlowW_r, 'Blues', 'Reds', 'Greens']
    lbls = ['(A) Political Cost', '(B) Punishment Cost', '(C) Total Cost',
            r'(D) Tolerance $\tau_{final}$', r'(E) Severity $\psi_{final}$',
            r'(F) Surveillance $\nu_{final}$']
    for i, (axi, datum, (pmin, pmax), cmap, lbl) in \
            enumerate(zip(axes_bd, data, lims, cmaps, lbls)):
        im = axi.pcolormesh(deltas, betas, datum.T, vmin=pmin, vmax=pmax,
                            cmap=cmap, shading='auto')
        fig.colorbar(im, ax=axi)
        axi.set_title(lbl, weight='bold')
        if i in [2, 5]:
            axi.set_xlabel(r'Mean Desired Dissent $\delta$')
        else:
            axi.tick_params(labelbottom=False)
        if i <= 2:
            axi.set_ylabel(r'Mean Boldness $\beta$')
        else:
            axi.tick_params(labelleft=False)

    fig.savefig(osp.join('..', 'figs', f'BO_sweep_N{N}_R{R}_{pi}_S{seed}.png'))


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', action='store_true',
                        help=('If present, run the sweep experiment (ignores '
                              '--delta, --beta, --tau, --psi, and --nu); '
                              'otherwise; runs one independent trial (ignores '
                              '--granularity, --trials, --threads)'))
    parser.add_argument('-N', '--num_ind', type=int, default=100000,
                        help='Number of individuals in the population')
    parser.add_argument('-R', '--n_calls', type=int, default=100,
                        help='Number of rounds to simulate in a single trial')
    parser.add_argument('-D', '--delta', type=float, default=0.25,
                        help='Mean population desired dissent > 0')
    parser.add_argument('-B', '--beta', type=float, default=0.5,
                        help='Mean population boldness > 0')
    parser.add_argument('-P', '--pi', choices=['uniform', 'proportional'],
                        default='uniform', help='Punishment function')
    parser.add_argument('-A', '--alpha', type=float, default=1.0,
                        help='Authority\'s adamancy > 0')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random number generation')
    parser.add_argument('--granularity', type=int, default=50,
                        help='Number of parameter values to sweep over')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials to run per parameter setting')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to parallelize over')
    args = parser.parse_args()

    # Run a single trial or sweep experiment.
    rng = np.random.default_rng(args.seed)
    if args.sweep:
        bo_sweep(N=args.num_ind, R=args.n_calls, pi=args.pi, alpha=args.alpha,
                 seed=args.seed, granularity=args.granularity, 
                 trials=args.trials, threads=args.threads)
        plot_sweep(N=args.num_ind, R=args.n_calls, pi=args.pi, alpha=args.alpha, seed=args.seed)
        
    else:
        (best_total_cost, best_pol_cost, best_pun_cost, best_params, res) = \
            run_bayesian_optimization(
                N=args.num_ind, R=args.n_calls, delta=args.delta,
                beta=args.beta, pi=args.pi, alpha=args.alpha,
                seed=args.seed
            )
        
        print(f"Lowest Total Cost: {best_total_cost:.4f}")
        print(f"  Political Cost (raw): {best_pol_cost:.4f}")
        print(f"  Punishment Cost: {best_pun_cost:.4f}")
        print(f"  Optimal tau: {best_params[0]:.4f}")
        print(f"  Optimal psi: {best_params[1]:.4f}")
        print(f"  Optimal nu: {best_params[2]:.4f}")
        
        # This plots the "Best cost so far" vs. "Number of calls"
        plot_convergence(res)
        
        # This plots all the points the optimizer tried
        plot_evaluations(res, dimensions=['tau', 'psi', 'nu'])
        
        # This shows the optimizer's "Best Guess Map"
        plot_objective(res, dimensions=['tau', 'psi', 'nu'])
        
        # This makes the plots appear on your screen
        plt.show()
    
