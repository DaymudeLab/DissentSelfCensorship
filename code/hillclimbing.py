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
from tqdm.contrib.concurrent import process_map


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


def rmhc_trial(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, seed):
    """
    Runs a single simulation trial of the model where individuals' desired
    dissents and boldness constants are exponentially-distributed but fixed and
    the authority adapts its parameters based on random mutation hill climbing.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param delta: a float mean population desired dissent (> 0)
    :param beta: a float mean population boldness (> 0)
    :param pi: 'uniform' or 'variable' punishment
    :param tau0: the authority's float initial tolerance (in [0,1])
    :param psi0: the authority's float initial punishment severity (> 0)
    :param nu0: the authority's float initial surveillance (in [0,1])
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation

    :returns: a 3xR array of the authority's parameter values
    :returns: a 1xR array of the authority's political costs
    :returns: a 1xR array of the authority's punishment costs
    :returns: a 1xN array of individuals' dissent desires
    :returns: a 1xN array of individuals' boldness constants
    """
    # Set up random number generation.
    rng = np.random.default_rng(seed)

    # Initialize the population's desired dissents according to an exponential
    # distribution truncated to [0, 1] with the given mean.
    deltas = texponential(rng, bound=1, scale=delta, size=N)

    # Initialize the population's boldness constants according to an
    # exponential distribution with the given mean.
    betas = rng.exponential(scale=beta, size=N)

    # Set bounds on the authority's parameters.
    bounds = np.array([[0, 1],          # tau
                       [1e-9, np.inf],  # psi
                       [0, 1]])         # nu

    # Set up arrays to store everything that happens.
    params = np.zeros((3, R))
    pol_costs, pun_costs = np.zeros(R), np.zeros(R)

    # Pre-generate all random choices of which parameter to attempt to update
    # at each step; the hope is that doing this in batch is faster than doing
    # one at a time in each for loop iteration.
    param_choices = rng.integers(3, size=R)

    # Simulate the specified number of rounds, allowing the authority to adapt
    # its parameters using random mutation hill climbing (RMHC).
    for r in range(R):
        # If this is the first round, the authority simply uses its initial
        # parameters. Otherwise, it generates new candidate parameters to test.
        if r == 0:
            params[:, r] = [tau0, psi0, nu0]
        else:
            p = param_choices[r]  # Choose parameter to update.
            params[:, r] = params[:, r-1]
            params[p, r] = rng.uniform(max(bounds[p, 0], params[p, r] - eps),
                                       min(bounds[p, 1], params[p, r] + eps))
        tau, psi, nu = params[:, r]

        # The individuals act based on their desires and boldness constants and
        # the authority's current parameters.
        acts = opt_actions(deltas, betas, nu, pi, tau, psi)

        # The authority punishes any actions that it observes above tolerance.
        cond = (acts > tau) & (rng.random(N) < (nu + (1 - nu) * acts))
        if pi == 'uniform':
            punish = cond * psi
        elif pi == 'variable':
            punish = cond * psi * (acts - tau)
        else:
            assert False, f'ERROR: Invalid punishment function \"{pi}\"'

        # The authority's political cost for this round is the summed actions
        # and its punishment cost is the summed punishments.
        pol_costs[r] = acts.sum()
        pun_costs[r] = punish.sum()

        # If the authority's adamancy-weighted cost in this round is worse than
        # last round, reset to last round's parameters.
        if r > 0 and alpha * pol_costs[r] + pun_costs[r] > \
                alpha * pol_costs[r-1] + pun_costs[r-1]:
            params[:, r] = params[:, r-1]

    return params, pol_costs, pun_costs, deltas, betas


def sweep_worker(idx, db, N, R, pi, tau0s, psi0s, nu0s, alpha, eps, seeds):
    """
    Worker function handling the repeated RMHC trials for a single setting of
    (delta, beta).

    :param idx: a tuple (i, j) representing this parameter setting's index
    :param db: a tuple (delta, beta) of the float mean population desired
    dissent (> 0) and the float mean population boldness (> 0)
    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'variable' punishment
    :param tau0s: a 1xT array of the authority's float initial tolerances
    :param psi0s: a 1xT array of the authority's float initial severities
    :param nu0s: a 1xT array of the authority's float initial surveillances
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seeds: a 1xT array of int seeds for random number generation

    :returns: the tuple (i, j) representing this parameter setting's index
    :returns: a 3xR array of the authority's mean parameters per round
    :returns: a 3xR array of the authority's parameter standard deviations per
    round
    :returns: a 1xR array of the authority's mean political costs per round
    :returns: a 1xR array of the authority's political cost standard deviations
    per round
    :returns: a 1xR array of the authority's mean punishment costs per round
    :returns: a 1xR array of the authority's punishment cost standard
    deviations per round
    """
    # Set up worker-specific results arrays.
    w_params = np.zeros((len(seeds), 3, R))
    w_pol_costs = np.zeros((len(seeds), R))
    w_pun_costs = np.zeros((len(seeds), R))

    # Run the specified number of trials for this parameter setting.
    delta, beta = db
    for t in range(len(seeds)):
        w_params[t], w_pol_costs[t], w_pun_costs[t], _, _ = \
            rmhc_trial(N, R, delta, beta, pi, tau0s[t], psi0s[t], nu0s[t],
                       alpha, eps, seeds[t])

    # Return the index + means/standard deviations across trials.
    return (idx, w_params.mean(axis=0), w_params.std(axis=0),
            w_pol_costs.mean(axis=0), w_pol_costs.std(axis=0),
            w_pun_costs.mean(axis=0), w_pun_costs.std(axis=0))


def rmhc_sweep(N, R, pi, alpha, eps, seed, granularity, trials, threads):
    """
    Varying the population's mean desired dissent and boldness as independent
    variables and randomly initializing the authority's parameters, measure the
    authority's final parameter values after a fixed number of rounds. In each
    random trial, record the authority's parameters over time, the population's
    desired dissents and boldness constants, and the random seed. This is
    sufficient to reconstruct the entire trajectory of actions and punishments.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'variable' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation
    :param granularity: an int number of delta and beta values to sweep over
    :param trials: an int number of trials to run per parameter setting
    :param threads: an int number of threads to parallelize over
    """
    # Set up the independent variables.
    deltas = np.linspace(0.005, 0.495, granularity)
    betas = np.linspace(0.1, 10, granularity)

    # Set up random seeds and initial authority parameters for the trials.
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32, size=trials)
    tau0s = rng.random(size=trials)
    psi0s = rng.random(size=trials)
    nu0s = rng.random(size=trials)

    # Set up results containers: for each (delta, beta) pair, we store the mean
    # and standard deviation of the three parameters, political costs, and
    # punishment costs in each round across all trials.
    params = np.zeros((granularity, granularity, 2, 3, R))
    pol_costs = np.zeros((granularity, granularity, 2, R))
    pun_costs = np.zeros((granularity, granularity, 2, R))

    # Run the experiment with the specified number of parallel threads.
    idxs = list(product(range(granularity), range(granularity)))
    dbs = list(product(deltas, betas))
    p = process_map(sweep_worker, idxs, dbs, repeat(N), repeat(R), repeat(pi),
                    repeat(tau0s), repeat(psi0s), repeat(nu0s), repeat(alpha),
                    repeat(eps), repeat(seeds), max_workers=threads,
                    chunksize=1)
    for (i, j), w_params_mean, w_params_std, w_pol_costs_mean, \
            w_pol_costs_std, w_pun_costs_mean, w_pun_costs_std in p:
        params[i, j, 0] = w_params_mean
        params[i, j, 1] = w_params_std
        pol_costs[i, j, 0] = w_pol_costs_mean
        pol_costs[i, j, 1] = w_pol_costs_std
        pun_costs[i, j, 0] = w_pun_costs_mean
        pun_costs[i, j, 1] = w_pol_costs_std

    # Dump all results to file.
    resultsdir = osp.join('..', 'results', f'sweep_N{N}_R{R}_{pi}_S{seed}')
    dump_np(osp.join(resultsdir, 'deltas.npy'), deltas)
    dump_np(osp.join(resultsdir, 'betas.npy'), betas)
    dump_np(osp.join(resultsdir, 'params.npy'), params)
    dump_np(osp.join(resultsdir, 'pol_costs.npy'), pol_costs)
    dump_np(osp.join(resultsdir, 'pun_costs.npy'), pun_costs)


def plot_trial(taus, psis, nus, pol_costs, pun_costs, alpha, pi, delta, beta):
    """
    Plot the evolution of authority costs & parameters in a single RMHC trial.

    :param taus: a 1xR array of the authority's tolerance values
    :param psis: a 1xR array of the authority's severity values
    :param nus: a 1xR array of the authority's surveillance values
    :param pol_costs: a 1xR array of the authority's political costs
    :param pun_costs: a 1xR array of the authority's punishment costs
    :param alpha: the authority's float adamancy (> 0)
    :param pi: the authority's string punishment function
    :param delta: the float mean population desired dissent (> 0)
    :param beta: the float mean population boldness (> 0)
    """
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, dpi=300,
                           tight_layout=True)
    R = len(taus)

    # Plot costs (negative utility) over time.
    ax[0].plot(np.arange(R), alpha * pol_costs, label=r'$\alpha \times$'
               'Political Cost', c=cm.vikO(0.7))
    ax[0].plot(np.arange(R), pun_costs, label='Punishment Cost',
               c=cm.vikO(0.3))
    ax[0].plot(np.arange(R), alpha * pol_costs + pun_costs, label='Total Cost',
               c=cm.vikO(0))
    ax[0].legend()
    ax[0].set(title=r"Hill Climbing Authority ($\pi$ " f"= {pi}, " r"$\alpha$ "
              f"= {alpha}) vs. Population " r"$\delta_i \sim$" f"Exp({delta}),"
              r" $\beta_i \sim$" f"Exp({beta}" r"$^{-1}$)",
              ylabel='Costs')

    # Plot parameters over time.
    ax[1].plot(np.arange(R), taus, label=r'Tolerance $\tau$', c=cm.batlowS(2))
    ax[1].plot(np.arange(R), psis, label=r'Severity $\psi$', c=cm.batlowS(3))
    ax[1].plot(np.arange(R), nus, label=r'Surveillance $\nu$', c=cm.batlowS(4))
    ax[1].legend()
    ax[1].set(xlim=[0, R], xlabel='Round', ylabel='Parameter Value')

    fig.savefig(osp.join('..', 'figs', 'rmhc_trial.png'))


def plot_sweep(N, R, pi, alpha, eps, seed):
    """
    Plots the results of an RMHC sweep experiment showing the authority's
    final average parameter values and costs per (mean desired dissent, mean
    boldness) pair and the authority's total cost over time for a median
    desired dissent and all boldnesses.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'variable' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation
    """
    # Load results from file.
    resultsdir = osp.join('..', 'results', f'sweep_N{N}_R{R}_{pi}_S{seed}')
    deltas = load_np(osp.join(resultsdir, 'deltas.npy'))
    betas = load_np(osp.join(resultsdir, 'betas.npy'))
    params = load_np(osp.join(resultsdir, 'params.npy'))
    pol_costs = load_np(osp.join(resultsdir, 'pol_costs.npy'))
    pun_costs = load_np(osp.join(resultsdir, 'pun_costs.npy'))

    # Set up the figure.
    fig = plt.figure(figsize=(16, 5.1), dpi=300, facecolor='w',
                     layout='constrained')
    gs = fig.add_gridspec(2, 5)
    axes_bd = [fig.add_subplot(gs[i]) for i in product(range(2), range(3))]
    ax_tb = fig.add_subplot(gs[:, 3:5])

    # Plot average final parameter values and final costs.
    data = [alpha * pol_costs[:, :, 0, -1], pun_costs[:, :, 0, -1],
            alpha * pol_costs[:, :, 0, -1] + pun_costs[:, :, 0, -1],
            params[:, :, 0, 0, -1], params[:, :, 0, 1, -1],
            params[:, :, 0, 2, -1]]
    lims = [(0, None), (0, None), (0, None), (0, 1), (0, None), (0, 1)]
    cmaps = [cm.devon_r, cm.bilbao_r, cm.lipari_r, 'Blues', 'Reds', 'Greens']
    lbls = [r'(A) $\alpha \times$Political Cost', '(B) Punishment Cost',
            '(C) Total Cost', r'(D) Tolerance $\tau$', r'(E) Severity $\psi$',
            r'(F) Surveillance $\nu$']
    for i, (axi, datum, (pmin, pmax), cmap, lbl) in \
            enumerate(zip(axes_bd, data, lims, cmaps, lbls)):
        im = axi.pcolormesh(deltas, betas, datum.T, vmin=pmin, vmax=pmax,
                            cmap=cmap, shading='auto')
        fig.colorbar(im, ax=axi)
        axi.set_title(lbl, weight='bold')
        if i <= 2:
            axi.tick_params(labelbottom=False)
        else:
            axi.set_xlabel(r'Mean Desired Dissent $\delta$')
        if i in [0, 3]:
            axi.set_ylabel(r'Mean Boldness $\beta$')
        else:
            axi.tick_params(labelleft=False)

    # Plot average total cost vs. (round, mean boldness).
    d = len(deltas) // 2
    im = ax_tb.pcolormesh(np.arange(R), betas, alpha * pol_costs[d, :, 0] +
                          pun_costs[d, :, 0], vmin=0, vmax=None,
                          cmap=cm.lipari_r, shading='auto')
    fig.colorbar(im, ax=ax_tb)
    ax_tb.set_title(r'(G) Total Cost Over Time ($\delta$' +
                    f' = {deltas[d]:.2f})', weight='bold')
    ax_tb.set(xlim=(0, R), ylim=(0, betas.max()), xlabel='Rounds',
              ylabel=r'Mean Boldness $\beta$')

    fig.savefig(osp.join('..', 'figs', f'sweep_N{N}_R{R}_{pi}_S{seed}.png'))


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
    parser.add_argument('-R', '--rounds', type=int, default=10000,
                        help='Number of rounds to simulate in a single trial')
    parser.add_argument('-D', '--delta', type=float, default=0.25,
                        help='Mean population desired dissent > 0')
    parser.add_argument('-B', '--beta', type=float, default=0.5,
                        help='Mean population boldness > 0')
    parser.add_argument('-P', '--pi', choices=['uniform', 'variable'],
                        default='uniform', help='Punishment function')
    parser.add_argument('-T', '--tau', type=float, default=0.25,
                        help='Authority\'s initial tolerance in [0,1]')
    parser.add_argument('-S', '--psi', type=float, default=0.1,
                        help='Authority\'s initial severity > 0')
    parser.add_argument('-V', '--nu', type=float, default=0.1,
                        help='Authority\'s initial surveillance in [0,1]')
    parser.add_argument('-A', '--alpha', type=float, default=1.0,
                        help='Authority\'s adamancy > 0')
    parser.add_argument('-E', '--epsilon', type=float, default=0.05,
                        help='Window radius for authority parameter updates')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random number generation')
    parser.add_argument('--granularity', type=int, default=50,
                        help='Number of parameter values to sweep over')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials to run per parameter setting')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to parallelize over')
    args = parser.parse_args()

    # Run a single trial or sweep experiment.
    rng = np.random.default_rng(args.seed)
    if args.sweep:
        rmhc_sweep(N=args.num_ind, R=args.rounds, pi=args.pi, alpha=args.alpha,
                   eps=args.epsilon, seed=args.seed,
                   granularity=args.granularity, trials=args.trials,
                   threads=args.threads)
        plot_sweep(N=args.num_ind, R=args.rounds, pi=args.pi, alpha=args.alpha,
                   eps=args.epsilon, seed=args.seed)
    else:
        (taus, psis, nus), pol_costs, pun_costs, deltas, betas = \
            rmhc_trial(N=args.num_ind, R=args.rounds, delta=args.delta,
                       beta=args.beta, pi=args.pi, tau0=args.tau,
                       psi0=args.psi, nu0=args.nu, alpha=args.alpha,
                       eps=args.epsilon, seed=args.seed)
        plot_trial(taus, psis, nus, pol_costs, pun_costs, args.alpha, args.pi,
                   args.delta, args.beta)
