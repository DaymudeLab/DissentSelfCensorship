# Project:  CensorshipDissent
# Filename: hillclimbing.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
hillclimbing: An adaptive authority experiment using random hill climbing.
"""

from opt_action import opt_actions

import argparse
from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def rmhc_trial(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, rng):
    """
    Runs a single simulation trial of the model where individuals' desired
    dissents and boldness constants are exponentially-distributed but fixed and
    the authority adapts its parameters based on random mutation hill climbing.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param delta: a float mean population desired dissent (in [0,1])
    :param beta: a float mean population boldness (> 0)
    :param pi: 'uniform' or 'variable' punishment
    :param tau0: the authority's float initial tolerance (in [0,1])
    :param psi0: the authority's float initial punishment severity (> 0)
    :param nu0: the authority's float initial surveillance (in [0,1])
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param rng: a numpy.random.Generator, or None if one should be created here

    :returns: a 3xR array of the authority's parameter values
    :returns: a 1xR array of the authority's political costs
    :returns: a 1xR array of the authority's punishment costs
    :returns: a 1xN array of individuals' dissent desires
    :returns: a 1xN array of individuals' boldness constants
    """
    # Set up a random number generator if one was not provided.
    if rng is None:
        rng = np.random.default_rng()

    # Initialize the population's desired dissents according to an exponential
    # distribution with the given mean, using rejection sampling to ensure
    # that all desired dissents lie in [0,1].
    deltas = np.array([])
    while len(deltas) < N:
        x = rng.exponential(scale=delta, size=N)
        deltas = np.append(deltas, x[np.where(x <= 1)])
    deltas = deltas[:N]

    # Initialize the population's boldness constants according to an
    # exponential distribution with the given mean.
    betas = rng.exponential(scale=beta, size=N)

    # Calculate bounds on the authority's parameters.
    bounds = np.array([[0, 1],              # tau
                       [1e-9, max(betas)],  # TODO: psi
                       [0, 1]])             # nu

    # Set up arrays to store everything that happens.
    params = np.zeros((3, R))
    pol_costs, pun_costs = np.zeros(R), np.zeros(R)

    # Simulate the specified number of rounds, allowing the authority to adapt
    # its parameters using random mutation hill climbing (RMHC).
    for r in range(R):
        # If this is the first round, the authority simply uses its initial
        # parameters. Otherwise, it generates new candidate parameters to test.
        if r == 0:
            params[:, r] = [tau0, psi0, nu0]
        else:
            p = rng.choice([0, 1, 2])  # Choose parameter to update.
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


def sweep_experiment(N, R, pi, alpha, eps, seed):
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
    :param seed: a int seed for random number generation
    """
    pass


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
    :param delta: the float mean population desired dissent (in [0,1])
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
              f"= {alpha}) vs. Population " r"$\delta \sim \mathcal{E}$"
              f"({delta}), " r"$\beta \sim \mathcal{E}$" f"({beta})",
              ylabel='Costs')

    # Plot parameters over time.
    ax[1].plot(np.arange(R), taus, label=r'Tolerance $\tau$', c=cm.batlowS(2))
    ax[1].plot(np.arange(R), psis, label=r'Severity $\psi$', c=cm.batlowS(3))
    ax[1].plot(np.arange(R), nus, label=r'Surveillance $\nu$', c=cm.batlowS(4))
    ax[1].legend()
    ax[1].set(xlim=[0, R], xlabel='Round', ylabel='Parameter Value')

    fig.savefig(osp.join('..', 'figs', 'rmhc_trial.pdf'))


def plot_sweep():
    pass


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', action='store_true',
                        help=('If present, run the sweep experiment (ignores '
                              '--delta, --beta, --tau0, --psi0, and --nu0); '
                              'otherwise; runs one independent trial (ignores '
                              '--trials, --threads)'))
    parser.add_argument('-N', '--num_ind', type=int, default=100000,
                        help='Number of individuals in the population')
    parser.add_argument('-R', '--rounds', type=int, default=1000,
                        help='Number of rounds to simulate in a single trial')
    parser.add_argument('-D', '--delta', type=float, default=0.5,
                        help='Mean population desired dissent in [0,1]')
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
    parser.add_argument('-A', '--alpha', type=float, default=0.25,
                        help='Authority\'s adamancy > 0')
    parser.add_argument('-E', '--epsilon', type=float, default=0.05,
                        help='Window radius for authority parameter updates')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random number generation')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of trials to run per parameter setting')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to parallelize over')
    args = parser.parse_args()

    # Run a single trial or sweep experiment.
    rng = np.random.default_rng(args.seed)
    if args.sweep:
        pass
    else:
        (taus, psis, nus), pol_costs, pun_costs, deltas, betas = \
            rmhc_trial(N=args.num_ind, R=args.rounds, delta=args.delta,
                       beta=args.beta, pi=args.pi, tau0=args.tau,
                       psi0=args.psi, nu0=args.nu, alpha=args.alpha,
                       eps=args.epsilon, rng=rng)
        plot_trial(taus, psis, nus, pol_costs, pun_costs, args.alpha, args.pi,
                   args.delta, args.beta)
