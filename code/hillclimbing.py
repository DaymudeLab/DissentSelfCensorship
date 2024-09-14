# Project:  CensorshipDissent
# Filename: hillclimbing.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
hillclimbing: An adaptive authority experiment using random hill climbing.
"""

from opt_action import opt_action

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from tqdm import trange


def rmhc_trial(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, rng):
    """
    Runs a single simulation trial of the model where individuals' desired
    dissents and boldness constants are exponentially-distributed but fixed and
    the authority adapts its parameters based on random mutation hill climbing.

    :param N: an int number of individuals
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
    :returns: an NxR array of individuals' actions
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
    acts = np.zeros((N, R))

    # Simulate the specified number of rounds, allowing the authority to adapt
    # its parameters using random mutation hill climbing (RMHC).
    for r in trange(R):
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
        acts[:, r] = [opt_action(deltas[i], betas[i], nu, pi, tau, psi)
                      for i in range(N)]

        # The authority punishes any actions that it observes above tolerance.
        pr_obs = nu + (1 - nu) * acts[:, r]
        punish = np.zeros(N)
        for i in range(N):
            if acts[i, r] > tau and rng.random() < pr_obs[i]:
                if pi == 'uniform':
                    punish[i] = psi
                elif pi == 'variable':
                    punish[i] = psi * (acts[i, r] - tau)
                else:
                    assert False, f"ERROR: Invalid punishment function {pi}"

        # The authority's political cost for this round is the summed actions
        # and its punishment cost is the summed punishments.
        pol_costs[r] = acts[:, r].sum()
        pun_costs[r] = punish.sum()

        # If the authority's adamancy-weighted cost in this round is worse than
        # last round, reset to last round's parameters.
        if r > 0 and alpha * pol_costs[r] + pun_costs[r] > \
                alpha * pol_costs[r-1] + pun_costs[r-1]:
            params[:, r] = params[:, r-1]

    return params, pol_costs, pun_costs, acts


if __name__ == "__main__":
    # Start with something simple.
    N = 10000
    R = 1000
    delta = 0.5
    beta = 0.5
    pi = 'uniform'
    tau0 = 0
    psi0 = 0.1
    nu0 = 0.5
    alpha = 0.25
    eps = 0.05
    (taus, psis, nus), pol_costs, pun_costs, acts = \
        rmhc_trial(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, None)

    # Plot results.
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, dpi=300,
                           tight_layout=True)

    # Plot costs (negative utility) over time.
    ax[0].plot(np.arange(R), alpha * pol_costs, label=r'$\alpha \times$'
               'Political Cost')
    ax[0].plot(np.arange(R), pun_costs, label='Punishment Cost')
    ax[0].plot(np.arange(R), alpha * pol_costs + pun_costs, label='Total Cost')
    ax[0].legend()
    ax[0].set(title=r"Hill Climbing Authority ($\pi$ " f"= {pi}, " r"$\alpha$ "
              f"= {alpha}) vs. Population " r"$\delta \sim \mathcal{E}$"
              f"({delta}), " r"$\beta \sim \mathcal{E}$" f"({beta})")

    # Plot parameters over time.
    ax[1].plot(np.arange(R), taus, label=r'Tolerance $\tau$')
    ax[1].plot(np.arange(R), psis, label=r'Severity $\psi$')
    ax[1].plot(np.arange(R), nus, label=r'Surveillance $\nu$')
    ax[1].legend()
    ax[1].set(xlim=[0, R], xlabel='Round')

    fig.savefig(osp.join('..', 'figs', 'hillclimbing.pdf'))
