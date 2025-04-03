# Project:  CensorshipDissent
# Filename: engine.py
# Authors:  Anish Nahar (anahar1@asu.edu) and Joshua J. Daymude (jdaymude@asu.edu).

"""
engine: Simulates an authority and a network of individuals that adapt their
desired dissents or actions based on their neighbors in the network.
"""

from opt_action import opt_action

import numpy as np


############################### ADAPTATION RULES ##############################

def b2sim(G, deltas, acts, shape=1):
    """
    Calculates one round of individuals adapting their boldnesses based on how
    similar their desired dissent is to their neighbor's actions, where more
    similarity yields higher boldness. The 'shape' parameter, if written as 1/c,
    causes boldness to equal 1 when the average unsigned desire-to-nbr_action
    distance is 1/(c+1).

    :param G: a networkx graph
    :param deltas: an array of individuals' float desired dissents (in [0,1])
    :param acts: an array of individuals' float actions (in [0,1])
    :param shape: a float shape parameter for the adaptation function
    """
    new_betas = np.zeros(len(deltas))
    for i in range(len(deltas)):
        new_betas[i] = np.mean(np.abs(deltas[i] - acts[G[i]]))
    new_betas = np.minimum(np.maximum(new_betas, 1e-6), 1 - 1e-6)
    return shape * (1 / new_betas - 1)


def b2a(G, acts, shape=1):
    """
    Calculates one round of individuals adapting their boldnesses based on how
    dissenting their most dissenting neighbor's action is, where more dissent
    yields higher boldness. The 'shape' parameter causes boldness to equal 1
    when the maximum neighbor action is 1/(shape+1).

    :param G: a networkx graph
    :param acts: an array of individuals' float actions (in [0,1])
    :param shape: a float shape parameter for the adaptation function
    """
    new_betas = np.zeros(len(acts))
    for i in range(len(acts)):
        new_betas[i] = 1 - np.max(acts[G[i]])
    new_betas = np.minimum(np.maximum(new_betas, 1e-6), 1 - 1e-6)
    return shape * (1 / new_betas - 1)


def d2d(G, deltas, w=0.5):
    """
    Calculates one round of individuals adapting their desired dissents based
    on their current desired dissents and those of their neighbors.

    :param G: a networkx graph
    :param deltas: an array of individuals' float desired dissents (in [0,1])
    :param w: a float relative weight for an individual's own desired dissent
              compared to their neighbors' average desired dissent (in [0,1])
    :returns: an updated array of individuals' float desired dissents (in [0,1])
    """
    new_deltas = np.zeros(len(deltas))
    for i in range(len(deltas)):
        new_deltas[i] = w * deltas[i] + (1 - w) * np.mean(deltas[G[i]])

    return new_deltas


def d2a(G, deltas, acts, w=0.5):
    """
    Calculates one round of individuals adapting their desired dissents based
    on their current desired dissents and their neighbors' current actions.

    :param G: a networkx graph
    :param deltas: an array of individuals' float desired dissents (in [0,1])
    :param acts: an array of individuals' float actions (in [0,1])
    :param w: a float relative weight for an individual's own desired dissent
              compared to their neighbors' average action (in [0,1])
    :returns: an updated array of individuals' float desired dissents (in [0,1])
    """
    new_deltas = np.zeros(len(deltas))
    for i in range(len(deltas)):
        new_deltas[i] = w * deltas[i] + (1 - w) * np.mean(acts[G[i]])

    return new_deltas


def a2a(G, opt_acts, acts, w=0.5):
    """
    Calculates one round of individuals' actions based their current optimal
    action and their neighbors' actions from the previous round.

    :param G: a networkx graph
    :param opt_acts: an array of individuals' float desired dissents (in [0,1])
    :param acts: an array of individuals' float actions (in [0,1])
    :param w: a float relative weight for an individual's own optimal action
              compared to their neighbors' average action (in [0,1])
    :returns: an array of individuals' actions for the current round (in [0,1])
    """
    new_acts = np.zeros(len(acts))
    for i in range(len(acts)):
        new_acts[i] = w * opt_acts[i] + (1 - w) * np.mean(acts[G[i]])

    return new_acts


############################## SIMULATION ENGINE ##############################

def engine(G, N=100, R=100, rule='d2d', w=0.5, deltas=np.linspace(0, 1, 100),
           betas=np.repeat(1, 100), nu=0.5, pi='proportional', tau=0.25,
           sigma_tau=0.05, psi=1.75, sigma_psi=0.05, rng=None):
    """
    Runs a simulation of the censorship-dissent model on a social network of
    individuals whose topology has powerlaw-distributed node degrees.

    :param G: a networkx graph
    :param N: an int number of individuals
    :param R: an int number of rounds to simulate
    :param rule: a string adaptation rule in ['d2d', 'd2a', 'a2a']
    :param w: a float weight for an individual's own preference relative to any
              one of their neighbors in an adaptation rule (in [0,1])
    :param deg0: see 'm' in nx.powerlaw_cluster_graph (in [1,N])
    :param ptri: see 'p' in nx.powerlaw_cluster_graph (in [0,1])
    :param deltas: an array of individuals' float desired dissents (in [0,1])
    :param betas: an array of individuals' float boldnesses (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'proportional' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param sigma_tau: the float stddev for tolerance observation noise (> 0)
    :param psi: the authority's float punishment severity (> 0)
    :param sigma_psi: the float stddev for severity observation noise (> 0)
    :param rng: a numpy.random.Generator, or None if one should be created here

    :returns: an Nx(R+1) array of dissent desire histories
    :returns: an Nx(R+1) array of boldness histories
    :returns: an NxR array of action histories
    """
    assert rule in ['b2sim', 'b2a', 'd2d', 'd2a', 'a2a'], f'ERROR: Unrecognized rule \"{rule}\"'

    # Set up arrays to record desired dissent, boldness, and action histories.
    delta_hist = np.zeros((N, R+1))
    delta_hist[:, 0] = deltas
    beta_hist = np.zeros((N, R+1))
    beta_hist[:, 0] = betas
    act_hist = np.zeros((N, R))

    # Set up random number generator if one was not provided.
    if rng is None:
        rng = np.random.default_rng()

    # Simulate the specified number of rounds.
    for r in range(R):
        # Calculate the individuals' optimal actions based on their desired
        # dissents, boldnesses, and (independently noisy) observations of the
        # authority's tolerance and severity.
        acts = np.zeros(N)
        noisy_taus = np.minimum(np.maximum(rng.normal(tau, sigma_tau, N), 0), 1)
        noisy_psis = np.maximum(rng.normal(psi, sigma_psi, N), 1e-3)
        for i in range(N):
            acts[i] = opt_action(deltas[i], betas[i], nu, pi, noisy_taus[i],
                                 noisy_psis[i])

        # Apply the specified adaptation rule.
        if rule == 'b2sim':  # "solidarity", adapt boldness to similarity
            betas = b2sim(G, deltas, acts, shape=w)
        elif rule == 'b2a':  # "solidarity", adapt boldness to actions
            betas = b2a(G, acts, shape=w)
        elif rule == 'd2d':  # "sharing around tables", adapt desire to desires
            deltas = d2d(G, deltas, w)
        elif rule == 'd2a':  # "socialization", adapt desire to actions
            deltas = d2a(G, deltas, acts, w)
        elif r > 0:  # "when in Rome", adapt optimal action to previous actions
            acts = a2a(G, acts, act_hist[:, r-1], w)

        # Record desired dissents and actions in history.
        delta_hist[:, r+1] = np.copy(deltas)
        beta_hist[:, r+1] = np.copy(betas)
        act_hist[:, r] = acts

    return delta_hist, beta_hist, act_hist
