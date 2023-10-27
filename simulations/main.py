# Project:  CensorshipDissent
# Filename: main.py
# Authors:  Anish Nahar (anahar1@asu.edu)

"""
main: Runs the simulation for a network of individuals and an authority 
with a set of rules that defines one's actions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os.path as osp

class Individual:
    def __init__(self, delta, beta):
        self.delta = delta
        self.beta = beta

def opt_dissent(delta, beta, nu, pi, t, s):
    """
    Computes an individual's optimal action dissent as a function of their
    parameters, the authority's parameters, and their noisy estimates of the
    authority's tolerance and severity.

    :param delta: the individual's float desire to dissent (in [0,1])
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param s: the authority's float punishment severity (> 0)
    :returns: the individual's float optimal action dissent (in [0,delta])
    """
    if pi == 'constant':
        # Compliant/defiant individuals act = desire; the rest self-censor.
        tip_constant = (beta * t + nu * s) / (beta - (1 - nu) * s)
        if delta <= t or (beta > (1 - nu) * s and delta > tip_constant):
            return delta
        else:
            return t
    elif pi == 'linear':
        # If the authority's surveillance is perfect, we simply compare the
        # individual's boldness to the authority's severity.
        tip_linear = (t + (beta - nu * s) / ((1 - nu) * s)) / 2 if nu < 1 else 0
        if delta <= t or (nu == 1 and beta >= s) or (nu < 1 and delta <= tip_linear):
            return delta
        elif (nu == 1 and beta < s) or (nu < 1 and t >= tip_linear):
            return t
        else:
            return tip_linear
    else:
        print('ERROR: Unrecognized punishment function \'' + pi + '\'')

def experiment(individuals, num_individuals, nu, pi, t, s,  num_rounds):
    actions_history = []

    # Step 1: All individuals calculate and enact a_i.

    # Step 2: The authority calculates and enacts punishments.

    # Step 3: Individuals update one of (d_i, beta_i) according to an update rule.

    # Step 4: The authority updates one of (tau, s) for the next round.

    # Measurement:
    # Utilities (authority and individuals).
    # All parameters.
    # Individuals' actions. (Imagine a plot with time vs. everyone's action trajectories).
    
    for round in range(num_rounds):
        # Step 1
        


        # Step 2

        # Step 3

        # Step 4
        break


if __name__ == "__main__":
    # Setup.

    # Set of parameters for individuals
    desires = np.linspace(0, 1, 1000)
    beta = 2

    # Create Individual objects
    individuals = [Individual(delta, beta) for delta in desires]

    # Run experiment
    experiment(individuals, 1000, nu=0.5, pi='constant', t=0.25, s=0.6, num_rounds=100)

    exit(0)
