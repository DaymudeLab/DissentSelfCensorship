# Project:  CensorshipDissent
# Filename: opt_dissent.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
opt_dissent: Calculates and plots an individual's optimal action dissent and
corresponding utility as a function of their own parameters and noisy estimates
of the authority's tolerance and severity.
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

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


def plot_opt_dissent(ax, beta, nu, pi, t, s, color='tab:blue'):
    """
    Plot an individual's optimal action dissent vs. their desire to dissent.

    :param ax: a matplotlib.Axes object to plot on
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param s: the authority's float punishment severity (> 0)
    :param color: any color recognized by matplotlib
    """
    # Evaluate the individual's optimal action dissent over a range of desires.
    deltas = np.linspace(0, 1, 10000)
    acts = np.array([opt_dissent(delta, beta, nu, pi, t, s) for delta in deltas])
    ax.plot(deltas, acts, c=color)

    # Add visual elements and set axes information depending on the scenario.
    if pi == 'constant':
        if beta <= (1 - nu) * s:
            ax.set(xticks=[0, t, 1], yticks=[0, t, 1],
                   xticklabels=['0', r'$\hat{t}_{i,r}$', '1'],
                   yticklabels=['0', r'$\hat{t}_{i,r}$', '1'])
        else:
            # Visualize the discontinuity where defiance kicks in.
            tip_constant = (beta * t + nu * s) / (beta - (1 - nu) * s)
            ax.scatter([tip_constant, tip_constant], [t, tip_constant],
                       c=[color, 'white'], edgecolor=color, zorder=3)
            ax.set(xticks=[0, t, tip_constant, 1],
                   yticks=[0, t, tip_constant, 1],
                   xticklabels=['0', r'$\hat{t}_{i,r}$', r'$d_{i,r}^{con}$', '1'],
                   yticklabels=['0', r'$\hat{t}_{i,r}$', r'$d_{i,r}^{con}$', '1'])
    elif pi == 'linear':
        tip_linear = (t + (beta - nu * s) / ((1 - nu) * s)) / 2 if nu < 1 else 0
        if nu < 1 and tip_linear > t:
            ax.set(xticks=[0, t, tip_linear, 1], yticks=[0, t, tip_linear, 1],
                   xticklabels=['0', r'$\hat{t}_{i,r}$', r'$d_{i,r}^{lin}$', '1'],
                   yticklabels=['0', r'$\hat{t}_{i,r}$', r'$d_{i,r}^{lin}$', '1'])
        else:
            ax.set(xticks=[0, t, 1], yticks=[0, t, 1],
                   xticklabels=['0', r'$\hat{t}_{i,r}$', '1'],
                   yticklabels=['0', r'$\hat{t}_{i,r}$', '1'])

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()


if __name__ == "__main__":
    # Combine subplots for constant pi into one figure and save.
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 4), dpi=300, facecolor='white',
                           tight_layout=True)

    # Plot optimal action dissent and utility when defiance exists.
    plot_opt_dissent(ax[0], beta=1, nu=0.2, pi='constant', t=0.25, s=0.6)
    ax[0].set(title=r'(a) $\beta_i > (1 - \nu) \cdot \hat{s}_{i,r}$',
              xlabel=r'Desire to Dissent, $\delta_i$',
              ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')

    # Plot optimal action dissent and utility when defiance does not exist.
    plot_opt_dissent(ax[1], beta=1, nu=0.2, pi='constant', t=0.25, s=2)
    ax[1].set(title=r'(b) otherwise',
              xlabel=r'Desire to Dissent, $\delta_i$')

    fig.savefig(osp.join('..', 'figs', 'opt_dissent_constant.png'))

    # Combine subplots for linear pi into one figure and save.
    fig, ax = plt.subplots(1, 3, figsize=(10.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)

    # Plot optimal action dissent and utility when v < 1 and defiance exists.
    plot_opt_dissent(ax[0], beta=1, nu=0.2, pi='linear', t=0.25, s=1)
    ax[0].set(title=r'(a) $\nu < 1$ and $d_{i,r}^{lin} > \hat{t}_{i,r}$',
              xlabel=r'Desire to Dissent, $\delta_i$',
              ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')

    # Plot optimal action dissent and utility when v = 1 and beta >= s.
    plot_opt_dissent(ax[1], beta=1, nu=1, pi='linear', t=0.25, s=0.6)
    ax[1].set(title=r'(b) $\nu = 1$ and $\beta_i \geq \hat{s}_{i,r}$',
              xlabel=r'Desire to Dissent, $\delta_i$')

    # Plot optimal action dissent and utility in the remaining case.
    plot_opt_dissent(ax[2], beta=1, nu=1, pi='linear', t=0.25, s=2)
    ax[2].set(title=r'(c) otherwise',
              xlabel=r'Desire to Dissent, $\delta_i$')

    fig.savefig(osp.join('..', 'figs', 'opt_dissent_linear.png'))
