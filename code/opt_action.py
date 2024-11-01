# Project:  CensorshipDissent
# Filename: opt_action.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
opt_action: Calculates and plots an individual's optimal action and
corresponding utility as a function of their own desired dissent and boldness
and noisy estimates of the authority's tolerance and severity.
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def duni(beta, nu, tau, psi):
    """
    Computes the critical point between (full) self-censorship and defiance
    under uniform punishment.

    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the float critical point for uniform punishment
    """
    return (beta * tau + nu * psi) / (beta - (1 - nu) * psi)


def dvar(beta, nu, tau, psi):
    """
    Computes the critical point between self-censorship and defiance as well as
    the level of self-censorship enacted under variable punishment.

    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the float critical point for variable punishment
    """
    return (tau + (beta - nu * psi) / ((1 - nu) * psi)) / 2 if nu < 1 else 0


def opt_action(delta, beta, nu, pi, tau, psi):
    """
    Computes an individual's optimal action as a function of their parameters,
    the authority's parameters, and their noisy estimates of the authority's
    tolerance and severity.

    :param delta: the individual's float desire to dissent (in [0,1])
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'variable' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the individual's float optimal action (in [0,delta])
    """
    assert pi in ['uniform', 'variable'], f'ERROR: Unrecognized punishment function \"{pi}\"'

    if pi == 'uniform':
        # Compliant/defiant individuals act = desire; the rest self-censor.
        d = duni(beta, nu, tau, psi)
        if delta <= tau or (beta > (1 - nu) * psi and delta > d):
            return delta
        else:
            return tau
    elif pi == 'variable':
        # If the authority's surveillance is perfect, we simply compare the
        # individual's boldness to the authority's severity.
        d = dvar(beta, nu, tau, psi)
        if delta <= tau or (nu == 1 and beta >= psi) or (nu < 1 and delta <= d):
            return delta
        elif (nu == 1 and beta < psi) or (nu < 1 and tau >= d):
            return tau
        else:
            return d


def opt_actions(deltas, betas, nu, pi, tau, psi):
    """
    A vectorized version of the above opt_action().

    :param deltas: an array of individuals' float desires to dissent (in [0,1])
    :param betas: an array of individuals' float boldness constants (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'variable' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the individuals' float optimal actions (in [0,delta_i])
    """
    if pi == 'uniform':
        dunis = (betas * tau + nu * psi) / (betas - (1 - nu) * psi)
        cond = (deltas <= tau) | ((betas > (1 - nu) * psi) & (deltas >= dunis))
        return cond * deltas + np.logical_not(cond) * tau
    elif pi == 'variable':
        if nu == 1:
            dvars = np.zeros(len(deltas))
        else:
            dvars = (tau + (betas - nu * psi) / ((1 - nu) * psi)) / 2
        cond1 = (deltas <= tau) | ((nu == 1) & (betas >= psi)) | \
                ((nu < 1) & (deltas <= dvars))
        cond2 = (deltas > tau) & (((nu == 1) & (betas < psi)) |
                                  ((nu < 1) & (tau >= dvars)))
        return cond1 * deltas + cond2 * tau + np.logical_not(cond1 | cond2) * dvars
    else:
        assert False, f'ERROR: Unrecognized punishment function \"{pi}\"'


def plot_opt_action(ax, beta, nu, pi, tau, psi, color='tab:blue'):
    """
    Plot an individual's optimal action vs. their desired dissent.

    :param ax: a matplotlib.Axes object to plot on
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'variable' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :param color: any color recognized by matplotlib
    """
    # Evaluate the individual's optimal action over a range of desires.
    deltas = np.linspace(0, 1, 10000)
    acts = np.array([opt_action(delta, beta, nu, pi, tau, psi) for delta in deltas])
    ax.plot(deltas, acts, c=color)

    # Add visual elements and set axes information depending on the scenario.
    if pi == 'uniform':
        if beta <= (1 - nu) * psi:
            ax.set(xticks=[0, tau, 1], yticks=[0, tau, 1],
                   xticklabels=['0', r'$\hat{\tau}_{i,r}$', '1'],
                   yticklabels=['0', r'$\hat{\tau}_{i,r}$', '1'])
        else:
            # Visualize the discontinuity where defiance kicks in.
            d = duni(beta, nu, tau, psi)
            ax.scatter([d, d], [tau, d], c=[color, 'white'], edgecolor=color,
                       zorder=3)
            ax.set(xticks=[0, tau, d, 1], yticks=[0, tau, d, 1],
                   xticklabels=['0', r'$\hat{\tau}_{i,r}$', r'$d_{i,r}^{uni}$', '1'],
                   yticklabels=['0', r'$\hat{\tau}_{i,r}$', r'$d_{i,r}^{uni}$', '1'])
    elif pi == 'variable':
        d = dvar(beta, nu, tau, psi)
        if nu < 1 and d > tau:
            ax.set(xticks=[0, tau, d, 1], yticks=[0, tau, d, 1],
                   xticklabels=['0', r'$\hat{\tau}_{i,r}$', r'$d_{i,r}^{var}$', '1'],
                   yticklabels=['0', r'$\hat{\tau}_{i,r}$', r'$d_{i,r}^{var}$', '1'])
        else:
            ax.set(xticks=[0, tau, 1], yticks=[0, tau, 1],
                   xticklabels=['0', r'$\hat{\tau}_{i,r}$', '1'],
                   yticklabels=['0', r'$\hat{\tau}_{i,r}$', '1'])

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()


if __name__ == "__main__":
    # Combine subplots for uniform punishment into one figure and save.
    fig, ax = plt.subplots(2, 1, figsize=(4, 7.5), dpi=300, facecolor='white',
                           tight_layout=True)

    # Plot optimal action and utility when defiance exists.
    plot_opt_action(ax[0], beta=1, nu=0.2, pi='uniform', tau=0.25, psi=0.6)
    ax[0].set_title(r'(A) $\beta_i > (1 - \nu) \cdot \hat{\psi}_{i,r}$',
                    weight='bold')
    ax[0].set(ylabel=r'Optimal Action $a_{i,r}^*$')

    # Plot optimal action and utility when defiance does not exist.
    plot_opt_action(ax[1], beta=1, nu=0.2, pi='uniform', tau=0.25, psi=2)
    ax[1].set_title(r'(B) otherwise', weight='bold')
    ax[1].set(xlabel=r'Desired Dissent $\delta_i$',
              ylabel=r'Optimal Action $a_{i,r}^*$')

    fig.savefig(osp.join('..', 'figs', 'opt_action_uniform.pdf'))

    # Combine subplots for variable punishment into one figure and save.
    fig, ax = plt.subplots(1, 3, figsize=(10.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)

    # Plot optimal action and utility when v < 1 and defiance exists.
    plot_opt_action(ax[0], beta=1, nu=0.2, pi='variable', tau=0.25, psi=1)
    ax[0].set_title(r'(A) $\nu < 1$ and $\hat{\tau}_{i,r} < d_{i,r}^{var}$',
                    weight='bold')
    ax[0].set(xlabel=r'Desired Dissent $\delta_i$',
              ylabel=r'Optimal Action $a_{i,r}^*$')

    # Plot optimal action and utility when v = 1 and beta >= s.
    plot_opt_action(ax[1], beta=1, nu=1, pi='variable', tau=0.25, psi=0.6)
    ax[1].set_title(r'(B) $\nu = 1$ and $\beta_i \geq \hat{\psi}_{i,r}$',
                    weight='bold')
    ax[1].set(xlabel=r'Desired Dissent $\delta_i$')

    # Plot optimal action and utility in the remaining case.
    plot_opt_action(ax[2], beta=1, nu=1, pi='variable', tau=0.25, psi=2)
    ax[2].set_title(r'(C) otherwise', weight='bold')
    ax[2].set(xlabel=r'Desired Dissent $\delta_i$')

    fig.savefig(osp.join('..', 'figs', 'opt_action_variable.pdf'))
