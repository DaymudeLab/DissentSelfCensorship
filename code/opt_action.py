# Project:  CensorshipDissent
# Filename: opt_action.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
opt_action: Calculates and plots an individual's optimal action and
corresponding utility as a function of their own desired dissent and boldness
and noisy estimates of the authority's tolerance and severity.
"""

from cmcrameri import cm
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


def dlin(beta, nu, tau, psi):
    """
    Computes the critical point between self-censorship and defiance as well as
    the level of self-censorship enacted under linear punishment.

    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the float critical point for linear punishment
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
    :param pi: 'uniform' or 'linear' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the individual's float optimal action (in [0,delta])
    """
    if pi == 'uniform':
        # Compliant/defiant individuals act = desire; the rest self-censor.
        d = duni(beta, nu, tau, psi)
        if delta <= tau or (beta > (1 - nu) * psi and delta > d):
            return delta
        else:
            return tau
    elif pi == 'linear':
        # If the authority's surveillance is perfect, we simply compare the
        # individual's boldness to the authority's severity.
        d = dlin(beta, nu, tau, psi)
        if delta <= tau or (nu == 1 and beta >= psi) or (nu < 1 and delta <= d):
            return delta
        elif (nu == 1 and beta < psi) or (nu < 1 and tau >= d):
            return tau
        else:
            return d
    else:
        assert False, f'ERROR: Unrecognized punishment function \"{pi}\"'


def opt_actions(deltas, betas, nu, pi, tau, psi):
    """
    A vectorized version of the above opt_action().

    :param deltas: an array of individuals' float desires to dissent (in [0,1])
    :param betas: an array of individuals' float boldness constants (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'linear' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    :returns: the individuals' float optimal actions (in [0,delta_i])
    """
    if pi == 'uniform':
        dunis = (betas * tau + nu * psi) / (betas - (1 - nu) * psi)
        cond = (deltas <= tau) | ((betas > (1 - nu) * psi) & (deltas >= dunis))
        return cond * deltas + np.logical_not(cond) * tau
    elif pi == 'linear':
        if nu == 1:
            dlins = np.zeros(len(deltas))
        else:
            dlins = (tau + (betas - nu * psi) / ((1 - nu) * psi)) / 2
        cond1 = (deltas <= tau) | ((nu == 1) & (betas >= psi)) | \
                ((nu < 1) & (deltas <= dlins))
        cond2 = (deltas > tau) & (((nu == 1) & (betas < psi)) |
                                  ((nu < 1) & (tau >= dlins)))
        return cond1 * deltas + cond2 * tau + np.logical_not(cond1 | cond2) * dlins
    else:
        assert False, f'ERROR: Unrecognized punishment function \"{pi}\"'


def plot_opt_action(ax, beta, nu, pi, tau, psi):
    """
    Plot an individual's optimal action vs. their desired dissent, using color
    to differentiate compliant, self-censoring, and defiant behavior.

    :param ax: a matplotlib.Axes object to plot on
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'linear' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    """
    # Evaluate the individual's optimal action over a range of desires.
    deltas, betas = np.linspace(0, 1, 10000), np.repeat(beta, 10000)
    acts = opt_actions(deltas, betas, nu, pi, tau, psi)

    # Plot compliant, self-censoring, and defiant behaviors.
    colors=[cm.batlow(0.2), cm.batlow(0.5), cm.batlow(0.8)]
    ax.plot(np.ma.masked_where(acts >= tau, deltas),
            np.ma.masked_where(acts >= tau, acts), c=colors[0])
    ax.plot(np.ma.masked_where(acts >= deltas, deltas),
            np.ma.masked_where(acts >= deltas, acts), c=colors[1])
    ax.plot(np.ma.masked_where(np.logical_or(acts <= tau, acts < deltas), deltas),
            np.ma.masked_where(np.logical_or(acts <= tau, acts < deltas), acts),
            c=colors[2])

    # Add visual elements and set axes information depending on the scenario.
    if pi == 'uniform':
        if beta <= (1 - nu) * psi:
            ax.set(xticks=[0, tau, 1], yticks=[0, tau, 1],
                   xticklabels=['0', r'$\tau_r$', '1'],
                   yticklabels=['0', r'$\tau_r$', '1'])
        else:
            # Visualize the discontinuity where defiance kicks in.
            d = duni(beta, nu, tau, psi)
            ax.scatter([d, d], [tau, d], c=[colors[1], 'white'],
                       edgecolor=[colors[1], colors[2]], zorder=3)
            ax.set(xticks=[0, tau, d, 1], yticks=[0, tau, d, 1],
                   xticklabels=['0', r'$\tau_r$', r'$d_{i,r}^{uni}$', '1'],
                   yticklabels=['0', r'$\tau_r$', r'$d_{i,r}^{uni}$', '1'])
    elif pi == 'linear':
        d = dlin(beta, nu, tau, psi)
        if nu < 1 and d > tau:
            ax.set(xticks=[0, tau, d, 1], yticks=[0, tau, d, 1],
                   xticklabels=['0', r'$\tau_r$', r'$d_{i,r}^{lin}$', '1'],
                   yticklabels=['0', r'$\tau_r$', r'$d_{i,r}^{lin}$', '1'])
        else:
            ax.set(xticks=[0, tau, 1], yticks=[0, tau, 1],
                   xticklabels=['0', r'$\tau_r$', '1'],
                   yticklabels=['0', r'$\tau_r$', '1'])

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()


if __name__ == "__main__":
    # Combine subplots for uniform punishment into one figure and save.
    fig, ax = plt.subplots(2, 1, figsize=(4, 7.5), dpi=300, facecolor='white',
                           tight_layout=True)

    # Plot optimal action and utility when defiance exists.
    plot_opt_action(ax[0], beta=1, nu=0.2, pi='uniform', tau=0.25, psi=0.6)
    ax[0].set_title(r'(A) $\beta_i > (1 - \nu_r) \cdot \psi_r$', weight='bold')
    ax[0].set(ylabel=r'Optimal Action $a_{i,r}^*$')

    # Plot optimal action and utility when defiance does not exist.
    plot_opt_action(ax[1], beta=1, nu=0.2, pi='uniform', tau=0.25, psi=2)
    ax[1].set_title(r'(B) otherwise', weight='bold')
    ax[1].set(xlabel=r'Desired Dissent $\delta_i$',
              ylabel=r'Optimal Action $a_{i,r}^*$')

    fig.savefig(osp.join('..', 'figs', 'opt_action_uniform.pdf'))

    # Combine subplots for linear punishment into one figure and save.
    fig, ax = plt.subplots(1, 3, figsize=(10.75, 4), dpi=300,
                           facecolor='white', tight_layout=True)

    # Plot optimal action and utility when v < 1 and defiance exists.
    plot_opt_action(ax[0], beta=1, nu=0.2, pi='linear', tau=0.25, psi=1)
    ax[0].set_title(r'(A) $\nu_r < 1$ and $\tau_r < d_{i,r}^{lin}$',
                    weight='bold')
    ax[0].set(xlabel=r'Desired Dissent $\delta_i$',
              ylabel=r'Optimal Action $a_{i,r}^*$')

    # Plot optimal action and utility when v = 1 and beta >= s.
    plot_opt_action(ax[1], beta=1, nu=1, pi='linear', tau=0.25, psi=0.6)
    ax[1].set_title(r'(B) $\nu_r = 1$ and $\beta_i \geq \psi_r$',
                    weight='bold')
    ax[1].set(xlabel=r'Desired Dissent $\delta_i$')

    # Plot optimal action and utility in the remaining case.
    plot_opt_action(ax[2], beta=1, nu=1, pi='linear', tau=0.25, psi=2)
    ax[2].set_title(r'(C) otherwise', weight='bold')
    ax[2].set(xlabel=r'Desired Dissent $\delta_i$')

    fig.savefig(osp.join('..', 'figs', 'opt_action_linear.pdf'))
