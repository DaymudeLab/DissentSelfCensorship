# Project:  CensorshipDissent
# Filename: opt_dissent_tauknown.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
opt_dissent_tauknown: Calculates and plots individuals' optimal action dissents
and corresponding maximum utilities when the authority's tolerance is known.
"""

import matplotlib.pyplot as plt
import os.path as osp


def opt_dissent(ax, pi, tau, s, beta, color='tab:blue'):
    """
    Plot an individual's optimal action dissent as a function of their dissent
    desire when the authority's dissent threshold is known.

    :param ax: a matplotlib.Axes object to plot on
    :param pi: 'constant' or 'linear' punishment
    :param tau: a float authority's dissent threshold in (in [0,1])
    :param s: a float authority's punishment severity (> 0)
    :param beta: a float boldness constant (> 0)
    :param color: any color recognized by matplotlib
    """
    if pi == 'constant':
        ax.plot([0, tau, tau + s/beta], [0, tau, tau], c=color)
        ax.plot([tau + s/beta, tau + s/beta], [tau, tau + s/beta], \
                c=color, linestyle=':')
        ax.plot([tau + s/beta, 1], [tau + s/beta, 1], c=color)
        ax.scatter([tau + s/beta, tau + s/beta], [tau, tau + s/beta], \
                   c=['white', color], edgecolor=color, zorder=3)
        ax.set(xticks=[0, tau, tau + s/beta, 1], \
               yticks=[0, tau, tau + s/beta, 1], \
               xticklabels=['0', r'$\tau$', r'$\tau + s/\beta$', '1'], \
               yticklabels=['0', r'$\tau$', r'$\tau + s/\beta$', '1'])
    elif pi == 'linear':
        ax.plot([0, tau], [0, tau], c=color)
        if s < beta:
            ax.plot([tau, 1], [tau, 1], c=color)
        else:
            ax.plot([tau, 1], [tau, tau], c=color)
        ax.set(xticks=[0, tau, 1], \
               yticks=[0, tau, 1], \
               xticklabels=['0', r'$\tau$', '1'], \
               yticklabels=['0', r'$\tau$', '1'])
    else:
        print('ERROR: Unrecognized punishment form \'' + pi + '\'')

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()


def max_utility(ax, pi, tau, s, beta, color='tab:green'):
    """
    Plot an individual's maximum utility as a function of their dissent desire
    when the authority's dissent threshold is known.

    :param ax: a matplotlib.Axes object to plot on
    :param pi: 'constant' or 'linear' punishment
    :param tau: a float authority's dissent threshold in (in [0,1])
    :param s: a float authority's punishment severity (> 0)
    :param beta: a float boldness constant (> 0)
    :param color: any color recognized by matplotlib
    """
    if pi == 'constant':
        ax.plot([0, tau, tau + s/beta, 1], [beta, beta, beta - s, beta - s], \
                c=color)
        ax.set(xticks=[0, tau, tau + s/beta, 1], \
               yticks=[0, beta - s, beta], \
               xticklabels=['0', r'$\tau$', r'$\tau + s/\beta$', '1'], \
               yticklabels=['0', r'$\beta - s$', r'$\beta$'])
    elif pi == 'linear':
        ax.plot([0, tau], [beta, beta], c=color)
        if s < beta:
            ax.plot([tau, 1], [beta, beta - s*(1 - tau)], c=color)
        else:
            ax.plot([tau, 1], [beta, beta * tau], c=color)
        ax.set(xticks=[0, tau, 1], \
               yticks=[0, beta - s*(1 - tau) if s < beta else beta*tau, beta], \
               xticklabels=['0', r'$\tau$', '1'], \
               yticklabels=['0', r'$\beta - s + s\tau$' if s < beta else \
                            r'$\beta\tau$', r'$\beta$'])
    else:
        print('ERROR: Unrecognized punishment form \'' + pi + '\'')

    ax.set(xlim=[0, 1], ylim=[None, beta + 0.1])
    ax.grid()


if __name__ == "__main__":
    # Combine the subplots into one figure and save.
    fig, ax = plt.subplots(2, 3, sharex='col', figsize=(12, 7), dpi=300, \
                           facecolor='white', tight_layout=True)

    # Plot optimal action dissent and utility for constant pi.
    opt_dissent(ax[0,0], pi='constant', tau=0.25, s=0.6, beta=1)
    max_utility(ax[1,0], pi='constant', tau=0.25, s=0.6, beta=1)
    ax[0,0].set(title=r'(a) $\pi$ constant', \
                ylabel=r'Optimal Action Dissent, $a_{{i,t}}^*$')
    ax[1,0].set(xlabel=r'Desire to Dissent, $d_i$', \
                ylabel=r'Maximum Utility, $U_{{i,t}}^*$')

    # Plot optimal action dissent and utility for linear pi when s < beta.
    opt_dissent(ax[0,1], pi='linear', tau=0.25, s=0.6, beta=1)
    max_utility(ax[1,1], pi='linear', tau=0.25, s=0.6, beta=1)
    ax[0,1].set(title=r'(b) $\pi$ linear with $s < \beta$')
    ax[1,1].set(xlabel=r'Desire to Dissent, $d_i$')

    # Plot optimal action dissent and utility for linear pi when s >= beta.
    opt_dissent(ax[0,2], pi='linear', tau=0.25, s=1.6, beta=1)
    max_utility(ax[1,2], pi='linear', tau=0.25, s=1.6, beta=1)
    ax[0,2].set(title=r'(c) $\pi$ linear with $s \geq \beta$')
    ax[1,2].set(xlabel=r'Desire to Dissent, $d_i$')

    fig.savefig(osp.join('figs', 'opt_dissent_tauknown.png'))
