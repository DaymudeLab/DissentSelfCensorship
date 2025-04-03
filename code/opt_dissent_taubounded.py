# Project:  CensorshipDissent
# Filename: opt_dissent_taubounded.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
opt_dissent_taubounded: Calculates and plots individuals' optimal action
dissents and corresponding maximum utilities when the authority's tolerance is
not known exactly, but instead is bounded above and below.
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def opt_dissent(ax, pi, tau_l, tau_u, s, beta, color='tab:blue'):
    """
    Plot an individual's optimal action dissent as a function of their dissent
    desire when the authority's dissent threshold is bounded.

    :param ax: a matplotlib.Axes object to plot on
    :param pi: 'uniform' or 'variable' punishment
    :param tau_l: a float lower bound on tau (in [0,tau])
    :param tau_u: a float upper bound on tau (in [tau,1])
    :param s: a float authority's punishment severity (> 0)
    :param beta: a float boldness constant (> 0)
    :param color: any color recognized by matplotlib
    """
    if pi == 'uniform':
        ax.plot([0, tau_l], [0, tau_l], c=color)
        if tau_u >= tau_l + s/beta:
            ax.plot([tau_l, 1], [tau_l, 1], c=color)
        else:
            ax.plot([tau_l, tau_l + s/beta], [tau_l, tau_l], c=color)
            ax.plot([tau_l + s/beta, tau_l + s/beta], [tau_l, tau_l + s/beta], \
                    c=color, linestyle=':')
            ax.plot([tau_l + s/beta, 1], [tau_l + s/beta, 1], c=color)
            ax.scatter([tau_l + s/beta, tau_l + s/beta], [tau_l, tau_l + s/beta],\
                       c=['white', color], edgecolor=color, zorder=3)
        ax.set(xticks=[0, tau_l, tau_u, tau_l + s/beta, 1], \
               yticks=[0, tau_l, tau_l + s/beta, tau_u, 1], \
               xticklabels=['0', r'$\tau_\ell$', r'$\tau_u$', \
                            r'$\tau_\ell + s/\beta$', '1'], \
               yticklabels=['0', r'$\tau_\ell$', r'$\tau_\ell + s/\beta$', \
                            r'$\tau_u$', '1'])
    elif pi == 'variable':
        ax.plot([0, tau_l], [0, tau_l], c=color)
        if s < beta:
            ax.plot([tau_l, 1], [tau_l, 1], c=color)
        else:
            crit = tau_l + (beta/s)*(tau_u - tau_l)
            ax.plot([tau_l, crit, 1], [tau_l, crit, crit], c=color)
        ax.set(xticks=[0, tau_l] + ([crit] if s >= beta else []) + [tau_u, 1], \
               yticks=[0, tau_l] + ([crit] if s >= beta else []) + [tau_u, 1], \
               xticklabels=['0', r'$\tau_\ell$'] + \
                           ([r'$\tau_\ell + \frac{\beta(\tau_u - \tau_\ell)}{s}$'] \
                           if s >= beta else []) + [r'$\tau_u$', '1'], \
               yticklabels=['0', r'$\tau_\ell$'] + \
                           ([r'$\tau_\ell + \frac{\beta(\tau_u - \tau_\ell)}{s}$'] \
                           if s >= beta else []) + [r'$\tau_u$', '1'])
    else:
        print('ERROR: Unrecognized punishment form \'' + pi + '\'')

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()


def max_utility(ax, pi, tau_l, tau_u, s, beta, color='tab:green'):
    """
    Plot an individual's maximum utility as a function of their dissent desire
    when the authority's dissent threshold is bounded.

    :param ax: a matplotlib.Axes object to plot on
    :param pi: 'uniform' or 'variable' punishment
    :param tau_l: a float lower bound on tau (in [0,tau])
    :param tau_u: a float upper bound on tau (in [tau,1])
    :param s: a float authority's punishment severity (> 0)
    :param beta: a float boldness constant (> 0)
    :param color: any color recognized by matplotlib
    """
    if pi == 'uniform':
        ax.plot([0, tau_l], [beta, beta], c=color)
        if tau_u >= tau_l + s/beta:
            ax.plot([tau_l, tau_u, 1], [beta, beta - s, beta - s], c=color)
        else:
            ax.plot([tau_l, tau_l + s/beta, 1], [beta, beta - s, beta - s], \
                    c=color)
        ax.set(xticks=[0, tau_l, tau_u, tau_l + s/beta, 1], \
               yticks=[0, beta - s, beta], \
               xticklabels=['0', r'$\tau_\ell$', r'$\tau_u$', \
                            r'$\tau_\ell + s/\beta$', '1'], \
               yticklabels=['0', r'$\beta - s$', r'$\beta$'])
    elif pi == 'variable':
        ax.plot([0, tau_l], [beta, beta], c=color)
        if s < beta:
            di = np.linspace(tau_l, tau_u, 1000)
            eui = beta - s * (di - tau_l)**2 / (2*(tau_u - tau_l))
            ax.plot(di, eui, c=color)
            ax.plot([tau_u, 1], [beta - (s/2)*(tau_u - tau_l), \
                    beta - s + (s/2)*(tau_l + tau_u)], c=color)
            ax.set(yticks=[0, beta - (s/2)*(tau_u - tau_l), \
                           beta - s + (s/2)*(tau_l + tau_u), beta], \
                   yticklabels=['0', r'$\beta - \frac{s(\tau_u - \tau_\ell)}{2}$', \
                                r'$\beta - s + \frac{s(\tau_\ell + \tau_u)}{2}$', \
                                r'$\beta$'])
        else:
            crit = tau_l + (beta/s)*(tau_u - tau_l)
            di = np.linspace(tau_l, crit, 1000)
            eui = beta - s * (di - tau_l)**2 / (2*(tau_u - tau_l))
            ax.plot(di, eui, c=color)
            y1 = beta*(1 + (tau_u - tau_l)*(0.5 - beta)/s)
            y2 = beta*(tau_l + (tau_u - tau_l)/(2*s))
            ax.plot([crit, 1], [y1, y2], c=color)
            ax.set(yticks=[0, y2, y1, beta], \
                   yticklabels=['0', r'$\beta(\tau_\ell + \frac{\tau_u - \tau_\ell}{2s})$', \
                                r'$\beta + \frac{\beta(\tau_u - \tau_\ell)(1/2 - \beta)}{s}$', \
                                r'$\beta$'])
        ax.set(xticks=[0, tau_l] + ([crit] if s >= beta else []) + [tau_u, 1], \
               xticklabels=['0', r'$\tau_\ell$'] + \
                           ([r'$\tau_\ell + \frac{\beta(\tau_u - \tau_\ell)}{s}$'] \
                           if s >= beta else []) + [r'$\tau_u$', '1'])
    else:
        print('ERROR: Unrecognized punishment form \'' + pi + '\'')

    ax.set(xlim=[0, 1], ylim=[0, beta + 0.1])
    ax.grid()


if __name__ == "__main__":
    # Combine the subplots into one figure and save.
    fig, ax = plt.subplots(2, 4, sharex='col', figsize=(17.5, 7), dpi=300, \
                           facecolor='white', tight_layout=True)

    # Plot optimal action dissent and utility for uniform pi and
    # s/beta <= tau_u - tau_l.
    opt_dissent(ax[0,0], 'uniform', tau_l=0.2, tau_u=0.65, s=0.3, beta=1)
    max_utility(ax[1,0], 'uniform', tau_l=0.2, tau_u=0.65, s=0.3, beta=1)
    ax[0,0].set(title=r'(a) $\pi$ uniform with $s/\beta \leq \tau_u - \tau_\ell$',\
                ylabel=r'Optimal Action Dissent, $\tilde{{a}}_{{i,t}}^*$')
    ax[1,0].set(xlabel=r'Desire to Dissent, $d_i$', \
                ylabel=r'Maximum Expected Utility, $E[U_{{i,t}}]^*$')

    # Plot optimal action dissent and utility for uniform pi and
    # s/beta > tau_u - tau_l.
    opt_dissent(ax[0,1], 'uniform', tau_l=0.2, tau_u=0.65, s=0.6, beta=1)
    max_utility(ax[1,1], 'uniform', tau_l=0.2, tau_u=0.65, s=0.6, beta=1)
    ax[0,1].set(title=r'(b) $\pi$ uniform with $s/\beta > \tau_u - \tau_\ell$')
    ax[1,1].set(xlabel=r'Desire to Dissent, $d_i$')

    # Plot optimal action dissent and utility for variable pi and s < beta.
    opt_dissent(ax[0,2], 'variable', tau_l=0.2, tau_u=0.65, s=0.6, beta=1)
    max_utility(ax[1,2], 'variable', tau_l=0.2, tau_u=0.65, s=0.6, beta=1)
    ax[0,2].set(title=r'(c) $\pi$ variable with $s < \beta$')
    ax[1,2].set(xlabel=r'Desire to Dissent, $d_i$')

    # Plot optimal action dissent and utility for variable pi and s >= beta.
    opt_dissent(ax[0,3], 'variable', tau_l=0.2, tau_u=0.65, s=1.6, beta=1)
    max_utility(ax[1,3], 'variable', tau_l=0.2, tau_u=0.65, s=1.6, beta=1)
    ax[0,3].set(title=r'(d) $\pi$ variable with $s \geq \beta$')
    ax[1,3].set(xlabel=r'Desire to Dissent, $d_i$')

    fig.savefig(osp.join('figs', 'opt_dissent_taubounded.png'))
