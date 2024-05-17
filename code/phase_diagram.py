# Project:  CensorshipDissent
# Filename: phase_diagram.py
# Authors:  Anish Nahar (anahar1@asu.edu) and Joshua J. Daymude (jdaymude@asu.edu).

"""
phase_diagram: Uses an individual's optimal action to plot phase diagrams of
compliance, self-censorship, and defiance as a function of an individual's
desire to dissent vs. the model's other parameters.
"""

from opt_action import opt_action

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def phase(delta, beta, nu, pi, tau, psi):
    """
    Compute a phase value in [0,1] for the given set of parameters where 0 is
    compliance, 0.25 is total self-censorship, and 1 is defiance.

    :param delta: the individual's float desire to dissent (in [0,1])
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    """
    opt_act = opt_action(delta, beta, nu, pi, tau, psi)
    if delta <= tau:  # Compliance.
        return 0
    elif opt_act >= tau and opt_act < delta:  # Self-censorship.
        return 0.25 + 0.75 * ((opt_act - tau) / (delta - tau))
    else:  # Defiance.
        return 1


def plot_phase_diagram(ax, yparam, beta, nu, pi, tau, psi):
    """
    Plot a phase diagram of compliance, self-censorship, and defiance in the
    space of desires to dissent vs. the given parameter.

    :param ax: a matplotlib.Axes object to plot on
    :param yparam: the y-axis parameter, from ['beta', 'nu', 'tau', 'psi']
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    """
    # Set an appropriate range for the y-axis parameter.
    ylim = 3 if yparam in ['beta', 'psi'] else 1

    # Create an array representing the 2D parameter space.
    size = 1000
    phases = np.zeros((size, size))
    deltas = np.linspace(0, 1, size)
    yvals = np.linspace(0, ylim, size)

    # Compute the individual's optimal action dissent for each point in space,
    # then convert it to a compliant, self-censoring, or defiant phase.
    for i, yval in enumerate(yvals):
        for j, delta in enumerate(deltas):
            if yparam == 'beta':
                phases[i, j] = phase(delta, yval, nu, pi, tau, psi)
            elif yparam == 'nu':
                phases[i, j] = phase(delta, beta, yval, pi, tau, psi)
            elif yparam == 'tau':
                phases[i, j] = phase(delta, beta, nu, pi, yval, psi)
            elif yparam == 'psi':
                phases[i, j] = phase(delta, beta, nu, pi, tau, yval)
            else:
                assert(False, f'ERROR: Unrecognized y-parameter \'{yparam}\'')

    # Plot the phases as colors.
    colors = [(0, 'orange'), (0.25, 'red'), (1, 'darkred')]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('cphase', colors)
    im = ax.pcolormesh(deltas, yvals, phases, cmap=cmap, shading='auto')

    # Set axes information that is common across plots.
    ax.set(xlim=[0, 1], ylim=[0, ylim])
    ax.set_box_aspect(1)

    return im


if __name__ == "__main__":
    # Combine subplots for all parameters and punishment functions.
    fig, ax = plt.subplots(2, 4, figsize=(13, 5.5), dpi=300, sharex='col',
                           facecolor='white', layout='constrained')
    params = ['tau', 'nu', 'psi', 'beta']
    labels = [r'(A) Tolerance $\tau$',
              r'(B) Surveillance $\nu$',
              r'(C) Severity $\psi$',
              r'(D) Boldness $\beta_i$']

    # Plot the phase diagrams for each parameter and punishment function.
    for i, (param, label) in enumerate(zip(params, labels)):
        # Plot sweeps.
        im = plot_phase_diagram(ax[0, i], param, beta=1, nu=0.5, pi='constant',
                                tau=0.25, psi=0.6)
        im = plot_phase_diagram(ax[1, i], param, beta=1, nu=0.5, pi='linear',
                                tau=0.25, psi=1.5)
        # Set axes information.
        ax[0, i].set_title(label, weight='bold')
        ax[1, i].set(xlabel=r'Desire to Dissent, $\delta_i$')
        if i == 0:
            ax[0, i].set(ylabel=r'Constant Punishment $\pi$')
            ax[1, i].set(ylabel=r'Linear Punishment $\pi$')
        if params[i] != 'tau':
            ax[1, i].set_xticks([0, 0.25, 1])
            ax[1, i].set_xticklabels(['0', r'$\tau$', '1'])

    # Create and configure a colorbar shared by all axes.
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label('')
    cbar.set_ticks([0, 0.25, 1])
    cbar.set_ticklabels(['Compliant', 'Self-Censoring', 'Defiant'])

    fig.savefig(osp.join('..', 'figs', 'phase_diagram.png'))
