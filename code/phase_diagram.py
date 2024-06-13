# Project:  CensorshipDissent
# Filename: phase_diagram.py
# Authors:  Anish Nahar (anahar1@asu.edu) and Joshua J. Daymude (jdaymude@asu.edu).

"""
phase_diagram: Uses an individual's optimal action to plot phase diagrams of
compliance, self-censorship, and defiance as a function of an individual's
desire to dissent vs. the model's other parameters.
"""

from opt_action import duni, dvar, opt_action

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
    :param pi: 'uniform' or 'variable' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    """
    opt_act = opt_action(delta, beta, nu, pi, tau, psi)
    if delta <= tau:  # Compliance.
        return 0
    elif opt_act >= tau and opt_act < delta:  # Self-censorship.
        return 0.5 + 0.5 * ((opt_act - tau) / (delta - tau))
    else:  # Defiance.
        return 1


def plot_phase_diagram(ax, xparam, beta, nu, pi, tau, psi):
    """
    Plot a phase diagram of compliance, self-censorship, and defiance in the
    space of desires to dissent vs. the given parameter.

    :param ax: a matplotlib.Axes object to plot on
    :param xparam: the x-axis parameter, from ['beta', 'nu', 'tau', 'psi']
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'uniform' or 'variable' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    """
    # Set an appropriate range for the y-axis parameter.
    xlim = 2.5 if xparam in ['beta', 'psi'] else 1

    # Create an array representing the 2D parameter space.
    size = 1001  # Avoids divide-by-zero visual artifacts for duni/dvar.
    phases = np.zeros((size, size))
    xvals = np.linspace(0, xlim, size)
    deltas = np.linspace(0, 1, size)

    # Compute the individual's optimal action dissent for each point in space,
    # then convert it to a compliant, self-censoring, or defiant phase.
    for i, delta in enumerate(deltas):
        for j, xval in enumerate(xvals):
            if xparam == 'beta':
                phases[i, j] = phase(delta, xval, nu, pi, tau, psi)
            elif xparam == 'nu':
                phases[i, j] = phase(delta, beta, xval, pi, tau, psi)
            elif xparam == 'tau':
                phases[i, j] = phase(delta, beta, nu, pi, xval, psi)
            elif xparam == 'psi':
                phases[i, j] = phase(delta, beta, nu, pi, tau, xval)
            else:
                assert False, f'ERROR: Unrecognized parameter \'{xparam}\''

    # Plot the phases as colors.
    colors = [(0, 'orange'), (0.5, 'red'), (1, 'darkred')]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('cphase', colors)
    im = ax.pcolormesh(xvals, deltas, phases, cmap=cmap, shading='auto')

    # Plot duni or dvar, the critical points, over the heatmap.
    dfun = duni if pi == 'uniform' else dvar
    if xparam == 'beta':
        ds = [dfun(xval, nu, tau, psi) for xval in xvals]
    elif xparam == 'nu':
        ds = [dfun(beta, xval, tau, psi) for xval in xvals]
    elif xparam == 'tau':
        ds = [dfun(beta, nu, xval, psi) for xval in xvals]
    elif xparam == 'psi':
        ds = [dfun(beta, nu, tau, xval) for xval in xvals]
    ax.plot(xvals, ds, color='white')

    # Set axes information that is common across plots.
    ax.set(xlim=[0, xlim], ylim=[0, 1])
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
        im = plot_phase_diagram(ax[0, i], param, beta=1, nu=0.5, pi='uniform',
                                tau=0.25, psi=0.6)
        im = plot_phase_diagram(ax[1, i], param, beta=1, nu=0.5, pi='variable',
                                tau=0.25, psi=1.5)
        # Set axes information.
        ax[1, i].set_xlabel(label, weight='bold', fontsize='large')
        if i == 0:
            ax[0, i].set(ylabel=r'Uniform Punishment $\pi$')
            ax[1, i].set(ylabel=r'Variable Punishment $\pi$')
        if param != 'tau':
            ax[0, i].set(yticks=[0, 0.25, 1], yticklabels=['0', r'$\tau$', '1'])
            ax[1, i].set(yticks=[0, 0.25, 1], yticklabels=['0', r'$\tau$', '1'])
    fig.supylabel(r'Desired Dissent $\delta_i$')

    # Create and configure a colorbar shared by all axes.
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Compliant', 'Fully Self-\nCensoring', 'Defiant'])

    fig.savefig(osp.join('..', 'figs', 'phase_diagram.png'))
