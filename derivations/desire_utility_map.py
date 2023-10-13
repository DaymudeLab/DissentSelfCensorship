# Project:  CensorshipDissent
# Filename: desire_map.py
# Authors:  Anish Nahar (anahar1@asu.edu)

"""
desire_map: Calculates and plots an individual's desire to dissent and
corresponding utility and map the individual into compliant, self-censoring, 
partial self-cesnoring , defiant
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def heat_map_value(delta, t, opt_a):
    """
    Get the color of heat map based on the optimal action of the individual, relative to their desire to dissent and the authority's tolerance.

    :param delta: the individual's float desire to dissent (in [0,1])
    :param t: the authority's float tolerance (in [0,1])
    :param opt_a: the individual's float optimal action dissent (in [0,delta])
    """

    # Compliant Individuals
    if delta <= t:
        return 0
    # Self-Censoring Individuals
    elif opt_a == t and delta > t:
        return t
    # Partial Self-Censoring Individuals
    elif opt_a > t and opt_a < delta:
        position_percentage = ((opt_a - t) / (delta - t)) * 100

        # Interpolate between t and 1 based on position_percentage
        return t + (position_percentage / 100) * (1 - t)
    # Defiant Individuals
    elif opt_a == delta and opt_a > t:
        return 1

def heat_map_value_for_t(delta, t, opt_a):
    """
    Get the color of heat map based on the optimal action of the individual, relative to their desire to dissent and the authority's tolerance.

    :param delta: the individual's float desire to dissent (in [0,1])
    :param t: the authority's float tolerance (in [0,1])
    :param opt_a: the individual's float optimal action dissent (in [0,delta])
    """

    # Compliant Individuals
    if delta <= t:
        return 0
    # Self-Censoring Individuals
    elif opt_a == t and delta > t:
        return 0.25
    # Partial Self-Censoring Individuals
    elif opt_a > t and opt_a < delta:
        position_percentage = ((opt_a - t) / (delta - t)) * 100

        # Interpolate between t and 1 based on position_percentage
        return 0.25 + (position_percentage / 100) * (1 - 0.25)
    # Defiant Individuals
    elif opt_a == delta and opt_a > t:
        return 1

def plot_opt_dissent_vs_utility(ax, beta, nu, pi, t, s, utility_str):
    """
    Plot the authority's surveillance vs individual's desire to dissent.

    :param ax: a matplotlib.Axes object to plot on
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param s: the authority's float punishment severity (> 0)
    :param utility_str: the string to use for the utility axis (["nu", "t", "s", "beta"])
    """

    y_limit = 1
    if utility_str == "beta" or utility_str == "s":
        y_limit = 5

    # Evaluate the individual's optimal action dissent over a range of nu and delta.
    desires = np.linspace(0, 1, 1000)

    acts = np.empty((1000, 1000))


    utility = np.linspace(0, y_limit, 1000)

    
    # Get the heat map values for the given utility
    if utility_str == "nu":
        for i, nu in enumerate(utility):
            for j, delta in enumerate(desires):
                acts[i, j] = heat_map_value(delta, t, opt_dissent(delta, beta, nu, pi, t, s))
    elif utility_str == "t":
        for i, t in enumerate(utility):
            for j, delta in enumerate(desires):
                acts[i, j] = heat_map_value_for_t(delta, t, opt_dissent(delta, beta, nu, pi, t, s))
    elif utility_str == "s":
        for i, s in enumerate(utility):
            for j, delta in enumerate(desires):
                acts[i, j] = heat_map_value(delta, t, opt_dissent(delta, beta, nu, pi, t, s))
    elif utility_str == "beta":
        for i, beta in enumerate(utility):
            for j, delta in enumerate(desires):
                acts[i, j] = heat_map_value(delta, t, opt_dissent(delta, beta, nu, pi, t, s))



    if utility_str != "t":
        # Define custom colors for the colormap
        cmap_colors = [(0.0, 'orange'), (t, 'red'), (1.0, 'darkred')]

        # Create a custom colormap using LinearSegmentedColormap
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_heatmap', cmap_colors)

        im = ax.pcolormesh(desires, utility, acts, cmap=cmap, shading='auto')

        # Dotted line for tolerance
        #ax.axvline(t, color='white', linestyle='--')

        # X-axis ticks
        ax.set_xticks([0, t, 1])
        ax.set_xticklabels(['0.0', r'$t_r$', '1.0'])
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([0, t, 1])
    else:
        # Define custom colors for the colormap
        cmap_colors = [(0.0, 'orange'), (0.25, 'red'), (1.0, 'darkred')]

        # Create a custom colormap using LinearSegmentedColormap
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_heatmap', cmap_colors)

        im = ax.pcolormesh(desires, utility, acts, cmap=cmap, shading='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([0, 0.25, 1])

    ax.set(xlim=[0, 1], ylim=[0, y_limit])

    cbar.set_label('')
    cbar.set_ticklabels(['Compliant', 'Self-Censoring', 'Defiant'])

if __name__ == "__main__":
    titles = [r'Surveillance, $\nu$', r'Tolerance, $t_r$', r'Severity, $s_r$', r'Boldness, $\beta_i$']
    utilities = ["nu", "t", "s", "beta"]

    for i, utility in enumerate(utilities):
        # Combine subplots for constant pi into one figure and save.
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=300, facecolor='white',
                            tight_layout=True)
        
        # Plot desire to dissent and surveillance for constant punishment.
        plot_opt_dissent_vs_utility(ax[0], nu=0.5, beta=1, pi='constant', t=0.25, s=0.6, utility_str=utility)
        ax[0].set(title=r'(a) Constant Punishment',
            xlabel=r'Desire to Dissent, $\delta_i$',
            ylabel=titles[i])
    
        # Plot desire to dissent and surveillance for linear punishment.
        plot_opt_dissent_vs_utility(ax[1], nu=0.5, beta=1, pi='linear', t=0.25, s=1, utility_str=utility)
        ax[1].set(title=r'(b) Linear Punishment',
            xlabel=r'Desire to Dissent, $\delta_i$')
        
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)
        
        fig.savefig(osp.join('.', 'figs', f'desire_{utility}_heat_map.png'))