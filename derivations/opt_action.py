# Project:  CensorshipDissent
# Filename: opt_action.py
# Authors:  Anish Nahar (anahar1@asu.edu) and Joshua J. Daymude (jdaymude@asu.edu).

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


def plot_opt_dissent_vs_beta(ax, delta, nu, pi, t, s, color='tab:blue'):
    """
    Plot an individual's optimal action dissent vs. their boldness.

    :param ax: a matplotlib.Axes object to plot on
    :param delta: the individual's float desire to dissent (in [0,1])
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param s: the authority's float punishment severity (> 0)
    :param color: any color recognized by matplotlib
    """
    # Evaluate the individual's optimal action dissent over a range boldness
    betas = np.linspace(0, 1, 10000)

    acts = np.array([opt_dissent(delta, beta, nu, pi, t, s) for beta in betas])
    ax.plot(betas, acts, c=color)

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()

def plot_opt_dissent_vs_s(ax, delta, nu, pi, t, beta, color='tab:blue'):
    """
    Plot an individual's optimal action dissent vs. authority's severity.

    :param ax: a matplotlib.Axes object to plot on
    :param delta: the individual's float desire to dissent (in [0,1])
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param color: any color recognized by matplotlib
    """
    # Evaluate the individual's optimal action dissent over a range of severity.
    severitys = np.linspace(0, 1, 10000)

    acts = np.array([opt_dissent(delta, beta, nu, pi, t, s) for s in severitys])
    ax.plot(severitys, acts, c=color)

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()

def plot_opt_dissent_vs_t(ax, delta, nu, pi, beta, s, color='tab:blue'):
    """
    Plot an individual's optimal action dissent vs. authority's tolerance.

    :param ax: a matplotlib.Axes object to plot on
    :param delta: the individual's float desire to dissent (in [0,1])
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param s: the authority's float punishment severity (> 0)
    :param color: any color recognized by matplotlib
    """
    # Evaluate the individual's optimal action dissent over a range of tolerances.
    tolerances = np.linspace(0, 1, 10000)

    acts = np.array([opt_dissent(delta, beta, nu, pi, t, s) for t in tolerances])
    ax.plot(tolerances, acts, c=color)

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()

def plot_opt_dissent_vs_nu(ax, delta, beta, pi, t, s, color='tab:blue'):
    """
    Plot an individual's optimal action dissent vs. their authority's surveillance.

    :param ax: a matplotlib.Axes object to plot on
    :param delta: the individual's float desire to dissent (in [0,1])
    :param beta: the individual's float boldness (> 0)
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param s: the authority's float punishment severity (> 0)
    :param color: any color recognized by matplotlib
    """
    # Evaluate the individual's optimal action dissent over a range of nu.
    nus = np.linspace(0, 1, 10000)

    acts = np.array([opt_dissent(delta, beta, nu, pi, t, s) for nu in nus])
    ax.plot(nus, acts, c=color)

    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.grid()

def subplot_boldness(delta_values):
    """
    Combine subplots for optimal action dissent and beta, for both linear and constant punishment with varying deltas = [0.25, 0.5, 0.75, 1]
    """

    # Combine subplots for constant pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and beta when delta = d and pi = constant
        plot_opt_dissent_vs_beta(ax[i], delta=d, nu=0.2, pi='constant', t=0.25, s=0.6)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Boldness, $\beta_i$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')
    
    fig.savefig(osp.join('.', 'figs', 'opt_action_beta_constant.png'))

    # Combine subplots for linear pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and beta when delta = d and pi = linear
        plot_opt_dissent_vs_beta(ax[i], delta=d, nu=0.2, pi='linear', t=0.25, s=0.6)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Boldness, $\beta_i$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')

    fig.savefig(osp.join('.', 'figs', 'opt_action_beta_linear.png'))

def subplot_severity(delta_values):
    """
    Combine subplots for optimal action dissent and severity, for both linear and constant punishment with varying deltas = [0.25, 0.5, 0.75, 1]
    """
    
    # Combine subplots for constant pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and severity when delta = d and pi = constant
        plot_opt_dissent_vs_s(ax[i], delta=d, nu=0.2, pi='constant', t=0.25, beta=1)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Severity, $s_r$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')
    
    fig.savefig(osp.join('.', 'figs', 'opt_action_severity_constant.png'))

    # Combine subplots for linear pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and severity when delta = d and pi = linear
        plot_opt_dissent_vs_s(ax[i], delta=d, nu=0.2, pi='linear', t=0.25, beta=1)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Severity, $s_r$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')

    fig.savefig(osp.join('.', 'figs', 'opt_action_severity_linear.png'))

def subplot_tolerance(delta_values):
    """
    Combine subplots for optimal action dissent and tolerance, for both linear and constant punishment with varying deltas = [0.25, 0.5, 0.75, 1]
    """
    
    # Combine subplots for constant pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and tolerance when delta = d and pi = constant
        plot_opt_dissent_vs_t(ax[i], delta=d, nu=0.2, pi='constant', beta=1, s=0.6)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Tolerance, $t_r$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')
    
    fig.savefig(osp.join('.', 'figs', 'opt_action_tolerance_constant.png'))

    # Combine subplots for linear pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and tolerance when delta = d and pi = linear
        plot_opt_dissent_vs_t(ax[i], delta=d, nu=0.2, pi='linear', beta=1, s=0.6)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Tolerance, $t_r$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')

    fig.savefig(osp.join('.', 'figs', 'opt_action_tolerance_linear.png'))

def subplot_nu(delta_values):
    """
    Combine subplots for optimal action dissent and nu, for both linear and constant punishment with varying deltas = [0.25, 0.5, 0.75, 1]
    """
    
    # Combine subplots for constant pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and nu when delta = d and pi = constant
        plot_opt_dissent_vs_nu(ax[i], delta=d, beta=1, pi='constant', t=0.25, s=0.6)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Surveillance, $v$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')
    
    fig.savefig(osp.join('.', 'figs', 'opt_action_nu_constant.png'))

    # Combine subplots for linear pi into one figure and save.
    fig, ax = plt.subplots(1, 4, figsize=(15.75, 4), dpi=300, facecolor='white',
                           tight_layout=True)
    
    for i, d in enumerate(delta_values):
        # Plot optimal action dissent and nu when delta = d and pi = linear
        plot_opt_dissent_vs_nu(ax[i], delta=d, beta=1, pi='linear', t=0.25, s=0.6)
        ax[i].set(title=f'(a) $\delta_i = {d}$',
                xlabel=r'Surveillance, $v$',
                ylabel=r'Optimal Action Dissent, $a_{i,r}^*$')

    fig.savefig(osp.join('.', 'figs', 'opt_action_nu_linear.png'))

if __name__ == "__main__":
    delta_values = [0.25, 0.5, 0.75, 1.0]

    subplot_boldness(delta_values)

    subplot_severity(delta_values)

    subplot_tolerance(delta_values)

    subplot_nu(delta_values)