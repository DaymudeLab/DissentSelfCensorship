import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path as osp

def load_np(fname):
    """
    Reads a numpy array from file.
    """
    with open(fname, 'rb') as f:
        return np.load(f)

# Define the phase function
def phase(action, delta, beta=1, nu=None, pi=None, tau=0.25, psi=None):
    if delta <= tau:  # Compliance.
        return 0
    elif action >= tau and action < delta:  # Self-censorship.
        return 0.25 + 0.75 * ((action - tau) / (delta - tau))
    else:  # Defiance.
        return 1

def calc_action(actions, pi, psi, mu_delta, nu, T, N):
    """
    Calculate the average action between each trial and individual
    """
    sum_action = 0
    for i in range(T):
        for j in range(N):
            sum_action += actions[pi, psi, mu_delta, nu, i, j]

    return (sum_action*1.0)/(N*T)

def plot_phase_diagram(ax, actions, psi, beta, pi, tau):
    """
    Plot a phase diagram of compliance, self-censorship, and defiance in the
    space of desires to dissent vs. the given parameter.

    :param ax: a matplotlib.Axes object to plot on
    :param beta: the individual's float boldness (> 0)
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param tau: the authority's float tolerance (in [0,1])
    :param psi: the authority's float punishment severity (> 0)
    """

    # Create an array representing the 2D parameter space.
    size = 50
    phases = np.zeros((size, size))
    deltas = np.linspace(0.1, 0.9, size)
    nus = np.linspace(0, 1, size)
    
    trials = 25 # Trials
    num_individual = 500 # number of individuals
    
    # Compute the individual's optimal action dissent for each point in space,
    # then convert it to a compliant, self-censoring, or defiant phase.
    for i, nu in enumerate(nus):
        for j, delta in enumerate(deltas):
            action = calc_action(actions=actions, pi=pi, psi=psi, mu_delta=j, nu=i, T=trials, N=num_individual)
            phases[i, j] = action
            #phases[i, j] = phase(action, delta, beta, nu, pi, tau, psi)

    # Plot the phases as colors.
    colors = [(0, 'yellow'), (0.25, 'orange'), (0.5, 'red'), (0.75, 'darkred'), (1, 'black')]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('cphase', colors)
    im = ax.pcolormesh(deltas, nus, phases, cmap=cmap, vmin=0, vmax=1)

    # Set axes information that is common across plots.
    ax.set(xlim=[0.1, 0.9], ylim=[0, 1])
    ax.set_box_aspect(1)
    
    return im


if __name__ == "__main__":
    # Load the .npy files
    loaded_deltas = load_np('other_code/anish_thesis_deltas.npy')
    loaded_actions = load_np('other_code/anish_thesis_acts.npy')

    # Combine subplots for all parameters and punishment functions.
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.5), dpi=300, sharex='col',
                           facecolor='white', layout='constrained')
    
    psi_array = [0.5, 1, 1.5]
    labels = [r'(A) $\psi_r$ < $\beta_i$',
              r'(B) $\psi_r$ = $\beta_i$',
              r'(C) $\psi_r$ > $\beta_i$']

    diagram = 'action' # ['action', 'desire']

    # Plot the phase diagrams for psi and punishment function.
    for i, (psi, label) in enumerate(zip(psi_array, labels)):
        # Plot sweeps.
        if diagram == 'action':
            im = plot_phase_diagram(ax[0, i], actions=loaded_actions, psi=i, beta=1, pi=0,
                                    tau=0.25) # pi = constant
            im = plot_phase_diagram(ax[1, i], actions=loaded_actions, psi=i, beta=1, pi=1,
                                    tau=0.25) # pi = linear
        elif diagram == 'desire':
            im = plot_phase_diagram(ax[0, i], actions=loaded_deltas, psi=i, beta=1, pi=0,
                                    tau=0.25) # pi = constant
            im = plot_phase_diagram(ax[1, i], actions=loaded_deltas, psi=i, beta=1, pi=1,
                                    tau=0.25) # pi = linear
        else:
            exit(1)

        # Set axes information.
        ax[0, i].set_title(label, weight='bold')
        ax[1, i].set(xlabel=r'Desire to Dissent, $\delta_i$')
        if i == 0:
            ax[0, i].set(ylabel=r'(i) Constant Punishment $\pi$' + '\n' +'\n' + r'Surveillance $\nu$')
            ax[1, i].set(ylabel=r'(ii) Linear Punishment $\pi$' + '\n' + '\n' + r'Surveillance $\nu$')
        
        ax[1, i].set_xticks([0.1, 0.25, 0.9])
        ax[1, i].set_xticklabels(['0.1', r'$\tau_r$', '0.9'])

    

    # Create and configure a colorbar shared by all axes.
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label('')

    if diagram == 'action':
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        fig.savefig(osp.join('figs', 'd2a_action_phase_diagram.png'))
    elif diagram == 'desire':
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        fig.savefig(osp.join('figs', 'd2a_desire_phase_diagram.png'))
    else:
        cbar.set_ticks([0, 0.25, 1])
        cbar.set_ticklabels(['Compliant', 'Self-Censoring', 'Defiant'])
        fig.savefig(osp.join('figs', 'd2a_trans_action_phase_diagram.png'))