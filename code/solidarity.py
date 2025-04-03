# Project:  CensorshipDissent
# Filename: solidarity.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
solidarity: Network simulation experiment for the solidarity adaptation rule.
"""

from engine import engine

from cmcrameri import cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os.path as osp


if __name__ == '__main__':
    # Set up random number generator.
    rng = np.random.default_rng()

    # Generate the network.
    N = 1000
    N_A = int(0.035 * N)
    sizes = [N_A, N - N_A]
    probs = [[0.95, 5 / (N - N_A)], [5 / (N - N_A), 0.05]]
    G = nx.stochastic_block_model(sizes, probs, seed=rng)
    # G = nx.path_graph(N)

    # Set up other parameters.
    R = 5
    deltas = np.append(np.minimum(np.maximum(rng.normal(0.9, 0.1, N_A), 0), 1),
                       np.minimum(np.maximum(rng.normal(0.2, 0.1, N - N_A), 0), 1))
    # deltas = np.append(np.repeat(0.9, N_A), np.repeat(0.5, N - N_A))
    betas = np.append(np.repeat(2, N_A), np.repeat(0.75, N - N_A))

    # Run simulation.
    deltas, betas, acts = engine(
        G, N=N, R=R, rule='b2a', w=1, deltas=deltas, betas=betas, nu=0.5,
        pi='uniform', tau=0.1, sigma_tau=0.0, psi=1, sigma_psi=0.0, rng=rng)

    # Plot the network.
    fig, ax = plt.subplots(dpi=300, tight_layout=True)
    colors = [cm.batlowS(i) for i in range(N)]
    nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G, 'sfdp'), node_size=10,
            node_color=colors, width=0.1)
    fig.savefig(osp.join('..', 'figs', 'b2sim_network.png'))

    # Plot desires, boldnesses, and actions over time.
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), dpi=300, sharex=True,
                           tight_layout=True)
    for i in range(N):
        ax[0].plot(np.arange(R+1), deltas[i], color=colors[i])
        ax[1].plot(np.arange(R+1), betas[i], color=colors[i])
        ax[2].plot(np.arange(R)+1, acts[i], color=colors[i])

    # Set axes information and save.
    ax[0].set(ylim=[-0.02, 1.02], ylabel=r'Desire $\delta_{i,r}$')
    ax[1].set(ylim=[0, 10], ylabel=r'Boldness $\beta_{i,r}$')
    ax[2].set(xlim=[0, R], xlabel='Round',
              ylim=[-0.02, 1.02], ylabel=r'Action $a_{i,r}$')
    fig.savefig(osp.join('..', 'figs', 'b2sim_evo.png'))
