import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from opt_action import opt_actions
from engine import d2d, b2sim, d2sim, b2a, d2max, d2a, b2b, a2a

def coadaptive_engine(G, N=10000, R=2000, rule='d2d', w=0.5, 
                      deltas=None, betas=None, 
                      pi='uniform', alpha=1.0, eps=0.05,
                      tau0=0.25, psi0=0.1, nu0=0.1,
                      rng=None):
    """
    Simulates BOTH an adaptive authority (using RMHC) and an adaptive 
    network of individuals (using rules above).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if deltas is None: deltas = np.linspace(0, 1, N)
    if betas is None: betas = np.repeat(1.0, N)

    # Setting up History Arrays
    delta_hist = np.zeros((N, R+1))
    delta_hist[:, 0] = deltas
    beta_hist = np.zeros((N, R+1))
    beta_hist[:, 0] = betas
    act_hist = np.zeros((N, R))
    
    # Authority tracking
    params = np.zeros((3, R)) # 0: tau, 1: psi, 2: nu
    pol_costs = np.zeros(R)
    pun_costs = np.zeros(R)
    bounds = np.array([[0, 1], [1e-9, np.inf], [0, 1]])
    param_choices = rng.integers(3, size=R)

    # Main Co-adaptive Loop
    for r in range(R):
        #  AUTHORITY MUTATES
        if r == 0:
            params[:, r] = [tau0, psi0, nu0]
        else:
            p = param_choices[r] # Choose param to update
            params[:, r] = params[:, r-1]
            params[p, r] = rng.uniform(max(bounds[p, 0], params[p, r] - eps),
                                       min(bounds[p, 1], params[p, r] + eps))
        tau, psi, nu = params[:, r]

        # INDIVIDUALS ACT
       
        acts = opt_actions(deltas, betas, nu, pi, tau, psi)
        act_hist[:, r] = acts

        # AUTHORITY EVALUATES COST
        cond = (acts > tau) & (rng.random(N) < (nu + (1 - nu) * acts))
        if pi == 'uniform':
            punish = cond * psi
        elif pi == 'proportional':
            punish = cond * psi * (acts - tau)

        pol_costs[r] = acts.sum()
        pun_costs[r] = punish.sum()

        # If the mutation was worse, revert to last round's parameters
        if r > 0 and alpha * pol_costs[r] + pun_costs[r] > alpha * pol_costs[r-1] + pun_costs[r-1]:
            params[:, r] = params[:, r-1]

        # NETWORK ADAPTS!
        if rule == 'b2sim':
            betas = b2sim(G, deltas, acts, shape=w)
        elif rule == 'd2sim':
            betas = d2sim(G, deltas, acts, shape=w)
        elif rule == 'b2a':
            betas = b2a(G, acts, shape=w)
        elif rule == 'd2max':
            betas = d2max(G, acts, shape=w)
        elif rule == 'd2d':
            deltas = d2d(G, deltas, w)
        elif rule == 'd2a':
            deltas = d2a(G, deltas, acts, w)
        elif rule == 'b2b':
            betas = b2b(G, betas, w)
        elif r > 0 and rule == 'a2a':
            acts = a2a(G, acts, act_hist[:, r-1], w)

        # Record histories
        delta_hist[:, r+1] = np.copy(deltas)
        beta_hist[:, r+1] = np.copy(betas)

    return delta_hist, beta_hist, act_hist, params, pol_costs, pun_costs
if __name__ == "__main__":
    # Setup graph and population
    N, R = 10000, 2000
    G = nx.barabasi_albert_graph(n=N, m=2, seed=42)
    rng = np.random.default_rng(42)

    # Init pop: 5% active dissenters
    N_A = int(0.05 * N)
    deltas = np.append(np.clip(rng.normal(0.9, 0.1, N_A), 0, 1),
                       np.clip(rng.normal(0.2, 0.1, N - N_A), 0, 1))
    betas = np.append(np.repeat(2.0, N_A), np.repeat(0.75, N - N_A))

    # Run the engine
    delta_hist, beta_hist, act_hist, params, pol_costs, pun_costs = coadaptive_engine(
        G, N=N, R=R, rule='d2d', w=0.5, deltas=deltas, betas=betas, 
        pi='uniform', alpha=1.0, eps=0.05, tau0=0.25, psi0=0.5, nu0=0.2, rng=rng
    )

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, dpi=200)

    # 1. Network Averages
    axs[0].plot(np.mean(delta_hist[:, :-1], axis=0), label='Mean Desired Dissent ($\\delta$)', color='purple')
    axs[0].plot(np.mean(act_hist, axis=0), label='Mean Action ($a$)', color='orange')
    axs[0].set_ylabel("Network Averages")
    axs[0].set_title("Co-Adaptive Dynamics: Authority vs Network (BA Graph, d2d)")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # 2. Authority Parameters
    axs[1].plot(params[0, :], label='Tolerance ($\\tau$)', color='blue')
    axs[1].plot(params[1, :], label='Severity ($\\psi$)', color='red')
    axs[1].plot(params[2, :], label='Surveillance ($\\nu$)', color='green')
    axs[1].set_ylabel("Authority Params")
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # 3. Authority Costs
    axs[2].plot(pol_costs, label='Political Cost', color='darkorange')
    axs[2].plot(pun_costs, label='Punishment Cost', color='brown')
    axs[2].plot(pol_costs + pun_costs, label='Total Cost', color='black', linestyle='--')
    axs[2].set_ylabel("Authority Costs")
    axs[2].set_xlabel("Round ($r$)")
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('coadaptive_dynamics.png')
    print("Graph saved as 'coadaptive_dynamics.png'")