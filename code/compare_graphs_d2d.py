# Project:  CensorshipDissent
# Filename: compare_graphs_d2d.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
This script uses the same engine (rule="d2d"), only changing the graph generation method (SBM/WS/BA), and repeatedly runs the simulation multiple times for different N_A fractions. It takes the mean and standard deviation of the final mean δ.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from engine import engine


# ---------------------------
# Graph generators (SBM / WS / BA)
# ---------------------------
def make_graph(kind: str, n: int, seed: int,
               N_A: int,
               ws_k: int = 6, ws_p: float = 0.10,
               ba_m: int = 2,
               sbm_P=None):
    """
    Create a connected-ish graph with node labels 0..N-1 to match engine's indexing style.
    """
    kind = kind.upper()

    if kind == "WS":
        # connected version avoids isolated nodes (d2d uses mean over neighbors)
        return nx.connected_watts_strogatz_graph(n=n, k=ws_k, p=ws_p, seed=seed)

    if kind == "BA":
        return nx.barabasi_albert_graph(n=n, m=ba_m, seed=seed)

    if kind == "SBM":
        # Two communities aligned with "active vs non-active" just for community structure
        sizes = [N_A, n - N_A]
        P = sbm_P if sbm_P is not None else [[0.15, 0.01],
                                             [0.01, 0.05]]
        G = nx.stochastic_block_model(sizes, P, seed=seed, directed=False, selfloops=False, sparse=True)

        # SBM may be disconnected -> take largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        # relabel to 0..N-1 to keep indexing safe in engine (acts[G[i]] style)
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        return G

    raise ValueError(f"Unknown kind: {kind}")


# ---------------------------
# Initialization (same idea as solidarity.py)
# ---------------------------
def init_population(rng: np.random.Generator, N: int, N_A: int):
    """
    Create deltas/betas with first N_A "active" having higher delta/beta.
    """
    deltas = np.append(
        np.clip(rng.normal(0.9, 0.1, N_A), 0, 1),
        np.clip(rng.normal(0.2, 0.1, N - N_A), 0, 1),
    )
    betas = np.append(np.repeat(2.0, N_A), np.repeat(0.75, N - N_A))
    return deltas, betas


# ---------------------------
# One experiment: final mean delta for given params
# ---------------------------
def final_mean_delta(kind: str,
                     n: int,
                     N_A_frac: float,
                     R: int = 50,
                     runs: int = 10,
                     base_seed: int = 0,
                     # fixed engine params:
                     rule: str = "d2d",
                     w: float = 0.5,
                     nu: float = 0.5,
                     pi: str = "uniform",
                     tau: float = 0.1, sigma_tau: float = 0.0,
                     psi: float = 1.0, sigma_psi: float = 0.0,
                     # graph params:
                     ws_k: int = 6, ws_p: float = 0.10,
                     ba_m: int = 2,
                     sbm_P=None):
    finals = []

    for r in range(runs):
        seed = base_seed + r
        rng = np.random.default_rng(seed)

        # build graph
        N_A = max(1, int(N_A_frac * n))
        G = make_graph(kind, n=n, seed=seed, N_A=N_A,
                       ws_k=ws_k, ws_p=ws_p,
                       ba_m=ba_m, sbm_P=sbm_P)

        # graph size might change if we took largest CC
        N = G.number_of_nodes()
        N_A = max(1, int(N_A_frac * N))

        # init conditions
        deltas0, betas0 = init_population(rng, N, N_A)

        delta_hist, beta_hist, act_hist = engine(
            G, N=N, R=R, rule=rule, w=w,
            deltas=deltas0, betas=betas0,
            nu=nu, pi=pi, tau=tau, sigma_tau=sigma_tau, psi=psi, sigma_psi=sigma_psi,
            rng=rng
        )

        # final mean delta (this is what your Figure 2 needs)
        finals.append(float(delta_hist.mean(axis=0)[-1]))

    return float(np.mean(finals)), float(np.std(finals))


# ---------------------------
# Sweep helpers
# ---------------------------
def sweep_NA_three_models(outdir="figs_final",
                          n=500, R=50, runs=10,
                          NA_fracs=None,
                          # fixed model params:
                          ws_k=6, ws_p=0.10,
                          ba_m=2,
                          sbm_P=None):
    os.makedirs(outdir, exist_ok=True)
    if NA_fracs is None:
        NA_fracs = np.linspace(0.01, 0.15, 10)

    models = ["SBM", "WS", "BA"]
    results = {m: {"mean": [], "std": []} for m in models}

    for frac in NA_fracs:
        for m in models:
            mean, std = final_mean_delta(
                m, n=n, N_A_frac=float(frac),
                R=R, runs=runs,
                ws_k=ws_k, ws_p=ws_p,
                ba_m=ba_m,
                sbm_P=sbm_P
            )
            results[m]["mean"].append(mean)
            results[m]["std"].append(std)

    # plot (Figure 2A)
    plt.figure(dpi=200)
    for m in models:
        y = np.array(results[m]["mean"])
        yerr = np.array(results[m]["std"])
        plt.errorbar(NA_fracs, y, yerr=yerr, marker="o", capsize=3, label=m)

    plt.xlabel("N_A fraction")
    plt.ylabel("Final mean delta (d2d)")
    plt.title("Final mean delta vs N_A (SBM vs WS vs BA)")
    plt.grid(True)
    plt.legend()
    outpath = os.path.join(outdir, "final_vs_NA_three_models.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    return results

def sweep_WS_p(outdir="figs_final",
               n=500, R=50, runs=10,
               NA_frac=0.035,
               ws_k=6,
               ps=None):
    os.makedirs(outdir, exist_ok=True)
    if ps is None:
        ps = np.linspace(0, 1, 11)

    means, stds = [], []
    for p in ps:
        mean, std = final_mean_delta("WS", n=n, N_A_frac=NA_frac, R=R, runs=runs, ws_k=ws_k, ws_p=float(p))
        means.append(mean); stds.append(std)

    plt.figure(dpi=200)
    plt.errorbar(ps, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel("p (rewiring probability)")
    plt.ylabel("Final mean delta (d2d)")
    plt.title(f"WS: Final mean delta vs p (k={ws_k}, N_A={NA_frac})")
    plt.grid(True)
    outpath = os.path.join(outdir, "final_vs_p_WS.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    print("Saved:", outpath)


def sweep_BA_m(outdir="figs_final",
               n=500, R=50, runs=10,
               NA_frac=0.035,
               ms=None):
    os.makedirs(outdir, exist_ok=True)
    if ms is None:
        ms = [1, 2, 4, 8]

    means, stds = [], []
    for m in ms:
        mean, std = final_mean_delta("BA", n=n, N_A_frac=NA_frac, R=R, runs=runs, ba_m=int(m))
        means.append(mean); stds.append(std)

    plt.figure(dpi=200)
    plt.errorbar(ms, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel("m (edges per new node)")
    plt.ylabel("Final mean delta (d2d)")
    plt.title(f"BA: Final mean delta vs m (N_A={NA_frac})")
    plt.grid(True)
    outpath = os.path.join(outdir, "final_vs_m_BA.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    print("Saved:", outpath)


if __name__ == "__main__":
    # (final vs N_A with three models)
    sweep_NA_three_models(
        outdir="figs_final",
        n=500, R=50, runs=10,
        NA_fracs=np.linspace(0.01, 1, 10),
        ws_k=6, ws_p=0.10,
        ba_m=2,
        sbm_P=[[0.15, 0.01],
               [0.01, 0.05]]
    )

sweep_WS_p(outdir="figs_final", n=500, R=50, runs=10, NA_frac=0.035, ws_k=6)
sweep_BA_m(outdir="figs_final", n=500, R=50, runs=10, NA_frac=0.035)
