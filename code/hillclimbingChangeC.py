# Project:  CensorshipDissent
# Filename: hillclimbing.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
hillclimbing: An adaptive authority experiment using random hill climbing.
"""

from opt_action import opt_actions

import argparse
from cmcrameri import cm
from helper import dump_np, load_np
from itertools import product, repeat
from math import expm1, isclose
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def texponential(rng, bound, scale, size):
    """
    Generates `size` samples from a truncated exponential distribution with
    mean `scale` and range [0, bound]. The underlying sampling method uses
    rejection sampling from a non-truncated exponential distribution whose mean
    is approximated to yield the desired truncated distribution's mean.

    :param rng: a numpy.random.Generator instance for random number generation
    :param bound: a float upper bound of the truncated exponential distribution
    :param scale: a float mean of the truncated exponential distribution; must
    be in (0, bound / 2)
    :param size: an int number of samples to generate
    :returns: a 1xsize array of float random samples
    """
    # Use bisection search to obtain the mean of an exponential distribution
    # that, when rejection sampled to [0, bound], effectively samples from the
    # truncated exponential distribution on [0, bound] with the desired mean.
    # Specifically, the mean X of the desired truncated exponential and the
    # mean Y of the corresponding non-truncated exponential are related as:
    # X = Y - bound / (exp(bound / Y) - 1). This method approximates Y given X.
    lower, upper = scale, 2*scale
    while upper - bound / expm1(bound / upper) < scale:
        lower = upper
        upper *= 2
    while True:
        mid = (upper + lower) / 2
        approx_mean = mid - bound / expm1(bound / mid)
        if isclose(scale, approx_mean):
            break
        elif approx_mean < scale:
            lower = mid
        else:  # approx_mean >= scale
            upper = mid

    # Generate the desired number of samples using rejection sampling.
    samples = np.array([])
    while len(samples) < size:
        x = rng.exponential(scale=mid, size=size)
        samples = np.append(samples, x[np.where(x <= bound)])

    return samples[:size]


def self_censor_mask(acts, deltas, atol=1e-12, rtol=1e-9):
    """
    Returns a boolean mask of individuals who self-censored this round, i.e.
    whose actual action fell strictly below their desired dissent (guarding
    against floating-point ties).
    """
    return (acts < deltas) & np.logical_not(
        np.isclose(acts, deltas, atol=atol, rtol=rtol)
    )


def trigger_masks(update_trigger, punished_mask, acts, deltas):
    """
    Returns separate masks for desire and boldness updates.

    'punished'    -> both desire and boldness use punishment
    'self_censor' -> both desire and boldness use self-censorship
    'mixed'       -> desire uses self-censorship, boldness uses punishment
    """
    self_censor = self_censor_mask(acts, deltas)
    if update_trigger == 'punished':
        return punished_mask, punished_mask
    if update_trigger == 'self_censor':
        return self_censor, self_censor
    if update_trigger == 'mixed':
        return self_censor, punished_mask
    assert False, f'ERROR: Invalid update trigger "{update_trigger}"'


def rmhc_trial(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, seed,
               k_mutate=1, pair='random', k3_method='sphere', C=0.0,
               update_trigger='punished', update_target='desire',
               boldness_pct=0.01):
    """
    Runs a single simulation trial of the model where individuals' desired
    dissents and boldness constants are exponentially-distributed but fixed and
    the authority adapts its parameters based on random mutation hill climbing.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param delta: a float mean population desired dissent (> 0)
    :param beta: a float mean population boldness (> 0)
    :param pi: 'uniform' or 'proportional' punishment
    :param tau0: the authority's float initial tolerance (in [0,1])
    :param psi0: the authority's float initial punishment severity (> 0)
    :param nu0: the authority's float initial surveillance (in [0,1])
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation
    :param update_trigger: what condition drives the population update —
                           'punished'    : both desire and boldness use those
                                           caught/observed by the authority
                           'self_censor' : both desire and boldness use those
                                           whose action < desired dissent
                           'mixed'       : desire uses self-censorship, while
                                           boldness uses punishment
    :param update_target: what gets updated each round —
                          'desire'   : only desired dissent shifts
                          'boldness' : only boldness shifts
                          'all'      : both desire and boldness shift
    :param boldness_pct: multiplicative boldness change fraction; only used
                         when update_target is 'boldness' or 'all'

    :returns: a 3xR array of the authority's parameter values
    :returns: a 1xR array of the authority's political costs
    :returns: a 1xR array of the authority's punishment costs
    :returns: a 1xN array of individuals' final dissent desires
    :returns: a 1xN array of individuals' final boldness constants
    :returns: a 1xR array of mean population desired dissent per round
    :returns: a 1xR array of mean population boldness per round
    """
    # Set up random number generation.
    rng = np.random.default_rng(seed)

    # Initialize the population's desired dissents according to an exponential
    # distribution truncated to [0, 1] with the given mean.
    deltas = texponential(rng, bound=1, scale=delta, size=N)

    # Initialize the population's boldness constants according to an
    # exponential distribution with the given mean.
    betas = rng.exponential(scale=beta, size=N)

    # Set bounds on the authority's parameters.
    bounds = np.array([[0, 1],          # tau
                       [1e-9, np.inf],  # psi
                       [0, 1]])         # nu

    # Set up arrays to store everything that happens.
    params = np.zeros((3, R))
    pol_costs, pun_costs = np.zeros(R), np.zeros(R)
    avg_desires = np.zeros(R)  # mean population desired dissent per round
    avg_betas   = np.zeros(R)  # mean population boldness per round

    #define plot params
    cands_history = []

    # Pre-generate all random choices of which parameter to attempt to update
    # at each step; the hope is that doing this in batch is faster than doing
    # one at a time in each for loop iteration.
    #param_choices = rng.integers(3, size=R)

    # Simulate the specified number of rounds, allowing the authority to adapt
    # its parameters using random mutation hill climbing (RMHC).
    for r in range(R):
        # If this is the first round, the authority simply uses its initial
        # parameters. Otherwise, it generates new candidate parameters to test.
        if r == 0:
            params[:, r] = [tau0, psi0, nu0]
        else:
            #params[:, r] = params[:, r-1]
            candidate_params = np.copy(params[:, r-1])

        
            #choose how many params gonna changeeeee

            # --k-mutate [1,2,3]: Sets whether to mutate 1, 2, or all 3 parameters at each step.

            # --pair [tp,tn,pn]: For k=2, specifies which parameter pair to mutate (t=tolerance, p=psi, n=nu).

            # --k3-method [box,sphere]: For k=3, sets the mutation method to sample from a cube (box) or a sphere.
            
            if k_mutate == 1:
                idx = rng.integers(3)

                low  = max(bounds[idx, 0], candidate_params[idx] - eps)
                high = min(bounds[idx, 1], candidate_params[idx] + eps)
                candidate_params[idx] = rng.uniform(low, high)

            elif k_mutate == 2:
                if pair == 'random':
                    idx_pair = rng.choice(3, size=2, replace=False)
                else:
                    pair_map = {'tp': [0,1], 'tn': [0,2], 'pn': [1,2]}
                    # 0:tau,1:psi,2:nu
                    idx_pair = pair_map[pair]

                for p in idx_pair:
                    low  = max(bounds[p, 0], candidate_params[p] - eps)
                    high = min(bounds[p, 1], candidate_params[p] + eps)
                    candidate_params[p] = rng.uniform(low, high)
            # k_mutate == 3
            else:
            # Mutate all three parameters
                if k3_method == 'box':
                        # Sample uniformly from a cube around the current point.
                    for p in range(3):
                        low  = max(bounds[p, 0], candidate_params[p] - eps)
                        high = min(bounds[p, 1], candidate_params[p] + eps)
                        candidate_params[p] = rng.uniform(low, high)
                    
                elif k3_method == 'sphere':
                        
                    angle = rng.normal(size=3) #sample from normal distribution
                    angle /= np.linalg.norm(angle) #get unit vector

                    #magnitude = eps * (rng.random()**(1/3.0)) #uniform in sphere, inverse transform sampling, 4/3 pi r^3 volume
                    magnitude = eps
                    move = angle * magnitude
                    candidate_params += move

                     # Clip the new values to stay within bounds.
                candidate_params[0] = np.clip(candidate_params[0], bounds[0, 0], bounds[0, 1])
                candidate_params[1] = max(candidate_params[1], bounds[1, 0])
                candidate_params[2] = np.clip(candidate_params[2], bounds[2, 0], bounds[2, 1])

                # Set the current parameters to the new candidate for this round.
            params[:, r] = candidate_params
            
            #record params
            cands_history.append(candidate_params.copy())

        # The individuals act based on their desires and boldness constants and
        # the authority's current parameters.
        tau, psi, nu = params[:, r]
        acts = opt_actions(deltas, betas, nu, pi, tau, psi)

        # The authority punishes any actions that it observes above tolerance.
        cond = (acts > tau) & (rng.random(N) < (nu + (1 - nu) * acts))
        if pi == 'uniform':
            punish = cond * psi
        elif pi == 'proportional':
            punish = cond * psi * (acts - tau)
        else:
            assert False, f'ERROR: Invalid punishment function \"{pi}\"'

        # The authority's political cost for this round is the summed actions
        # and its punishment cost is the summed punishments.
        pol_costs[r] = acts.sum()
        pun_costs[r] = punish.sum()

        # If the authority's adamancy-weighted cost in this round is worse than
        # last round, reset to last round's parameters.
        if r > 0 and alpha * pol_costs[r] + pun_costs[r] > \
                alpha * pol_costs[r-1] + pun_costs[r-1]:
            params[:, r] = params[:, r-1]

        # Population update: pick the trigger mask first, then apply to
        # whichever targets are selected.
        #   update_trigger 'punished'   -> mask = those caught by authority
        #   update_trigger 'self_censor'-> mask = those who held back
        #   update_target  'desire'     -> shift desired dissent only
        #   update_target  'boldness'   -> shift boldness only
        #   update_target  'all'        -> shift both
        if C > 0 or update_target in ('boldness', 'all'):
            desire_mask, boldness_mask = trigger_masks(
                update_trigger, cond, acts, deltas
            )

            if update_target in ('desire', 'all'):
                deltas = np.where(desire_mask,
                                  np.minimum(1, deltas + C),
                                  np.maximum(0, deltas - C))

            if update_target in ('boldness', 'all'):
                betas = np.where(boldness_mask,
                                 np.maximum(1e-9, betas * (1 - boldness_pct)),
                                 np.maximum(1e-9, betas * (1 + boldness_pct)))

        # Record mean population desire and boldness after this round's update.
        avg_desires[r] = deltas.mean()
        avg_betas[r]   = betas.mean()

    #after getting all params, plot!
    #_plot_candidates_2d(cands_history)

    return params, pol_costs, pun_costs, deltas, betas, avg_desires, avg_betas

def fast_c_worker(k, C, N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps,
                  seed, update_trigger, update_target, boldness_pct, trials=5):
    """
    Worker function to run multiple C vs Cost trials and average them.

    Also captures the full parameter and cost trajectories from the first
    (representative) trial so that plot_trial can be called per C value.

    :returns: k, C, update_trigger, update_target, boldness_pct, mean final cost,
              rep_params, rep_pol_costs, rep_pun_costs,
              rep_avg_desires, rep_avg_betas
    """
    final_costs = []
    rep_params = rep_pol_costs = rep_pun_costs = None
    rep_avg_desires = rep_avg_betas = None

    for t in range(trials):
        # Give each trial a unique seed.
        trial_seed = seed + t if seed is not None else None

        params, pol_costs, pun_costs, _, _, avg_desires, avg_betas = rmhc_trial(
            N, R, delta, beta, pi, tau0, psi0, nu0, alpha,
            eps, trial_seed, k_mutate=k, k3_method='sphere', C=C,
            update_trigger=update_trigger, update_target=update_target,
            boldness_pct=boldness_pct
        )
        final_costs.append(alpha * pol_costs[-1] + pun_costs[-1])

        if t == 0:
            rep_params      = params
            rep_pol_costs   = pol_costs
            rep_pun_costs   = pun_costs
            rep_avg_desires = avg_desires
            rep_avg_betas   = avg_betas

    return (k, C, update_trigger, update_target, boldness_pct,
            np.mean(final_costs),
            rep_params, rep_pol_costs, rep_pun_costs,
            rep_avg_desires, rep_avg_betas)

def plot_fast_c_vs_cost(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps,
                        seed, threads, update_trigger='punished',
                        update_target='desire', boldness_pct=0.01):
    """
    Sweeps C (and boldness_pct where relevant) and produces per-round plots.

    Summary plot  →  figs/fast_c_vs_cost_{trigger}_{target}.pdf
    Authority     →  figs/{trigger}_{target}_trials/
    Population    →  figs/{trigger}_{target}_population/

    Filename convention:
      rmhc_trial_k{k}_C{i:03d}_c{C:.6f}_bp{j:03d}_b{bp:.4f}_{trigger}_{target}.pdf

    bp sweep only runs when update_target is 'boldness' or 'all'.
    For 'desire' only, bp is fixed at 0.0 and skipped from the filename.

    :param update_trigger: 'punished' or 'self_censor'
    :param update_target:  'desire', 'boldness', or 'all'
    :param boldness_pct:   upper bound of the bp sweep (ignored for 'desire')
    """
    # C sweep only makes sense when desire is being updated.
    # bp sweep only makes sense when boldness is being updated.
    if update_target == 'boldness':
        C_values  = np.array([0.0])       # C has no effect, fix at 0
        bp_values = np.linspace(0, boldness_pct, 11)
    elif update_target == 'desire':
        C_values  = np.linspace(0, 0.01, 11)
        bp_values = np.array([0.0])       # bp has no effect, fix at 0
    else:  # 'all' — sweep both
        C_values  = np.linspace(0, 0.01, 11)
        bp_values = np.linspace(0, boldness_pct, 11)
    k_values = [1, 3]

    tasks = list(product(k_values, C_values, bp_values))
    ks  = [t[0] for t in tasks]
    Cs  = [t[1] for t in tasks]
    bps = [t[2] for t in tasks]

    trials_run = 5

    results = process_map(
        fast_c_worker, ks, Cs,
        repeat(N), repeat(R), repeat(delta), repeat(beta), repeat(pi),
        repeat(tau0), repeat(psi0), repeat(nu0), repeat(alpha), repeat(eps),
        repeat(seed), repeat(update_trigger), repeat(update_target),
        bps, repeat(trials_run),
        max_workers=threads, chunksize=1,
        desc=f"Running Fast C Pass [{update_trigger} → {update_target}]"
    )

    bp_colors   = {round(bp, 10): cm.batlow(x)
                   for bp, x in zip(bp_values,
                                    np.linspace(0.1, 0.9, len(bp_values)))}
    line_styles = {1: '-', 3: '--'}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300, facecolor='w')

    if update_target == 'boldness':
        # x-axis = bp, one line per k
        for k in k_values:
            series = [res[5] for res in results if res[0] == k]
            ax.plot(bp_values * 100, series,
                    color=cm.batlow(0.2 if k == 1 else 0.8),
                    linestyle=line_styles[k],
                    marker='o', markersize=3,
                    label=f'k={k}')
        ax.set_xlabel('Boldness Update % (bp)')
        ax.set_xlim([0, max(bp_values) * 100])
    else:
        # x-axis = C, lines coloured by bp
        for k in k_values:
            for bp in bp_values:
                series = [res[5] for res in results
                          if res[0] == k and np.isclose(res[4], bp)]
                lbl = f'k={k}, bp={bp*100:.0f}%'
                ax.plot(C_values, series,
                        color=bp_colors[round(bp, 10)],
                        linestyle=line_styles[k],
                        marker='o', markersize=3, label=lbl)
        ax.set_xlabel('Constant C (Change in Desire)')
        ax.set_xlim([0, max(C_values)])

    ax.set_ylabel('Final Total Cost (avg over trials)')
    ax.set_title(f'RMHC: Final Cost  '
                 f'[trigger={update_trigger}, target={update_target}]')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize='x-small', ncol=2, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig.savefig(osp.join('..', 'figs',
                         f'fast_c_vs_cost_{update_trigger}_{update_target}.pdf'))
    plt.close(fig)

    trials_dir = osp.join('..', 'figs', f'{update_trigger}_{update_target}_trials')
    pop_dir    = osp.join('..', 'figs', f'{update_trigger}_{update_target}_population')
    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(pop_dir,    exist_ok=True)

    result_lookup = {
        (res[0], round(res[1], 10), round(res[4], 10)): res
        for res in results
    }

    for k in k_values:
        for i, C_val in enumerate(C_values):
            for j, bp in enumerate(bp_values):
                key = (k, round(C_val, 10), round(bp, 10))
                res = result_lookup.get(key)
                if res is None:
                    continue

                (_, _, _, _, _, _, rep_params, rep_pol_costs, rep_pun_costs,
                 rep_avg_desires, rep_avg_betas) = res
                taus, psis, nus = rep_params

                fname = (f'rmhc_trial_k{k}_C{i:03d}_c{C_val:.6f}'
                         f'_bp{j:03d}_b{bp:.4f}'
                         f'_{update_trigger}_{update_target}.pdf')

                # Authority plot: costs + params per round
                plot_trial(taus, psis, nus, rep_pol_costs, rep_pun_costs,
                           alpha, pi, delta, beta,
                           title=True, k_mutate=k, k3_method='sphere',
                           C=C_val, savepath=osp.join(trials_dir, fname))

                # Population plot: avg desire (top) + avg boldness (bottom)
                fig_p, ax_p = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                                           dpi=300, layout='constrained')
                rounds = np.arange(len(rep_avg_desires))

                ax_p[0].plot(rounds, rep_avg_desires,
                             c=plt.cm.Purples(0.8),
                             label=r'Mean Desired Dissent $\bar{\delta}_r$')
                ax_p[0].axhline(delta, linestyle='--', linewidth=0.8,
                                color='grey', label=r'Initial mean $\delta$')
                ax_p[0].legend(loc='best', fontsize='small')
                ax_p[0].set(ylim=[0, 1], ylabel=r'Mean Desired Dissent',
                            title=(f'Population Dynamics  (k={k}, '
                                   f'C={C_val:.4f}, bp={bp*100:.0f}%, '
                                   f'{update_trigger}→{update_target})'))

                ax_p[1].plot(rounds, rep_avg_betas,
                             c=plt.cm.Oranges(0.7),
                             label=r'Mean Boldness $\bar{\beta}_r$')
                ax_p[1].axhline(beta, linestyle='--', linewidth=0.8,
                                color='grey', label=r'Initial mean $\beta$')
                ax_p[1].legend(loc='best', fontsize='small')
                ax_p[1].set(xlim=[0, len(rep_avg_desires)],
                            xlabel=r'Round $r$',
                            ylabel=r'Mean Boldness', ylim=[0, None])

                fig_p.savefig(osp.join(pop_dir, fname))
                plt.close(fig_p)


def sweep_worker(idx, db, N, R, pi, tau0s, psi0s, nu0s, alpha, eps, seeds, k_mutate, pair, k3_method):
    """
    Worker function handling the repeated RMHC trials for a single setting of
    (delta, beta).

    :param idx: a tuple (i, j) representing this parameter setting's index
    :param db: a tuple (delta, beta) of the float mean population desired
    dissent (> 0) and the float mean population boldness (> 0)
    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'proportional' punishment
    :param tau0s: a 1xT array of the authority's float initial tolerances
    :param psi0s: a 1xT array of the authority's float initial severities
    :param nu0s: a 1xT array of the authority's float initial surveillances
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seeds: a 1xT array of int seeds for random number generation

    :returns: the tuple (i, j) representing this parameter setting's index
    :returns: a 3xR array of the authority's mean parameters per round
    :returns: a 3xR array of the authority's parameter standard deviations per
    round
    :returns: a 1xR array of the authority's mean political costs per round
    :returns: a 1xR array of the authority's political cost standard deviations
    per round
    :returns: a 1xR array of the authority's mean punishment costs per round
    :returns: a 1xR array of the authority's punishment cost standard
    deviations per round
    """
    # Set up worker-specific results arrays.
    w_params = np.zeros((len(seeds), 3, R))
    w_pol_costs = np.zeros((len(seeds), R))
    w_pun_costs = np.zeros((len(seeds), R))

    # Run the specified number of trials for this parameter setting.
    delta, beta = db
    for t in range(len(seeds)):
        w_params[t], w_pol_costs[t], w_pun_costs[t], _, _, _, _ = \
            rmhc_trial(N, R, delta, beta, pi, tau0s[t], psi0s[t], nu0s[t],
                       alpha, eps, seeds[t], k_mutate, pair, k3_method)

    # Return the index + means/standard deviations across trials.
    return (idx, w_params.mean(axis=0), w_params.std(axis=0),
            w_pol_costs.mean(axis=0), w_pol_costs.std(axis=0),
            w_pun_costs.mean(axis=0), w_pun_costs.std(axis=0))


def rmhc_sweep(N, R, pi, alpha, eps, seed, granularity, trials, threads, k_mutate, pair, k3_method):
    """
    Varying the population's mean desired dissent and boldness as independent
    variables and randomly initializing the authority's parameters, measure the
    authority's final parameter values after a fixed number of rounds. In each
    random trial, record the authority's parameters over time, the population's
    desired dissents and boldness constants, and the random seed. This is
    sufficient to reconstruct the entire trajectory of actions and punishments.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation
    :param granularity: an int number of delta and beta values to sweep over
    :param trials: an int number of trials to run per parameter setting
    :param threads: an int number of threads to parallelize over
    """
    # Set up the independent variables.
    deltas = np.linspace(0.005, 0.495, granularity)
    betas = np.linspace(0.1, 10, granularity)

    # Set up random seeds and initial authority parameters for the trials.
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32, size=trials)
    tau0s = rng.random(size=trials)
    psi0s = rng.random(size=trials)
    nu0s = rng.random(size=trials)

    # Set up results containers: for each (delta, beta) pair, we store the mean
    # and standard deviation of the three parameters, political costs, and
    # punishment costs in each round across all trials.
    params = np.zeros((granularity, granularity, 2, 3, R))
    pol_costs = np.zeros((granularity, granularity, 2, R))
    pun_costs = np.zeros((granularity, granularity, 2, R))

    # Run the experiment with the specified number of parallel threads.
    idxs = list(product(range(granularity), range(granularity)))
    dbs = list(product(deltas, betas))
    p = process_map(sweep_worker, idxs, dbs, repeat(N), repeat(R), repeat(pi),
                    repeat(tau0s), repeat(psi0s), repeat(nu0s), repeat(alpha),
                    repeat(eps), repeat(seeds), repeat(k_mutate), repeat(pair), repeat(k3_method), max_workers=threads,
                    chunksize=1)
    for (i, j), w_params_mean, w_params_std, w_pol_costs_mean, \
            w_pol_costs_std, w_pun_costs_mean, w_pun_costs_std in p:
        params[i, j, 0] = w_params_mean
        params[i, j, 1] = w_params_std
        pol_costs[i, j, 0] = w_pol_costs_mean
        pol_costs[i, j, 1] = w_pol_costs_std
        pun_costs[i, j, 0] = w_pun_costs_mean
        pun_costs[i, j, 1] = w_pol_costs_std

    # Dump all results to file.
    resultsdir = osp.join('..', 'results', f'sweep_N{N}_R{R}_{pi}_S{seed}')
    dump_np(osp.join(resultsdir, 'deltas.npy'), deltas)
    dump_np(osp.join(resultsdir, 'betas.npy'), betas)
    dump_np(osp.join(resultsdir, 'params.npy'), params)
    dump_np(osp.join(resultsdir, 'pol_costs.npy'), pol_costs)
    dump_np(osp.join(resultsdir, 'pun_costs.npy'), pun_costs)


def plot_trial(taus, psis, nus, pol_costs, pun_costs, alpha, pi, delta, beta,
               title=False, k_mutate=None, pair=None, k3_method=None,
               C=None, savepath=None):
    """
    Plot the evolution of authority costs & parameters in a single RMHC trial.

    :param taus: a 1xR array of the authority's tolerance values
    :param psis: a 1xR array of the authority's severity values
    :param nus: a 1xR array of the authority's surveillance values
    :param pol_costs: a 1xR array of the authority's political costs
    :param pun_costs: a 1xR array of the authority's punishment costs
    :param alpha: the authority's float adamancy (> 0)
    :param pi: 'uniform' or 'proportional' punishment
    :param delta: the float mean population desired dissent (> 0)
    :param beta: the float mean population boldness (> 0)
    :param title: True iff the plot should have a title detailing parameters
    :param C: optional float dissent-change constant; shown in title when given
    :param savepath: optional full output path; if None, uses the default
    figs/rmhc_trial{suffix}.pdf path
    """
    fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True, dpi=300,
                           layout='constrained')
    R = len(taus)

    # Plot costs (negative utility) over time.
    ax[0].plot(np.arange(R), alpha * pol_costs, label='Political Cost',
               c=cm.vikO(0.3))
    ax[0].plot(np.arange(R), pun_costs, label='Punishment Cost',
               c=cm.vikO(0.7))
    ax[0].plot(np.arange(R), alpha * pol_costs + pun_costs, label='Total Cost',
               c=cm.vikO(0))
    ax[0].legend(loc='best', fontsize='small')
    ax[0].set(ylabel='Costs')
    if title:
        c_str = f', C={C:.6f}' if C is not None else ''
        ax[0].set_title(r"Hill Climbing Authority ($\pi$ " f"={pi}, "
                        r"$\alpha$ " f"= {alpha}{c_str}) vs. Population "
                        r"$\delta_i \sim$" f"Exp({delta}), " r" $\beta_i \sim$"
                        f"Exp({beta}" r"$^{-1}$")

    # Plot parameters over time.
    ax[1].plot(np.arange(R), taus, label=r'Tolerance $\tau_r$',
               c=plt.cm.Blues(0.9))
    ax[1].plot(np.arange(R), psis, label=r'Severity $\psi_r$',
               c=plt.cm.Reds(0.8))
    ax[1].plot(np.arange(R), nus, label=r'Surveillance $\nu_r$',
               c=plt.cm.Greens(0.7))
    ax[1].legend(loc='best', fontsize='small')
    ax[1].set(xlim=[0, R], xlabel=r'Round $r$', ylabel='Parameter Value')

    #To specific which k in the file name
    suffix = ""
    if k_mutate is not None:
        suffix += f"_k{k_mutate}"
    if pair is not None and k_mutate == 2:
        suffix += f"_{pair}"
    if k3_method is not None and k_mutate == 3:
        suffix += f"_{k3_method}"

    # Use the caller-supplied path when given (e.g. from plot_fast_c_vs_cost),
    # otherwise fall back to the default location.
    if savepath is not None:
        fig.savefig(savepath)
    else:
        fig.savefig(osp.join('..', 'figs', f'rmhc_trial{suffix}.pdf'))
    plt.close(fig)  # prevent 62 open figures accumulating in memory


def plot_sweep(N, R, pi, alpha, eps, seed):
    """
    Plots the results of an RMHC sweep experiment showing the authority's
    final average parameter values and costs per (mean desired dissent, mean
    boldness) pair.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param eps: the float update window radius for RMHC
    :param seed: an int seed for random number generation
    """
    # Load results means from file.
    resultsdir = osp.join('..', 'results', f'sweep_N{N}_R{R}_{pi}_S{seed}')
    deltas = load_np(osp.join(resultsdir, 'deltas.npy'))
    betas = load_np(osp.join(resultsdir, 'betas.npy'))
    params = load_np(osp.join(resultsdir, 'params.npy'))[:, :, 0]
    pol_costs = alpha * load_np(osp.join(resultsdir, 'pol_costs.npy'))[:, :, 0]
    pun_costs = load_np(osp.join(resultsdir, 'pun_costs.npy'))[:, :, 0]
    total_costs = pol_costs + pun_costs

    # Set up the figure.
    fig = plt.figure(figsize=(6.8, 8), dpi=300, facecolor='w',
                     layout='constrained')
    gs = fig.add_gridspec(3, 2)
    axes_bd = [fig.add_subplot(gs[i[::-1]])
               for i in product(range(2), range(3))]

    # Plot average final parameter values and final costs.
    data = [pol_costs[:, :, -1], pun_costs[:, :, -1], total_costs[:, :, -1],
            params[:, :, 0, -1], params[:, :, 1, -1], params[:, :, 2, -1]]
    lims = [(0, None), (0, None), (0, None), (0, 1), (0, None), (0, 1)]
    cmaps = [cm.devon_r, cm.bilbao_r, cm.batlowW_r, 'Blues', 'Reds', 'Greens']
    lbls = ['(A) Political Cost', '(B) Punishment Cost', '(C) Total Cost',
            r'(D) Tolerance $\tau_{final}$', r'(E) Severity $\psi_{final}$',
            r'(F) Surveillance $\nu_{final}$']
    for i, (axi, datum, (pmin, pmax), cmap, lbl) in \
            enumerate(zip(axes_bd, data, lims, cmaps, lbls)):
        im = axi.pcolormesh(deltas, betas, datum.T, vmin=pmin, vmax=pmax,
                            cmap=cmap, shading='auto')
        fig.colorbar(im, ax=axi)
        axi.set_title(lbl, weight='bold')
        if i in [2, 5]:
            axi.set_xlabel(r'Mean Desired Dissent $\delta$')
        else:
            axi.tick_params(labelbottom=False)
        if i <= 2:
            axi.set_ylabel(r'Mean Boldness $\beta$')
        else:
            axi.tick_params(labelleft=False)

    fig.savefig(osp.join('..', 'figs', f'sweep_N{N}_R{R}_{pi}_S{seed}.png'))


def plot_suppression_times(N, R, pi, alpha, seed, window=500, threshold=0.25,
                           xmax=None):
    """
    Plots the authority's times to suppression as a function of boldness.

    :param N: an int number of individuals in the population
    :param R: an int number of rounds to simulate
    :param pi: 'uniform' or 'proportional' punishment
    :param alpha: the authority's float adamancy (> 0)
    :param seed: an int seed for random number generation
    :param window: an int sliding window size for measuring suppression
    :param threshold: a fraction of political cost below which is suppression
    :param xmax: a maximum value for the x-axis, or None if inferred
    """
    # Load results means from file.
    resultsdir = osp.join('..', 'results', f'sweep_N{N}_R{R}_{pi}_S{seed}')
    deltas = load_np(osp.join(resultsdir, 'deltas.npy'))
    betas = load_np(osp.join(resultsdir, 'betas.npy'))
    pol_costs = alpha * load_np(osp.join(resultsdir, 'pol_costs.npy'))[:, :, 0]

    # Compute suppression times for total costs.
    rep_times = np.zeros(pol_costs.shape[0:2], dtype=int)
    for idx in tqdm(list(np.ndindex(rep_times.shape))):
        max_cost = texponential(np.random.default_rng(seed), bound=1,
                                scale=deltas[idx[0]], size=N).sum()
        step = window
        while step <= len(pol_costs[idx]):
            if pol_costs[idx][step-window:step].mean() < threshold * max_cost:
                break
            else:
                step += 1
        rep_times[idx] = step

    # Plot the figure.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300, facecolor='w',
                           layout='tight')
    colors = [cm.lipari(i) for i in np.linspace(0, 1, len(deltas))]
    for i, delta in enumerate(deltas):
        if i in [0, 1, 2, len(deltas) - 3, len(deltas) - 2, len(deltas) - 1]:
            ax.plot(betas, rep_times[i, :], color=colors[i],
                    label=r'$\delta$ = ' + f'{delta:.3f}')
        elif i == 3:
            ax.plot(betas, rep_times[i, :], color=colors[i],
                    label=r'$\delta$ = ...')
        else:
            ax.plot(betas, rep_times[i, :], color=colors[i])

    if xmax is None:
        xmax = betas.max()
    ax.set(xlabel=r'Mean Boldness $\beta$',
           ylabel='Time to Suppression (Rounds)',
           xlim=[0, xmax], ylim=[0, None])
    ax.legend()

    fig.savefig(osp.join('..', 'figs',
                         f'suppression_times_N{N}_R{R}_{pi}_S{seed}.pdf'))


def policy_trial(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, seed,
                 k_mutate=1, k3_method='sphere', C=0.0, boldness_pct=0.01,
                 snapshot_every=500):
    """
    RMHC trial where the population reacts to the authority's policy directly.

    Desire update — each of the three authority parameters is compared to
    each individual's personal threshold every round.  For each parameter:
      tau < tau_thresh_i   → desire + C   (policy too tight)
      tau >= tau_thresh_i  → desire - C   (policy lenient enough)
      nu  > nu_thresh_i    → desire + C   (too much surveillance)
      nu  <= nu_thresh_i   → desire - C   (surveillance acceptable)
      psi > psi_thresh_i   → desire + C   (severity too high)
      psi <= psi_thresh_i  → desire - C   (severity acceptable)
    Net change is the sum of the three contributions (−3C to +3C per round).

    Boldness update — based on punishment:
      punished             → betas * (1 - boldness_pct)
      not punished         → betas * (1 + boldness_pct)

    Personal thresholds are drawn once at initialisation:
      tau_thresh ~ Uniform(0, 1)
      nu_thresh  ~ Uniform(0, 1)
      psi_thresh ~ Exponential(mean=1.0)

    Snapshots of desire distribution are taken every `snapshot_every` rounds,
    recording the fraction of the population with desire ≈ 0, in (0,1), ≈ 1.

    :returns: params, pol_costs, pun_costs, deltas, betas,
              avg_desires (1xR), avg_betas (1xR),
              snapshots dict with keys:
                'rounds'   — 1D array of round indices
                'at_zero'  — fraction with desire < 1e-6
                'middle'   — fraction with 1e-6 <= desire <= 1-1e-6
                'at_one'   — fraction with desire > 1-1e-6
    """
    rng = np.random.default_rng(seed)

    # Initialise population.
    deltas = texponential(rng, bound=1, scale=delta, size=N)
    betas  = rng.exponential(scale=beta, size=N)

    # Personal policy thresholds — drawn once, fixed for the whole trial.
    tau_thresh = rng.uniform(0, 1,    size=N)
    nu_thresh  = rng.uniform(0, 1,    size=N)
    psi_thresh = rng.exponential(1.0, size=N)

    # Authority parameter bounds.
    bounds = np.array([[0, 1], [1e-9, np.inf], [0, 1]])

    params      = np.zeros((3, R))
    pol_costs   = np.zeros(R)
    pun_costs   = np.zeros(R)
    avg_desires = np.zeros(R)
    avg_betas   = np.zeros(R)

    # Snapshot containers — record every snapshot_every rounds.
    snap_rounds  = []
    snap_at_zero = []
    snap_middle  = []
    snap_at_one  = []

    for r in range(R):
        # ---- Authority RMHC step (identical logic to rmhc_trial) ----
        if r == 0:
            params[:, r] = [tau0, psi0, nu0]
        else:
            candidate_params = np.copy(params[:, r-1])
            if k_mutate == 1:
                idx  = rng.integers(3)
                low  = max(bounds[idx, 0], candidate_params[idx] - eps)
                high = min(bounds[idx, 1], candidate_params[idx] + eps)
                candidate_params[idx] = rng.uniform(low, high)
            else:  # k_mutate == 3, sphere
                angle  = rng.normal(size=3)
                angle /= np.linalg.norm(angle)
                candidate_params += angle * eps
            candidate_params[0] = np.clip(candidate_params[0], bounds[0,0], bounds[0,1])
            candidate_params[1] = max(candidate_params[1], bounds[1,0])
            candidate_params[2] = np.clip(candidate_params[2], bounds[2,0], bounds[2,1])
            params[:, r] = candidate_params

        # ---- Population acts ----
        tau, psi, nu = params[:, r]
        with np.errstate(divide='ignore', invalid='ignore'):
            acts = opt_actions(deltas, betas, nu, pi, tau, psi)
        acts = np.clip(np.nan_to_num(acts, nan=0.0, posinf=1.0, neginf=0.0),
                       0.0, 1.0)

        # ---- Authority observes and punishes ----
        cond = (acts > tau) & (rng.random(N) < (nu + (1 - nu) * acts))
        if pi == 'uniform':
            punish = cond * psi
        elif pi == 'proportional':
            punish = cond * psi * (acts - tau)
        else:
            assert False, f'ERROR: Invalid punishment function "{pi}"'

        pol_costs[r] = acts.sum()
        pun_costs[r] = punish.sum()

        # ---- Authority rolls back if costs got worse ----
        if r > 0 and alpha * pol_costs[r] + pun_costs[r] > \
                     alpha * pol_costs[r-1] + pun_costs[r-1]:
            params[:, r] = params[:, r-1]
            # Re-read tau/psi/nu after possible rollback so population
            # update uses the params the authority actually kept.
            tau, psi, nu = params[:, r]

        # ---- Desire update: each param contributes ±C independently ----
        if C > 0:
            delta_desire = np.zeros(N)
            delta_desire += np.where(tau  <  tau_thresh,  C, -C)
            delta_desire += np.where(nu   >  nu_thresh,   C, -C)
            delta_desire += np.where(psi  >  psi_thresh,  C, -C)
            deltas = np.clip(deltas + delta_desire, 0.0, 1.0)

        # ---- Boldness update: based on punishment ----
        if boldness_pct > 0:
            betas = np.where(cond,
                             np.maximum(1e-9, betas * (1 - boldness_pct)),
                             np.maximum(1e-9, betas * (1 + boldness_pct)))

        # ---- Record per-round averages ----
        avg_desires[r] = deltas.mean()
        avg_betas[r]   = betas.mean()

        # ---- Snapshot every snapshot_every rounds ----
        if (r + 1) % snapshot_every == 0 or r == R - 1:
            snap_rounds.append(r + 1)
            snap_at_zero.append((deltas  < 1e-6).mean())
            snap_at_one.append( (deltas  > 1 - 1e-6).mean())
            snap_middle.append( ((deltas >= 1e-6) & (deltas <= 1 - 1e-6)).mean())

    snapshots = {
        'rounds':  np.array(snap_rounds),
        'at_zero': np.array(snap_at_zero),
        'middle':  np.array(snap_middle),
        'at_one':  np.array(snap_at_one),
    }
    return (params, pol_costs, pun_costs, deltas, betas,
            avg_desires, avg_betas, snapshots)


def policy_worker(k, C, boldness_pct, N, R, delta, beta, pi, tau0, psi0, nu0,
                  alpha, eps, seed, trials=5):
    """
    Worker for plot_policy_c.  Runs `trials` policy_trial calls, averages the
    final total cost, and keeps the full trajectory from the first trial.

    :returns: (k, C, boldness_pct, mean_final_cost,
               rep_params, rep_pol_costs, rep_pun_costs,
               rep_avg_desires, rep_avg_betas, rep_snapshots)
    """
    final_costs = []
    rep = {}

    for t in range(trials):
        trial_seed = seed + t if seed is not None else None
        (params, pol_costs, pun_costs, _, _,
         avg_desires, avg_betas, snapshots) = policy_trial(
            N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps, trial_seed,
            k_mutate=k, k3_method='sphere', C=C, boldness_pct=boldness_pct
        )
        final_costs.append(alpha * pol_costs[-1] + pun_costs[-1])
        if t == 0:
            rep = dict(params=params, pol_costs=pol_costs,
                       pun_costs=pun_costs, avg_desires=avg_desires,
                       avg_betas=avg_betas, snapshots=snapshots)

    return (k, C, boldness_pct, np.mean(final_costs),
            rep['params'], rep['pol_costs'], rep['pun_costs'],
            rep['avg_desires'], rep['avg_betas'], rep['snapshots'])


def plot_policy_c(N, R, delta, beta, pi, tau0, psi0, nu0, alpha, eps,
                  seed, threads, max_C=0.01, max_bp=0.10):
    """
    Sweeps C (5 values, 0→max_C) and boldness_pct (5 values, 0→max_bp)
    for the policy-responsive population model.

    Summary plot  →  figs/policy_summary.pdf
    Authority     →  figs/policy_trials/
    Population    →  figs/policy_trials_population/
                     (avg desire + avg boldness + desire snapshot chart,
                      all in one 3-panel PDF per run)

    Filename convention:
      policy_trial_k{k}_C{i:02d}_c{C:.6f}_bp{j:02d}_b{bp:.4f}.pdf

    :param max_C:  upper bound of the C sweep
    :param max_bp: upper bound of the boldness_pct sweep
    """
    C_values  = np.linspace(0, max_C,  5)
    bp_values = np.linspace(0, max_bp, 5)
    k_values  = [1, 3]

    tasks = list(product(k_values, C_values, bp_values))
    ks  = [t[0] for t in tasks]
    Cs  = [t[1] for t in tasks]
    bps = [t[2] for t in tasks]

    trials_run = 5

    results = process_map(
        policy_worker, ks, Cs, bps,
        repeat(N), repeat(R), repeat(delta), repeat(beta), repeat(pi),
        repeat(tau0), repeat(psi0), repeat(nu0), repeat(alpha), repeat(eps),
        repeat(seed), repeat(trials_run),
        max_workers=threads, chunksize=1,
        desc="Running Policy C Sweep"
    )

    bp_colors   = {round(bp, 10): cm.batlow(x)
                   for bp, x in zip(bp_values,
                                    np.linspace(0.1, 0.9, len(bp_values)))}
    line_styles = {1: '-', 3: '--'}

    fig_s, ax_s = plt.subplots(figsize=(8, 5), dpi=300, facecolor='w')
    for k in k_values:
        for bp in bp_values:
            series = [res[3] for res in results
                      if res[0] == k and np.isclose(res[2], bp)]
            ax_s.plot(C_values, series,
                      color=bp_colors[round(bp, 10)],
                      linestyle=line_styles[k],
                      marker='o', markersize=3,
                      label=f'k={k}, bp={bp*100:.0f}%')
    ax_s.set_xlabel('Constant C (Policy Response Strength)')
    ax_s.set_ylabel('Final Total Cost (avg over trials)')
    ax_s.set_title('Policy-Responsive Model: C vs Final Cost')
    ax_s.set_xlim([0, max(C_values)])
    ax_s.set_ylim(bottom=0)
    ax_s.legend(fontsize='x-small', ncol=2, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig_s.savefig(osp.join('..', 'figs', 'policy_summary.pdf'))
    plt.close(fig_s)

    trials_dir = osp.join('..', 'figs', 'policy_trials')
    pop_dir    = osp.join('..', 'figs', 'policy_trials_population')
    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(pop_dir,    exist_ok=True)

    result_lookup = {
        (res[0], round(res[1], 10), round(res[2], 10)): res
        for res in results
    }

    for k in k_values:
        for i, C_val in enumerate(C_values):
            for j, bp in enumerate(bp_values):
                key = (k, round(C_val, 10), round(bp, 10))
                res = result_lookup.get(key)
                if res is None:
                    continue

                (_, _, _, _, rep_params, rep_pol_costs, rep_pun_costs,
                 rep_avg_desires, rep_avg_betas, rep_snaps) = res
                taus, psis, nus = rep_params

                fname = (f'policy_trial_k{k}'
                         f'_C{i:02d}_c{C_val:.6f}'
                         f'_bp{j:02d}_b{bp:.4f}.pdf')

                # -- Authority plot: costs + params (reuse plot_trial) --
                plot_trial(taus, psis, nus, rep_pol_costs, rep_pun_costs,
                           alpha, pi, delta, beta,
                           title=True, k_mutate=k, k3_method='sphere',
                           C=C_val, savepath=osp.join(trials_dir, fname))

                # -- Population plot: 3 panels in one figure --
                fig_p, ax_p = plt.subplots(3, 1, figsize=(6, 7), sharex=False,
                                           dpi=300, layout='constrained')
                rounds = np.arange(len(rep_avg_desires))

                # Panel 1: avg desire per round
                ax_p[0].plot(rounds, rep_avg_desires,
                             c=plt.cm.Purples(0.8),
                             label=r'Mean Desired Dissent $\bar{\delta}_r$')
                ax_p[0].axhline(delta, linestyle='--', linewidth=0.8,
                                color='grey', label=r'Initial mean $\delta$')
                ax_p[0].legend(loc='best', fontsize='small')
                ax_p[0].set(xlim=[0, len(rounds)], ylim=[0, 1],
                            ylabel=r'Mean Desired Dissent',
                            title=(f'Policy-Responsive Dynamics  '
                                   f'(k={k}, C={C_val:.4f}, '
                                   f'bp={bp*100:.0f}%)'))

                # Panel 2: avg boldness per round
                ax_p[1].plot(rounds, rep_avg_betas,
                             c=plt.cm.Oranges(0.7),
                             label=r'Mean Boldness $\bar{\beta}_r$')
                ax_p[1].axhline(beta, linestyle='--', linewidth=0.8,
                                color='grey', label=r'Initial mean $\beta$')
                ax_p[1].legend(loc='best', fontsize='small')
                ax_p[1].set(xlim=[0, len(rounds)],
                            ylabel=r'Mean Boldness', ylim=[0, None])

                # Panel 3: snapshot distribution every 500 rounds
                snap_r  = rep_snaps['rounds']
                at_zero = rep_snaps['at_zero']
                middle  = rep_snaps['middle']
                at_one  = rep_snaps['at_one']

                ax_p[2].stackplot(snap_r,
                                  at_zero, middle, at_one,
                                  labels=['Desire ≈ 0', '0 < Desire < 1',
                                          'Desire ≈ 1'],
                                  colors=[plt.cm.Blues(0.4),
                                          plt.cm.Greys(0.4),
                                          plt.cm.Reds(0.5)],
                                  alpha=0.85)
                ax_p[2].legend(loc='upper right', fontsize='small')
                ax_p[2].set(xlim=[snap_r[0], snap_r[-1]], ylim=[0, 1],
                            xlabel=r'Round $r$',
                            ylabel='Fraction of Population')

                fig_p.savefig(osp.join(pop_dir, fname))
                plt.close(fig_p)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', action='store_true',
                        help=('If present, run the sweep experiment (ignores '
                              '--delta, --beta, --tau, --psi, and --nu); '
                              'otherwise; runs one independent trial (ignores '
                              '--granularity, --trials, --threads)'))
    parser.add_argument('-N', '--num_ind', type=int, default=100000,
                        help='Number of individuals in the population')
    parser.add_argument('-R', '--rounds', type=int, default=10000,
                        help='Number of rounds to simulate in a single trial')
    parser.add_argument('-D', '--delta', type=float, default=0.25,
                        help='Mean population desired dissent > 0')
    parser.add_argument('-B', '--beta', type=float, default=0.5,
                        help='Mean population boldness > 0')
    parser.add_argument('-P', '--pi', choices=['uniform', 'proportional'],
                        default='uniform', help='Punishment function')
    parser.add_argument('-T', '--tau', type=float, default=0.25,
                        help='Authority\'s initial tolerance in [0,1]')
    parser.add_argument('-S', '--psi', type=float, default=0.1,
                        help='Authority\'s initial severity > 0')
    parser.add_argument('-V', '--nu', type=float, default=0.1,
                        help='Authority\'s initial surveillance in [0,1]')
    parser.add_argument('-A', '--alpha', type=float, default=1.0,
                        help='Authority\'s adamancy > 0')
    parser.add_argument('-E', '--epsilon', type=float, default=0.05,
                        help='Window radius for authority parameter updates')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random number generation')
    parser.add_argument('--granularity', type=int, default=50,
                        help='Number of parameter values to sweep over')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials to run per parameter setting')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to parallelize over')
                        
    #k=1,k=2,k=3
    parser.add_argument('--k-mutate', type=int, choices=[1,2,3], default=1,
                    help='How many parameters to mutate per step (1, 2, or 3).')

    parser.add_argument('--pair', type=str, choices=['random','tp','tn','pn'], default='random',
                    help='When k=2: choose which two to mutate. tp=(tau,psi), tn=(tau,nu), pn=(psi,nu).')

    parser.add_argument('--run-all-k', action='store_true',
                    help='If set, run trials for k=1,2,3 sequentially.')
    parser.add_argument('--k3-method', type=str, choices=['box', 'sphere'], default='sphere',
                    help='When k=3: mutation method (box or sphere).')
    

    parser.add_argument('--fast-c', action='store_true',
                        help='Run the fast 1-pass C vs Cost plot')
    parser.add_argument('--policy-c', action='store_true',
                        help=('Run the policy-responsive population sweep: '
                              'desire reacts to authority params via personal '
                              'thresholds; boldness reacts to punishment'))
    parser.add_argument('--policy-max-c', type=float, default=0.01,
                        help='Upper bound of C sweep for --policy-c (default 0.01)')
    parser.add_argument('--policy-max-bp', type=float, default=0.10,
                        help=('Upper bound of boldness_pct sweep for '
                              '--policy-c (default 0.10)'))
    parser.add_argument('--update-trigger',
                        choices=['punished', 'self_censor', 'mixed'],
                        default='punished',
                        help=('What condition drives the population update: '
                              '"punished" uses those caught by the authority; '
                              '"self_censor" uses those who held back; '
                              '"mixed" uses self-censorship for desire and '
                              'punishment for boldness'))
    parser.add_argument('--update-target',
                        choices=['desire', 'boldness', 'all'],
                        default='desire',
                        help=('What gets updated each round: '
                              '"desire" shifts desired dissent only; '
                              '"boldness" shifts boldness only; '
                              '"all" shifts both'))
    parser.add_argument('--boldness-pct', type=float, default=0.10,
                        help=('Upper bound of the boldness_pct sweep when '
                              '--update-target is boldness or all (default 0.10)'))

    
    args = parser.parse_args()

    # Run a single trial or sweep experiment.
    rng = np.random.default_rng(args.seed)
    if args.policy_c:
        plot_policy_c(N=args.num_ind, R=args.rounds, delta=args.delta,
                      beta=args.beta, pi=args.pi, tau0=args.tau,
                      psi0=args.psi, nu0=args.nu, alpha=args.alpha,
                      eps=args.epsilon, seed=args.seed,
                      threads=args.threads,
                      max_C=args.policy_max_c,
                      max_bp=args.policy_max_bp)
    elif args.fast_c:
        plot_fast_c_vs_cost(N=args.num_ind, R=args.rounds, delta=args.delta,
                            beta=args.beta, pi=args.pi, tau0=args.tau,
                            psi0=args.psi, nu0=args.nu, alpha=args.alpha,
                            eps=args.epsilon, seed=args.seed,
                            threads=args.threads,
                            update_trigger=args.update_trigger,
                            update_target=args.update_target,
                            boldness_pct=args.boldness_pct)
    elif args.sweep:
        rmhc_sweep(N=args.num_ind, R=args.rounds, pi=args.pi, alpha=args.alpha,
                   eps=args.epsilon, seed=args.seed,
                   granularity=args.granularity, trials=args.trials,
                   threads=args.threads, k_mutate=args.k_mutate,
           pair=args.pair, k3_method=args.k3_method)
        plot_sweep(N=args.num_ind, R=args.rounds, pi=args.pi, alpha=args.alpha,
                   eps=args.epsilon, seed=args.seed)
        plot_suppression_times(N=args.num_ind, R=args.rounds, pi=args.pi,
                               alpha=args.alpha, seed=args.seed)
    else:
        (taus, psis, nus), pol_costs, pun_costs, deltas, betas, avg_desires, avg_betas = \
            rmhc_trial(N=args.num_ind, R=args.rounds, delta=args.delta,
                       beta=args.beta, pi=args.pi, tau0=args.tau,
                       psi0=args.psi, nu0=args.nu, alpha=args.alpha,
                       eps=args.epsilon, seed=args.seed,
                       k_mutate=args.k_mutate, pair=args.pair,
                       k3_method=args.k3_method,
                       update_trigger=args.update_trigger,
                       update_target=args.update_target,
                       boldness_pct=args.boldness_pct)
        plot_trial(taus, psis, nus, pol_costs, pun_costs, args.alpha, args.pi,
                   args.delta, args.beta, title=False,
                   k_mutate=args.k_mutate, pair=args.pair, k3_method=args.k3_method)