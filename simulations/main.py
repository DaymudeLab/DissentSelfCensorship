# Project:  CensorshipDissent
# Filename: main.py
# Authors:  Anish Nahar (anahar1@asu.edu)

"""
main: Runs the simulation for a network of individuals and an authority 
with a set of rules that defines one's actions.
"""

from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

class Individual:
    def __init__(self, delta, beta):
        self.delta = delta
        self.beta = beta
        self.actions = []

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


def socialization(G, neighbors, neighbors_num, r, d_r):
    """
    Each individual updates their desired dissent for the next round to a weighted average of
    their current desired dissent and their neighbors' actions. In addition to the above terminology,
    let a_i,r be the action taken by individual I_i in round r. As in Rule 1, we can generalize with a
    weight parameter w > 0 capturing the strength of self-preference
    d_i,r+1 = (w * d_r + sum_of_neighors_a_ir) / (w + N(I_i))

    :param G:
    :param neighbors:
    :param neighbors_num:
    :param r:
    :param d_r:
    :returns: 
    """
    w = neighbors_num

    sum_neighbors_last_action = 0
    for n in neighbors:
        sum_neighbors_last_action += G.nodes[n]['action'][r-1]

    new_desire = (w * d_r + sum_neighbors_last_action) / (w + neighbors_num)

    return new_desire


def sharing_around_tables(G, neighbors, neighbors_num, r, d_r):
    """
    Each individual updates their desired dissent for the next round to a weighted average of
    their current desired dissent and those of their neighbors (N).
    d_i,r+1 = (N * d_r + sum_of_neighors_d_r) / (1 + N)
    
    :param G:
    :param neighbors:
    :param neighbors_num:
    :param r:
    :param d_r:
    :returns: 
    """
    w = neighbors_num

    sum_neighbors_desire = 0
    for n in neighbors:
        sum_neighbors_desire += G.nodes[n]['desire']

    new_desire = (w * d_r + sum_neighbors_desire) / (w + neighbors_num)

    return new_desire

def when_in_rome(G, neighbors, neighbors_num, r, a_ir):
    """
    Each individual I_i first computes their optimal action a*_i,r for round r as a function of their
    (unchanging) desired dissent and boldness. But instead of acting according to a*_i,r, they act
    according to a weighted average of a*_i,r and their neighbors' actions in the previous round. As in
    Rules 2-3, we can generalize with a weight parameter w > 0 capturing the strength of self-preference
    a_i,r+1 = (w * a*_i,r + sum_of_neighors_a_i,r) / (w + N(I_i))

    :param G:
    :param neighbors:
    :param neighbors_num:
    :param r:
    :param a_ir:
    :returns: 
    """
    w = neighbors_num

    sum_neighbors_last_action = 0
    for n in neighbors:
        sum_neighbors_last_action += G.nodes[n]['action'][r-1]

    new_action = (w * a_ir + sum_neighbors_last_action) / (w + neighbors_num)

    return new_action


def create_action_hist_plot(action_hist, rule):
    """
    Creates the action vs round plot for "r" rounds

    :param action_hist:
    :param rule:
    """

    rounds = list(range(len(action_hist[0])))

    plt.figure(figsize=(10, 6))

    n = len(action_hist)
    colormap = cm.batlow

    for key, values in action_hist.items():
        plt.plot(rounds, values, label=f'Individual {(key+1)}', marker='.', color=colormap(key/n))

    plt.xlabel('Round')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.axhline(y=0.25, color='gray', linestyle='--', label='t') # Add a dotted line at y=0.25 and label it as "t"
    plt.ylabel('Action')
    plt.title('Action vs Round')

    plt.savefig(osp.join('.', f'figs/simulation/{rule}', f'action_vs_round.png'), dpi=300, bbox_inches='tight')


def create_desire_hist_plot(desire_dict, rule):
    """
    Creates the desire vs round plot for "r" rounds

    :param desire_dict:
    :param rule:
    """

    rounds = list(range(len(desire_dict[0])))

    plt.figure(figsize=(10, 6))

    n = len(desire_dict)
    colormap = cm.batlow

    for key, values in desire_dict.items():
        plt.plot(rounds, values, label=f'Individual {(key+1)}', marker='.', color=colormap(key/n))

    plt.xlabel('Round')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.axhline(y=0.25, color='gray', linestyle='--', label='t') # Add a dotted line at y=0.25 and label it as "t"
    plt.ylabel('Desire')
    plt.title('Desire vs Round')

    plt.savefig(osp.join('.', f'figs/simulation/{rule}', f'desire_vs_round.png'), dpi=300, bbox_inches='tight')


def color_mapping(action, desire):
    """
    Determines the color of the node and the text inside the node
    :param action: 
    :param desire: 
    """

    if action <= 0.25:
        return 'lightblue', 'black'
    elif action < desire:
        normalized_value = (action - 0.25) / (desire - 0.25)
        color = plt.cm.Reds(normalized_value)
        
        # check if the color is dark enough to use white text
        text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
        return color, text_color
    else:
        return 'darkred', 'white'

def experiment(individuals, num_individuals, nu, pi, t, s,  num_rounds, change_rule):
    """
    Runs the experiment for a network of individuals and an authority
    :param individuals: list of Individual objects
    :param num_individuals: number of individuals
    :param nu: the authority's float surveillance (in [0,1])
    :param pi: 'constant' or 'linear' punishment
    :param t: the authority's float tolerance (in [0,1])
    :param s: the authority's float punishment severity (> 0)
    :param num_rounds: number of rounds
    :param change_rule: the rule to be used to change the desires of the individuals
    """

    action_hist = {}
    desire_dict = {}
    for i in range(num_individuals):
        action_hist[i] = []
        desire_dict[i] = []

    G = nx.powerlaw_cluster_graph(individuals_num, m=2, p=0.3)
    
    # Create the power law graph
    pos = nx.spring_layout(G) 
    
    if change_rule == 'socialization':
        for round in range(num_rounds):

            for i, individual in enumerate(individuals):
                if round != 0:
                    individual.delta = socialization(G=G, neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round, d_r=individual.delta)
                    
                action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=np.random.normal(loc=t, scale=0.05) , s=np.random.normal(loc=s, scale=0.05) )
                individual.actions.append(action)
                               
            # Add the Individual objects as nodes with their desires as node attributes
            for i, individual in enumerate(individuals):
                G.nodes[i]['desire'] = individual.delta
                G.nodes[i]['action'] = individual.actions
                desire_dict[i].append(individual.delta)
                #if round == num_rounds - 1:
                action_hist[i] = individual.actions
            
            plt.figure(figsize=(15, 12))

            node_colors, text_colors = zip(*[color_mapping(G.nodes[i]['action'][round], G.nodes[i]['desire']) for i in G.nodes])

            nx.draw(G, pos, node_size=100, with_labels=False, node_color=node_colors, edgecolors='black')

            # Add action values as labels inside each node
            #for node, text_color in zip(G.nodes, text_colors):
            #    nx.draw_networkx_labels(G, pos, labels={node: f'{G.nodes[node]["action"][round]:.4f}'}, font_size=10, font_color=text_color)

            # cbar legend
            cmap_colors = [(0.0, 'lightblue'), (0.25, (1.0, 0.9607843137254902, 0.9411764705882353)), (1.0, 'darkred')]
            cmap = LinearSegmentedColormap.from_list('custom_heatmap', cmap_colors)
            sm = ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, pad=0.02)
            cbar.set_label('')
            cbar.set_ticks([0, 0.25, 1])
            cbar.set_ticklabels(['Compliant', 'Completely Censoring', 'Defiant'])

            plt.title(f'Socialization Rule. Round {round}', fontsize=30)

            plt.savefig(osp.join('.', 'figs/simulation/socialization', f'round_{(round+1)}.png'), dpi=150, bbox_inches='tight')

            plt.close()

    elif change_rule == 'sharing_around_tables':
        
        for round in range(num_rounds):
            
            for i, individual in enumerate(individuals):
                if round != 0:
                    individual.delta = sharing_around_tables(G=G, neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round, d_r=individual.delta)
                    
                action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=np.random.normal(loc=t, scale=0.05) , s=np.random.normal(loc=s, scale=0.05) )
                individual.actions.append(action)
                               
            # Add the Individual objects as nodes with their desires as node attributes
            for i, individual in enumerate(individuals):
                G.nodes[i]['desire'] = individual.delta
                G.nodes[i]['action'] = individual.actions
                desire_dict[i].append(individual.delta)
                action_hist[i] = individual.actions
            
            plt.figure(figsize=(15, 12))

            node_colors, text_colors = zip(*[color_mapping(G.nodes[i]['action'][round], G.nodes[i]['desire']) for i in G.nodes])

            nx.draw(G, pos, node_size=100, with_labels=False, node_color=node_colors, edgecolors='black')

            # Add action values as labels inside each node
            #for node, text_color in zip(G.nodes, text_colors):
            #    nx.draw_networkx_labels(G, pos, labels={node: f'{G.nodes[node]["action"][round]:.4f}'}, font_size=10, font_color=text_color)

            # cbar legend
            cmap_colors = [(0.0, 'lightblue'), (0.25, (1.0, 0.9607843137254902, 0.9411764705882353)), (1.0, 'darkred')]
            cmap = LinearSegmentedColormap.from_list('custom_heatmap', cmap_colors)
            sm = ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, pad=0.02)
            cbar.set_label('')
            cbar.set_ticks([0, 0.25, 1])
            cbar.set_ticklabels(['Compliant', 'Completely Censoring', 'Defiant'])

            plt.title(f'Sharing Around Tables Rule. Round {round}', fontsize=30)
            plt.savefig(osp.join('.', 'figs/simulation/sharing_around_tables', f'round_{(round+1)}.png'), dpi=150, bbox_inches='tight')

            plt.close()

    elif change_rule == 'when_in_rome':
        
        for round in range(num_rounds):
            
            for i, individual in enumerate(individuals):                    
                action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=np.random.normal(loc=t, scale=0.05) , s=np.random.normal(loc=s, scale=0.05) )
                if round != 0:
                    action = when_in_rome(G=G, neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round, a_ir=action)
                individual.actions.append(action)
            

            # Add the Individual objects as nodes with their desires as node attributes
            for i, individual in enumerate(individuals):
                G.nodes[i]['desire'] = individual.delta
                G.nodes[i]['action'] = individual.actions
                desire_dict[i].append(individual.delta)
                action_hist[i] = individual.actions

            
            plt.figure(figsize=(15, 12))

            node_colors, text_colors = zip(*[color_mapping(G.nodes[i]['action'][round], G.nodes[i]['desire']) for i in G.nodes])

            nx.draw(G, pos, node_size=100, with_labels=False, node_color=node_colors, edgecolors='black')

            # Add action values as labels inside each node
            #for node, text_color in zip(G.nodes, text_colors):
            #    nx.draw_networkx_labels(G, pos, labels={node: f'{G.nodes[node]["action"][round]:.4f}'}, font_size=10, font_color=text_color)

            # cbar legend
            cmap_colors = [(0.0, 'lightblue'), (0.25, (1.0, 0.9607843137254902, 0.9411764705882353)), (1.0, 'darkred')]
            cmap = LinearSegmentedColormap.from_list('custom_heatmap', cmap_colors)
            sm = ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, pad=0.02)
            cbar.set_label('')
            cbar.set_ticks([0, 0.25, 1])
            cbar.set_ticklabels(['Compliant', 'Completely Censoring', 'Defiant'])

            plt.title(f'When in Rome Rule. Round {round}', fontsize=30)
            
            plt.savefig(osp.join('.', 'figs/simulation/when_in_rome', f'round_{(round+1)}.png'), dpi=150, bbox_inches='tight')

            plt.close()

    else:
        print('ERROR: Unrecognized change rule \'' + change_rule + '\'')
    
    create_action_hist_plot(action_hist, change_rule)
    create_desire_hist_plot(desire_dict, change_rule)
    


if __name__ == "__main__":
    # Setup.
    individuals_num = 1000
    
    # Set of parameters for individuals
    desires = np.linspace(0, 1, individuals_num)
    
    # Create Individual objects
    individuals = [Individual(delta, beta=np.random.uniform(1, 2)) for delta in desires]
    
    # Run experiment
    rule = 'sharing_around_tables'
    rule = 'when_in_rome'
    rule = 'socialization'
    experiment(individuals, individuals_num, nu=0.5, pi='linear', t=0.25, s=1.75, num_rounds=100, change_rule = rule)
    
    exit(0)