# Project:  CensorshipDissent
# Filename: main.py
# Authors:  Anish Nahar (anahar1@asu.edu)

"""
main: Runs the simulation for a network of individuals and an authority 
with a set of rules that defines one's actions.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os.path as osp
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

class Individual:
    def __init__(self, delta, beta):
        self.delta = delta
        self.first_action = 0
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


def irr_dissent(G, first_action, neighbors, neighbors_num, r):
    """
    Computes an individual's irrational action dissent as a function of their
    first action, their neighbor's new action, and their neighbors

    :param first_action:
    :param neighbors:
    """

    sum_neighbors_last_action = 0
    for n in neighbors:
        sum_neighbors_last_action += G.nodes[n]['action'][r-1]

    a_ir = (neighbors_num * first_action + sum_neighbors_last_action) / (2 * neighbors_num)

    return a_ir
    

def experiment(individuals, num_individuals, nu, pi, t, s,  num_rounds):
    
    G = nx.powerlaw_cluster_graph(individuals_num, m=2, p=0.3)
    
    # Create the power law graph
    pos = nx.spring_layout(G) 
    
    for round in range(num_rounds):

        if round == 0:
            for i, individual in enumerate(individuals):
                individual.first_action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=t, s=s)
                individual.actions.append(individual.first_action)
        else:
            for i, individual in enumerate(individuals):
                irr_action = irr_dissent(G=G, first_action=G.nodes[i]['first_action'], neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round)
                individual.actions.append(irr_action)
            

        # Add the Individual objects as nodes with their desires as node attributes
        for i, individual in enumerate(individuals):
            G.nodes[i]['desire'] = individual.delta
            G.nodes[i]['first_action'] = individual.first_action
            G.nodes[i]['action'] = individual.actions
        
    
        # Visualize
        desires = nx.get_node_attributes(G, 'desire')  
        
        plt.figure(figsize=(10, 10))
        
        cmap = LinearSegmentedColormap.from_list('custom_heatmap', [(0.0, 'orange'), (0.5, 'red'), (1.0, 'darkred')])
        nx.draw(G, pos, node_size=1000, with_labels=False, node_color=list(desires.values()), cmap=cmap, edgecolors='black')

        # Add action values as labels inside each node
        node_labels = {node: f'{G.nodes[node]["action"][round]:.2f}' for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

        # scalar mappable for the colorbar
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  
        
        # Show the colorbar
        cbar = plt.colorbar(sm, cax=None, ax=None, orientation='vertical')
        cbar.set_label('Desire', rotation=90, labelpad=10)

        plt.savefig(osp.join('.', 'figs/simulation', f'round_{(round+1)}.png'), dpi=300, bbox_inches='tight')

        plt.close()


if __name__ == "__main__":
    # Setup.
    individuals_num = 50
    
    # Set of parameters for individuals
    desires = np.linspace(0, 1, individuals_num)
    beta = 2
    
    # Create Individual objects
    individuals = [Individual(delta, beta) for delta in desires]
    
    # Run experiment
    experiment(individuals, individuals_num, nu=0.5, pi='linear', t=0.25, s=0.6, num_rounds=50)
    
    exit(0)
