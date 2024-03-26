# Project:  CensorshipDissent
# Filename: plotgraph.py
# Authors:  Anish Nahar (anahar1@asu.edu)

from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


def color_mapping(action, desire):
    """
    Determines the color of the node and the text inside the node
    :param action: the individual's float action dissent (in [0,delta])
    :param desire: the individual's float desire to dissent (in [0,1])
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

def graph_plot(G, pos, round, rule):
    """
    Creates the networkx graph plot for the given round
    :param G: the networkx graph containing individuals' data
    :param pos: the position of the nodes in the graph
    :param round: the current round [0, num_rounds-1]
    :param rule: the update rule used [socialization, sharing_around_tables, when_in_rome]
    """

    fig, ax = plt.subplots(figsize=(15, 12))

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

    # Create a new axis for the color bar
    cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('')
    cbar.set_ticks([0, 0.25, 1])
    cbar.set_ticklabels(['Compliant', 'Completely Censoring', 'Defiant'])
    ax.set_title(f'{rule} - Round {round}', fontsize=30)

    plt.savefig(osp.join('.', 'figs/simulation/' + rule, f'round_{(round+1)}.png'), dpi=150, bbox_inches='tight')

    plt.close()

def action_round_plot(action_hist, rule):
    """
    Creates the action vs round plot for "r" rounds

    :param action_hist: action dictionary containing each individual's actions for each round
    :param rule: the update rule [socialization, sharing_around_tables, when_in_rome]
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


def desire_round_plot(desire_dict, rule):
    """
    Creates the desire vs round plot for "r" rounds

    :param desire_dict: action dictionary containing each individual's actions for each round
    :param rule: the update rule [socialization, sharing_around_tables, when_in_rome]
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
