# Project:  CensorshipDissent
# Filename: simulation.py
# Authors:  Anish Nahar (anahar1@asu.edu)

"""
main: Runs the simulation for a network of individuals and an authority 
with a set of rules that defines one's actions.
"""

from plotgraph import action_round_plot, desire_round_plot, graph_plot
import numpy as np
import networkx as nx

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

    :param G: the networkx graph containing the individuals' data
    :param neighbors: the neighbor nodes of the individual
    :param neighbors_num: the total number of neighbors of the individual
    :param r: current round
    :param d_r: the individual's desired dissent in the previous round
    :returns: the new desire according to the socialization rule
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
    d_i,r+1 = (w * d_r + sum_of_neighors_d_r) / (w + N)
    
    :param G: the networkx graph containing the individuals' data
    :param neighbors: the neighbor nodes of the individual
    :param neighbors_num: the total number of neighbors of the individual
    :param r: current round
    :param d_r: the individual's desired dissent in the previous round
    :returns: the new desire based on the Sharing Around Tables rule
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

    :param G: the networkx graph containing the individuals' data
    :param neighbors: the neighbor nodes of the individual
    :param neighbors_num: the total number of neighbors of the individual
    :param r: current round
    :param a_ir: the individual's optimal action in the current round
    :returns: the new action based on the When in Rome rule
    """
    
    w = neighbors_num

    sum_neighbors_last_action = 0
    for n in neighbors:
        sum_neighbors_last_action += G.nodes[n]['action'][r-1]

    new_action = (w * a_ir + sum_neighbors_last_action) / (w + neighbors_num)

    return new_action

def simulation(individuals, num_individuals, nu, pi, t, s,  num_rounds, change_rule):
    """
    Runs the simulation for a network of individuals and an authority
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

    G = nx.powerlaw_cluster_graph(num_individuals, m=2, p=0.3)
    
    # Create the power law graph
    pos = nx.spring_layout(G) 
    
    if change_rule == 'socialization':
        for round in range(num_rounds):
            # Update the desires and calculate the action of the individuals based on the socialization rule
            for i, individual in enumerate(individuals):
                if round != 0:
                    individual.delta = socialization(G=G, neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round, d_r=individual.delta)
                
                # Optimal Action based on updated values of the individual and the authority   
                action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=np.random.normal(loc=t, scale=0.05) , s=np.random.normal(loc=s, scale=0.05) )
                individual.actions.append(action)
                               
            # Add the Individual objects as nodes with their desires and actions as node attributes
            for i, individual in enumerate(individuals):
                G.nodes[i]['desire'] = individual.delta
                G.nodes[i]['action'] = individual.actions
                desire_dict[i].append(individual.delta)
                #if round == num_rounds - 1:
                action_hist[i] = individual.actions
            
            graph_plot(G, pos, round, change_rule)

    elif change_rule == 'sharing_around_tables':
        
        for round in range(num_rounds):
            # Update the desires and calculate the action of the individuals based on the rule
            for i, individual in enumerate(individuals):
                if round != 0: # first round desire is not changed by rule
                    individual.delta = sharing_around_tables(G=G, neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round, d_r=individual.delta)
                
                # Optimal Action based on updated values of the individual and the authority     
                action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=np.random.normal(loc=t, scale=0.05) , s=np.random.normal(loc=s, scale=0.05) )
                individual.actions.append(action)
                               
            # Add the Individual objects as nodes with their desires and actions as node attributes
            for i, individual in enumerate(individuals):
                G.nodes[i]['desire'] = individual.delta
                G.nodes[i]['action'] = individual.actions
                desire_dict[i].append(individual.delta)
                action_hist[i] = individual.actions
            
            graph_plot(G, pos, round, change_rule)

    elif change_rule == 'when_in_rome':
        # Update the actions of the individuals based on the rule
        for round in range(num_rounds):
            
            for i, individual in enumerate(individuals): 
                # Optimal Action based on values of the individual and the authority                   
                action = opt_dissent(delta=individual.delta, beta=individual.beta, nu=nu, pi=pi, t=np.random.normal(loc=t, scale=0.05) , s=np.random.normal(loc=s, scale=0.05) )
                
                # Update the action based on the rule after first round
                if round != 0:
                    action = when_in_rome(G=G, neighbors=G.neighbors(i), neighbors_num=G.degree(i), r=round, a_ir=action)
                
                individual.actions.append(action)
            

            # Add the Individual objects as nodes with their desires and actionsas node attributes
            for i, individual in enumerate(individuals):
                G.nodes[i]['desire'] = individual.delta
                G.nodes[i]['action'] = individual.actions
                desire_dict[i].append(individual.delta)
                action_hist[i] = individual.actions

            graph_plot(G, pos, round, change_rule)

    else:
        print('ERROR: Unrecognized change rule \'' + change_rule + '\'')
    
    action_round_plot(action_hist, change_rule)
    desire_round_plot(desire_dict, change_rule)