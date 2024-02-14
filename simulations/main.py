# Project:  CensorshipDissent
# Filename: main.py
# Authors:  Anish Nahar (anahar1@asu.edu)

import numpy as np
from simulation import simulation

# Individual class
class Individual:
    def __init__(self, delta, beta):
        self.delta = delta
        self.beta = beta
        self.actions = []


if __name__ == "__main__":
    # Setup.
    individuals_num = 1000
    
    # Set of parameters for individuals
    desires = np.linspace(0, 1, individuals_num)
    
    # Create Individual objects
    individuals = [Individual(delta, beta=np.random.uniform(1, 2)) for delta in desires]
    
    # Update Rule to be used
    rule = 'sharing_around_tables'
    #rule = 'when_in_rome'
    #rule = 'socialization'
    
    print('Running simulation for rule: ' + rule + ' ...')
    
    # Run simulation
    simulation(individuals, individuals_num, nu=0.5, pi='linear', t=0.25, s=1.75, num_rounds=100, change_rule = rule)
    
    print(f'Simulation complete. Graph stored in figs/simulations/{rule} folder')
    
    exit(0)