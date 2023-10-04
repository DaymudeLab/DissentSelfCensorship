import uuid
import numpy as np

"""
Authority Class
"""
class Authority:

    def __init__(self):
        self.pi = None
        self.tau = None
        self.s = None
        self.alpha = None
        self.v = None
    
    def utility_authority(a, alpha, pi, tau, s):
        """

        :param a: a array of float actions a_i in [0, d_i]
        :param alpha: a float adamancy parameter > 0
        :param pi: a string punishment function in ["constant", "linear"]
        :param tau: a tolerance in [0, 1]
        :param s: a severity parameter > 0
        """
        utility = 0

        for a_i in a:
            p_i = Authority.calc_punishment(a_i, pi, tau, s)
            utility += alpha * a_i + p_i
        
        return -utility

    
    def calc_punishment(a_i, pi, tau, s):
        """

        :param a: a array of float actions a_i in [0, d_i]
        :param pi: a string punishment function in ["constant", "linear"]
        :param tau: a tolerance in [0, 1]
        :param s: a severity parameter > 0
        """

        if a_i > tau:
            if pi == "constant":
                return s
            else:
                return s * (a_i - tau)
        
        else:
            return 0

    def probability(v, a_i):
        """
        Calculates the probability of a punishment being enacted.
        """
        return v + (1 - v) * a_i
    
    def noise():

        return np.random.normal(0, 1)

    def opt_tolerance():
        """
        Calculates authority's optimal tolerance as a function of known parameters.
        """
        pass

    def opt_severity():
        """
        Calculates authority's optimal severity as a function of known parameters.
        """
        pass

    
    def get_tolerance():
        """
        Returns the authority's tolerance.
        """
        return Authority.tau + Authority.noise()


    def get_severity():
        """
        Returns the authority's severity.
        """
        return Authority.s + Authority.noise()


"""
Individual Class
"""
class Individual:

    def __init__(self):
        self.id = uuid.uuid4()
        self.b_i = None
        self.d_i = None

    def opt_action(d_i, b_i, pi, tau, s, v):
        """
        Calculates individual i's optimal action as a function of known parameters.

        :param d_i: a float desire to dissent in [0, 1]
        :param b_i: a boldness parameter > 0
        :param pi: a string punishment function in ["constant", "linear"]
        :param tau: a tolerance in [0, 1]
        :param s: a severity parameter > 0
        """

        if pi == "constant":
            if d_i <= tau or (b_i > (1 - v) and  d_i > (b_i * tau + v * s)/(b_i - (1 - v) * s)):
                return d_i
            else:
                return tau
        else:
            if d_i <= tau or (v == 1 and s <= b_i) or (v < 1 and Individual.a_peak(tau, v, s, b_i) >= d_i):
                return d_i
            if d_i > tau and ((v == 1 and s > b_i) or (v < 1 and Individual.a_peak(tau, v, s, b_i) <= tau)):
                return tau
            else:
                return Individual.a_peak(tau, v, s, b_i)

    def a_peak(tau, v, s, b_i):
        """
        Calculates individual's peak utility as a function of known parameters.
        """

        return 0.5 (tau - (v * s - b_i) / (s * (1 - v)))
    

    def utility_individual(b_i, a_i, d_i, p_i):
        """
        Calculates individual utility.

        :param d_i: a float desire to dissent in [0, 1]
        :param a_i: a float action a_i in [0, d_i]
        :param b_i: a boldness parameter > 0
        :param p_i: a float punishment in [0, s]
        """

        return b_i (1 - d_i + a_i) - p_i


def experiment(individuals, num_individuals, authority,  num_rounds):
    actions_history = []

    # Step 1: All individuals calculate and enact a_i.

    # Step 2: The authority calculates and enacts punishments.

    # Step 3: Individuals update one of (d_i, beta_i) according to an update rule.

    # Step 4: The authority updates one of (tau, s) for the next round.

    # Measurement:
    # Utilities (authority and individuals).
    # All parameters.
    # Individuals' actions. (Imagine a plot with time vs. everyone's action trajectories).
    
    for round in range(num_rounds):
        # Step 1
        actions = [individuals[i].opt_action(individuals[i].d_i, individuals[i].b_i, authority.pi, authority.tau, authority.s, authority.v) for i in range(num_individuals)]
        
        actions_history.append(actions)


        # Step 2

        # Step 3

        # Step 4


if __name__ == "__main__":
    # Setup.

    # Total amount of indivduals.
    num_individuals = 10000 

    # Create an array of Individual objects with evenly spaced desires
    individuals = [Individual() for _ in range(num_individuals)]

    # Generate evenly spaced desires for a large number of individuals
    desires = np.linspace(0, 1, num_individuals) 

    
    
    for i in range(num_individuals):
        individuals[i].d_i = desires[i]
        individuals[i].b_i = np.random.uniform(0, 10)

    # Create an Authority object with random values
    authority = Authority()
    authority.tau = np.random.rand()
    authority.s = np.random.uniform(0, 10)
    authority.alpha = np.random.uniform(0, 10)
    authority.v = np.random.rand()
    authority.pi = "constant"

    # Run experiment
    experiment(individuals, num_individuals, authority, 100)

    exit(0)
