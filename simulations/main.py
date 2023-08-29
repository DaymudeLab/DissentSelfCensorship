import uuid

"""
Authority Class
"""
class Authority:

    def __init__(self):
        self.punishment = None
        self.tolerance = None
        self.severity = None
        self.alpha = None
    
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


"""
Individual Class
"""
class Individual:

    def __init__(self):
        self.id = uuid.uuid4()
        self.action = None
        self.beta = None
        self.desire = None

    def opt_action(d_i, b_i, pi, tau, s):
        """
        Calculates individual i's optimal action as a function of known parameters.

        :param d_i: a float desire to dissent in [0, 1]
        :param b_i: a boldness parameter > 0
        :param pi: a string punishment function in ["constant", "linear"]
        :param tau: a tolerance in [0, 1]
        :param s: a severity parameter > 0
        """
        pass

    def utility_individual(b_i, a_i, d_i, p_i):
        """
        Calculates individual utility.

        :param d_i: a float desire to dissent in [0, 1]
        :param a_i: a float action a_i in [0, d_i]
        :param b_i: a boldness parameter > 0
        :param p_i: a float punishment in [0, s]
        """

        return b_i (1 - d_i + a_i) - p_i


if __name__ == "__main__":
    for round in range(100):
        # Step 1: All individuals calculate and enact a_i.

        # Step 2: The authority calculates and enacts punishments.

        # Step 3: Individuals update one of (d_i, beta_i) according to an update rule.

        # Step 4: The authority updates one of (tau, s) for the next round.

        # Measurement:
        # Utilities (authority and individuals).
        # All parameters.
        # Individuals' actions. (Imagine a plot with time vs. everyone's action trajectories).
        # Total (or normalized) amount of dissent.
        pass
