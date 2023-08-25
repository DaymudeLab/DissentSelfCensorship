

def utility_ind(a_i, d_i, ):
    pass


def utility_authority(a, alpha, pi, tau, s):
    """

    :param a: a array of float actions a_i in [0, d_i]
    :param alpha: a float adamancy parameter > 0
    :param pi: a string punishment function in ["constant", "linear"]
    :param tau: a tolerance in [0, 1]
    :param s: a severity parameter > 0
    """
    p_i = punishment(a, pi, tau, s)
    # TODO:


def punishment(a, pi, tau, s):
    """

    :param a: a array of float actions a_i in [0, d_i]
    :param pi: a string punishment function in ["constant", "linear"]
    :param tau: a tolerance in [0, 1]
    :param s: a severity parameter > 0
    """
    pass


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


def opt_tolerance():
    pass


def opt_severity():
    pass


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
