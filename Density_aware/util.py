import numpy as np
from scipy.optimize import fsolve

# velocity upper bound from Wu et al (https://flow-project.github.io/papers/wu17a.pdf )
# This is an approximation 

def v_eq_max_function(v, *args):
    """Return the error between the desired and actual equivalent gap."""
    num_vehicles, length = args

    # maximum gap in the presence of one rl vehicle
    s_eq_max = (length - num_vehicles * 5) / (num_vehicles - 1)

    v0 = 30
    s0 = 2
    tau = 1
    gamma = 4

    error = s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5

    return error

def get_velocity_upper_bound(num_vehicles, length):
    """Return the velocity upper bound for the given number of vehicles."""
    v_guess = 4
    return fsolve(v_eq_max_function, np.array(v_guess), args=(num_vehicles, length))[0]

def get_desired_velocity(num_vehicles, length):
    """Return the desired velocity for the given number of vehicles."""
    scaler = 0.93 # 93% of the upper bound may be desired? 
    print("Scaler: ", scaler)
    return get_velocity_upper_bound(num_vehicles, length) * scaler


# Shock
# Define shock models 

def shock_model(identifier):
    # Accel/ Decel value, duration, frequency (in the interval between shock start and shock end)
    # Duration: In seconds, for which each shock is applied
    # Frequency: In the interval, how many shocks are applied
    #  
    if identifier == 1:
        return (-0.2, 10, 10)
        #return (0.2, 20, 20)

    elif identifier == 2:
        return (0.2, 20, 10)

    elif identifier == 3:
        return (0.2, 20, 10)

    else: 
        raise ValueError("Shock model identifier not recognized")
