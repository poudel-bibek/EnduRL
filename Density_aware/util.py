import numpy as np
from scipy.optimize import fsolve

# velocity upper bound from Wu et al
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