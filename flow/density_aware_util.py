import random
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

def get_desired_velocity(num_vehicles, length, method_name = None):
    """
    Desired velocity is gotten as the uniform flow equillibrium velocity
    Only some controllers require this
    """

    # some known values are hard coded: 
    if length == 220:
        # reduce to 2.7 for FS
        if method_name == "fs":
            return 2.7
        else:
            return 3.0 

    elif length == 230:
        return 3.45

    elif length == 260:
        # From hit and trial, for 
        return 4.55 # 4.82 is the value from LORR paper, other sources # For 60% (13 cars controlled always unstable at this velocity). Change to 4.55 for that

    elif length == 270:
        return 5.2

    else: 
        scaler = 0.93 # 93% of the upper bound may be desired? 
        print("Scaler: ", scaler)
        return get_velocity_upper_bound(num_vehicles, length) * scaler


# Shock
# Define shock models 

def get_shock_model(identifier, length = None, network_scaler=1, bidirectional=False, high_speed = False):
    # Network scaler 6 used in the bottleneck
    # Accel/ Decel value, duration, frequency (in the interval between shock start and shock end)
    # Duration: In seconds, for which each shock is applied
    # Frequency: In the interval, how many shocks are applied
    # if identifier == 1:
    #     return (-1.4, 2, 10)
    
    if identifier == 2:
        # Thiese ranges are obtained form data
        # sample frequency 
        frequency = network_scaler*np.random.randint(10, 30) # value of 10 means once shock every 3000/10 = 300 steps, 5 = 600 steps, 15 = 200 steps
        
        intensity_collect = [] 
        duration_collect = []
        if high_speed:
            intensity_abs_min = 1.5
            intensity_abs_max = 4.0
        else:
            intensity_abs_min = 1
            intensity_abs_max = 3.0
        print("Frequency:", frequency)

        for i in range(frequency):
            if bidirectional:
                # between (-abs_max to -abs_min) and (abs_min to abs_max) but not between (-abs_min to abs_min)
                intensity = random.uniform(-intensity_abs_max, intensity_abs_max)
                while intensity > -intensity_abs_min and intensity < intensity_abs_min:
                    intensity = random.uniform(-intensity_abs_max, intensity_abs_max)
                
            else:
                intensity = random.uniform(-intensity_abs_max, -intensity_abs_min)

            print("Intensity:", intensity)

            durations = np.linspace(0.1, 2.5, 20) # In seconds
            
            abs_intensity = abs(intensity)
            intensity_bucket = np.linspace(intensity_abs_min, intensity_abs_max,len(durations))
            loc = np.searchsorted(intensity_bucket, abs_intensity)

            left = loc 
            right = len(durations) - loc
            probabilities_left = np.linspace(0.0, 10, left)
            # print("Probabilities left:", probabilities_left, probabilities_left.sum())

            probabilities_right = np.linspace(10, 0.0, right)
            # print("Probabilities right:", probabilities_right, probabilities_right.sum())

            probabilities = np.concatenate((probabilities_left, probabilities_right))
            probabilities /= probabilities.sum()
            #print("Probabilities:", probabilities, probabilities.sum())

            duration = round(np.random.choice(durations, 1, p=probabilities)[0], 1)
            print("Duration:", duration)

            intensity_collect.append(intensity)
            duration_collect.append(duration)

        # return intensity, durations (second), frequency 
        return (np.asarray(intensity_collect), np.asarray(duration_collect), frequency)

    # Stability test
    elif identifier == -1:
        # velocity, duration, frequency
        # Stability tests have velocity manipulation, so the first param here is speed at the velocity dip
        # Duration and frequency are also used 
        # Just apply once is enough
        if length ==220:
            vel_set = 2.0
            duration = 1

        elif length == 270:
            vel_set = 3.0
            duration = 2

        elif length == 260:
            vel_set = 3.0
            duration = 2
            
        else: 
            vel_set = 5.0
            duration = 2
        print("\n\nVelocity set: ", vel_set)
        return (vel_set, duration, 1)
        #return (2, 10, 10)

    else: 
        raise ValueError("Shock model identifier not recognized")

## Shock utils
def get_time_steps_stability(duration, frequency, shock_start_time, shock_end_time):
        # Convert duration to env steps
        duration = duration*10

        # Based on this frequency, get the time steps at which the shock is applied
        start_times = np.linspace(shock_start_time, shock_end_time - duration, frequency, dtype=int)
        end_times = np.linspace(shock_start_time + duration, shock_end_time, frequency, dtype=int)
        shock_time_steps = np.stack((start_times, end_times), axis=1)

        print("Start times: ", start_times)
        print("End times: ", end_times)
        print("Shock times: \n", shock_time_steps)

        # TODO: Perform overlap tests and warn if there is overlap
        # if start_times[1] < end_times[0]:
        #     import sys
        #     sys.exit()

        return shock_time_steps

def get_time_steps(durations, frequency, shock_start_time, shock_end_time):
        # Convert duration to env steps
        durations = durations*10
        print("Durations: ", durations)

        # Based on this frequency, get the time steps at which the shock is applied
        start_times = np.linspace(shock_start_time, shock_end_time - durations[-1], frequency, dtype=int)
        end_times = []

        for i in range(frequency):
            end_times.append(start_times[i] + durations[i])
        shock_time_steps = np.stack((start_times, end_times), axis=1)

        print("Start times: ", start_times)
        print("End times: ", end_times)
        print("Shock times: \n", shock_time_steps)

        # TODO: Perform overlap tests and warn if there is overlap
        # if start_times[1] < end_times[0]:
        #     import sys
        #     sys.exit()

        return shock_time_steps

# use 
# sm = shock_model(2)
# get_time_steps(durations, frequency, 8000, 10000)
#print(sm[0][1])