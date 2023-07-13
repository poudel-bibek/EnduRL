"""
Density Aware RL environment 
Latest idea: Use RL to learn not only the optimal acceleration, but also the optimal gap. 

"""
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.base import Env

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.tuple import Tuple

import torch
import torch.nn as nn
from time import strftime
from copy import deepcopy
import numpy as np
import random

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}

class DensityAwareRLEnv(Env):
    """
    Docs here
    TODO: 
        Things work only for single agent
        Update additional command
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        
        self.LOCAL_ZONE = 50 # m, arbitrarily set
        self.VEHICLE_LENGTH = 5 #m can use self.k.vehicle.get_length(veh_id)
        self.MAX_SPEED = 10 # m/s
        self.velocity_track = []
        self.label_meaning = ["Leaving", "Forming", "Free Flow", "Congested", "Undefined", "No vehicle in front"] 
        # Set this on init and reset
        self.data_storage = []
        self.tse_model = self.load_tse_model()
        self.tse_output = None
        self.tse_output_encoded = None

    @property
    def action_space(self):
        """ 
        Acceleration are capped in between max and min accelerations
        For (time-headway) gap, we can either set max gap and min gap or NOT
        """
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)
        

    @property
    def observation_space(self):
        """
        This is set according to the local zone
        shape = (LZ/length_of_vehicle) , 2
        """
        
        # For TSE data collection
        # return Box(low=-float('inf'), 
        #            high=float('inf'), 
        #            shape=((self.LOCAL_ZONE // self.VEHICLE_LENGTH), 2), 
        #            dtype = np.float32)

        # For RL training, flatten all observations and add a single one in the end
        shp = (3 + 6,) # 5 categories of labels one hot encoded
        #print(f"\nObservation shape: {shp}\n")
        return Box(low=-float('inf'), 
                   high=float('inf'), 
                   shape=shp, 
                   dtype = np.float32)
    
        
    def _apply_rl_actions(self, rl_actions):
        """ 
        
        """
        print(f"\n\nRL action received: {rl_actions}")

        # Original acceleration action
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.2*np.mean(vel) - 4*magnitude
        print(f"First Reward: {reward}")
        
        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign > 0:
                forming_penalty = penalty_scalar*magnitude
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration
            elif sign == 0:
                forming_penalty = fixed_penalty
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty

        print(f"Last Reward: {reward}")
        return reward

    
    # Helper 1
    def sort_vehicle_list(self, vehicles_in_zone):
        return sorted(vehicles_in_zone, key=lambda x: (x[:-1], -int(x[-1])) if len(x) > 1 else (x, 0))

    # Helper 2: Get Monotonoicity based label. VEMA based label has too many parameters to manually set or estimate
    # The monotonocity based label is supported by the assymetric driving theory. 
    # That when accelerating, humans leave a larger gap in front and while decelerating, they leave a smaller gap
    # This is more intuitive to understand as well
    # This is based on positions whereas the VEMA (Exponential moving average velocity) is based on velocity
    # This basis of assymetric driving theory will be easier to scale (to more densities) as well
    def get_monotonicity_label(self, distances):
        """
        Get the normalized distance (sorted from farthest to closest) of the vehicles in front
        Put else undefined condition as well for more than no vehicles in front

        """
        
        # Determine the monotonicity of the difference of distances
        # Monotonicity condition is not strict i.e. less than/ greater than, equal to
        differences = [distances[i] - distances[i+1] for i in range(len(distances)-1)]
        num_diff = len(differences)
        print("Differences: ", differences)
        
        # Since the distance measures is center to center (length of vehicle + effective gap) 
        min_gap = 6.8/self.LOCAL_ZONE 

        # For leaving, all vehicles should participate in monotonic increase (away from RL)
        # So that its actually clear to accelerate (and wont have to brake again immediately later)
        leaving = []
        for i in range(num_diff-1, 0, -1):
            # Nearest to Farthest, the difference is increasing
            if differences[i-1] > differences[i]: # Greater than or equal to (if its equal to then it will be in both free flow and leaving)
                leaving.append(True)
            else: 
                leaving.append(False)

        # For forming, any vehicle can participate in monotonic decrease (away from RL)
        # To account for both formation of congestion that travels upstream and occurs within the local zone
        forming = []
        for i in range(num_diff-1, 0, -1):
            # Nearest to Farthest, the difference is decreasing
            if (1.1)*differences[i-1] < differences[i]: # Buffer
                forming.append(True)
            else:
                forming.append(False)

        # Once a condition is met immediately return 
        if len(distances) == 0:
            return 5 # No vehicles in front, This never occurs 
        
        # First check increasing or decreasing (leaving for forming)
        elif all(leaving):
            print("Leaving")
            return 0 # Leaving congestion
        
        elif any(forming):
            print("Forming")
            return 1
        
        # Then check with thresholds (congested or free flow)
        # threshold for free flow 
        elif all([difference >= 1.5*min_gap for difference in differences]):
            print("Free flow")
            return 2
        
        # threshold for congestion: all vehicles will have more or less the small multiple of minGap distance
        # When there is congestion, there will be many vehicles in the local zone, so instead of all check any?
        # This check is performed after the leaving check is performed, so it will not be confused with leaving
        elif all([difference <= 1.2*min_gap for difference in differences]):
            print("Congested")
            return 3

        else:
            print("Undefined")
            return 4
        
    # Helper 3: Get TSE output
    def get_tse_output(self, current_obs):
        """
        Get the output of Traffic State Estimator Neural Network
        """
        current_obs = torch.from_numpy(current_obs).flatten()

        with torch.no_grad():
            outputs = self.tse_model(current_obs.unsqueeze(0))

        # print("TSE output: ", outputs)
        # return outputs.numpy() # Logits

        _, predicted_label = torch.max(outputs, 1)
        predicted_label = predicted_label.numpy()
        return predicted_label
        

    # Helper 4: Load TSE model 
    def load_tse_model(self, ):
        """
        Load the Traffic State Estimator Neural Network and its trained weights
        """
        class TSE_Net(nn.Module):
            def __init__(self, input_size, num_classes):
                super(TSE_Net, self).__init__() 
                self.fc1 = nn.Linear(input_size, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 16)
                self.relu = nn.ReLU()
                self.fc3 = nn.Linear(16, num_classes)
                
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.fc3(out)
                return out

        input_size = 10*2
        num_classes = 6
        url = "https://huggingface.co/matrix-multiply/Congestion_Stage_Classifier/resolve/main/best_csc_model.pt"
        saved_best_net = TSE_Net(input_size, num_classes)

        state_dict = torch.hub.load_state_dict_from_url(url)
        saved_best_net.load_state_dict(state_dict)
        saved_best_net.eval()

        return saved_best_net


    def get_state(self):
        """ 
        Relative position difference (normalized by the ring length)
        Absolute velocity (normalized by the max speed)
        """
        rl_id = self.k.vehicle.get_rl_ids()[0]
        rl_pos = self.k.vehicle.get_x_by_id(rl_id)
        current_length = self.k.network.length()

        # Get the list of all vehicles in the local zone (sorted from farthest to closest)
        vehicles_in_zone = self.sort_vehicle_list(self.k.vehicle.get_veh_list_local_zone(rl_id, 
                                                                                         current_length, 
                                                                                         self.LOCAL_ZONE )) # Direction i front by default
        
        # For TSE data collection
        #observation = np.full(self.observation_space.shape, -1.0)

        # For RL training
        observation_tse = np.full((10, 2), -1.0)

        num_vehicle_in_zone = len(vehicles_in_zone)
        distances = []
        if num_vehicle_in_zone > 0:
            for i in range(len(vehicles_in_zone)):
                # Distance is measured center to center between the two vehicles (if -5 present, distance if bumper to bumper)
                rel_pos = (self.k.vehicle.get_x_by_id(vehicles_in_zone[i]) - rl_pos) % current_length
                norm_pos = rel_pos / self.LOCAL_ZONE # This is actually the normalized distance
                distances.append(norm_pos)

                vel = self.k.vehicle.get_speed(vehicles_in_zone[i])
                norm_vel = vel / self.MAX_SPEED

                observation_tse[i] = [norm_pos, norm_vel]
                
        observation_tse = np.array(observation_tse, dtype=np.float32)
        
        # For using TSE model: add TSE output to appropriate observation
        self.tse_output = self.get_tse_output(observation_tse)
        self.tse_output_encoded = np.zeros(6) 
        self.tse_output_encoded[self.tse_output] = 1

        print(f"TSE output: {self.tse_output}, one hot encoded: {self.tse_output_encoded}, meaning: {self.label_meaning[self.tse_output[0]]}")

        # Original observations
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

        # normalizers
        max_speed = 15.
        if self.env_params.additional_params['ring_length'] is not None:
            max_length = self.env_params.additional_params['ring_length'][1]
        else:
            max_length = self.k.network.length()

        observation = np.array([
            self.k.vehicle.get_speed(rl_id) / max_speed,
            (self.k.vehicle.get_speed(lead_id) -
             self.k.vehicle.get_speed(rl_id)) / max_speed,
            (self.k.vehicle.get_x_by_id(lead_id) -
             self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()
            / max_length
        ])

        observation = np.append(observation, self.tse_output_encoded)
        print(f"Observations new: {observation, observation.shape}\n")

        # For training TSE model: Get data for TSE NN training
        #print("Observation\n", observation)
        # timestep = self.step_counter
        # label = self.get_monotonicity_label(distances)
        # print(f"Writing data: {timestep}, {label}, {observation}")
        # self.data_storage.append([timestep, label, observation])
        # # For data collection, make  warmup 2500
        # if self.step_counter == self.env_params.warmup_steps - 1: #leave this self.env_params.horizon 
        #     np.save("./tse_data/tse_data_{}.npy".format(strftime("%Y-%m-%d-%H:%M:%S")), np.array(self.data_storage))

        return observation

    def reset(self):
        """ 
        """
        # skip if ring length is None
        if self.env_params.additional_params['ring_length'] is None:
            return super().reset()

        # update the network
        initial_config = InitialConfig(bunching=50, min_gap=0)
        length = random.randint(
            self.env_params.additional_params['ring_length'][0],
            self.env_params.additional_params['ring_length'][1])
        additional_net_params = {
            'length':
                length,
            'lanes':
                self.net_params.additional_params['lanes'],
            'speed_limit':
                self.net_params.additional_params['speed_limit'],
            'resolution':
                self.net_params.additional_params['resolution']
        }

        net_params = NetParams(additional_params=additional_net_params)

        self.network = self.network.__class__(
            self.network.orig_name, self.network.vehicles,
            net_params, initial_config)

        self.k.vehicle = deepcopy(self.initial_vehicles)

        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        print('\n-----------------------')
        print('ring length:', net_params.additional_params['length'])
        print('-----------------------')

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)
        
        self.step_counter = 0
        self.data_storage = []
        
        # perform the generic reset function
        return super().reset()


    def additional_command(self):
        """ 
        Define which vehicles are observed for visualization purposes.
        According to the local density range
        """
        
        # specify observed vehicles
        rl_id = self.k.vehicle.get_rl_ids()[0]
        current_length = self.k.network.length()
        vehicles_in_zone = self.k.vehicle.get_veh_list_local_zone(rl_id, current_length, self.LOCAL_ZONE )
        for veh_id in vehicles_in_zone:
            self.k.vehicle.set_observed(veh_id)
        