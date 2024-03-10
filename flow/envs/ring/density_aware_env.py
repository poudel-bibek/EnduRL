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

import os
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

        
        self.LOCAL_ZONE = 35 # 50 #m, arbitrarily set
        self.VEHICLE_LENGTH = 5 #m can use self.k.vehicle.get_length(veh_id)
        self.MAX_SPEED = 10 # m/s
        self.velocity_track = []
        self.label_meaning = ["Leaving", "Forming", "Free Flow", "Congested", "Undefined", "No vehicle in front"] 

        # Set this on init and reset
        self.data_storage = []
        self.csc_model = self.load_csc_model()
        self.csc_output = None
        self.csc_output_encoded = None
        self.estimated_free_speed = 0

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
        
        ########## FOR CSC DATA COLLECTION ##########
        # return Box(low=-float('inf'), 
        #            high=float('inf'), 
        #            shape=((self.LOCAL_ZONE // self.VEHICLE_LENGTH), 2), 
        #            dtype = np.float32)

        ########## FOR REGULAR TRAINING ##########
        return Box(low=-float('inf'), 
                   high=float('inf'), 
                   shape=(9,), # 3 regular + 6 categories of labels one hot encoded
                   dtype = np.float32)
    
        
    def _apply_rl_actions(self, rl_actions):
        """ 
        
        """
        ##############
        # For Efficiency (Fuel and Throughput). Only present in test time. For multi-agents, even for the trained agent's zero acceleration behavior, change in controllersforaware (Not from here)
        # For the first 300 steps after warmup, estimate the free flow speed in the local zone (Leverage CSC)
        # if speed is greater than desired speed, then just maintain the speed
        # rl_speed = self.k.vehicle.get_speed(self.k.vehicle.get_rl_ids()[0])
        # if self.estimated_free_speed!=0 and rl_speed >= self.estimated_free_speed:
        #     rl_actions = [0.0]
        
        #print(f"RL action received: {rl_actions}")
            
        ##############
        # Original acceleration action
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        ########## FOR CSC DATA COLLECTION ##########
        # return 0

        ########## FOR REGULAR TRAINING ##########
        if rl_actions is None: 
            return 0
        
        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0
        
        ##############
        # Reward for safety and stability

        # rl_accel = rl_actions[0]
        # magnitude = np.abs(rl_accel)
        # sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0
        # #print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # reward = 0.2*np.mean(vel) - 4*magnitude # The general acceleration penalty
        # #print(f"First Reward: {reward}")

        # # Forming Shaping
        # penalty_scalar = -5
        # fixed_penalty = -1

        # # 0 = Leaving, 1 = Forming, 2 = Free Flow, 3 = Congested, 4 = Undefined, 5 = No vehicle in front
        # if self.csc_output[0] == 1:
        #     if sign >= 0:
        #         forming_penalty = min(fixed_penalty, penalty_scalar*magnitude)
        #         #print(f"Forming: {forming_penalty}")
        #         reward += forming_penalty # If congestion is fomring, penalize acceleration
    
        # return reward

        ###########
        # Reward for efficiency  
        rl_accel = rl_actions[0]
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0
        reward = np.mean(vel) - 2*magnitude # The general acceleration penalty

        # Congested, forming and undefined shaping 
        penalty_scalar = -10
        penalty_scalar_2 = -10
        fixed_penalty = -1

        # Maintaining velocity is fine
        # 0 = Leaving, 1 = Forming, 2 = Free Flow, 3 = Congested, 4 = Undefined, 5 = No vehicle in front
        if self.csc_output[0] == 3 : # Congested
            if sign > 0: # This equal to sign must be removed. To make more like FS sign=0 has to be enabled
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                #print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        # Leaving shaping 
        elif self.csc_output[0] == 0:
            # We want the acceleration to be positive
            if sign < 0:
                leaving_penalty = penalty_scalar_2*magnitude
                #print(f"Leaving: {leaving_penalty}")
                reward += leaving_penalty

        return reward

    # Helper: Get Monotonoicity based label. 
    # The monotonocity based label is supported by the assymetric driving theory. 
    # That when accelerating, humans leave a larger gap in front and while decelerating, they leave a smaller gap. This is more intuitive to understand as well
    def get_monotonicity_label(self, distances):
        """
        The normalized distance (sorted from RL at index 0 and furthest vehicle at index n) of the vehicles in front
        Put else undefined condition as well for more than no vehicles in front
        """
        
        # Difference of distances. E.g., RL - the one in front.
        differences = [-1*(distances[i] - distances[i+1]) for i in range(len(distances)-1)] # Rl has travelled the least distance so -1 in front
        num_diff = len(differences)
        print("Differences: ", differences)

        # No vehicle in front
        if num_diff == 0:
            print("No vehicles in front")
            return 5
        
        # Since the distance measures is center to center (length of vehicle + effective gap) 
        min_gap = 6.8/self.LOCAL_ZONE 
        
        # For leaving, all vehicles should participate in monotonic increase (as we move away from RL). i.e., diff at 1 is higher than at 0, 2 is higher than 1..etc
        # So that its actually clear to accelerate (and wont have to brake again immediately later)
        leaving = []
        for i in range(num_diff-1):
            # Nearest to Farthest, the difference is increasing
            if (1.05)*differences[i] < differences[i+1]:
                leaving.append(True)
            else:
                leaving.append(False)

        # For forming, any vehicle can participate in monotonic decrease (as we away from RL)
        # To account for both formation of congestion that travels upstream and occurs within the local zone
        forming = []
        for i in range(num_diff-1):
            if differences[i] > (1.1)*differences[i+1]:
                forming.append(True)
            else:
                forming.append(False)

        identifier = []
        # Once a condition is met immediately return 
        if len(distances) == 0:
            identifier.append(5) # No vehicles in front, This never occurs because we include RL in distances
        
        # Check increasing or decreasing (leaving for forming)
        if all(leaving):
            print("Leaving")
            identifier.append(0) # Leaving congestion
        
        if any(forming): # TODO idea: Any is a problem. Because when RL approaches a platoon of HVs this will be true. It could be any, excluding the vehicle in front
            print("Forming")
            identifier.append(1)
        
        # Then check with thresholds (congested or free flow)
        # threshold for free flow 
        if all([difference >= 1.45*min_gap for difference in differences]):
            print("Free flow")
            identifier.append(2)
        
        # threshold for congestion: all vehicles will have more or less the small multiple of minGap distance
        # When there is congestion, there will be many vehicles in the local zone, so instead of all check any?
        # This check is performed after the leaving check is performed, so it will not be confused with leaving
        if all([difference <= 1.2*min_gap for difference in differences]):
            print("Congested")
            identifier.append(3)

        if len(identifier) == 0:
            print("Undefined")
            identifier.append(4)
            
        # If both leaving and free flow, then free flow (0 vs 2) i.e., Higher should win
        if 0 in identifier and 2 in identifier:
            identifier.remove(0)
        # If both forming and congested, then congested (1 vs 3) i.e., Higher should win
        if 1 in identifier and 3 in identifier:
            identifier.remove(1)
        # If both forming and free flow then free flow (1 vs 2) i.e., Higher should win
        if 1 in identifier and 2 in identifier:
            identifier.remove(1)
            
        return_val = identifier[0]
        return return_val
        
    # Helper 3: Get csc output
    def get_csc_output(self, current_obs):
        """
        Get the output of Traffic State Estimator Neural Network
        """
        current_obs = torch.from_numpy(current_obs).flatten()

        with torch.no_grad():
            outputs = self.csc_model(current_obs.unsqueeze(0))

        # print("csc output: ", outputs)
        # return outputs.numpy() # Logits

        _, predicted_label = torch.max(outputs, 1)
        predicted_label = predicted_label.numpy()
        return predicted_label
        

    # Helper 4: Load csc model 
    def load_csc_model(self, ):
        """
        Load the Traffic State Estimator Neural Network and its trained weights
        """
        class csc_Net(nn.Module):
            def __init__(self, input_size, num_classes):
                super(csc_Net, self).__init__() 
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
        url = "https://huggingface.co/matrix-multiply/Congestion_Stage_Classifier/resolve/main/ring_best_csc_model.pt"
        saved_best_net = csc_Net(input_size, num_classes)

        state_dict = torch.hub.load_state_dict_from_url(url)
        saved_best_net.load_state_dict(state_dict)
        saved_best_net.eval()

        return saved_best_net


    def get_state(self):
        """ 
        Relative position difference (normalized by the ring length)
        Absolute velocity (normalized by the max speed)
        """
        ########## FOR CSC DATA COLLECTION ##########
        # rl_id = self.k.vehicle.get_rl_ids()[0]
        # rl_pos = self.k.vehicle.get_x_by_id(rl_id)
        # current_length = self.k.network.length()

        # # This is sorted as ['human_0', 'human_1', 'human_2', 'human_3', 'rl_0'], with human_3 as furthest
        # sorted_veh_ids = self.k.vehicle.get_veh_list_local_zone(rl_id, self.k.network.length(), self.LOCAL_ZONE)

        # # sorting needs to be RL at index 0 with furthest vehicle at index n
        # sorted_veh_ids.remove('rl_0')  
        # sorted_veh_ids.insert(0, 'rl_0')
        # print(f"Vehicles in zone: {sorted_veh_ids}")

        # observation_csc = np.full((10, 2), -1.0)
        # distances = []
        # for i in range(len(sorted_veh_ids)):
        #     # Get the distance of the vehicle from the RL vehicle
        #     rel_pos = (self.k.vehicle.get_x_by_id(sorted_veh_ids[i]) - rl_pos) % current_length
        #     norm_pos = rel_pos / self.LOCAL_ZONE # This is actually the normalized distance
        #     distances.append(norm_pos)

        #     vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
        #     norm_vel = vel / self.MAX_SPEED

        #     observation_csc[i] = [norm_pos, norm_vel]

        # timestep = self.step_counter
        # label = self.get_monotonicity_label(distances)
        # print(f"Writing data: {timestep}, {label}, {observation_csc}")
        # self.data_storage.append([timestep, label, observation_csc])

        # # Make the RL stop for some times towards last timesteps of warmup, to include more variety in data
        # # Because RL vehicle Stopped and others moving, this scenario is hard to obtain.
        # stop_timesteps = [(2900, 3000), (3400, 3500)]
        
        # # If timestep is in ranges of stop_timesteps, then stop the RL vehicle
        # if any([timestep in range(start, end) for start, end in stop_timesteps]):
        #     self.k.vehicle.apply_acceleration(rl_id, -1.0)

        # if self.step_counter == self.env_params.warmup_steps - 1: 
        #      # if does not exist 
        #     if not os.path.exists("./csc_data"):
        #         os.makedirs("./csc_data")
        #     time_now = strftime("%Y-%m-%d-%H:%M:%S")
        #     np.save(f"./csc_data/length_{int(current_length)}/csc_data_{time_now}.npy", np.array(self.data_storage))

        # return observation_csc
    
        ########## FOR REGULAR TRAINING ##########
        rl_id = self.k.vehicle.get_rl_ids()[0]
        rl_pos = self.k.vehicle.get_x_by_id(rl_id)
        current_length = self.k.network.length()

        # This is sorted as ['human_0', 'human_1', 'human_2', 'human_3', 'rl_0'], with human_3 as furthest
        sorted_veh_ids = self.k.vehicle.get_veh_list_local_zone(rl_id, self.k.network.length(), self.LOCAL_ZONE)
        # sorting needs to be RL at index 0 with furthest vehicle at index n
        sorted_veh_ids.remove('rl_0')  
        sorted_veh_ids.insert(0, 'rl_0')

        #distances = []
        observation_csc = np.full((10, 2), -1.0)
        for i in range(len(sorted_veh_ids)):
            # Get the distance of the vehicle from the RL vehicle
            rel_pos = (self.k.vehicle.get_x_by_id(sorted_veh_ids[i]) - rl_pos) % current_length
            norm_pos = rel_pos / self.LOCAL_ZONE # This is actually the normalized distance
            #distances.append(norm_pos)

            vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
            norm_vel = vel / self.MAX_SPEED
            observation_csc[i] = [norm_pos, norm_vel]

        #label = self.get_monotonicity_label(distances)
        observation_csc = np.array(observation_csc, dtype = np.float32) # required for torch

        # For using csc model: add csc output to appropriate observation
        self.csc_output = self.get_csc_output(observation_csc)
        self.csc_output_encoded = np.zeros(6) 
        self.csc_output_encoded[self.csc_output] = 1 

        #print(f"csc output: {self.csc_output}, one hot encoded: {self.csc_output_encoded}, meaning: {self.label_meaning[self.csc_output[0]]}")

        # Zone count will count the RL agent icsclf as well
        # Observe the leader
        if len(sorted_veh_ids) > 1:
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
            self.k.vehicle.get_speed(rl_id) / max_speed,
            (self.k.vehicle.get_speed(lead_id) -
            self.k.vehicle.get_speed(rl_id)) / max_speed,
            (self.k.vehicle.get_x_by_id(lead_id) -
            self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()
            / max_length
            ])
        
        # Dont observe the leader
        else:
            observation = np.array([-1, -1, -1]) # the second -1 could be plausible above but unlikely

        # If time steps are less than warmup + 300 then estimate the free speed
        if (self.step_counter > self.env_params.warmup_steps and self.step_counter < self.env_params.warmup_steps + 300):
            # csc output is free flow
            if self.csc_output[0] == 2:
                estimate = 0.70*np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in sorted_veh_ids])
                if estimate > self.estimated_free_speed:
                    self.estimated_free_speed = estimate
        
        #print(f"Estimated free speed: {self.estimated_free_speed}")

        observation = np.append(observation, self.csc_output_encoded)
        #print(f"Observation: {observation, observation.shape}\n")
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
        