import torch
import torch.nn as nn
from copy import deepcopy

import numpy as np
#from flow.envs.base import Env
#from flow.envs.bottleneck import BottleneckEnv
from flow.envs.multiagent.base import MultiEnv
from gym.spaces.box import Box

"""
Irrespective of the lane that the RL vehicle finds icsclf in, the actions are same
"""

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # whether the toll booth should be active
    "disable_tb": True,
    # whether the ramp meter is active
    "disable_ramp_metering": True,
}

MAX_LANES = 8  # base number of largest number of lanes in the network

ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    "add_rl_if_exit": True,
}

class DensityAwareBottleneckEnv(MultiEnv):
    """
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """
        """
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.scaling = network.net_params.additional_params.get("scaling", 1) # If not found replace with 1
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")

        #self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids()) 
        #print(f"\nInitial RL vehicles: {self.rl_id_list, len(self.rl_id_list)}\n")

        # self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles) # Somehow this works
        # initial_rl_veh = [f"rl_{i}" for i in range(self.initial_vehicles.num_rl_vehicles)] # Hack
        # self.rl_id_list = deepcopy(initial_rl_veh)
        # print(f"\nInitial RL vehicles: {self.rl_id_list, len(self.rl_id_list)}\n")
        
        self.max_speed = self.k.network.max_speed()
        self.LOCAL_ZONE = 50 #m
        self.MAX_SPEED = 10 # This is just a normalizer for csc observations
        self.VEHICLE_LENGTH = 5 #m
        self.csc_model = self.load_csc_model()
        self.label_meanings = ['Leaving', 'Forming', 'Free Flow', 'Congested', 'Undefined', 'No vehicle in front']
        # Create a dictionary to store the id, csc output and action of each RL vehicle
        self.rl_storedict = {}

        # A nested dict for 2 zippers (edge :4 and :5)
        # RL is inside and wants to look for vehicle outsize the zipper
        self.lane_mapping_dict_outside = {"4":{ 0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
                                     "5":{ 0: 0, 1: 0, 2: 1, 3: 1,}}
        
        # RL is inside and wants to look for vehicle inside the zipper
        self.lane_mapping_dict_inside = {":4_0":{ 0: [0,1], 1: [0, 1], 2: [2, 3], 3: [2, 3], 4: [4, 5], 5: [4, 5], 6: [6, 7], 7: [6, 7]},
                                    ":5_0":{ 0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3],}}

        self.free_flow_speed = 0.0 # All the RL vehicles estimate a single free flow speed 

    @property
    def observation_space(self):
        """
        Since policies are distributed, define observations for each RL vehicle.
        """
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(9,), # First theree are default observations, next 6 are CSC observations encoded
            dtype=np.float32) 
    
    @property
    def action_space(self):
        """
        Although there are many agents, there is a singular neural network that is shared.
        """
        
        return Box(
                low=-np.abs(self.env_params.additional_params['max_decel']),
                high=self.env_params.additional_params['max_accel'],
                shape=(1,),
                dtype=np.float32)
        
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
        
    def load_csc_model(self, ):
        """
        Load the Traffic State Estimator Neural Network and its trained weights
        """
        class CSC_Net(nn.Module):
            def __init__(self, input_size, num_classes):
                super(CSC_Net, self).__init__() 
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
        url = "https://huggingface.co/matrix-multiply/Congestion_Stage_Estimator/resolve/main/best_cse_model.pt"
        saved_best_net = CSC_Net(input_size, num_classes)

        state_dict = torch.hub.load_state_dict_from_url(url)
        saved_best_net.load_state_dict(state_dict)
        saved_best_net.eval()

        return saved_best_net
    
    def get_default_observations(self, rl_id, new_positions):
        """
        We need to supply the position dict for the immediate leader
        """

        lead_id = self.k.vehicle.get_leader_bottleneck(rl_id, self.lane_mapping_dict_outside, self.lane_mapping_dict_inside, new_positions) # Leader itself maybe hard to get in the bottleneck. In zipper lanes
        if lead_id is not None:

            # print(f"RL id: {rl_id} Lead id: {lead_id}")
            # Use similar normalizers,  arbitrary high values should suffice
            max_length = 270

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / self.MAX_SPEED,

                (self.k.vehicle.get_speed(lead_id) - self.k.vehicle.get_speed(rl_id)) / self.MAX_SPEED,

                (self.k.vehicle.get_distance(lead_id) - self.k.vehicle.get_distance(rl_id)) / max_length # x is bad. Use distance here. This problem should be fixed

                ])
        
        else: # Current solution. In zipper lanes, if there is no leader, then we are not observing the leader
            #print(f"RL id: {rl_id} No leader")
            observation = np.array([-1, -1, -1])
        
        return observation
        
    def get_state(self):
        """
        First three observations are the default observations, last six are encoded csc output

        """

        new_positions = self.k.vehicle.corrected_position_zipper()
        observation = {}
        for rl_id in self.k.vehicle.get_rl_ids():

            # Get the CSC observations for each RL vehicle 
            # The order of vehicles should be increasing in the order of distance to the RL vehicle (include RL vehicle itself at 0th index)
            # RL needs to be at index 0 and the rest of the vehicles should be sorted in increasing order of distance
            sorted_veh_ids = self.k.vehicle.get_veh_list_local_zone_bottleneck(rl_id, self.LOCAL_ZONE, self.lane_mapping_dict_outside, self.lane_mapping_dict_inside, new_positions)
            # Due to the nature of zipper lanes, sometimes there can be more than 10 vehicles in the local zone.
            # In that case, we will only consider the first 10 vehicles in the local zone.
            sorted_veh_ids = sorted_veh_ids[:10] # Only truncates if there are more than 10
            #print(f"RL id: {rl_id} Sorted veh ids: {sorted_veh_ids}") #TODO: Verify this. Verified.
            
            if len(sorted_veh_ids) == 0:
                csc_output_encoded = np.zeros(6) # i.e., nothing
            
            else:
                # For csc, both relative position and relative velocity to the leaders in the zone are required. 
                # The default input to csc is set to -1
                observation_csc = np.full((10, 2), -1.0) # Max 10 vehicles, with 2 properties each. Make this LOCAL_ZONE dependent?
                rl_pos = self.k.vehicle.get_distance(rl_id) # get_x_by distance may be misleading so using this. #TODO: Verify this. Verified.
                
                for i in range(len(sorted_veh_ids)):
                    distance = self.k.vehicle.get_distance(sorted_veh_ids[i])
                    #x = self.k.vehicle.get_x_by_id(sorted_veh_ids[i])
                    #print(f"RL id: {rl_id} Distance: {distance}, x: {x}")

                    rel_pos = (distance - rl_pos)
                    norm_pos = rel_pos / self.LOCAL_ZONE

                    vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
                    norm_vel = vel / self.MAX_SPEED

                    observation_csc[i] = [norm_pos, norm_vel]

                observation_csc = np.array(observation_csc, dtype=np.float32)
                csc_output = self.get_csc_output(observation_csc)
                #print(f"RL id: {rl_id} CSC output: {self.label_meanings[csc_output[0]]}")

                csc_output_encoded = np.zeros(6)
                csc_output_encoded[csc_output] = 1 # i.e. something

            # Add items to the dict
            self.rl_storedict[rl_id] = {'csc_output': csc_output, 
                                        'sorted_ids': sorted_veh_ids,} 
            
            ############## EFFICIENCY (AT TEST TIME) ############## Comment during training
            # Add the free estimated flow speed for each RL.
            # csc output is free flow 
            # if csc_output[0] == 2: 
            #     # Get an estimate of the free flow speed 
            #     estimate = 0.60 * np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in sorted_veh_ids]) 
            #     # May need to change the scalar based on penetration rate. 0.60 for 0.05
            #     if estimate > self.free_flow_speed:
            #         self.free_flow_speed = estimate
            
            # Concatenate observations and return 
            default_obs = self.get_default_observations(rl_id, new_positions)
            obs_for_this_vehicle = np.concatenate((default_obs, csc_output_encoded), axis=None)
            # TODO: Make sure this is good. 
            observation[rl_id] = obs_for_this_vehicle

        #print(f"Free flow speed: {self.free_flow_speed}")
        return observation

    def compute_reward(self, rl_actions, **kwargs):
        """
        

        """
        if rl_actions is None:
            return {}
        
        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        # print(f"All IDS: {self.k.vehicle.get_ids()}") # Only the vehicles currently present in the sim are present.

        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return {}

        ##############
        # Reward for safety and stability
        # reward = {}
        # for rl_id in self.k.vehicle.get_rl_ids():
        #     if rl_id not in rl_actions.keys():
        #         #print(f"\nRL id: {rl_id} not in rl_actions\n")
        #         # TODO: Verify how many and which vehicles are not in rl_actions Verified. Only vehicles that just entered.
        #         # the vehicle just entered
        #         reward[rl_id] = 0.0
                
        #     else: 
        #         rl_action = rl_actions[rl_id] # Acceleration
        #         sign = np.sign(rl_action)
        #         magnitude = np.abs(rl_action)

        #         # Speed 
        #         rl_vel = self.k.vehicle.get_speed(rl_id)

        #         # Perhaps we need to add a small component of its own velocity so that it does not stop in zippers
        #         reward_value = 0.25*rl_vel- 2*magnitude

        #         penalty_scalar = -5 
        #         fixed_penalty = -1 
        #         csc_output = self.rl_storedict[rl_id]['csc_output'][0]
        #         #print(f"RL id: {rl_id} CSC output: {csc_output}, Meaning: {self.label_meanings[csc_output]}")

        #         # Shaping component 1
        #         if csc_output == 1: # Forming
        #             if sign >= 0:
        #                 forming_penalty = min(fixed_penalty, penalty_scalar * magnitude) # Min because both quantities are negative
        #                 #print(f"RL id: {rl_id} Forming penalty: {forming_penalty}")
        #                 reward_value += forming_penalty

        #         reward[rl_id] = reward_value[0]

        #print(f"Reward: {reward}")
        #print(f"\n Reward keys: {reward.keys()}")
        #return reward

        ##############
        # Reward for efficiency
        reward = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            if rl_id not in rl_actions.keys():
                reward[rl_id] = 0.0

            else:
                rl_action = rl_actions[rl_id]
                sign = np.sign(rl_action)
                magnitude = np.abs(rl_action)
                # In case of single agent ring, this reward component in efficiency would have been for for average velocity.
                # Here, its its own velocity. 
                rl_vel = self.k.vehicle.get_speed(rl_id)

                # Average velocity of vehicles in its lane? Behind it.
                # Just want it to behave that way, does not have to be part of observations
                reward_value = 0.75*rl_vel - 2*magnitude 

                penalty_scalar = -10
                penalty_scalar_2 = -10
                fixed_penalty = -1
                csc_output = self.rl_storedict[rl_id]['csc_output'][0]

                #print(f"RL id: {rl_id} CSC output: {csc_output}, Meaning: {self.label_meanings[csc_output]}")

                # Forming states are allowed.
                if csc_output == 3 : # Congested
                    if sign > 0: 
                        reward_value += min(fixed_penalty, penalty_scalar * magnitude)
                        #print(f"RL id: {rl_id} Congested penalty: {reward_value}")
                
                elif csc_output == 0: # Leaving
                    if sign < 0: 
                        # Fixed penalty
                        reward_value += penalty_scalar_2 * magnitude
                        #print(f"RL id: {rl_id} Leaving penalty: {reward_value}")
                
                reward[rl_id] = reward_value[0]

        #print(f"Reward: {reward}")
        return reward
    

    def _apply_rl_actions(self, rl_actions):
        """
        

        """
        for rl_id in self.k.vehicle.get_rl_ids():
            if rl_id not in rl_actions.keys():
                # the vehicle just entered, so ignore
                continue
            rl_action = rl_actions[rl_id]
            ############## SAFETY + STAILITY (AT TEST TIME) ##############

            ############## EFFICIENCY (AT TEST TIME) ############## Comment during training
            # If rl velocity greater than estimated free flow velocity, acceleration = 0
            # rl_vel = self.k.vehicle.get_speed(rl_id)
            # if rl_vel >= self.free_flow_speed:
            #     rl_action = 0.0

            self.k.vehicle.apply_acceleration(rl_id, rl_action)


    def additional_command(self):
        """
        Reintroduce any RL vehicle that may have exited in the last step.
        """

        super().additional_command()
        
        # This is too expensive just to show color
        # new_positions = self.k.vehicle.corrected_position_zipper()
        # for rl_id in self.k.vehicle.get_rl_ids():
        #     vehicles_in_zone = self.k.vehicle.get_veh_list_local_zone_bottleneck(rl_id, self.LOCAL_ZONE, new_positions)
        #     for veh_id in vehicles_in_zone:
        #         self.k.vehicle.set_observed(veh_id)

        #print(f"\nRL storedict {self.rl_storedict}")
        for rl_id in self.rl_storedict.keys():
            vehicles_in_zone = self.rl_storedict[rl_id]['sorted_ids']
            for veh_id in vehicles_in_zone:
                self.k.vehicle.set_observed(veh_id)
        

    def get_idm_accel(self, rl_id):
        """
        For testing
        """

        # default params
        self.v0=30
        self.T=1
        self.a=1
        self.b=1.5
        self.delta=4
        self.s0=2
        self.time_delay=0.0

        v = self.k.vehicle.get_speed(rl_id)
        lead_id = self.k.vehicle.get_leader(rl_id)
        h = self.k.vehicle.get_headway(rl_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = self.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        #print("IDM")
        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
    
    def reset(self):
        """
        
        """
        self.rl_storedict = {}
        return super().reset()
    