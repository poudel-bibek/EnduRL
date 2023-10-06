import torch
import torch.nn as nn
from copy import deepcopy

import numpy as np
#from flow.envs.base import Env
from flow.envs.bottleneck import BottleneckEnv
from gym.spaces.box import Box
from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import InFlows, NetParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams

"""
Irrespective of the lane that the RL vehicle finds itself in, the actions are same
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

class DensityAwareBottleneckEnv(BottleneckEnv):
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
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")

        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles) # Somehow this works
        #self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids()) # But this doesn't
        initial_rl_veh = [f"rl_{i}" for i in range(self.initial_vehicles.num_rl_vehicles)] # Hack
        self.rl_id_list = deepcopy(initial_rl_veh)

        self.max_speed = self.k.network.max_speed()

        # Max anticipated RL vehicle at a time
        self.max_rl_vehicles = 40
        self.LOCAL_ZONE = 50 #m
        self.MAX_SPEED = 10 # This is just a normalizer for TSE observations
        self.tse_model = self.load_tse_model()

        # create a dictionary to store the id, TSE output and action of each RL vehicle
        # Also reset this dict 
        self.rl_storedict = {}

    @property
    def observation_space(self):
        """
        """
        #num_edges = len(self.k.network.get_edge_list())
        #num_rl_veh = self.num_rl

        # Each RL is going to have 9 length vector as observations of shape shp = (3 + 6,) 
        return Box(low=-float('inf'),
                    high=float('inf'),
                    shape=(self.max_rl_vehicles, 9),
                    dtype=np.float32)
    
    @property
    def action_space(self):
        """
        Although there are many agents, there is a singular neural network that is shared.
        """
        return Box(
                low=-np.abs(self.env_params.additional_params['max_decel']),
                high=self.env_params.additional_params['max_accel'],
                shape=(self.max_rl_vehicles, ),
                dtype=np.float32)
    
    def sort_vehicle_list(self, vehicles_in_zone):
        """
        simply reverse the order because we need farthest to closest from left to right
        """

        if vehicles_in_zone is None:
            # Usually RL vehicle's own id will always be there
            return []
        else:
            # revese the order
            sorted_vehicles_in_zone = vehicles_in_zone[::-1]
            return sorted_vehicles_in_zone
        
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
        url = "https://huggingface.co/matrix-multiply/Congestion_Stage_Estimator/resolve/main/best_cse_model.pt"
        saved_best_net = TSE_Net(input_size, num_classes)

        state_dict = torch.hub.load_state_dict_from_url(url)
        saved_best_net.load_state_dict(state_dict)
        saved_best_net.eval()

        return saved_best_net
    
    def get_default_observations(self, rl_id):
        
        if self.k.vehicle.get_leader_bottleneck(rl_id) is not None:
            
            lead_id = self.k.vehicle.get_leader_bottleneck(rl_id)
            #print(f"RL id: {rl_id} Lead id: {lead}")

            # Use similar normalizers,  arbitrary high values should suffice
            max_speed = 20.
            max_length = 270

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,

                (self.k.vehicle.get_speed(lead_id) - self.k.vehicle.get_speed(rl_id)) / max_speed,

                (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(rl_id)) / max_length

                ])
        # If there is no leader, then we are not observing the leader
        else:
            observation = np.array([-1, -1, -1])
        
        return observation
    
    def bottle_sort(self, veh_list):
        """
        sort according to ids
        """
        names = [] 
        for veh in veh_list: 
            name = veh.split('_')[-1]
            if "flow" in veh:
                name = int(name.split('.')[-1])
            else:
                name = int(name)

            names.append(name)
        #print(f"Names: {names}")
        sorted_veh_list = [x for _,x in sorted(zip(names,veh_list))]
        return sorted_veh_list
        
    def get_state(self):
        """
        First three observations are the default observations, last six are encoded TSE output
        """
        # Update the RL id list, if there is flow_00 in the vehicle ids 
        # all_ids = self.k.vehicle.get_ids()
        # for veh_id in all_ids:
        #     if "flow_00" in veh_id:
        #         self.rl_id_list.append(veh_id)
        #         #self.num_rl += 1 # This updates automatically

        # simple solution 
        self.rl_id_list = self.k.vehicle.get_rl_ids()
        rl_ids = self.k.vehicle.get_rl_ids()

        # To have some order in the observations, sort the rl_ids in terms of x-position
        # But sometimes the position is not defined, so just sort by the last 2 digits of the id
        rl_ids = self.bottle_sort(rl_ids)
        #print(f"Total RL ids: {len(rl_ids)}, RL ids: {rl_ids}")

        #rl_obs = np.zeros((self.max_rl_vehicles, 9))
        rl_obs = []
        for i, rl_id in enumerate(rl_ids):
            #print(f"RL id: {rl_id}")
            # For this RL vehicle, get the sorted list of vehicles in its local zone (sorted from farthest to closest)
            # Already sorted in increasing distance order from the RL
            sorted_veh_ids = self.sort_vehicle_list(self.k.vehicle.get_veh_list_local_zone_bottleneck(rl_id, self.LOCAL_ZONE))
            # print(f"RL id: {rl_id} Sorted veh ids: {sorted_veh_ids}")
            # print("\n")

            if len(sorted_veh_ids) == 0:
                tse_output_encoded = np.zeros(6) # i.e., nothing
            else: 
                
                # For TSE, both relative position and relative velocity to the leaders in zone are required
                distances = []

                # The default input to TSE is set to -1
                observation_tse = np.full((10, 2), -1.0)

                rl_pos = self.k.vehicle.get_x_by_id(rl_id)

                # go through the vehicles in the zone
                for i in range(len(sorted_veh_ids)):
                    rel_pos = (self.k.vehicle.get_x_by_id(sorted_veh_ids[i]) - rl_pos)
                    norm_pos = rel_pos / self.LOCAL_ZONE
                    distances.append(norm_pos)

                    vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
                    norm_vel = vel / self.MAX_SPEED
                    observation_tse[i] = [norm_pos, norm_vel]

                observation_tse = np.array(observation_tse, dtype=np.float32)
        
                # For using TSE model: add TSE output to appropriate observation
                tse_output = self.get_tse_output(observation_tse)
                tse_output_encoded = np.zeros(6) 
                tse_output_encoded[tse_output] = 1 # i.e. something
                
            # Add the item to dict
            self.rl_storedict[rl_id] = {'tse_output': tse_output, 'action': 0}

            default_obs = self.get_default_observations(rl_id)
            obs_for_this_vehicle = np.append(default_obs, tse_output_encoded)
            rl_obs.append(obs_for_this_vehicle)

        # Append the rest of the observations with zeros to make it shape (max_rl_vehicles, 9)
        if len(rl_obs) < self.max_rl_vehicles:
            num_missing = self.max_rl_vehicles - len(rl_obs)
            for i in range(num_missing):
                rl_obs.append(np.zeros(9))

        rl_obs = np.reshape(rl_obs, (self.max_rl_vehicles, 9))
        #print(f"RL obs: {rl_obs}\n")
        #print(f"rl_storedict: {self.rl_storedict}")

        return rl_obs

    def compute_reward(self, rl_actions, **kwargs):
        """
        Although there are many agents, there is a singular reward.
        """
        #num_rl = self.k.vehicle.num_rl_vehicles

        if rl_actions is None: 
            return 0 
        
        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        # General reward, average velocity and acceleration magnitude penalty
        #print(f"Average absolute actions: {np.mean(np.abs(rl_actions))}")
        reward = 0.2*np.mean(vel) - 4*np.mean(np.abs(rl_actions))

        #print("Actions: ", rl_actions)
        rl_ids = self.k.vehicle.get_rl_ids()
        sorted_rl_ids = self.bottle_sort(rl_ids)

        # Safety + Stability reward 

        penalty_scalar = -5
        fixed_penalty = -0.5
      
        for i, rl_id in enumerate(sorted_rl_ids):
            self.rl_storedict[rl_id]['action'] = rl_actions[i]
            tse_output = self.rl_storedict[rl_id]['tse_output']

            #print(f" From reward function: RL id: {rl_id} TSE output: {tse_output}, Action: {rl_actions[i]}")
            sign = np.sign(rl_actions[i])
            magnitude = np.abs(rl_actions[i])
            if tse_output[0] == 1:
                if sign>=0:
                    # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                    forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                    #print(f"Forming: {forming_penalty}")
                    reward += forming_penalty # If congestion is fomring, penalize acceleration

        # Efficiency reward
        # if tse_output[0] == 3 or tse_output[0] == 4 or tse_output[0] == 1:

        return reward

    def _apply_rl_actions(self, actions):
        """
        Map the sorted RL ids to the actions
        """

        # Use the same method to sort RL, and then apply actions 
        rl_ids = self.k.vehicle.get_rl_ids()
        sorted_rl_ids = self.bottle_sort(rl_ids)

        # Find the same id in the stored dict and append the action
        for i, rl_id in enumerate(sorted_rl_ids):
            #action =  actions[i] #-1 #-1*np.ones(len(sorted_rl_ids))
            action = self.rl_storedict[rl_id]['action']

            bypass_action = self.get_idm_accel(rl_id)
            self.k.vehicle.apply_acceleration(rl_id, acc=bypass_action) #action
            #print(f"From apply_rl_actions: RL id: {rl_id} Action: {action}")
        

    def additional_command(self):
        """
        Reintroduce any RL vehicle that may have exited in the last step.
        """
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge='1',
                        type_id=str('rl'),
                        lane=str(lane_num),
                        pos="0",
                        speed="max")
                except Exception:
                    pass
        
        for rl_id in self.k.vehicle.get_rl_ids():
            vehicles_in_zone = self.k.vehicle.get_veh_list_local_zone_bottleneck(rl_id, self.LOCAL_ZONE)
            for veh_id in vehicles_in_zone:
                self.k.vehicle.set_observed(veh_id)

    def get_idm_accel(self, rl_id):
        """
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
        
    # def reset(self):
    #     """Reset the environment with a new inflow rate.

    #     The diverse set of inflows are used to generate a policy that is more
    #     robust with respect to the inflow rate. The inflow rate is update by
    #     creating a new network similar to the previous one, but with a new
    #     Inflow object with a rate within the additional environment parameter
    #     "inflow_range", which is a list consisting of the smallest and largest
    #     allowable inflow rates.

    #     **WARNING**: The inflows assume there are vehicles of type
    #     "followerstopper" and "human" within the VehicleParams object.
    #     """
    #     add_params = self.env_params.additional_params
    #     if add_params.get("reset_inflow"):
    #         inflow_range = add_params.get("inflow_range")
    #         flow_rate = np.random.uniform(
    #             min(inflow_range), max(inflow_range)) * self.scaling
            
    #     for _ in range(100):
    #         try:
    #             # introduce new inflows within the pre-defined inflow range
    #             inflow = InFlows()
    #             inflow.add(
    #                 veh_type="followerstopper",  # FIXME: make generic
    #                 edge="1",
    #                 vehs_per_hour=flow_rate * .1,
    #                 departLane="random",
    #                 departSpeed=10)
    #             inflow.add(
    #                 veh_type="human",
    #                 edge="1",
    #                 vehs_per_hour=flow_rate * .9,
    #                 departLane="random",
    #                 departSpeed=10)

    #             # all other network parameters should match the previous
    #             # environment (we only want to change the inflow)
    #             additional_net_params = {
    #                 "scaling": self.scaling,
    #                 "speed_limit": self.net_params.
    #                 additional_params['speed_limit']
    #             }
    #             net_params = NetParams(
    #                 inflows=inflow,
    #                 additional_params=additional_net_params)

    #             vehicles = VehicleParams()
    #             vehicles.add(
    #                 veh_id="human",  # FIXME: make generic
    #                 car_following_params=SumoCarFollowingParams(
    #                     speed_mode=9,
    #                 ),
    #                 lane_change_controller=(SimLaneChangeController, {}),
    #                 routing_controller=(ContinuousRouter, {}),
    #                 lane_change_params=SumoLaneChangeParams(
    #                     lane_change_mode=0,  # 1621,#0b100000101,
    #                 ),
    #                 num_vehicles=1 * self.scaling)
    #             vehicles.add(
    #                 veh_id="followerstopper",
    #                 acceleration_controller=(RLController, {}),
    #                 lane_change_controller=(SimLaneChangeController, {}),
    #                 routing_controller=(ContinuousRouter, {}),
    #                 car_following_params=SumoCarFollowingParams(
    #                     speed_mode=9,
    #                 ),
    #                 lane_change_params=SumoLaneChangeParams(
    #                     lane_change_mode=0,
    #                 ),
    #                 num_vehicles=1 * self.scaling)

    #             # recreate the network object
    #             self.network = self.network.__class__(
    #                 name=self.network.orig_name,
    #                 vehicles=vehicles,
    #                 net_params=net_params,
    #                 initial_config=self.initial_config,
    #                 traffic_lights=self.network.traffic_lights)
    #             observation = super().reset()

    #             # reset the timer to zero
    #             self.time_counter = 0

    #             return observation

    #         except Exception as e:
    #             print('error on reset ', e)

# class DensityAwareBottleneckEnv(BottleneckEnv):
#     """

#     """
#     def __init__(self, env_params, sim_params, network, simulator='traci'):
#         """Initialize the BottleneckEnv class."""
#         for p in ADDITIONAL_ENV_PARAMS.keys():
#             if p not in env_params.additional_params:
#                 raise KeyError(
#                     'Environment parameter "{}" not supplied'.format(p))
#         super().__init__(env_params, sim_params, network, simulator)
#         self.scaling = network.net_params.additional_params.get("scaling", 1) # If not found replace with 1
#         self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
#         self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")

#     @property
#     def observation_space(self):
#         return Box(
#             low=-float("inf"),
#             high=float("inf"),
#             shape=(1, ),
#             dtype=np.float32) 

#     @property
#     def action_space(self):
#         return Box(
#             low=-float("inf"),
#             high=float("inf"),
#             shape=(1, ),
#             dtype=np.float32) 

#     def _apply_rl_actions(self, rl_actions):
#         """
#         In Cathy's work there are 2 actions
#         1. Acceleration
#         2. Lane change

#         Multiple controllers: 
#         1. Desired velocity controller (set the max velocity of a controlled segment?)
#         2. Acceleration controller
#         3. 
#         """

#         pass


#     def compute_reward(self, rl_actions, **kwargs):
#         """

#         """

#         veh_ids = self.k.vehicle.get_ids()
#         speeds = self.k.vehicle.get_speed(veh_ids)
#         avg_speed = np.mean(speeds)
#         return avg_speed
    
#     def get_tse_output(self, current_obs):
#         """
#         Get the output of Traffic State Estimator Neural Network
#         """
#         current_obs = torch.from_numpy(current_obs).flatten()

#         with torch.no_grad():
#             outputs = self.tse_model(current_obs.unsqueeze(0))

#         # print("TSE output: ", outputs)
#         # return outputs.numpy() # Logits

#         _, predicted_label = torch.max(outputs, 1)
#         predicted_label = predicted_label.numpy()
#         return predicted_label
        

#     # Helper 4: Load TSE model 
#     def load_tse_model(self, ):
#         """
#         Load the Traffic State Estimator Neural Network and its trained weights
#         """
#         class TSE_Net(nn.Module):
#             def __init__(self, input_size, num_classes):
#                 super(TSE_Net, self).__init__() 
#                 self.fc1 = nn.Linear(input_size, 32)
#                 self.relu = nn.ReLU()
#                 self.fc2 = nn.Linear(32, 16)
#                 self.relu = nn.ReLU()
#                 self.fc3 = nn.Linear(16, num_classes)
                
#             def forward(self, x):
#                 out = self.fc1(x)
#                 out = self.relu(out)
#                 out = self.fc2(out)
#                 out = self.relu(out)
#                 out = self.fc3(out)
#                 return out

#         input_size = 10*2
#         num_classes = 6
#         url = "https://huggingface.co/matrix-multiply/Congestion_Stage_Estimator/resolve/main/best_cse_model.pt"
#         saved_best_net = TSE_Net(input_size, num_classes)

#         state_dict = torch.hub.load_state_dict_from_url(url)
#         saved_best_net.load_state_dict(state_dict)
#         saved_best_net.eval()

#         return saved_best_net
    
#     def get_state(self):
#         """
#         All RL vehicles at each timestep will have thier own observations. How to map that to a single policy?
#         Keep the observation and action size constant and Reintroduce RL vehicles that have exited in the system.
#         """
#         all_rl_ids = self.k.vehicle.get_rl_ids()
#         print("All RL ids: ", all_rl_ids)
#         #rl_id = self.k.vehicle.get_rl_ids()[0]

#         # Get sorted vehicle list in the local zone (from farthest to closest)
#         # Handle None conditons 
#         for rl_id in all_rl_ids:
#             if rl_id is None:
#                 continue
#             else:
#                 continue
        
#         print("Scaling: ", self.scaling)
#         print("Max lanes: ", MAX_LANES)
#         print("RL id list ", self.rl_id_list)
#         return np.array([0.0])

#     def reset(self):
#         """
#         For generating a policy that is "robust" to different inflow rates, the
#         flow rate is sampled in a range and the config is modified each time
#         """

#         # Do not move 2 the lines from below
#         observation = super().reset()
#         self.time_counter = 0
#         return observation

#     def additional_command(self):
#         """


#         """
#         super().additional_command()
#         # if the number of rl vehicles has decreased introduce it back in
#         num_rl = self.k.vehicle.num_rl_vehicles
#         if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
#             # find the vehicles that have exited
#             diff_list = list(
#                 set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
#             for rl_id in diff_list:
#                 # distribute rl cars evenly over lanes
#                 lane_num = self.rl_id_list.index(rl_id) % MAX_LANES * self.scaling

#                 # reintroduce it at the start of the network
#                 try:
#                     self.k.vehicle.add(
#                         veh_id=rl_id,
#                         edge='1',
#                         type_id=str('rl'),
#                         lane=str(lane_num),
#                         pos="0",
#                         speed="max")
#                 except Exception:
#                     pass

