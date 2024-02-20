import os
import torch
import torch.nn as nn
import numpy as np
from gym.spaces.box import Box
from time import strftime
from flow.envs.multiagent.base import MultiEnv

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

class DensityAwareIntersectionEnv(MultiEnv):
    """
    Observation space: Each RL observes all the vehicles in front of it in the zone.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # Something like this was used by Villarreal
        #self.num_rl = env_params.additional_params['num_rl']

        # Max anticipated RL vehicle at a time

        self.LOCAL_ZONE = 50 #m
        self.VEHICLE_LENGTH = 5 #m can use self.k.vehicle.get_length(veh_id) 
        self.MAX_SPEED= 10 # This is just a normalizer for CSC observations. Its 20 because the speed limit is higher. But in m/sec
        
        self.data_storage = []
        self.selected_rl_id = None
        self.CSC_model = self.load_csc_model()
        self.rl_storedict = {}

    @property
    def observation_space(self):
                   
        ########## FOR CSC DATA COLLECTION ##########
        return Box(low=-float('inf'), 
                   high=float('inf'), 
                   shape=((self.LOCAL_ZONE // self.VEHICLE_LENGTH), 2), 
                   dtype = np.float32)

        ########## FOR REGULAR TRAINING ##########
        # return Box(
        #     low=-float("inf"),
        #     high=float("inf"),
        #     shape=(9,), # First theree are default observations, next 6 are CSC observations encoded
        #     dtype=np.float32) 

    @property
    def action_space(self):
        return Box(
                low=-5,
                high=5,
                shape=(1,),
                dtype=np.float32)
        
    def get_default_observations(self, rl_id):
        """
        Get the default 3 observations for each RL vehicle.
        """

        if self.k.vehicle.get_leader(rl_id) is not None:
            lead_id = self.k.vehicle.get_leader(rl_id)

            #print(f"RL id: {rl_id} Lead id: {lead_id}")
            # Normalizers,  arbitrary high values should suffice
            max_length = 270

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / self.MAX_SPEED,

                (self.k.vehicle.get_speed(lead_id) - self.k.vehicle.get_speed(rl_id)) / self.MAX_SPEED,

                # But  the x value is not reliable and can be randomly very large. So replace it with distance travelled so far
                # Since we are taking the difference, it represents the same thing.
                #(self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(rl_id)) / max_length
                (self.k.vehicle.get_distance(lead_id) - self.k.vehicle.get_distance(rl_id)) / max_length

                ])
        # If there is no leader, then we are not observing the leader
        else:
            observation = np.array([-1, -1, -1])
        
        #print(f"RL id: {rl_id} Observation: {observation}")
        return observation
    
    def get_CSC_output(self, current_obs):
        """
        Get the output of Traffic State Estimator Neural Network
        """
        current_obs = torch.from_numpy(current_obs).flatten()

        with torch.no_grad():
            outputs = self.CSC_model(current_obs.unsqueeze(0))

        # print("CSC output: ", outputs)
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


    def _apply_rl_actions(self, rl_actions):
        """
    
        """
        #print(f"\n\nRL Actions: {rl_actions}\n\n")

        for rl_id in self.k.vehicle.get_rl_ids():
            if rl_id not in rl_actions.keys():
                # the vehicle just entered, so ignore
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[rl_id])

 
    # Helper 2: Get Monotonoicity based label. 
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
        
        if any(forming):
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


    def compute_reward(self, rl_actions, **kwargs):
        """
        There may be many RL vehicles, how to considolate the reward across all of them?
        Accumulate the penalty.
        """
        ########## FOR CSC DATA COLLECTION ##########
        return {self.selected_rl_id: 0}

        # ########## FOR REGULAR TRAINING ##########
        # ##############
        # # Reward for safety and stability

        # if rl_actions is None: 
        #     return {}
        
        # vel = np.array([
        #     self.k.vehicle.get_speed(veh_id)
        #     for veh_id in self.k.vehicle.get_ids()
        # ])

        # # Fail the current episode if these
        # if any(vel <-100) or kwargs['fail']:
        #     return {}
        
        # reward = {}
        # for rl_id in self.rl_storedict.keys():

        #     if rl_id not in rl_actions.keys():
        #         # the vehicle just entered
        #         reward[rl_id] = 0.0

        #     else: 
        #         rl_action = rl_actions[rl_id]
        #         sign = np.sign(rl_action)
        #         magnitude = np.abs(rl_action)

        #         reward_value = 0.2*np.mean(vel) - 4*np.abs(rl_action)

        #         # Shaping Component
        #         penalty_scalar = -5
        #         fixed_penalty = -0.5

        #         CSC_output = self.rl_storedict[rl_id]['CSC_output']
        #         if CSC_output ==1:
        #             if sign >= 0:
        #                 forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) # Min because both quantities are negative
        #                 reward_value += forming_penalty

        #         reward[rl_id] = reward_value[0]

        #print(f"\n\nReward: {reward}, \n\n")
        # print reward keys 
        #print(f"\n\nReward keys: {reward.keys()}, \n\n")
        #return reward
    
        ###########
        # Reward for efficiency 
                

    def get_state(self):
        """
        First three observations are the default observations, next 6 are CSC observations
        Sorting: RL ids dont have to be sorted in any order.
        """
        ########## FOR CSC DATA COLLECTION ##########
        all_rl_ids = self.k.vehicle.get_rl_ids() 

        # First try to remove 
        if self.selected_rl_id is not None:
            # remove selected RL id past its 650m distance
            if self.k.vehicle.get_distance(self.selected_rl_id) >= 650: #because not space to create local zone after that 
                self.k.vehicle.remove(self.selected_rl_id)
                self.selected_rl_id = None

            print(f"All RL ids: {all_rl_ids}")
        # second add if removed
        # If a RL id is selected, keep it. If not, select the RL id with the least distance travelled
        if self.selected_rl_id is None:

            # If we have some RL
            if len(all_rl_ids) > 0:
                
                # Sort to get the least distance RL
                self.selected_rl_id = all_rl_ids[0]
                for rl_id in all_rl_ids:
                    if self.k.vehicle.get_distance(rl_id) < self.k.vehicle.get_distance(self.selected_rl_id):
                        self.selected_rl_id = rl_id

            # NO RL ids have been isntantiated:
            else:
                return {self.selected_rl_id: np.zeros((self.LOCAL_ZONE // self.VEHICLE_LENGTH, 2))}
            
        rl_id = self.selected_rl_id # Only one RL vehicle at a time to collect data
        rl_dist = self.k.vehicle.get_distance(rl_id) # get x is broken, get distance is used.

        print(f"\n\n selected RL id: {rl_id}\n\n")
        observation_CSC = np.full((self.LOCAL_ZONE // self.VEHICLE_LENGTH, 2), -1.0) # (10, 2)

        timestep = self.step_counter
        # distances of vehicles in the local zone
        distances = []

        # This is sorted from closest to farthest and includes RL
        sorted_veh_ids = self.k.vehicle.get_veh_list_local_zone_intersection(rl_id, self.LOCAL_ZONE)
        self.rl_storedict[rl_id] = {'veh_in_zone': sorted_veh_ids }
        # This is sorted from RL vehicle at index 0 to farthest at index n

        for i in range(len(sorted_veh_ids)):
            # Get the distance of the vehicle from the RL vehicle
            rel_pos = self.k.vehicle.get_distance(sorted_veh_ids[i]) - rl_dist
            norm_pos = rel_pos / self.LOCAL_ZONE # Normalize it
            distances.append(norm_pos)

            vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
            norm_vel = vel / self.MAX_SPEED # Normalize it

            observation_CSC[i] = [norm_pos, norm_vel]


        label = self.get_monotonicity_label(distances)
        print(f"Writing data: {timestep}, {label}, {observation_CSC}")
        
        self.data_storage.append([timestep, label, observation_CSC])
        if self.step_counter == self.env_params.warmup_steps - 1: #leave this self.env_params.horizon 
            # if does not exist 
            if not os.path.exists("./csc_data"):
                os.makedirs("./csc_data")
            np.save("./csc_data/csc_data_{}.npy".format(strftime("%Y-%m-%d-%H:%M:%S")), np.array(self.data_storage))

        return {self.selected_rl_id: observation_CSC}

        ########## FOR REGULAR TRAINING ##########
        # self.rl_storedict = {}
        # observation = {}
        # for rl_id in self.k.vehicle.get_rl_ids():

        #     # Get the CSC observations for each RL vehicle
        #     # Increasing in the order of distance to the RL vehicle (including RL as well)
        #     # Get vehicle list in the local zone for intersection.
        #     sorted_veh_ids = self.k.vehicle.get_veh_list_local_zone_intersection(rl_id, self.LOCAL_ZONE)

        #     if len(sorted_veh_ids) == 0:
        #             CSC_output_encoded = np.zeros(6) # i.e., nothing

        #     else:
        #         # For CSC, both relative position and relative velocity to the leaders in zone are required
        #         # The default input to CSC is set to -1
        #         observation_CSC = np.full((10, 2), -1.0) # Max 2 vehicles with 2 properties
        #         rl_pos = self.k.vehicle.get_distance(rl_id) #self.k.vehicle.get_x_by_id(rl_id) 

        #         # Go through each sorted vehicle to get CSC output
        #         for i in range(len(sorted_veh_ids)):

        #             # Lets not use x to get relative positions, lets use total distance travelled
        #             # Since these vehicles are always ahead of RL, rel_pos will be a positive value
        #             rel_pos = self.k.vehicle.get_distance(sorted_veh_ids[i]) - rl_pos
        #             norm_pos = rel_pos / self.LOCAL_ZONE # Normalize it

        #             vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
        #             norm_vel = vel / self.MAX_SPEED # Normalize it
        #             observation_CSC[i] = [norm_pos, norm_vel]

        #         observation_CSC = np.array(observation_CSC, dtype = np.float32)

        #         # For using CSC model: add CSC output to appropriate observation
        #         CSC_output = self.get_CSC_output(observation_CSC)
        #         CSC_output_encoded = np.zeros(6) 
        #         CSC_output_encoded[CSC_output] = 1 # i.e. something

        #         self.rl_storedict[rl_id] = {'veh_in_zone': sorted_veh_ids , 'CSC_output': CSC_output, 'action': 0}

        #     # Concatenate them and return
        #     default_observation = self.get_default_observations(rl_id)
        #     obs_for_this_vehicle = np.concatenate((default_observation, CSC_output_encoded), axis=None)
        #     observation[rl_id] = obs_for_this_vehicle

        # #print(f"\n\nObservation: {observation}\n\n")
        # # print observation keys
        # #print(f"\n\nObservation keys: {observation.keys()}\n\n") 
        # return observation

    def reset(self):
        """
        Is this necessary? Why is it not defined in IntersectionRLPOEnv?
        """
        return super().reset()

    def additional_command(self):
        """
        Since RL vehicles are continuously introduces by the inflow, no need to reintroduce them.
        """
    
        # color the vehicles in the control zone
        for rl_id in self.rl_storedict.keys():
            vehicles_in_zone = self.rl_storedict[rl_id]['veh_in_zone']
            for veh_id in vehicles_in_zone:
                self.k.vehicle.set_observed(veh_id)

            
