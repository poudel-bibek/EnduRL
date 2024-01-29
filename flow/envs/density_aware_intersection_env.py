import torch
import torch.nn as nn
import numpy as np
from gym.spaces.box import Box
from flow.envs.base import Env

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
}

class DensityAwareIntersectionEnv(Env):
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
        self.max_rl_vehicles = 24 # Including both directions
        self.LOCAL_ZONE = 50 #m 
        self.MAX_SPEED= 10 # This is just a normalizer for TSE observations. Its 20 because the speed limit is higher. But in m/sec
        self.tse_model = self.load_tse_model()
        self.rl_storedict = {}
        self.sorted_rl_ids = []

    @property
    def observation_space(self):
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.max_rl_vehicles, 9), # First theree are default observations, next 6 are TSE observations encoded
            dtype=np.float32) 

    @property
    def action_space(self):
        return Box(
                low=-5,
                high=5,
                shape=(self.max_rl_vehicles,),
                dtype=np.float32)

    # Helpers for TSE 
    def sort_vehicle_list(self, vehicles_in_zone):
        """
        For the intersection, vehicles can be sorted depending on thier distance to the intersection
        """
        # TODO: Complete this.

        if vehicles_in_zone is None:
            # Usually RL vehicle's own id will always be there
            return []
        else:
            # revese the order
            sorted_vehicles_in_zone = vehicles_in_zone[::-1]
            return sorted_vehicles_in_zone
        
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


    def _apply_rl_actions(self, rl_actions):
        """
        In total, the "agent" produces 22 actions.
        Actual present RL vehicles may be less than 22.
        Just ignore the rest of the actions. Because the corresponding observations also were empty for those.
        """

        print("RL actions: ", rl_actions)

        # In the same sorted fashion, apply the actions 
        sorted_rl_ids = self.sorted_rl_ids

        print(f"RL actions: {rl_actions.shape}, Sorted RL ids: {len(sorted_rl_ids)}")

        for i, rl_id in enumerate(sorted_rl_ids):
            print("RL id: ", rl_id, " Action: ", rl_actions[i])
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])
            self.rl_storedict[rl_id]['action'] = rl_actions[i]

    def compute_reward(self, rl_actions, **kwargs):
        """
        There may be many RL vehicles, how to considolate the reward across all of them?
        Accumulate the penalty.
        """

        if rl_actions is None: 
            return 0 
        
        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        #################
        # Reward Safety + Stability 

        # 0.2* average_velocity - 4* acceleration
        # Since RL is optimizing for the network, consider all vehicles for average velocity
        reward = 0.2*np.mean(vel) - 4*np.mean(np.abs(rl_actions))

        # Shaping Component
        penalty_scalar = -5
        fixed_penalty = -0.5

        # If congestion is forming and the RL wants to accelerate
        # If forming and sign (acceleration) >= 0, then penalty
        sorted_rl_ids = self.sorted_rl_ids
        for i, rl_id in enumerate(sorted_rl_ids):
            tse_output = self.rl_storedict[rl_id]['tse_output'] 

            # State, action, reward. So make use of this action value that was stored in apply_actions
            action = self.rl_storedict[rl_id]['action']
            sign = np.sign(action)
            magnitude = np.abs(action)

            if tse_output ==1:
                if sign >= 0:
                    forming_penalty = min(fixed_penalty, penalty_scalar*magnitude)
                    reward += forming_penalty

        #################
        # Reward Efficiency 

        
        return reward

    def get_state(self):
        """
        First three observations are the default observations, next 6 are TSE observations
        
        Sorting: 
        idea 1: Based on the distance to the intersection
        RL vehicles at similar distances need to behave similarly
        However, thier exact observations will be different.

        idea 2: all the RL vehicles get the same observations..

        idea 3: Get distance travelled so far by each RL vehicle

        """

        # Get all the current RL vehicles in the network.
        # RL vehicles can be simply identified by the flow name (flow10 and flow 30). Or just by calling get_rl_ids()
        rl_ids = self.k.vehicle.get_rl_ids()
        #print(f" Total RL ids: {len(rl_ids)}:",  rl_ids)

        # Filter RL ids first
        valid_rl_ids = []
        for rl_id in rl_ids:
            if self.k.vehicle.get_distance(rl_id) == -1001:
                # Need to remove the RL ids after they are out of view
                self.k.vehicle.remove(rl_id)
                del self.rl_storedict[rl_id]
            else: 
                valid_rl_ids.append(rl_id)

        # Sort them. Based on the distance to the intersection 
        # Distance to intersection center
        distances = {}
        for rl_id in valid_rl_ids:
            # distance travelled so far approach
            distance = self.k.vehicle.get_distance(rl_id)
            distances[rl_id] = distance
            
        #print("Distances: ", distances)
        sorted_rl_ids = sorted(distances, key=distances.get)
        #print("Sorted RL ids: ", sorted_rl_ids)
        
        self.sorted_rl_ids = sorted_rl_ids

        # Initially there may be no RLs
        if len(sorted_rl_ids) == 0:
            # Give some default values
            rl_obs = []
        
        else:  
            rl_obs = []
            for rl_id in sorted_rl_ids:

                # Get the TSE observations for each RL vehicle
                # Increasing in the order of distance to the RL vehicle (including RL as well)
                # Get vehicle list in the local zone for intersection.
                sorted_veh_ids = self.k.vehicle.get_veh_list_local_zone_intersection(rl_id, self.LOCAL_ZONE)

                if len(sorted_veh_ids) == 0:
                    tse_output_encoded = np.zeros(6) # i.e., nothing

                else:
                    # For TSE, both relative position and relative velocity to the leaders in zone are required
                    # The default input to TSE is set to -1
                    observation_tse = np.full((10, 2), -1.0) # Max 2 vehicles with 2 properties
                    rl_pos = self.k.vehicle.get_distance(rl_id) #self.k.vehicle.get_x_by_id(rl_id) 

                    # Go through each sorted vehicle to get TSE output
                    for i in range(len(sorted_veh_ids)):

                        # Lets not use x to get relative positions, lets use total distance travelled
                        # Since these vehicles are always ahead of RL, rel_pos will be a positive value
                        rel_pos = self.k.vehicle.get_distance(sorted_veh_ids[i]) - rl_pos
                        norm_pos = rel_pos / self.LOCAL_ZONE # Normalize it

                        vel = self.k.vehicle.get_speed(sorted_veh_ids[i])
                        norm_vel = vel / self.MAX_SPEED # Normalize it
                        observation_tse[i] = [norm_pos, norm_vel]

                    observation_tse = np.array(observation_tse, dtype = np.float32)
                    #print("TSE input: ", observation_tse)

                    # For using TSE model: add TSE output to appropriate observation
                    tse_output = self.get_tse_output(observation_tse)
                    tse_output_encoded = np.zeros(6) 
                    tse_output_encoded[tse_output] = 1 # i.e. something

                    self.rl_storedict[rl_id] = {'veh_in_zone': sorted_veh_ids , 'tse_output': tse_output, 'action': 0}

                # Concatenate them and return
                default_observation = self.get_default_observations(rl_id)
                obs_for_this_vehicle = np.concatenate((default_observation, tse_output_encoded), axis=None)
                print("Observation for RL id: ", rl_id) #, " is: ", obs_for_this_vehicle)
                rl_obs.append(obs_for_this_vehicle)

        # Append the rest of the observations with 0 to make it shape (max_rl_vehicles, 9)
        if len(rl_obs) < self.max_rl_vehicles:
            num_missing = self.max_rl_vehicles - len(rl_obs)
            for i in range(num_missing):
                rl_obs.append(np.zeros(9))

        rl_obs = np.reshape(rl_obs, (self.max_rl_vehicles, 9))
        print("RL observations: ", rl_obs)
        print("\n\n\n")
        return rl_obs

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
            
