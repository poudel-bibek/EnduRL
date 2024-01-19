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
        self.max_rl_vehicles = 40
        self.LOCAL_ZONE = 50 #m 
        self.MAX_SPEED= 10 # This is just a normalizer for TSE observations
        self.tse_model = self.load_tse_model()

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
        Will this be useful in Intersection?
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

        """
        for i, rl_id in enumerate(self.rl_veh):

            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])


    def compute_reward(self, rl_actions, **kwargs):
        """

        """
        # Reward Safety + Stability 



        # Reward Efficiency 

        veh_ids = self.k.vehicle.get_ids()
        speeds = self.k.vehicle.get_speed(veh_ids)
        avg_speed = np.mean(speeds)
        return avg_speed

    def get_state(self):
        """

        """

        pass

    def reset(self):
        """
        Is this necessary? Why is it not defined in IntersectionRLPOEnv?
        """

        pass

    def additional_command(self):
        """
        Things like queue maintenance and stuff need to be done.

        """
        
        """
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]

        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)
        """
        pass