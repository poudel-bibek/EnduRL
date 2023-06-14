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

def calculate_threshold(network_length, num_vehicles, max_speed):
    """
    Calculate optimal gap for free flow conditions 
    Calculate the threshold based on the 2 second rule? solely based on the max speed?
    Right now the gap is calculated based on network length and number of vehicles
    """
    return (network_length - num_vehicles * 5) / num_vehicles 

def get_max_density(local_zone, min_gap=0.0, vehicle_length=5.0):
    """
    How many vehicles max can fit in the local zone?
    """
    max_vehicles = local_zone/(vehicle_length + min_gap)
    max_density = max_vehicles*1000/local_zone # density is in veh/km
    return max_density


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

        
        self.MAX_SPEED = self.k.network.max_speed() # Is the speed limit

        self.LOCAL_ZONE = 40 # m
        self.MAX_DENSITY = get_max_density(self.LOCAL_ZONE) # vehicle length is 5m, so max 200 vehicles in 1000m 
        self.MAX_DECEL = self.env_params.additional_params['max_decel']

        # This is used in observation
        self.history_length = 20
        self.density_history = [1e-3 for i in range(self.history_length)]

        # Detection flag
        self.approching_congestion = False

    @property
    def action_space(self):
        """ 
        """
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)
        

    @property
    def observation_space(self):
        
        shp = 3 + self.history_length + 1 
        return Box(low=-float('inf'), high=float('inf'), shape=(shp,), dtype = np.float32)

            
    def _apply_rl_actions(self, rl_actions):
        """ 
        """
        
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)
    

    def detect_conditions(self, tag):
        """
        Detect and turn on/ off the falgs 
        """
    
        changes = []
        for i in range(self.history_length-1):
            changes.append(self.density_history[i+1]/self.density_history[i]) # present/ past

        self.approching_congestion = False
        critical_density = 100 

        if tag ==1: 
            # Lower quality (Stage I )
            if any([change > 1.1 for change in changes]):
                self.approching_congestion = True
        else: 
            # Higher quality (Stage II)
            if self.density_history[0] >= critical_density and any([change > 1.1 for change in changes]):
                self.approching_congestion = True


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
        
        ##############
        response_scaler = 4
        penalty = -100 # Penalty higher in stage II

        reward = 0

        rl_id = self.k.vehicle.get_rl_ids()[0]
        current_length = self.k.network.length()
        current_density_up = self.k.vehicle.get_local_density(rl_id, current_length, self.LOCAL_ZONE, direction='back')

        if self.approching_congestion:
            # If the agent let density wave travel upstream 
            # We expect the agent to have already learned to slow down
            if current_density_up == self.density_history[-1]:
                reward += penalty
                
        reward += np.mean(vel)
        reward -= response_scaler*np.mean(np.abs(rl_actions))

        return float(reward)

    def get_state(self):
        """ 
        
        """

        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id)
        current_length = self.k.network.length()
        current_density_down = self.k.vehicle.get_local_density(rl_id, current_length, self.LOCAL_ZONE, direction='front')
        #print("Current density downstream: ", current_density_down)
        #current_density_up = self.k.vehicle.get_local_density(rl_id, current_length, self.LOCAL_ZONE, direction='back')
        #print("Current density upstream: ", current_density_up)
        

        self.detect_conditions(tag=0) 
        self.density_history.pop(0)
        self.density_history.append(current_density_down)
        # print("Max density: ", self.MAX_DENSITY)
        # print("Density history: ", self.density_history)
        # print("\n")

        observation = [] 
        observation.append(self.k.vehicle.get_speed(rl_id)/ self.MAX_SPEED)
        observation.append((self.k.vehicle.get_speed(lead_id) - self.k.vehicle.get_speed(rl_id))/self.MAX_SPEED)
        observation.append((self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(rl_id))/current_length)

        for i in self.density_history:
            obs = i/self.MAX_DENSITY
            #print("Obs: ", obs)
            observation.append(i/self.MAX_DENSITY)

        observation.append(int(self.approching_congestion))
        #observation.append(int(self.leaving_congestion))
        #observation.append(int(self.near_free_flow))
        
        #print("Observation = ", observation)
        return np.array(observation)

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

        # perform the generic reset function
        return super().reset()


    def additional_command(self):
        """ 
        Define which vehicles are observed for visualization purposes.
        According to the local density range
        """
        
        # specify observed vehicles
        rl_id = self.k.vehicle.get_rl_ids()[0]
        rl_position = self.k.vehicle.get_x_by_id(rl_id)
        
        local_zone = self.LOCAL_ZONE
        current_length = self.k.network.length()

        all_vehicle_ids = self.k.vehicle.get_ids()
        #veh_pos = [self.k.vehicle.get_x_by_id(v_id) for v_id in all_vehicle_ids]

        position_bound = [rl_position, rl_position + local_zone]
        if position_bound[1] > current_length:
            position_bound = [rl_position, current_length, (rl_position + local_zone - current_length)]
            observed_vehicles = [v_id for v_id in all_vehicle_ids if (self.k.vehicle.get_x_by_id(v_id) >= position_bound[0]\
                and self.k.vehicle.get_x_by_id(v_id) <= position_bound[1]) | (self.k.vehicle.get_x_by_id(v_id)>0.0 and self.k.vehicle.get_x_by_id(v_id)<=position_bound[2])]
        
        else:
            observed_vehicles = [v_id for v_id in all_vehicle_ids if self.k.vehicle.get_x_by_id(v_id) >= position_bound[0]\
                and self.k.vehicle.get_x_by_id(v_id) <= position_bound[1]]

        for veh_id in observed_vehicles:
            self.k.vehicle.set_observed(veh_id)
        


 # elif self.leaving_congestion:
        #     if direction == 1.0:
        #         reward += response_scaler * intensity
        #         #print("Three")
        #     else:
        #         reward -= penalty
        #         #print("Four")
        # else:
        #     reward += np.mean(vel)
        #     #print("Five")