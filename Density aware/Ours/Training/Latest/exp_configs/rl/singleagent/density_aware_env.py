"""
Density Aware RL environment 
Latest idea: Use RL to learn not only the optimal acceleration, but also the optimal gap. 

"""
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.base import Env

from gym.spaces.box import Box

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
        # CONSTANTS
        self.MAX_DENSITY = 200 # vehicle length is 5m, so max 200 vehicles in 1000m 
        self.MAX_SPEED = 15 # m/s
        self.LOCAL_ZONE = 80 # m
    @property
    def action_space(self):
        """ 
        Two actions: 
        1. Acceleration
        2. Gap
        """
        return Box(
            low=-1, 
            high=1, 
            shape=(2, ),
            dtype = np.float32)
        

    @property
    def observation_space(self):
        """ 
        Seven observations (All normalized):
        1. RL speed
        2. Difference in RL Lead speed
        3. Difference in RL and Lead position
        4. Difference in Lag and RL speed
        5. Difference in Lag and RL position
        6. Local density ahead
        7. Local density behind
        """

        return Box(
            low=-1, 
            high=1, 
            shape=(7, ),
            dtype = np.float32)

    def _apply_rl_actions(self, rl_actions):
        """ 
        Current logic: 
            Acceleration behavior is only respected if the gap is not perfect.
            If the gap is perfect, then the RL agent is forced to maintain the gap by maintaining velocity.
        """
        rl_id = self.k.vehicle.get_rl_ids()[0]
        desired_accel, desired_gap = rl_actions[0], rl_actions[1]

        # Does desired gap require scaling?

        # set desired min gap, this method works (but crashes?)
        self.k.vehicle.set_min_gap('rl_id', desired_gap) #TODO: Check if this works, how?

        # measure the current gap
        current_gap = self.k.vehicle.get_headway(rl_id)
        self.gap_error = desired_gap - current_gap

        if np.abs(self.gap_error)>=0.1:  #TODO: Check if this is a good threshold
            self.k.vehicle.apply_acceleration(rl_id, 0.0)
        else:
            self.k.vehicle.apply_acceleration(rl_id, desired_accel)
        
        
    def compute_reward(self, rl_actions, **kwargs):
        """ 
        
        """
        
        # measure the current gap
        return 0

    def get_state(self):
        """ 
        
        """
        # After warmup is done, this method gets called twice for some reason
        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id)
        lag_id = self.k.vehicle.get_follower(rl_id)

        observation = np.array([
            self.k.vehicle.get_speed(rl_id),
            self.k.vehicle.get_speed(lead_id) - self.k.vehicle.get_speed(rl_id),
            self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(rl_id),
            self.k.vehicle.get_speed(rl_id) - self.k.vehicle.get_speed(lag_id),
            self.k.vehicle.get_x_by_id(rl_id) - self.k.vehicle.get_x_by_id(lag_id),

            # These two are defined in flow.core.kernel.vehicle. 80 is the distance
            self.k.vehicle.get_local_density(rl_id, self.LOCAL_ZONE)/self.MAX_DENSITY,
            self.k.vehicle.get_local_density(rl_id, self.LOCAL_ZONE, direction='back')/self.MAX_DENSITY,
        ])

        print(observation)
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

        # perform the generic reset function
        return super().reset()


    def additional_command(self):
        """ 
        Define which vehicles are observed for visualization purposes.
        """
        
        # specify observed vehicles
        rl_id = self.k.vehicle.get_rl_ids()[0]
        lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
        self.k.vehicle.set_observed(lead_id)


