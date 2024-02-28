
import numpy as np
from gym.spaces.box import Box
import random
from scipy.optimize import fsolve
from copy import deepcopy

import torch
import torch.nn as nn
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.multiagent.base import MultiEnv
from flow.envs.ring.wave_attenuation import v_eq_max_function


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 3,
    # maximum deceleration of autonomous vehicles
    'max_decel': 3,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}

class MultiAgentDensityAwareRLEnv(MultiEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

        self.LOCAL_ZONE = 50 # m, arbitrarily set
        self.VEHICLE_LENGTH = 5 #m can use self.k.vehicle.get_length(veh_id)
        self.MAX_SPEED = 10 # m/s
        self.estimated_free_speed = 0

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """
        What do the followers observe? 
        Just the 3 regular observations with the Full platoon leader (not thier subjective leader)
        """
        NUM_AUTOMATED = 12 # Change to 3, 8, 12 .. accordingly i.e. number of RL agents 
        return Box(low=-float('inf'), 
                   high=float('inf'), 
                   shape=((NUM_AUTOMATED+1)*2,), # Since there is one leader and we want the total platoon info.
                   dtype = np.float32)

    
    def _apply_rl_actions(self, rl_actions):
        """
        # In Multi agent ring, for followers, at test time, when the leader has found a stabilizing velocity, followers just stay stable.
        """
        
        #print(f"RL actions: {rl_actions} \n")
        for rl_id in self.k.vehicle.get_rl_ids():
            rl_action = rl_actions[rl_id]

            ##############
            # For Safety + Stability. Only present in test time. In multi agents, for followers.
            if self.k.vehicle.get_speed(rl_id) >= 0.98*self.estimated_free_speed: # 0.98 for 20%, 0.95 for 40%, 0.90 for 60%
                rl_action = 0.0

            self.k.vehicle.apply_acceleration(rl_id, rl_action)

        
    
    def compute_reward(self, rl_actions, **kwargs):
            """
            Followers get rewarded for following the leader closely.
            While getting punished for too much acceleration events and rewarded for average velocity
            """
            
            # in the warmup steps
            if rl_actions is None:
                return {}

            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in self.k.vehicle.get_ids()
            ])

            if any(vel < -100) or kwargs['fail']:
                return {}
            
            rew = {}

            # get all ids and filter with "leader" in it
            #trained_rl_id = [rl_id for rl_id in self.k.vehicle.get_ids() if "leader" in rl_id][0]

            # Distance to the leader of the platoon is minimized
            for rl_id in self.k.vehicle.get_rl_ids():
                acceleration_magnitude = np.abs(rl_actions[rl_id])[0]
                
                immediate_leader = self.k.vehicle.get_leader(rl_id) 
                #print("RL id: ", rl_id)

                # Penalize the relative distance to the platoon leader
                dist = -2*((self.k.vehicle.get_x_by_id(immediate_leader) - self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()) + 10 # offset to make ranges similar
                #print(f"Distance penalty: {dist} \n")

                # Penalize the retaive speed magnitude to the leader
                vel_diff = -4*np.abs(self.k.vehicle.get_speed(immediate_leader) - self.k.vehicle.get_speed(rl_id))
                #print(f"Velocity difference: {vel_diff} \n")

                # Penalize the acceleration magnitude
                accel_penalty = -4*acceleration_magnitude
                #print(f"Acceleration penalty: {accel_penalty}")

                reward_val = dist + vel_diff + accel_penalty
                #print(f"Reward: {reward_val} \n")

                rew[rl_id] = reward_val #
                
                
            #print(f"Rewards: {rew} \n")
            return rew

    def get_state(self):
        """
        This is simply for followers, who do not need the CSC output.
        """

        ########## First get the observations for the leader agent. Start by getting csc ouput ##########
        # This is the way lead is assigned
        # get all ids and filter with "leader" in it
        trained_rl_id = [rl_id for rl_id in self.k.vehicle.get_ids() if "leader" in rl_id][0]
        current_length = self.k.network.length()
        max_length = current_length 

        ########## Observations for followers ##########

        platoon_ids = []
        platoon_ids.append(trained_rl_id)
        platoon_ids.extend(self.k.vehicle.get_rl_ids())
        #print(f"Platoon ids: {platoon_ids} \n")

        obs = {}
        # For each follower RL agent, the observation is the entire platoon state.
        for rl_id in self.k.vehicle.get_rl_ids(): 

            rl_pos = self.k.vehicle.get_x_by_id(rl_id)
            rl_vel = self.k.vehicle.get_speed(rl_id)

            current_obs = []
            # Get pos and vel for all vehicles in the platoon
            for rl_id_again in platoon_ids:
                current_obs.append([(self.k.vehicle.get_x_by_id(rl_id_again) - rl_pos) % current_length/ max_length, 
                                    (self.k.vehicle.get_speed(rl_id_again) - rl_vel)/ self.MAX_SPEED])
                
            obs[rl_id] = np.array(current_obs, dtype = np.float32).flatten()
            #print(f"RL id:{rl_id} Observation: {obs[rl_id]}, {obs[rl_id].shape} \n")
        
        ##############
        # For Safety + Stability. Its fine to have this ON at all times.
        if self.step_counter > self.env_params.warmup_steps:
            estimate = self.k.vehicle.get_speed(trained_rl_id)
            if estimate >= self.estimated_free_speed:
                    self.estimated_free_speed = estimate

        #print(f"Total Observations: {obs} \n")
        return obs


    def reset(self, new_inflow_rate=None):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # skip if ring length is None
        if self.env_params.additional_params['ring_length'] is None:
            return super().reset()

        # reset the step counter
        self.step_counter = 0

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

        # solve for the velocity upper bound of the ring
        v_guess = 4
        v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
                          args=(len(self.initial_ids), length))[0]

        print('\n-----------------------')
        print('ring length:', net_params.additional_params['length'])
        print('v_max:', v_eq_max)
        print('-----------------------')

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)

        # perform the generic reset function
        return super().reset()

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""

        # specify observed vehicles
        current_length = self.k.network.length()
        trained_rl_id = [rl_id for rl_id in self.k.vehicle.get_ids() if "leader" in rl_id][0]
        vehicles_in_zone = self.k.vehicle.get_veh_list_local_zone(trained_rl_id, current_length, self.LOCAL_ZONE )
        for veh_id in vehicles_in_zone:
            self.k.vehicle.set_observed(veh_id)