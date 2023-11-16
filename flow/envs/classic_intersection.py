"""
Intersection envinonment for the classic controllers in the intersection: 
BCM, LACC, FS, PI.
"""

import os 
import numpy as np
from gym.spaces.box import Box
from flow.controllers import BCMController, LACController, IDMController
from flow.controllers.controllers_for_daware import ModifiedIDMController
from flow.controllers.velocity_controllers import FollowerStopper, PISaturation
from flow.envs.base import Env
from flow.density_aware_util import get_shock_model, get_time_steps, get_time_steps_stability
from copy import deepcopy


# ADDITIONAL_ENV_PARAMS = {
#     # maximum acceleration for autonomous vehicles, in m/s^2
#     "max_accel": 3,
#     # maximum deceleration for autonomous vehicles, in m/s^2
#     "max_decel": 3,
# }

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,

    # if an RL vehicle exits, place it back at the front
    "add_classic_if_exit": True,
}

# Fully observed accel environment for intersection
class IntersectionAccelEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize BottleneckAccelEnv."""
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.add_classic_if_exit = env_params.get_additional_param("add_classic_if_exit")

        # Check the validity of this in the intersection
        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
        self.max_speed = self.k.network.max_speed()

    @property
    def observation_space(self):
        pass 

    @property
    def action_space(self):
        pass 
    
    def _apply_rl_actions(self, actions):
        pass

    def get_state(self):
        pass 

    def compute_reward(self, rl_actions, **kwargs):
        pass

    def additional_command(self):
        # reintroduce the vehicles that have exited
        pass


class classicIntersectionEnv(IntersectionAccelEnv):
    """
    Since the vehicles that enter and exit the network are chaning
    This needs to behave differently than the ring

    Vehicles of the classic type will have id with names flow_00.x 

    Not a closed network: Do we right away have the ids of all vehicles? No we only have the ids of the initially populated vehicles
    Make the initial population correctly (platoons)
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        self.warmup_steps = self.env_params.warmup_steps
        self.horizon = self.env_params.horizon

        #print(f"\n\n {self.network.name} \n")
        #network name is actually exp_tag (look at registry). A bit risky
        self.method_name = self.network.name.split('_')[1] # Gets is from the exp tag
        
        methods = ['bcm', 'lacc', 'idm', 'fs', 'piws']
        if self.method_name is None or self.method_name not in methods:
            raise ValueError("The 'method' argument is required and must be one of {}.".format(methods))
        
        self.control_dict = {'bcm': BCMController, 
                            'lacc': LACController, 
                            'idm': IDMController,
                            'fs': FollowerStopper,
                            'piws': PISaturation}
        
        # The controller to which to change to
        self.classic_controller = self.control_dict.get(self.method_name)

        self.shock_params = self.env_params.additional_params['shock_params']
        self.classic_params = self.env_params.additional_params['classic_params']

        # whether or not to shock
        self.shock = self.shock_params['shock']
        
        # when to start the shock
        self.shock_start_time = self.shock_params['shock_start_time'] 
        
        # when to end the shock
        self.shock_end_time = self.shock_params['shock_end_time'] 

        self.stability = self.shock_params['stability']
        
        # what model to use for the shock (intensity, duration, frequency)
        if self.stability:
            self.sm = get_shock_model(self.shock_params['shock_model'], self.network.net_params.additional_params["length"]) # This length is irrelevant here
        else: 
            self.sm = get_shock_model(self.shock_params['shock_model'], network_scaler=3, bidirectional=False, high_speed=False) 
        
         # count how many times, shock has been applies
        self.shock_counter = 0

        # Count duration of current shock (in timesteps)
        self.current_duration_counter = 0

        # Precise shock times
        if self.stability:
            self.shock_times = get_time_steps_stability(self.sm[1], self.sm[2], self.shock_start_time, self.shock_end_time)
        else:
            self.shock_times = get_time_steps(self.sm[1], self.sm[2], self.shock_start_time, self.shock_end_time)

        self.density_collector = []


    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(len(self.select_ids), ),
            dtype=np.float32)

    def step(self, rl_actions):
        """
        This is one central authority for all shocks, since each time step this is called once
        """ 
        if self.step_counter >= self.warmup_steps:
            veh_type = self.method_name
            all_vehicles = self.k.vehicle.get_ids()

            for veh_id in all_vehicles:
                # If vehicle IDs have _10 in them, they are classic going north, 
                # If they have _30 in them they are going south
                if 'flow_30.' in  veh_id or 'flow_10.' in  veh_id:
                    #print(f"Found classic vehicle: {veh_id}") 
                    if isinstance(self.k.vehicle.get_acc_controller(veh_id), ModifiedIDMController):
                        if 'classic_params' in self.env_params.additional_params:
                            controller = (self.classic_controller, \
                                self.env_params.additional_params['classic_params'])
                        
                        else:
                            controller = (self.classic_controller,{})
                        self.k.vehicle.set_vehicle_type(veh_id, veh_type, controller)

        # Shock 
        if self.shock and self.step_counter >= self.shock_start_time and self.step_counter <= self.shock_end_time:
            if self.stability:
                self.perform_shock_stability(self.shock_times)
            else: 
                self.perform_shock(self.shock_times)

        return super().step(rl_actions)

    def perform_shock(self, shock_times):
        """
        """
        pass

    def perform_shock_stability(self, shock_times):
        pass


    def additional_command(self):
        # Dont set observed for classic methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        # print("FAIL:", kwargs['fail'])
        return 1

