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
    # not used
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
                            'idm': ModifiedIDMController, # For IDM keep it as modified IDM
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
            #self.sm = get_shock_model(-1, length=220) 
            self.sm = (np.asarray([3]), np.asarray([1]), 1) # 1 second means 10 timesteps
        else: 
            self.sm = get_shock_model(self.shock_params['shock_model'], network_scaler=3, bidirectional=False, high_speed=False) # high_speed = True for intersection
            #print(f"Shock model: {self.sm}")
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
        self.sample_vehicles = 4 # How many vehicles to shock at a time
        self.shock_ids = [] 
        
        # The inflows are set differently for IDM and other types
        # The HV flow in which controller has to be changed. For IDM all flows are humans so its essentially the same as shockable flow
        self.controller_type_flow = ['flow_10.', 'flow_00.'] if self.method_name == 'idm' else ['flow_30.', 'flow_10.']

        # The HV flow in which to shock
        self.shockable_flow = ['flow_10.', 'flow_00.'] if self.method_name == 'idm' else ['flow_20.', 'flow_00.']

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
            
            if self.method_name!= "idm": # Dont need to do this for IDM. All IDM types are already ModifiedIDMController types
                #veh_type = "human" if self.method_name =="idm" else self.method_name # Slightly different than others
                veh_type = self.method_name
                all_vehicles = self.k.vehicle.get_ids()

                for veh_id in all_vehicles:
                    # If vehicle IDs have _10 in them, they are classic AVs going north, 
                    # If they have _30 in them they are classic AVs going south
                    #if 'flow_30.' in  veh_id or 'flow_10.' in  veh_id:
                    if any(x in veh_id for x in self.controller_type_flow):
                        #print(f"Found classic vehicle: {veh_id}") 

                        # This basically converts ModifiedIDMController types from shockable_flow to another type 
                        if isinstance(self.k.vehicle.get_acc_controller(veh_id), ModifiedIDMController):
                            if 'classic_params' in self.env_params.additional_params:
                                controller = (self.classic_controller, \
                                    self.env_params.additional_params['classic_params'])
                            
                            else:
                                controller = (self.classic_controller,{})
                            self.k.vehicle.set_vehicle_type(veh_id, veh_type, controller)

            # Shock is also only after warmup
            if self.shock and self.step_counter >= self.shock_start_time and self.step_counter <= self.shock_end_time:
                if self.stability:
                    self.perform_shock_stability(self.shock_times)
                else: 
                    self.perform_shock(self.shock_times)

        return super().step(rl_actions)

    def get_fresh_shock_ids(self, ):
        #north_south_hv_ids = [veh_id for veh_id in self.k.vehicle.get_ids() if 'flow_20.' in  veh_id or 'flow_00.' in  veh_id]
        north_south_hv_ids = [veh_id for veh_id in self.k.vehicle.get_ids() if any(x in veh_id for x in self.shockable_flow)]
        current_shockable_vehicles  = []

        for veh_id in north_south_hv_ids: 
            #print(f"Our id: {veh_id}, Egde: {self.k.vehicle.get_edge(veh_id)}")
            # These vehicles are already not of the AV controller type. And are already in correct edges. # Due to the 'flow_20.' and 'flow_00.' filter
            # We want vehicles before they cross the intersection and cause errors. 
            # If a vehicle is in an outgoing edge `right1_0` or `left0_0` only allowed to shock if position is less than 65
            if self.k.vehicle.get_edge(veh_id) == 'left0_0' or self.k.vehicle.get_edge(veh_id) == 'right1_0':
                #print(f"Veh id: {veh_id}, Edge: {self.k.vehicle.get_edge(veh_id)}, Position: {self.k.vehicle.get_position(veh_id)}")

                if self.k.vehicle.get_position(veh_id) < 165:
                    current_shockable_vehicles.append(veh_id)

            # Else if the vehicle is in an incoming egde `left1_0` or `right0_0` only allowed to shock if position is greater than 135
            elif self.k.vehicle.get_edge(veh_id) == 'left1_0' or self.k.vehicle.get_edge(veh_id) == 'right0_0':
                #print(f"Veh id: {veh_id}, Edge: {self.k.vehicle.get_edge(veh_id)}, Position: {self.k.vehicle.get_position(veh_id)}")
                if self.k.vehicle.get_position(veh_id) > 135:
                    current_shockable_vehicles.append(veh_id)

            else: 
                # If a vehicle is in the center, just add it
                current_shockable_vehicles.append(veh_id)

        # Now randomly select sample_vehicle number vehicles
        shock_ids = np.random.choice(current_shockable_vehicles, self.sample_vehicles, replace=False)
        #print(f"Shockable ids: {current_shockable_vehicles}")
        #print(f"Shock ids: {shock_ids}")
        return shock_ids
    
    def perform_shock(self, shock_times):
        """
        Human driven vehicles north, southbound perform shock.
        Can differentiate human driven vehicles from id. flow_20. and flow_00. are human driven vehicles north/ southbound
        
        Shock modality: shock XX vehicles at a time

        """

        # if vehicle IDs have a 'classic_00' in it, then its a classic vehicle
        if self.step_counter == shock_times[0][0]: # This occurs only once
            self.shock_ids = self.get_fresh_shock_ids()

        # Get controllers and set shock time to False for selected vehicles (default behavior)
        controllers = [self.k.vehicle.get_acc_controller(i) for i in self.shock_ids]
        for controller in controllers:
            controller.set_shock_time(False)

        # Increment shock counter only after shock for the duration was complete
        if self.shock_counter < len(self.sm[1]) and self.current_duration_counter >= self.sm[1][self.shock_counter]:
            self.shock_counter += 1

            # reset 
            self.current_duration_counter = 0
            self.shock_ids = self.get_fresh_shock_ids()

        # sm[1] is a list of intensities, sm[2] is a list of durations and sm[3] is frequency
        # if the shock counter is less than the frequency 
        if self.shock_counter < self.sm[2]: # '<' because shock counter starts from zero
            # if the we are in the precomputed shock times 
            if self.step_counter >= shock_times[self.shock_counter][0] and \
                self.step_counter <= shock_times[self.shock_counter][1]:
                print(f"Step = {self.step_counter}, Shock params: {self.sm[0][self.shock_counter], self.sm[1][self.shock_counter], self.sm[2]} applied to vehicle {self.shock_ids}\n")
                
                # Set shock time to True for selected vehicles
                for controller in controllers:
                    controller.set_shock_accel(self.sm[0][self.shock_counter])
                    controller.set_shock_time(True)

                # Change color to magenta
                for i in self.shock_ids:
                    self.k.vehicle.set_color(i, (255,0,255))
                
                self.current_duration_counter += 0.1 # increment current duration counter by one timestep seconds


    def perform_shock_stability(self, shock_times):
        """
        Just check the steps and shock the first one after initial population
        This is only called when the shock time starts anyway.
        """

        # if the vehicle id is flow_10.1 then shock it i.e., have its leader perform a velocity perturbation
        # Leader is flow_00.4 and follower is flow_00.5
        if self.step_counter == shock_times[0][0]: # This occurs only once
            self.shock_ids = ['flow_00.4']

        # Get controllers and set shock time to False for selected vehicles (default behavior)
        controllers = [self.k.vehicle.get_acc_controller(i) for i in self.shock_ids]
        for controller in controllers:
            controller.set_shock_time(False)

        # Increment shock counter only after shock for the duration was complete
        if self.shock_counter < len(self.sm[1]) and self.current_duration_counter >= self.sm[1][self.shock_counter]:
            self.shock_counter += 1

            # reset
            # Some things missing here because we only apply shocks once 
            self.k.vehicle.set_max_speed(self.shock_ids[0], 8) #V_enter
            
        if self.shock_counter < self.sm[2]: # '<' because shock counter starts from zero
            # if the we are in the precomputed shock times 
            if self.step_counter >= shock_times[self.shock_counter][0] and \
                self.step_counter <= shock_times[self.shock_counter][1]:

                print(f"Step = {self.step_counter}, Shock params: {self.sm[0][self.shock_counter], self.sm[1][self.shock_counter], self.sm[2]} applied to vehicle {self.shock_ids}\n")
                
                # Set shock time to True for selected vehicles
                for controller in controllers:
                    # The stability shock is a velocity perturbation
                    self.k.vehicle.set_max_speed(self.shock_ids[0], self.sm[0][0])
                    controller.set_shock_time(True)

                # Change color to magenta
                for i in self.shock_ids:
                    self.k.vehicle.set_color(i, (255,0,255))
                
                self.current_duration_counter += 0.1 # increment current duration counter by one timestep seconds


    def additional_command(self):
        # Dont set observed for classic methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        # print("FAIL:", kwargs['fail'])
        return 1

