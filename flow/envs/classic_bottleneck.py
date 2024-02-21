"""
Environment for classic controllers to run in the bottleneck environment

"""
import os 
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces.box import Box
from flow.controllers import BCMController, LACController, IDMController
from flow.controllers.controllers_for_daware import ModifiedIDMController
from flow.controllers.velocity_controllers import FollowerStopper, PISaturation
from flow.envs.base import Env
from flow.density_aware_util import get_shock_model, get_time_steps, get_time_steps_stability
from copy import deepcopy

# Check if this is even used anywhere?
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

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 15,
    # if an RL vehicle exits, place it back at the front
    "add_classic_if_exit": True,
}

# Fully observed accel environment for bottleneck
class BottleneneckAccelEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize BottleneckAccelEnv."""
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)
        self.add_classic_if_exit = env_params.get_additional_param("add_classic_if_exit")
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


class classicBottleneckEnv(BottleneneckAccelEnv):
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
        self.method_name = self.network.name.split('_')[2] # Gets is from the exp tag
        
        methods = ['bcm', 'lacc', 'idm', 'fs', 'piws']
        if self.method_name is None or self.method_name not in methods:
            raise ValueError("The 'method' argument is required and must be one of {}.".format(methods))
        
        self.control_dict = {'bcm': BCMController, 
                            'lacc': LACController, 
                            'idm': IDMController,
                            'fs': FollowerStopper,
                            'piws': PISaturation}

        # Probably dont need this
        self.num_controlled_dict = {'bcm': 4,
                                    'lacc': 9,
                                    'idm': 1,
                                    'fs': 1,
                                    'piws': 1}
        
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
            self.sm = get_shock_model(self.shock_params['shock_model'], network_scaler=2, bidirectional=True, high_speed=False) 
        
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
        # Dont allow '5' to avoid vehicles exiting from the network cause a NoneType error
        self.edges_allowed_list = ['3', '4_0', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7',  '4'] # 5_0, 5_1, and 5_2 are not allowed
        self.threshold_speed = 20.0 #0.5 # m/s # Just to make sure they are not actually stopped
        self.sample_vehicles = 4 # Number of vehicles to simultaneously shock in the network

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

        # All inflows are initialized as ModifiedIDMController
        # At warmup, or at later timesteps change vehicle type to method types as soon as they are spawned
        if self.step_counter >= self.warmup_steps:
            veh_type = self.method_name
            all_vehicles = self.k.vehicle.get_ids() # This list updates itself to vehicles that are currently in the network
            #print(f"\n\nAll vehicles: {all_vehicles}\n\n")
            for veh_id in all_vehicles:

                # First convert this vehicle
                if 'classic_00' in veh_id and isinstance(self.k.vehicle.get_acc_controller(veh_id), ModifiedIDMController):
                    # Inject parameters (if any) for classic controllers
                    if 'classic_params' in self.env_params.additional_params:
                        controller = (self.classic_controller, \
                            self.env_params.additional_params['classic_params'])
                    else:
                        controller = (self.classic_controller,{})

                    self.k.vehicle.set_vehicle_type(veh_id, veh_type, controller)

                    # Uncomment this if you want to operate the vehicles at min. no of vehicles required to stabilize in the ring.
                    # i.e., whenever, a vehicle in encountered. Converet 1 or 4 or 9 vehicles in front of it as well.
                    # requires a check if its not already a classic vehicle (i.e., not a ModifiedIDMController)

                    # current_vehicle = veh_id
                    # Then convert the (num_controlled - 1) vehicles in front of it
                    # for i in range(self.num_controlled_dict[self.method_name] - 1):
                    #     leader = self.k.vehicle.get_leader(current_vehicle)
                    #     if leader is not None and isinstance(self.k.vehicle.get_acc_controller(leader), ModifiedIDMController): # The leader may not be of type "classic_00" may be from another flow
                    #         # Inject parameters (if any) for classic controllers
                    #         if 'classic_params' in self.env_params.additional_params:
                    #             controller = (self.classic_controller,
                    #                 self.env_params.additional_params['classic_params'])
                    #         else:
                    #             controller = (self.classic_controller,{})
                    #         self.k.vehicle.set_vehicle_type(leader, veh_type, controller)
                    #     current_vehicle = leader 

                    # Uncomment this for FS
                    # Not sure if this will work here (because vehicle type is just changed)
                    # Sometimes (rarely) in FS, acceleration values can be None 
                    # These are the times when accelerations form the controller may not be useful and a human supervision would be best
                    # Under these conditions, we set acceleration to 0
                    # rl_actions = [self.k.vehicle.get_acc_controller(veh_id).get_accel(self)]
                    # if None in rl_actions:
                    #     print(f"\nWARNING: acceleration = None obtained after warmup, at timestep {self.step_counter}\n")
                    # rl_actions = np.asarray([float(i) if i is not None else 0.0 for i in rl_actions])

        # Density 
        density = len(self.k.vehicle.get_ids()) / (self.k.network.length()/1000)
        self.density_collector.append(density)
        # plot
        if self.step_counter == (self.warmup_steps + self.horizon -1):
            plt.plot(self.density_collector)
            plt.xlabel("Time Steps")
            plt.ylabel("Density")
            plt.title(f"Density profile for {self.method_name} controller, peak={round(max(self.density_collector),2)}")
            if not os.path.exists("plots"):
                os.mkdir("plots")
            plt.savefig(f"plots/density_{self.method_name}.png")
            plt.close()

        # Shock 
        if self.shock and self.step_counter >= self.shock_start_time and self.step_counter <= self.shock_end_time:
            if self.stability:
                self.perform_shock_stability(self.shock_times)
            else: 
                self.perform_shock(self.shock_times)
        
        return super().step(rl_actions)

    def perform_shock(self, shock_times):
        """
        The flow of vehicles (3600 veh/hr) is higher than in the ring.
        The density here is 3x higher than the ring, the effective number of lanes is also about 6x higher
        The number of vehicles in the period of interest is also much higher than the ring (difficult to measure exactly because of gradual increase of flow)

        Hence, at one timestep 4 vehicles are simultaneously shocked
        And the shocks are samples 6x more frequently than the ring

        To shock: 
            - The vehicle must be of the type ModifiedIDMController
            - Vehicles must be selected from the current list of available vehicles
            - Vehicle must be moving at a speed greater than 1 m/s
        """

        all_ids = self.k.vehicle.get_ids()

        # if vehicle IDs have a 'classic_00' in it, then its a classic vehicle
        if self.step_counter == shock_times[0][0]: # This occurs only once
            #current_shockable_vehicle_ids = [i for i in all_ids if 'classic_00' not in i and self.k.vehicle.get_edge(i) in self.edges_allowed_list and self.k.vehicle.get_speed(i) > self.threshold_speed] 
            current_shockable_vehicle_ids = [i for i in all_ids if 'classic_00' not in i and self.k.vehicle.get_edge(i) in self.edges_allowed_list and self.k.vehicle.get_speed(i) < self.threshold_speed] # and self.k.vehicle.get_leader(i) is not None]
            
            self.shock_ids = np.random.choice(current_shockable_vehicle_ids, self.sample_vehicles)
            print(f"\n\nShock ids: {self.shock_ids}\n\n")
        
        if len(self.shock_ids) > 0:
            controllers = [self.k.vehicle.get_acc_controller(i) for i in self.shock_ids]
            for controller in controllers:
                controller.set_shock_time(False)

        # Reset duration counter and increase shock counter, after completion of shock duration
        # Since the sim step here is 0.5, applying shock for 2 timesteps means 1 second
        # The durations can be anywhere between 0.1 to 2.5 at intervals of 0.1 but the current duration counter does not increment that way (make it >=)
        if self.shock_counter < len(self.sm[1]) and self.current_duration_counter >= self.sm[1][self.shock_counter]: 
            self.shock_counter += 1
            self.current_duration_counter = 0

            #current_shockable_vehicle_ids = [i for i in all_ids if 'classic_00' not in i and self.k.vehicle.get_edge(i) in self.edges_allowed_list and self.k.vehicle.get_speed(i) > self.threshold_speed]
            current_shockable_vehicle_ids = [i for i in all_ids if 'classic_00' not in i and self.k.vehicle.get_edge(i) in self.edges_allowed_list and self.k.vehicle.get_speed(i) < self.threshold_speed ]# and self.k.vehicle.get_leader(i) is not None]
            self.shock_ids = np.random.choice(current_shockable_vehicle_ids, self.sample_vehicles)
            print(f"\n\nShock ids: {self.shock_ids}\n\n")

        if self.shock_counter < self.sm[2]: # '<' because shock counter starts from zero
            if self.step_counter == shock_times[self.shock_counter][0]:
                current_shockable_vehicle_ids = [i for i in all_ids if 'classic_00' not in i and self.k.vehicle.get_edge(i) in self.edges_allowed_list and self.k.vehicle.get_speed(i) < self.threshold_speed] # and self.k.vehicle.get_leader(i) is not None]
                for controller in controllers:
                    controller.set_shock_accel(self.sm[0][self.shock_counter])
                    controller.set_shock_time(True)

            if self.step_counter >= shock_times[self.shock_counter][0] and \
                self.step_counter <= shock_times[self.shock_counter][1]:
                print(f"Step = {self.step_counter}, Shock params: {self.sm[0][self.shock_counter], self.sm[1][self.shock_counter], self.sm[2]} applied to vehicle {self.shock_ids}\n")
                
                for controller in controllers:
                    controller.set_shock_accel(self.sm[0][self.shock_counter])
                    controller.set_shock_time(True)

                # Change color to magenta
                for i in self.shock_ids:
                    self.k.vehicle.set_color(i, (255,0,255))

                self.current_duration_counter += 0.5 # increment current duration counter by one timestep seconds # sim_step = 0.5

            if self.step_counter == shock_times[self.shock_counter][1]:
                self.shock_ids = []

    def perform_shock_stability(self, shock_times):
        pass


    def additional_command(self):
        # Dont set observed for classic methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        # print("FAIL:", kwargs['fail'])
        return 1

