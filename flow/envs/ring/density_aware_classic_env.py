"""
Environment for classic models

"""
import numpy as np
from gym.spaces.box import Box
from flow.controllers import BCMController, LACController, IDMController
from flow.controllers.velocity_controllers import FollowerStopper, PISaturation
from flow.envs.ring.accel import AccelEnv

from flow.density_aware_util import get_shock_model, get_time_steps, get_time_steps_stability

class classicEnv(AccelEnv):
    """
    Docs here
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        self.warmup_steps = self.env_params.warmup_steps

        #network name is actually exp_tag (look at registry). A bit risky
        self.method_name = self.network.name.split('_')[0]
        
        methods = ['bcm', 'lacc', 'idm','fs','piws']
        if self.method_name is None or self.method_name not in methods:
            raise ValueError("The 'method' argument is required and must be one of {}.".format(methods))
        
        self.all_ids = [veh_id for veh_id in self.network.vehicles.ids]
        
        # Set the vehicles that are controlled by the method
        self.select_ids = [veh_id for veh_id in self.all_ids\
             if self.method_name in veh_id] #replace filter with a lambda function?
        
        if self.method_name =='idm':
            # In case of IDM, all 22 vehicles are IDM
            self.other_ids = self.all_ids
            self.select_ids = [] # Because we dont want to change vehicle type at warmup end
    
        else: 
            # For other classic controllers, remaining ids i.e., not controlled by the method
            self.other_ids = [veh_id for veh_id in self.all_ids if veh_id not in self.select_ids]

        self.control_dict = {'bcm': BCMController, 
                            'lacc': LACController, 
                            'idm': IDMController,
                            'fs': FollowerStopper,
                            'piws': PISaturation}
                            
        self.classic_controller = self.control_dict.get(self.method_name)
        self.shock_params = self.env_params.additional_params['shock_params']

        # whether or not to shock
        self.shock = self.shock_params['shock']
        
        # when to start the shock
        self.shock_start_time = self.shock_params['shock_start_time'] 
        
        # when to end the shock
        self.shock_end_time = self.shock_params['shock_end_time'] 

        self.stability = self.shock_params['stability']
        
        # what model to use for the shock (intensity, duration, frequency)
        if self.stability:
            self.sm = get_shock_model(self.shock_params['shock_model'], self.network.net_params.additional_params["length"])
        else: 
            self.sm = get_shock_model(self.shock_params['shock_model'], bidirectional=True)
            
        # count how many times, shock has been applies
        self.shock_counter = 0

        # Count duration of current shock (in timesteps)
        self.current_duration_counter = 0

        # Get the shock id of a single vehicle, randomly shuffled every time a shock is applied (not every time-step)
        self.single_shock_id = [] 

        # Precise shock times
        if self.stability:
            self.shock_times = get_time_steps_stability(self.sm[1], self.sm[2], self.shock_start_time, self.shock_end_time)
        else:
            self.shock_times = get_time_steps(self.sm[1], self.sm[2], self.shock_start_time, self.shock_end_time)

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
        Docs here
        This is one central authority for all shocks, since each time step this is called once
        """
        # TODO: Find a less embarassing way to do this
        # Tried inheriting from AccelEnv, Base and overriding parent functions, but that didn't work
        if self.step_counter == 0:
            if self.shock:

                # All vehicles that are not controller vehicles have the ability to shock
                self.shock_veh_ids = [veh_id for veh_id in self.other_ids \
                    if self.k.vehicle.get_acc_controller(veh_id).shock_vehicle == True]
                
                # Randomly choose one to perform shock, for a start
                self.single_shock_id = np.random.choice(self.shock_veh_ids, 1 )[0]

                if self.shock_veh_ids == []:
                    raise ValueError("No shock vehicles found")
        
        # Between shock start and end times, perform shock
        if self.shock and self.step_counter >= self.shock_start_time and self.step_counter <= self.shock_end_time: #<= is fine, handeled in perform_shock
            if self.stability:
                self.perform_shock_stability(self.shock_times)
            else: 
                self.perform_shock(self.shock_times)

        # At warmup, change vehicle type from all IDM to (method types)
        if self.step_counter == self.warmup_steps:
            veh_type = self.method_name
            for veh_id in self.select_ids:

                # Inject parameters (if any) for classic controllers 
                if 'classic_params' in self.env_params.additional_params:
                    controller = (self.classic_controller, \
                        self.env_params.additional_params['classic_params'])
                else:
                    controller = (self.classic_controller,{})

                # Use the funtion we wrote in Traci 
                self.k.vehicle.set_vehicle_type(veh_id, veh_type, controller)

        if self.step_counter >= self.warmup_steps:
            rl_actions = [self.k.vehicle.get_acc_controller(veh_id).get_accel(self)\
                 for veh_id in self.select_ids]

            # Sometimes (rarely) in FS, acceleration values can be None 
            # These are the times when accelerations form the controller may not be useful and a human supervision would be best
            # Under these conditions, we set acceleration to 0
          
            if None in rl_actions:
                print(f"\nWARNING: acceleration = None obtained after warmup, at timestep {self.step_counter}\n")
            rl_actions = np.asarray([float(i) if i is not None else 0.0 for i in rl_actions])
            
        return super().step(rl_actions)

    def perform_shock(self, shock_times):
        # Facts: We can only set intended acceleration, actual (realized) acceleration is computed by the simulator
        #print(f"Step: {self.step_counter}, Shock counter: {self.shock_counter}")

        # This is instantiated for every veh_id, we get for just the vehicle we selected as the shock vehicle
        controller = self.k.vehicle.get_acc_controller(self.single_shock_id) 
        # change color to white
        self.k.vehicle.set_color(self.single_shock_id, (255, 255, 255))
        
        # Default: at times when shock is not applied, get acceleration from IDM
        controller.set_shock_time(False) 

        # Reset duration counter and increase shock counter, after completion of a shock duration
        if self.current_duration_counter == self.sm[1][self.shock_counter]*10:
            self.shock_counter += 1
            self.current_duration_counter = 0

            #print("\n\nRESET COUNTER\n\n")

            # Random selection for next duration
            # Incase we want to set probabilities in the future, the line below can provide that as well
            self.single_shock_id = np.random.choice(self.shock_veh_ids, 1 )[0] 
            
        if self.shock_counter < self.sm[2]: # '<' because Shock counter starts counting from 0, sm[2] is the number of shocks
            if self.step_counter >= shock_times[self.shock_counter][0] and \
                self.step_counter <= shock_times[self.shock_counter][1]:
                print(f"Step = {self.step_counter}, Shock params: {self.sm[0][self.shock_counter], self.sm[1][self.shock_counter], self.sm[2]} applied to vehicle {self.single_shock_id}\n")
                
                controller.set_shock_accel(self.sm[0][self.shock_counter])
                controller.set_shock_time(True)

                # change color to magenta
                self.k.vehicle.set_color(self.single_shock_id, (255, 0, 255))
                self.current_duration_counter += 1
        
    # For stability
    def perform_shock_stability(self, shock_times):
        # Shock_time for each ModifiedIDM controller is set to False by default 
        # We manipulate the speed of the vehicle instead
        # Since we only want to vary the speed of the leader (human_0), no need to make use of random choice in single_shock_id
        if self.method_name =='idm':
            self.single_shock_id = 'idm_0'
            speed_limit = self.k.vehicle.get_max_speed('idm_1')

        else: 
            self.single_shock_id = 'human_0'
            
            # velocity shock model 
            # dip_velocity, duration, frequency = self.sm
            speed_limit = self.k.vehicle.get_max_speed('human_1')# Hacky, get the speed limit of other vehicles instead

        # When the velocity is not being dip, the max speed is set to speed limit (get from net_params, additional_params).
        self.k.vehicle.set_max_speed(self.single_shock_id, speed_limit) 

        # Reset duration counter and increase shock counter, after completion of a shock duration
        if self.current_duration_counter == self.sm[1]*10:
            self.shock_counter += 1
            self.current_duration_counter = 0
        
        if self.shock_counter < self.sm[2]: # '<' because Shock counter starts counting from 0, sm[2] is the number of shocks
            if self.step_counter >= shock_times[self.shock_counter][0] and \
                self.step_counter <= shock_times[self.shock_counter][1]:
                print(f"Step = {self.step_counter}, Shock params: {self.sm[0], self.sm[1], self.sm[2]} applied to vehicle {self.single_shock_id}\n")
                
                self.k.vehicle.set_max_speed(self.single_shock_id, self.sm[0])

                self.current_duration_counter += 1

    def additional_command(self):
        # Dont set observed for classic methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        # print("FAIL:", kwargs['fail'])
        return 1
