"""
Environment for traditional models

"""
import numpy as np
from gym.spaces.box import Box
from flow.controllers import BCMController, LACController, IDMController
from flow.controllers.velocity_controllers import FollowerStopper, PISaturation
from flow.envs.ring.accel import AccelEnv

class traditionalEnv(AccelEnv):
    """
    Docs here
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        self.warmup_steps = env_params.warmup_steps

        #network name is actually exp_tag (look at registry). A bit risky
        self.method_name = self.network.name.split('_')[0]

        methods = ['bcm', 'lacc', 'idm','fs','piws']
        if self.method_name is None or self.method_name not in methods:
            raise ValueError("The 'method' argument is required and must be one of {}.".format(methods))
        
        # Set the vehicles that are controlled by the method
        self.select_ids = [veh_id for veh_id in self.network.vehicles.ids\
             if self.method_name in veh_id] #replace filter with a lambda function?

        self.control_dict = {'bcm': BCMController, 
                            'lacc': LACController, 
                            'idm': IDMController,
                            'fs': FollowerStopper,
                            'piws': PISaturation}
                            
        self.traditional_controller = self.control_dict.get(self.method_name)

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
        """
        #print("step", self.step_counter)
        
        if self.step_counter == self.warmup_steps:
            veh_type = self.method_name
            for veh_id in self.select_ids:
                if 'traditional_params' in self.env_params.additional_params:
                    controller = (self.traditional_controller, \
                        self.env_params.additional_params['traditional_params'])
                else:
                    controller = (self.traditional_controller,{})
                self.k.vehicle.set_vehicle_type(veh_id, veh_type, controller)

        if self.step_counter >= self.warmup_steps:
            rl_actions = [self.k.vehicle.get_acc_controller(veh_id).get_accel(self)\
                 for veh_id in self.select_ids]
            # Sometimes in FS, rl_actions can be None:
            # Under these conditions accelerations may not be useful and a human supervision would be best
            # We set these conditions to 0 acceleration
            # Although the simulation requires RL actions to be None during warmup.. and uses this as an identifier sometimes,
            # After warmup, it is rarely so
            # rl_actions = np.nan_to_num(rl_actions, nan=0.0) 
            if None in rl_actions:
                print(f"\nWARNING: acceleration = None obtained after warmup, at timestep {self.step_counter}\n")
            rl_actions = np.asarray([float(i) if i is not None else 0.0 for i in rl_actions])
            
        #print("RL actions: ",rl_actions)
        return super().step(rl_actions)

 
    def additional_command(self):
        # Dont set observed for traditional methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        return 1