"""
Environment for traditional models

"""
import numpy as np
from gym.spaces.box import Box
from flow.controllers import BCMController, LACController, IDMController
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

        methods = ['bcm', 'lacc', 'idm']
        if self.method_name is None or self.method_name not in methods:
            raise ValueError("The 'method' argument is required and must be one of {}.".format(methods))
        
        # Set the vehicles that are controlled by the method
        self.select_ids = [veh_id for veh_id in self.network.vehicles.ids\
             if self.method_name in veh_id] #replace filter with a lambda function?

        self.control_dict = {'bcm': BCMController, 
                            'lacc': LACController, 
                            'idm': IDMController}
                            
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
        print("step", self.step_counter)
        
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
            rl_actions = np.asarray([self.k.vehicle.get_acc_controller(veh_id).get_accel(self)\
                 for veh_id in self.select_ids])

        return super().step(rl_actions)

 
    def additional_command(self):
        # Dont set observed for traditional methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        return 1