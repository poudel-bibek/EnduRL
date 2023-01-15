"""
Environment for traditional models

"""
import numpy as np
from gym.spaces.box import Box
from flow.controllers import BCMController
from flow.envs.ring.accel import AccelEnv

class traditionalEnv(AccelEnv):
    """
    Docs here
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        self.warmup_steps = env_params.warmup_steps
        self.bcm_vehicles = [f"bcm_{str(i)}" for i in range(4)]

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(len(self.bcm_vehicles), ),
            dtype=np.float32)

    def step(self, rl_actions):
        """
        Docs here
        """
        print("step", self.step_counter)
       
        if self.step_counter == self.warmup_steps:
            veh_type = "bcm"
            for veh_id in self.bcm_vehicles:
                controller = (BCMController,{'v_des':4.8}) 
                self.k.vehicle.set_vehicle_type(veh_id, veh_type, controller)

        if self.step_counter >= self.warmup_steps:
            rl_actions = np.asarray([self.k.vehicle.get_acc_controller(veh_id).get_accel(self) for veh_id in self.bcm_vehicles])

        return super().step(rl_actions)

 
    def additional_command(self):
        # Dont set observed for traditional methods
        pass

    def compute_reward(self, rl_actions, **kwargs):
        # Prevent large negative rewards or else sim quits
        return 1