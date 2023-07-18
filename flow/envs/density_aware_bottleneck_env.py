import numpy as np
from flow.envs.base import Env
from gym.spaces.box import Box

# Copy this from another file?
ADDITIONAL_ENV_PARAMS = {}

class DensityAwareBottleneckEnv(Env):
    """

    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        """Initialize the BottleneckEnv class."""
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        super().__init__(env_params, sim_params, network, simulator)


    @property
    def observation_space(self):
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32) 

    @property
    def action_space(self):
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32) 

    def _apply_rl_actions(self, rl_actions):
        """
        In Cathy's work there are 2 actions
        1. Acceleration
        2. Lane change

        Multiple controllers: 
        1. Desired velocity controller (set the max velocity of a controlled segment?)
        2. Acceleration controller
        3. 
        """

        pass


    def compute_reward(self, rl_actions, **kwargs):
        """

        """

        veh_ids = self.k.vehicle.get_ids()
        speeds = self.k.vehicle.get_speed(veh_ids)
        avg_speed = np.mean(speeds)
        return avg_speed

    def get_state(self):
        """

        """

        return np.array([0.0])

    def reset(self):
        """
        For generating a policy that is "robust" to different inflow rates, the
        flow rate is sampled in a range and the config is modified each time
        """

        # Do not move 2 the lines from below
        observation = super().reset()
        self.time_counter = 0
        return observation

    def additional_command(self):
        """


        """
        pass

