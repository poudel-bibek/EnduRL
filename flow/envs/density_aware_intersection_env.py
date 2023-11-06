import numpy as np
from flow.envs.base import Env


class DensityAwareIntersectionEnv(Env):
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

        pass

    def reset(self):
        """

        """

        pass

    def additional_command(self):
        """


        """
        pass