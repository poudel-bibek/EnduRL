"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiAgentWaveAttenuationPOEnv

# Bibek
from flow.envs.multiagent.ring.density_aware_env import MultiAgentDensityAwareRLEnv 
from flow.envs.multiagent.density_aware_intersection_env import DensityAwareIntersectionEnv
from flow.envs.multiagent.density_aware_bottleneck_env import DensityAwareBottleneckEnv

from flow.envs.multiagent.ring.accel import AdversarialAccelEnv
from flow.envs.multiagent.ring.accel import MultiAgentAccelPOEnv
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv
from flow.envs.multiagent.merge import MultiAgentMergePOEnv
from flow.envs.multiagent.i210 import I210MultiEnv

__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv',
    'MultiAgentAccelPOEnv',
    'MultiAgentWaveAttenuationPOEnv',
    'MultiAgentMergePOEnv',
    'I210MultiEnv',
    'MultiAgentDensityAwareRLEnv',
    'DensityAwareIntersectionEnv',
    'DensityAwareBottleneckEnv',
]
