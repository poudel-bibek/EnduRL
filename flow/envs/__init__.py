"""Contains all callable environments in Flow."""
from flow.envs.base import Env
from flow.envs.bay_bridge import BayBridgeEnv
from flow.envs.bottleneck import BottleneckAccelEnv, BottleneckEnv, \
    BottleneckDesiredVelocityEnv
from flow.envs.traffic_light_grid import TrafficLightGridEnv, \
    IntersectionRLPOEnv, TrafficLightGridTestEnv, TrafficLightGridBenchmarkEnv
from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.wave_attenuation import WaveAttenuationEnv, \
    WaveAttenuationPOEnv

#Bibek
from flow.envs.ring.density_aware_classic_env import classicEnv 
from flow.envs.ring.density_aware_env import DensityAwareRLEnv 
from flow.envs.classic_bottleneck import classicBottleneckEnv
from flow.envs.classic_intersection import classicIntersectionEnv

from flow.envs.merge import MergePOEnv
from flow.envs.test import TestEnv

# deprecated classes whose names have changed
from flow.envs.bottleneck_env import BottleNeckAccelEnv
from flow.envs.bottleneck_env import DesiredVelocityEnv
#from flow.envs.green_wave_env import PO_TrafficLightGridEnv
from flow.envs.green_wave_env import GreenWaveTestEnv


__all__ = [
    'Env',
    'AccelEnv',
    'LaneChangeAccelEnv',
    'LaneChangeAccelPOEnv',
    'TrafficLightGridTestEnv',
    'MergePOEnv',
    'BottleneckEnv',
    'BottleneckAccelEnv',
    'WaveAttenuationEnv',
    'WaveAttenuationPOEnv',
    'DensityAwareRLEnv', #Bibek
    'classicEnv', #Bibek
    'classicBottleneckEnv', #Bibek
    'classicIntersectionEnv', #Bibek
    'TrafficLightGridEnv',
    'IntersectionRLPOEnv', #Bibek
    'TrafficLightGridBenchmarkEnv',
    'BottleneckDesiredVelocityEnv',
    'TestEnv',
    'BayBridgeEnv',
    # deprecated classes
    'BottleNeckAccelEnv',
    'DesiredVelocityEnv',
    'PO_TrafficLightGridEnv',
    'GreenWaveTestEnv',
]
