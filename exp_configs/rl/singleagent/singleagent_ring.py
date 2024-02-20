"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.networks import RingNetwork

# Bibek: Load our custom environment
from flow.envs import DensityAwareRLEnv 

# time horizon of a single rollout
HORIZON = 4500 #20 For CSC #Bibek (for our purposes, 6000 total is good)
# number of rollouts per training iteration
N_ROLLOUTS = 5
# number of parallel workers
N_CPUS = 8

# We place one autonomous vehicle and 22 human-driven vehicles in the network
# Following Wu el. al. , humans have a minGap of 0.0 and RL has a default (2.5)
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.1 # Collisions are allowed at 0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="density_aware_rl",

    # name of the flow environment the experiment is running on
    env_name=DensityAwareRLEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=2500, #Bibek (for training only) # For CSC data collection, make this 3500
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [260, 260], # Do we train at various lengths?
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
