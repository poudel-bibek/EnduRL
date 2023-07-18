"""
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig, VehicleParams, InFlows, NetParams, TrafficLightParams
from flow.controllers import ContinuousRouter, SimLaneChangeController, RLController
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams

from flow.envs import DensityAwareBottleneckEnv
from flow.networks.poudel_bottleneck import PoudelBottleneckNetwork

HORIZON = 5000
N_CPUS = 1
N_ROLLOUTS = N_CPUS * 4 # Per iteration

vehicles = VehicleParams()

# fraction of AVs
AV_FRAC = 0.1
flow_rate = 5000 # vehicles per hour

# Randomly add some initializing human vehicles to initialize the simulation. 
# The vehicles have all lane changing enabled which is mode 1621

vehicles.add(
    veh_id="human",
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=7, # all_checks, 25, 31
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=1621,
    ),
    num_vehicles=1)

vehicles.add("rl",
    acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9, #same one as cathy
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1)

ADDITIONAL_NET_PARAMS = {
    "speed_limit": 20,
}

additional_env_params = {
    "target_velocity": 40,

    "max_accel": 3, # Instead of 1, -1
    "max_decel": 3,

    "lane_change_duration": 5,
    "add_rl_if_exit": False,
    "disable_tb": True,
    "disable_ramp_metering": True,
}

#ADDITIONAL_ENV_PARAMS = {}
additional_net_params = ADDITIONAL_NET_PARAMS.copy()

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="edge0",
    vehs_per_hour= (1- AV_FRAC) * flow_rate,
    depart_lane="random",
    depart_speed=10)

inflow.add(
    veh_type="rl",
    edge="edge0",
    vehs_per_hour= AV_FRAC * flow_rate,
    depart_lane="random",
    depart_speed=10)

traffic_lights = TrafficLightParams()

flow_params = dict(
    exp_tag = "DensityAwareBottleneck",

    env_name=DensityAwareBottleneckEnv,

    network=PoudelBottleneckNetwork,

    simulator='traci',

    sim = SumoParams(
                    sim_step=0.5, # For some reason, this is reduced 
                    render=True,
                    restart_instance=True
    ),

    env=EnvParams(
        warmup_steps=100, # original:set to 40
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    net = NetParams(
    inflows=inflow,
    additional_params=additional_net_params
    ),

    veh=vehicles,

    initial=InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        #edges_distribution=["2", "3", "4", "5"],
    ),

    tls=traffic_lights
)