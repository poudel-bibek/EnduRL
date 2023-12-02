"""
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig, VehicleParams, InFlows, NetParams, TrafficLightParams
from flow.controllers import ContinuousRouter, SimLaneChangeController, RLController, IDMController
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams

from flow.envs import DensityAwareBottleneckEnv
#from flow.networks.poudel_bottleneck import PoudelBottleneckNetwork
from flow.networks import BottleneckNetwork

# time horizon of a single rollout
HORIZON = 3600 # 3600

# number of parallel workers
N_CPUS = 8 # Instead of 8 at 1500. It will do a +1 later so actually its 9
# The Terminal log will count the total timesteps so it will show something like 180000 = (3600 + 4000)/2 * 50 
# It is specidied in Table 3 of the benchmarks paper that 50 rollouts per iteration 
# Also a batch size of 80000: 3600 timesteps per rollout, total = (3600/2) * 45 = 81000 # Close enough 
N_ROLLOUTS = N_CPUS * 4 # 45 rollouts per iteration

SCALING = 2 # The paper mentions N should be 3 with inflow = 3800 but in code, N is 2
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = .10

vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}), #IDMController
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=9.0,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id="human",
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(

    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)

controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    "target_velocity": 40,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "reset_inflow": True,
    "lane_change_duration": 5,
    "add_rl_if_exit": True,
    "max_accel": 3,
    "max_decel": 3,
    "inflow_range": [1200 * SCALING, 2500 * SCALING] # So that there is some randomness in the whole thing. Generalizes better
}

# flow rate
flow_rate = 1800 * SCALING

# percentage of flow coming out of each lane
inflow = InFlows()
inflow.add(
    veh_type="rl",
    edge="1",
    vehs_per_hour=flow_rate * AV_FRAC,
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    departLane="random",
    departSpeed=10)

traffic_lights = TrafficLightParams()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING, "speed_limit": 23}
net_params = NetParams(
    inflows=inflow,
    additional_params=additional_net_params)

flow_params = dict(
    # name of the experiment
    exp_tag="bottleneck_ours",

    # name of the flow environment the experiment is running on
    env_name=DensityAwareBottleneckEnv,

    # name of the network class the experiment is running on
    network=BottleneckNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1, 
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps= 4000, # 4000
        sims_per_step=1, # Sims per step huh
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=traffic_lights,
)











################################
# HORIZON = 3600
# N_CPUS = 1 # Increase this later
# N_ROLLOUTS = N_CPUS * 4 # Per iteration

# SCALING = 2 
# vehicles = VehicleParams()
# # fraction of AVs
# AV_FRAC = 0.1


# # Randomly add some initializing human vehicles to initialize the simulation. 
# # The vehicles have all lane changing enabled which is mode 1621

# vehicles.add(
#     veh_id="human",
#     lane_change_controller=(SimLaneChangeController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         speed_mode=7, # all_checks, 25, 31
#     ),
#     lane_change_params=SumoLaneChangeParams(
#         lane_change_mode=1621,
#     ),
#     num_vehicles=1)

# vehicles.add("rl",
#     acceleration_controller=(RLController, {}),
#     lane_change_controller=(SimLaneChangeController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     car_following_params=SumoCarFollowingParams(
#         speed_mode=9, #same one as cathy
#     ),
#     lane_change_params=SumoLaneChangeParams(
#         lane_change_mode=0,
#     ),
#     num_vehicles=1)

# ADDITIONAL_NET_PARAMS = {
#     "speed_limit": 23,
#     "scaling": SCALING,
# }

# additional_env_params = {
#     "target_velocity": 40,
#     "max_accel": 3, # Instead of 1, -1
#     "max_decel": 3,
#     "lane_change_duration": 5,
#     "add_rl_if_exit": True,
#     "disable_tb": True,
#     "disable_ramp_metering": True,
#     "inflow_range": [1200 * SCALING, 2500 * SCALING] # Variability in inflow during training
# }

# #ADDITIONAL_ENV_PARAMS = {}
# additional_net_params = ADDITIONAL_NET_PARAMS.copy()

# flow_rate = 1800 * SCALING # vehicles per hour

# inflow = InFlows()
# inflow.add(
#     veh_type="human",
#     edge="1", # This has to be edge0 in poudelbottleneck
#     vehs_per_hour= (1- AV_FRAC) * flow_rate,
#     depart_lane="random",
#     depart_speed=10)

# inflow.add(
#     veh_type="rl",
#     edge="1",
#     vehs_per_hour= AV_FRAC * flow_rate,
#     depart_lane="random",
#     depart_speed=10)

# traffic_lights = TrafficLightParams()

# flow_params = dict(
#     exp_tag = "DensityAwareBottleneck",

#     env_name=DensityAwareBottleneckEnv,

#     network=BottleneckNetwork,

#     simulator='traci',

#     sim = SumoParams(
#                     sim_step=0.1, # For some reason, this is reduced. In ours its 0.1
#                     render=True,
#                     restart_instance=True
#     ),

#     env=EnvParams(
#         warmup_steps=4000, # original:set to 40
#         sims_per_step=1,
#         horizon=HORIZON,
#         additional_params=additional_env_params,
#     ),

#     net = NetParams(
#     inflows=inflow,
#     additional_params=additional_net_params
#     ),

#     veh=vehicles,

#     initial=InitialConfig(
#         spacing="uniform",
#         min_gap=5,
#         lanes_distribution=float("inf"),
#         #edges_distribution=["2", "3", "4", "5"],
#     ),

#     tls=traffic_lights
# )