"""
This is not the original singleagent_bottleneck
Rather, this is bottleneck_2 mentioned in Vintinsky et. al. in http://proceedings.mlr.press/v87/vinitsky18a/vinitsky18a.pdf

Major differences: 
- Horizon 1500 vs 1000
- Scaling is set to 2 by default
- RL vehicle is named differently
- humans have a speed mode of 9 instead of all_checks
- inflow range (in env_params) is dependent on the scaling factor



"""

"""
Benchmark for bottleneck2.

Bottleneck in which the actions are specifying a desired velocity in a segment
of space for a large bottleneck.
The autonomous penetration rate in this example is 10%.

- **Action Dimension**: (40, )
- **Observation Dimension**: (281, )
- **Horizon**: 1000 steps
"""

from flow.envs import BottleneckDesiredVelocityEnv
from flow.networks import BottleneckNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter

# time horizon of a single rollout
HORIZON = 3600 # 1500 # 8000 is still too long for training. At eval can be done at 8000

# number of parallel workers
N_CPUS = 8 # Instead of 8 at 1500. It will do a +1 later so actually its 9
# The Terminal log will count the total timesteps so it will show something like 180000 = (3600 + 4000)/2 * 50 
# It is specidied in Table 3 of the benchmarks paper that 50 rollouts per iteration 
# Also a batch size of 80000: 3600 timesteps per rollout, total = (3600/2) * 45 = 81000 # Close enough 
N_ROLLOUTS = N_CPUS * 5 # 45 rollouts per iteration

SCALING = 2 # The paper mentions N should be 3 with inflow = 3800 but in code, N is 2
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = .10

vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id="human",
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)

controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    "target_velocity": 15,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "reset_inflow": False,
    "lane_change_duration": 5,
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
    departSpeed=6)
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    departLane="random",
    departSpeed=6)

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
    exp_tag="bottleneck_2",

    # name of the flow environment the experiment is running on
    env_name=BottleneckDesiredVelocityEnv,

    # name of the network class the experiment is running on
    network=BottleneckNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.5, 
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps= 4000, # Bibek: Change from 40 # 40 was not nearly enough to let a congestion form. 
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
