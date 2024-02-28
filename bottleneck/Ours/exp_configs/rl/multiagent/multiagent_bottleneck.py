"""
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig, VehicleParams, InFlows, NetParams, TrafficLightParams
from flow.controllers import ContinuousRouter, SimLaneChangeController, RLController, IDMController
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams

from flow.envs.multiagent import DensityAwareBottleneckEnv
#from flow.networks.poudel_bottleneck import PoudelBottleneckNetwork # The bottleneck network that we constructed ourself.
from flow.networks import BottleneckNetwork

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# Time horizon of a single rollout
HORIZON = 1300 

# Number of parallel workers
N_CPUS = 4
N_ROLLOUTS = N_CPUS * 2 # 24 rollouts per iteration

SCALING = 2 # The paper mentions N should be 3 with inflow = 3800 but in code, N is 2
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True

AV_FRAC = 0.05 # NEED to change this here everytime for a new training instance.

vehicles = VehicleParams()
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}), #IDMController
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=0.1, # Hack: # 
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id="human",
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
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
    "reset_inflow": True,
    "lane_change_duration": 5,
    "add_rl_if_exit": True,
    "max_accel": 5,
    "max_decel": 5,
    "inflow_range": [3400, 3800] # Just like training at a fixed density.
    #"inflow_range": [1300 * SCALING, 2600 * SCALING] # So that there is some randomness in the whole thing. Generalizes better
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

additional_net_params = {"scaling": SCALING, "speed_limit": 17}
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
        sim_step=0.5, # 0.1 is too much 
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps= 100,
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

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy():
    # Generate a policy in RLlib
    return PPOTFPolicy, obs_space, act_space, {}

# Setup single policy for all vehicles
POLICY_GRAPHS = {'shared_policy': gen_policy()}

def policy_mapping_fn(agent_id):
    # All agents use the shared policy
    return 'shared_policy'

# Only the shared policy needs to be trained
POLICIES_TO_TRAIN = ['shared_policy']