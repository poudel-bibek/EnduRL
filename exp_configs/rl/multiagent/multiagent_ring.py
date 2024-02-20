from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import RLController, IDMController, ContinuousRouter, TrainedAgentController
from flow.envs.multiagent import MultiAgentDensityAwareRLEnv

from flow.networks import RingNetwork
from flow.utils.registry import make_create_env

# time horizon of a single rollout
HORIZON = 4500
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 4
WARMUP_STEPS = 2500

# number of automated vehicles. Unfortunately it has to be changed here everytime.
NUM_AUTOMATED = 12 # 3 for 20%, 8 for 40%, 12 for 60% (1 less than actual)
num_human = 22 - (NUM_AUTOMATED + 1) # 1 for the trained leader

vehicles = VehicleParams()
for i in range(NUM_AUTOMATED):
    # Add one automated vehicle.
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.1 # Collisions are allowed at 0
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1,
        color='red')
    
# Add the trained leader. These must point to correct leader at both train and test times
# A correct copy of the exp_configs must be present in the same folder as test_rllib.py
########## FOR SAFETY + STABILITY ##########
vehicles.add(
    veh_id="rl_leader",
    acceleration_controller=(TrainedAgentController, {
                                                    "local_zone" : 50.0,
                                                    "directory" : "/mnt/c/Users/09_gi/Desktop/Beyond-Simulated-Drivers/ring/Ours/Trained_policies/5_percent/", 
                                                    "policy_name" : "PPO_DensityAwareRLEnv-v0_5dfded14_2024-02-07_09-58-351869u4p3", 
                                                    "checkpoint_num" : "168",
                                                    "num_cpus" : 5, # 1 greater than N_CPUS above
                                                    "warmup_steps" : WARMUP_STEPS,
                 }),

    car_following_params=SumoCarFollowingParams(
        min_gap=0.1 # Collisions are allowed at 0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1,
    color='red')

########## FOR EFFICIENCY ##########
# vehicles.add(
#     veh_id="rl_leader",
#     acceleration_controller=(TrainedAgentController, {
#                                                     "local_zone" : 50.0,
#                                                     "directory" : "/mnt/c/Users/09_gi/Desktop/Beyond-Simulated-Drivers/ring/Ours/Trained_policies/5_percent/", 
#                                                     "policy_name" : "PPO_DensityAwareRLEnv-v0_6106dcf6_2024-02-08_12-42-07_7vnf1n4", 
#                                                     "checkpoint_num" : "200",
#                                                     "num_cpus" : 5, # 1 greater than N_CPUS above
#                                                     "warmup_steps" : WARMUP_STEPS,
#                  }),

#     car_following_params=SumoCarFollowingParams(
#         min_gap=0.1 # Collisions are allowed at 0
#     ),
#     routing_controller=(ContinuousRouter, {}),
#     num_vehicles=1,
#     color='red')

# Bibek: Originally this is added in a way to not make a platoon i.e., distributed configuration
vehicles.add(
    veh_id="human_{}".format(i),
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=num_human)


flow_params = dict(
    # name of the experiment
    exp_tag= "density_aware_multiagent_ring", #"multiagent_ring",

    # name of the flow environment the experiment is running on
    env_name= MultiAgentDensityAwareRLEnv, 

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
        warmup_steps= WARMUP_STEPS,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": [260, 260],
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


create_env, env_name = make_create_env(params=flow_params, version=0)
register_env(env_name, create_env)
test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy():
    # Generate a policy in RLlib
    return PPOTFPolicy, obs_space, act_space, {}

# Leader and follower do not share policies
POLICY_GRAPHS = {'follower': gen_policy()}

def policy_mapping_fn(agent_id):
    #map policy to agent
    return 'follower'

POLICIES_TO_TRAIN = ['follower']


