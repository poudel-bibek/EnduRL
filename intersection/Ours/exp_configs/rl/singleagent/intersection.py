"""
Benchmark paper is controlling the traffic lights, and does not have RVs

"""

from flow.envs import DensityAwareIntersectionEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, RLController
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter

# time horizon of a single rollout
HORIZON = 3600 # Default 400 in the paper. The horizon for trainig and test can be separate. At test set it to 3800
WARMUP = 400 

# inflow rate of vehicles at every edge
EDGE_INFLOW = 1300 # Default 300 veh/hr/lane. Similar to Villarreal et al.,  set it to 1000

# enter speed for departing vehicles
V_ENTER = 5 #8 

# number of row of bidirectional lanes
N_ROWS = 1 # Default is 3
# number of columns of bidirectional lanes
N_COLUMNS = 1 # Default is 3

# length of inner edges in the grid network
INNER_LENGTH = 180

# length of final edge in route
LONG_LENGTH = 350 # Default is 100, too short for depart velocity of 30

# length of edges that vehicles start on
SHORT_LENGTH = 350 # Default is 300, Make 200 for uniformity

# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

# number of rollouts per training iteration
N_ROLLOUTS = 10 # CHANGE, Default is 10
# number of parallel workers
N_CPUS =  8 # CHANGE, Default is is 10

# Same as Villarreal et al.
rv_penetration = 0.2

# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS)

# add Rl. Similar to Villarreal et al. 
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=0)


# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    # 75% of the EDGE_INFLOW here (North-South)
    if edge == "left1_0" or edge == "right0_0":
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour= (1- rv_penetration)*0.75*EDGE_INFLOW, #RVs are only deployed North-South
            departLane="free",
            depart_speed=V_ENTER)

        inflow.add(
                veh_type="rl",
                edge=edge,
                vehs_per_hour= (rv_penetration)*0.75*EDGE_INFLOW, #RVs are only deployed North-South
                depart_lane="free",
                depart_speed=V_ENTER
            )
        
    # 25% of the EDGE_INFLOW here
    else:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour= 0.25*EDGE_INFLOW,
            departLane="free",
            depart_speed=V_ENTER
        )


flow_params = dict(
    # name of the experiment
    exp_tag="intersection",

    # name of the flow environment the experiment is running on
    env_name=DensityAwareIntersectionEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps= WARMUP,
        additional_params={
            "target_velocity": V_ENTER + 5,
            # These are TL specific parameters?
            "switch_time": 3,
            "num_observed": 2,
            "discrete": False,
            "tl_type": "actuated",
            "num_rl": 5,
            

            # More params needed here seems like
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params={
            "speed_limit": V_ENTER + 5,
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
            "traffic_lights": False, #False, # Bibek
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)
