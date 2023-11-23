"""

"""
import os 
from flow.envs.classic_intersection import classicIntersectionEnv
from flow.controllers import GridRouter
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers.controllers_for_daware import ModifiedIDMController

# enter speed for departing vehicles
V_ENTER = 12 # Previously 30. This is m/s. 20 m/s = 44 mph, 72 km/hr

# number of row of bidirectional lanes
N_ROWS = 1 # Default is 3
# number of columns of bidirectional lanes
N_COLUMNS = 1 # Default is 3

# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

# length of inner edges in the grid network
INNER_LENGTH = 180 # Not relevant here

# length of final edge in route
LONG_LENGTH = 200 # Default is 100, too short for depart velocity of 30

# length of edges that vehicles start on
SHORT_LENGTH = 200 # Default is 300, Make 200 for uniformity

rv_penetration = 0.2

def config_piws(args, **kwargs):

    # inflow rate of vehicles at every edge
    EDGE_INFLOW = args.inflow # Default 300 veh/hr/lane. Similar to Villarreal et al.,  set it to 1000

    vehicles = VehicleParams()

    vehicles.add(
        veh_id=kwargs['method_name'],
        acceleration_controller=(ModifiedIDMController, {
            "noise": args.noise, 
        }),
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way", # respect intersection rules
        ),
        num_vehicles=0, # Make them zero initially
        color = 'yellow'
        )
    
    # Even this is ModifiedIDMController that changes after warmup
    vehicles.add(
        veh_id="human",
        acceleration_controller=(ModifiedIDMController, {
            "shock_vehicle": True, # Just because it was initialized as a shock vehicle does not mean it will shock
            "noise": args.noise ,
        }),
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
        )

    kwargs['classic_parms'] = {  # Add more if necessary
                                    }
    
    if args.gen_emission:
        save_path = 'test_time_rollout/piws_stability' if args.stability else 'test_time_rollout/piws'
        sim_params = SumoParams(
            sim_step=0.1,
            render=True, # Bibek, these are not going to be trained, True is fine
            print_warnings=True,
            restart_instance=True,
            emission_path=os.path.abspath(os.path.join(os.getcwd(),save_path)))

    else:
        sim_params = SumoParams(
            sim_step=0.1, 
            render=True, # Bibek, these are not going to be trained, True is fine
            print_warnings=True,
            restart_instance=True,
            emission_path=None)

    # inflows of vehicles are place on all outer edges (listed here)
    outer_edges = []
    outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
    outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
    outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
    outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

    # We want a decent sized queue to form in East-Westbound direction
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
                    veh_type=kwargs['method_name'],
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
    exp_tag= f"intersection_{kwargs['method_name']}",

    # name of the flow environment the experiment is running on
    env_name=classicIntersectionEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim= sim_params,

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=args.warmup, # Bibek: Changed to 100 from 40
        sims_per_step=1,
        horizon=args.horizon,
        additional_params={
            "target_velocity": V_ENTER + 5,
            # These are TL specific parameters?
            "switch_time": 3,
            "num_observed": 2,
            "discrete": False,
            "tl_type": "actuated",
            "num_rl": 5,
            "add_classic_if_exit": True,
            "shock_params": kwargs['shock_params'],
            "classic_params": kwargs['classic_parms'] # Hacky way to pass
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

    # parameters specifying the positioning of vehicles upon initialization
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
        ),
    )

    return flow_params 



