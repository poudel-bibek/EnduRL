
import os 
from flow.envs.classic_bottleneck import classicBottleneckEnv
from flow.networks import BottleneckNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import ContinuousRouter
from flow.controllers.controllers_for_daware import ModifiedIDMController

def config_piws(args, **kwargs):
    vehicles = VehicleParams()
    #num_controlled = 4 if args.num_controlled is None else args.num_controlled # Minimum

    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    AV_FRAC = args.av_frac
    SCALING = 2

    vehicles.add(
        veh_id=kwargs['method_name'],
        acceleration_controller=(ModifiedIDMController, {
            "noise": args.noise, 
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
        num_vehicles=1 * SCALING,
        color = 'yellow') 

    vehicles.add(
        veh_id="human",
        acceleration_controller=(ModifiedIDMController, {
            "shock_vehicle": True, # Just because it was initialized as a shock vehicle does not mean it will shock
            "noise": args.noise ,
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9, # min_gap is not specified here
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
        num_vehicles=1 * SCALING)

    kwargs['classic_parms'] = {  # Add more if necessary
                                    }
    
    if args.gen_emission:
        save_path = 'test_time_rollout/pi_stability' if args.stability else 'test_time_rollout/pi'
        sim_params = SumoParams(
            sim_step=args.sim_step,
            render=True, # Bibek, these are not going to be trained, True is fine
            print_warnings=True,
            restart_instance=True,
            emission_path=os.path.abspath(os.path.join(os.getcwd(),save_path)))

    else:
        sim_params = SumoParams(
            sim_step=args.sim_step,
            render=True, # Bibek, these are not going to be trained, True is fine
            print_warnings=True,
            restart_instance=True,
            emission_path=None)

    additional_env_params = {
        "target_velocity": 15, # Change that to match the ring? Or match the noise levels accordingly
        "disable_tb": True,
        "disable_ramp_metering": True,
        "symmetric": False,
        "reset_inflow": False,
        "lane_change_duration": 5,
        "max_accel": 3,
        "max_decel": 3, # These are used by RL?
        "add_classic_if_exit": True, # 
        "classic_params": kwargs['classic_parms'], # Hacky way to pass
        "shock_params": kwargs['shock_params'], # Hacky way to pass
    }

    # flow rate
    flow_rate = 1800 * SCALING # What density profile would this flow rate lead to? # Changed from 2500

    # percentage of flow coming out of each lane
    inflow = InFlows()

    inflow.add(
        name = "classic",
        veh_type= kwargs['method_name'],
        edge="1",
        #/ num_controlled, # Uncomment if want to operatate at min. no . of vehicles required to stabilize in ring, Because later we scale them up
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

    flow_params = dict(
    # name of the experiment
    exp_tag= f"bottleneck_2_{kwargs['method_name']}",

    # name of the flow environment the experiment is running on
    env_name=classicBottleneckEnv,

    # name of the network class the experiment is running on
    network=BottleneckNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim= sim_params,

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=args.warmup, # Bibek: Changed to 100 from 40
        sims_per_step=1,
        horizon=args.horizon,
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
    
    return flow_params 
