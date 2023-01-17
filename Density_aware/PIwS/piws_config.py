"""
PIwS
"""

import random
import os

from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.envs.ring.density_aware_traditional_env import traditionalEnv

from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import SumoParams
from flow.core.params import EnvParams

def config_piws(args, **kwargs):

    vehicles = VehicleParams()

    if args.length is None:
        kwargs['length'] = random.randint(220, 270)
    else: 
        kwargs['length'] = args.length

    print("length: ", kwargs['length'])
    
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),

        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)

    vehicles.add(
        veh_id= kwargs['method_name'],
         acceleration_controller=(IDMController, {
            "noise": 0.2,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles= 1 if args.num_controlled is None else args.num_controlled, # Minimum 
        color = 'yellow')

    # Add specific properties of vehicles with this method_name id
    # These params from LORR paper
    kwargs['traditional_parms'] = {}

    if args.gen_emission:
        sim_params = SumoParams(sim_step=0.1, 
                                render=args.render,
                                emission_path=os.path.abspath(os.path.join(os.getcwd(),'test_time_rollout')))
    else:
        sim_params = SumoParams(sim_step=0.1, 
                                render=args.render,
                                emission_path=None)

    env_params = EnvParams(
        horizon=args.horizon,
        warmup_steps= 3500 if args.warmup is None else args.warmup,
        evaluate = True, # To prevent abrupt fails
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "target_velocity": 10,
            "sort_vehicles": False,
            "traditional_params": kwargs['traditional_parms'], # Hacky way to pass
        },
    )

    net_params = NetParams(
        additional_params={
            "length": kwargs['length'],
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        },)

    initial_config = InitialConfig(
        spacing="uniform",
        bunching=0,
        perturbation=0,
    )

    flow_params = dict(
        exp_tag= kwargs['method_name'],
        env_name= traditionalEnv, #AccelEnv,
        network=RingNetwork,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
    )

    return flow_params

