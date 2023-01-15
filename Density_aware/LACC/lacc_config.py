"""
Clustered BCM:
    minimum number of vehicles required: 
    Time to stabilize: 

BCM with even distribution:
    minimum number of vehicles required:
    Time to stabilize:

Desired velocity varies according to ring length:
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

def config_lacc(args, **kwargs):
    name = "bcm"
    vehicles = VehicleParams()

    if args.length is None:
        kwargs['length'] = random.randint(220, 270)
    else: 
        kwargs['length'] = args.length

    print("length: ", kwargs['length'])
    # Noise (And other sourve of randomness: speedDev, speedFactor, what is sigma?)
    # Max accel and decel 
    # Min gap (Seting min gap to 0 causes congestion to form early)
    # What should max_accel and max_decel be?
    # Set BCM controllers default as IDM, 
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),

        routing_controller=(ContinuousRouter, {}),
        num_vehicles=18)

    vehicles.add(
        veh_id="lacc",
         acceleration_controller=(IDMController, {
            "noise": 0.2,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=4,
        color = 'yellow')

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
        warmup_steps=args.warmup,
        evaluate = True, # To prevent abrupt fails
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "target_velocity": 10,
            "sort_vehicles": False,
            "fail_on_negative_reward": False, # Set this for traditional
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
        exp_tag=name,
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

