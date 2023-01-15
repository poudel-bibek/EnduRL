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

# This is for clustered version of BCM
# SET desired velocity for BCM according to ring length
# Current assumption, desired velocity can be set from the  velocity upper bound

def config_bcm(args, **kwargs):
    vehicles = VehicleParams()

    if args.length is None:
        kwargs['length'] = random.randint(220, 270)
    else: 
        kwargs['length'] = args.length

    print("length: ", kwargs['length'])
    # Noise (And other sourve of randomness: speedDev, speedFactor, what is sigma?)
    # Max accel and decel 
    # Target velocity : Desired velocity for all vehicles in the network (Accel env default is 10)
    # Desired velocity (LORR paper, for a 260m ring length, desired velocity is 4.8). Desired velocity is picked as the equillibrum velocity for the ring length
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
        veh_id=kwargs['method_name'],
         acceleration_controller=(IDMController, {
            "noise": 0.2,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0,
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles= 4 if args.num_controlled is None else args.num_controlled, # Minimum
        color = 'yellow')

    # Add specific properties of vehicles with this method_name id
    kwargs['traditional_parms'] = {'v_des':4.8, # Add more if necessary
                                    }

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
        warmup_steps= 3000 if args.warmup is None else args.warmup,
        evaluate = True, # To prevent abrupt fails
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "target_velocity": 10,
            "sort_vehicles": False,
            "fail_on_negative_reward": False, # Set this for traditional
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
        network= RingNetwork,
        simulator= 'traci',
        sim= sim_params,
        env= env_params,
        net= net_params,
        veh= vehicles,
        initial= initial_config,
    )

    return flow_params

