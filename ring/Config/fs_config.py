"""
FS
"""

import random
import os

from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController
from flow.controllers.controllers_for_daware import ModifiedIDMController

from flow.controllers.routing_controllers import ContinuousRouter
from flow.envs.ring.density_aware_classic_env import classicEnv

from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import SumoParams
from flow.core.params import EnvParams

# Again, the desired velocity is set from the velocity upper bound 
from flow.density_aware_util import get_desired_velocity

def config_fs(args, **kwargs):

    vehicles = VehicleParams()
    num_controlled = 1 if args.num_controlled is None else args.num_controlled # Minimum

    if args.length is None:
        kwargs['length'] = random.randint(220, 270)
    else: 
        kwargs['length'] = args.length

    print("length: ", kwargs['length'])
    
    vehicles.add(
            veh_id="human",
            acceleration_controller=(ModifiedIDMController, {
                "shock_vehicle": True, # Just because it was initialized as a shock vehicle does not mean it will shock
                "noise": args.noise,
            }),
            car_following_params=SumoCarFollowingParams(
                min_gap= args.min_gap,
            ),

            routing_controller=(ContinuousRouter, {}),
            num_vehicles= 22 - num_controlled) 

    vehicles.add(
        veh_id= kwargs['method_name'],
        acceleration_controller=(ModifiedIDMController, {
            "noise": args.noise,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap= args.min_gap,
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles= num_controlled,
        color = 'yellow')
    
    # if args.num_controlled == 13: # Unstable for the desired velocity of 4.82 in platoon configuration.
    #     # Add 5 humans, 7 controlled, 4 humans, 6 controlled
    #     vehicles.add(
    #         veh_id="human_0",
    #         acceleration_controller=(ModifiedIDMController, {
    #             "shock_vehicle": True,
    #             "noise": args.noise,
    #         }),
    #         car_following_params=SumoCarFollowingParams(
    #             min_gap= args.min_gap,
    #         ),
    #         routing_controller=(ContinuousRouter, {}),
    #         num_vehicles=5)
        
    #     vehicles.add(
    #         veh_id= f"{kwargs['method_name']}_0",
    #         acceleration_controller=(ModifiedIDMController, {
    #             "noise": args.noise,
    #         }),
    #         car_following_params=SumoCarFollowingParams(
    #             min_gap= args.min_gap,
    #         ),
    #         routing_controller=(ContinuousRouter, {}),
    #         num_vehicles=7,
    #         color = 'yellow')
        
    #     vehicles.add(
    #         veh_id="human_1",
    #         acceleration_controller=(ModifiedIDMController, {
    #             "shock_vehicle": True, 
    #             "noise": args.noise,
    #         }),
    #         car_following_params=SumoCarFollowingParams(
    #             min_gap= args.min_gap,
    #         ),
    #         routing_controller=(ContinuousRouter, {}),
    #         num_vehicles=4)
        
    #     vehicles.add(
    #         veh_id= f"{kwargs['method_name']}_1",
    #         acceleration_controller=(ModifiedIDMController, {
    #             "noise": args.noise,
    #         }),
    #         car_following_params=SumoCarFollowingParams(
    #             min_gap= args.min_gap,
    #         ),
    #         routing_controller=(ContinuousRouter, {}),
    #         num_vehicles=6,
    #         color = 'yellow')
    # else:
    #     vehicles.add(
    #         veh_id="human",
    #         acceleration_controller=(ModifiedIDMController, {
    #             "shock_vehicle": True, # Just because it was initialized as a shock vehicle does not mean it will shock
    #             "noise": args.noise,
    #         }),
    #         car_following_params=SumoCarFollowingParams(
    #             min_gap= args.min_gap,
    #         ),

    #         routing_controller=(ContinuousRouter, {}),
    #         num_vehicles=21 if args.stability else 22 - num_controlled) # 17 for stable

    #     vehicles.add(
    #         veh_id= kwargs['method_name'],
    #         acceleration_controller=(ModifiedIDMController, {
    #             "noise": args.noise,
    #         }),
    #         car_following_params=SumoCarFollowingParams(
    #             min_gap= args.min_gap,
    #         ),
    #         routing_controller=(ContinuousRouter, {}),
    #         num_vehicles= num_controlled,
    #         color = 'yellow')

    # Add specific properties of vehicles with this method_name id
    desired_velocity = get_desired_velocity(len(vehicles.ids), kwargs['length'], args.method)
    print("Desired Velocity: ", desired_velocity, "m/s")

    kwargs['classic_parms'] = {'v_des': desired_velocity, # Add more if necessary
                                    }

    if args.gen_emission:
        save_path = 'test_time_rollout/fs_stability' if args.stability else 'test_time_rollout/fs'
        sim_params = SumoParams(sim_step=0.1, 
                                render=args.render,
                                emission_path=os.path.abspath(os.path.join(os.getcwd(),save_path)))
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
            "classic_params": kwargs['classic_parms'], # Hacky way to pass
            "shock_params": kwargs['shock_params'], # Hacky way to pass        
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
        env_name= classicEnv, #AccelEnv,
        network=RingNetwork,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
    )

    return flow_params

