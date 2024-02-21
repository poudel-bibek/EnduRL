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
from flow.controllers.controllers_for_daware import ModifiedIDMController

from flow.controllers.routing_controllers import ContinuousRouter
from flow.envs.ring.density_aware_classic_env import classicEnv

from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import SumoParams
from flow.core.params import EnvParams

# This is for clustered version of BCM
# SET desired velocity for BCM according to ring length
# Current assumption, desired velocity can be set from the  velocity upper bound
from flow.density_aware_util import get_desired_velocity

def config_bcm(args, **kwargs):
    
    vehicles = VehicleParams()
    num_controlled = 4 if args.num_controlled is None else args.num_controlled # Minimum

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
        acceleration_controller=(ModifiedIDMController, {
            "shock_vehicle": True, # Just because it was initialized as a shock vehicle does not mean it will shock
            "noise": args.noise ,
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap= args.min_gap, # Test 2.5 , 
            #speed_mode=0, # Test 0, 7
        ),

        routing_controller=(ContinuousRouter, {}),
        num_vehicles= 22- num_controlled) # 14 for stable

    vehicles.add(
        veh_id=kwargs['method_name'],
         acceleration_controller=(ModifiedIDMController, {
            "noise": args.noise, 
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=args.min_gap, 
            #speed_mode=0,# Test
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles= num_controlled, 
        color = 'yellow')
    
    # Add specific properties of vehicles with this method_name id
    desired_velocity = get_desired_velocity(len(vehicles.ids), kwargs['length'])
    print("Desired Velocity: ", desired_velocity, "m/s")

    # Add specific properties of vehicles with this method_name id
    kwargs['classic_parms'] = {'v_des': desired_velocity, # Add more if necessary
                                    }
    if args.gen_emission:
        save_path = 'test_time_rollout/bcm_stability' if args.stability else 'test_time_rollout/bcm'
        sim_params = SumoParams(sim_step=0.1, 
                                render=args.render,
                                emission_path=os.path.abspath(os.path.join(os.getcwd(),save_path))) # will create directory if not exist
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
            "target_velocity": 15,
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
        network= RingNetwork,
        simulator= 'traci',
        sim= sim_params,
        env= env_params,
        net= net_params,
        veh= vehicles,
        initial= initial_config,
    )

    return flow_params

### Test stuff: 
# car_following_params=SumoCarFollowingParams(
#             min_gap=args.min_gap, # Change in others
        #     sigma= 0.0,
        #     speed_dev= 0.0,
        #     speed_factor= 1.0,
        #     tau= 1.0,
        #     speed_mode=7, # Test
        # ),
