import random 
import argparse
from flow.core.experiment import Experiment

from Config.bcm_config import config_bcm
from Config.lacc_config import config_lacc
from Config.idm_config import config_idm
from Config.fs_config import config_fs
from Config.piws_config import config_piws

from common_args import update_arguments

def run(args, **kwargs):
   
    config_dict = {'bcm': config_bcm, 
                    'lacc': config_lacc, 
                    'idm': config_idm,
                    'fs': config_fs,
                    'piws': config_piws}

    # args.method should be one from the keys list, if not throw error
    methods = list(config_dict.keys())
    if args.method is None or args.method not in methods:
        raise ValueError("The 'method' argument is required and must be one of {}.".format(methods))  

    # Add kwargs if necessary
    # Add shock params: 
    if args.stability:
        # If stability test, by default set shock to True
        args.shock = True
        kwargs['shock_params'] = {'shock': args.shock,
                            'shock_start_time': args.shock_start_time,
                            'shock_end_time': args.shock_end_time,
                            'shock_model': -1, # Shock model identifier is -1 for stability experiments 
                            'stability': args.stability, } 
    else: 
        kwargs['shock_params'] = {'shock': args.shock,
                            'shock_start_time': args.shock_start_time,
                            'shock_end_time': args.shock_end_time,
                            'shock_model': args.shock_model,
                            'stability': args.stability,} 

    kwargs['method_name'] = args.method
    config_func = config_dict.get(kwargs['method_name'])

    if config_func: 
        # To make random selection of ring length
        for i in range(args.num_rollouts):
            exp = Experiment(config_func(args, **kwargs))
            _ = exp.run(1, convert_to_csv=False)

    else:
        raise ValueError("Invalid Method")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--num_controlled', type=int, default=None)

    # Collisions only occur at min_gap = 0.0
    # Default value of min gap is 2.5m, since vehicle length is 5m
    # A collision is registered when gaps between vehicles are less than min_gap
    # Should we make IDM min_gap = 0.0 and controller min_gap = 2.5m (set nothing, it will take default)
    # If we do set a high min_gap, vehicles will never go close enough to collide (or reduce TTC to a risky value)
    # 0.2 is small enough?

    
    parser = update_arguments(parser)
    args = parser.parse_args()
    run(args)

