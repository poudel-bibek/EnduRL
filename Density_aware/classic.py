import random 
import argparse
from flow.core.experiment import Experiment

from BCM.bcm_config import config_bcm
from LACC.lacc_config import config_lacc
from IDM.idm_config import config_idm
from FS.fs_config import config_fs
from PIwS.piws_config import config_piws


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
    parser.add_argument('--num_rollouts', type=int, default=1)
    # store_true gen_emission
    parser.add_argument('--gen_emission', action='store_true', default=False)

    parser.add_argument('--horizon', type=int, default=15000)
    # Dont set default warmup, different controllers require different values set specific in config
    parser.add_argument('--warmup', type=int, default=None) 
    parser.add_argument('--length', type=int, default=None)

    parser.add_argument('--shock', action='store_true', default=False)
    parser.add_argument('--shock_start_time', type=int, default=8000)
    parser.add_argument('--shock_end_time', type=int, default=11500)
    parser.add_argument('--shock_model', type=int, default= 1)

    #TODO: remove all external sources of randomness, make system deterministic
    # Speed_dev and Sigma?
    parser.add_argument('--noise', action='store_true', default=True) 
    parser.add_argument('--render', action='store_true', default=True)

    parser.add_argument('--num_controlled', type=int, default=None)

    # Collisions only occur at min_gap = 0.0
    parser.add_argument('--min_gap', type=float, default=0.2) # Small value to prevent collisions (Are collisions causing sim to stop?)
    args = parser.parse_args()
    run(args)
