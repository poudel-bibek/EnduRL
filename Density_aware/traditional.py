import random 
import argparse
from flow.core.experiment import Experiment

from BCM.bcm_config import config_bcm
from LACC.lacc_config import config_lacc


from flow.controllers import IDMController, BCMController

def run(args, **kwargs):

    config_dict = {'bcm': config_bcm, 
                    'lacc': config_lacc, 
                    'idm': config_bcm} # Change this to config_idm

    # args.method should be one from the list ['bcm', 'lacc', 'idm'], if not throw error
    methods = ['bcm', 'lacc', 'idm']
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

    parser.add_argument('--horizon', type=int, default=12000)
    # Dont set default warmup, different controllers require different values set specific in config
    parser.add_argument('--warmup', type=int, default=None) 
    parser.add_argument('--length', type=int, default=None)

    parser.add_argument('--shock', action='store_true', default=False)
    parser.add_argument('--shock_start_time', type=int, default=2500)
    parser.add_argument('--shock_end_time', type=int, default=2500)

    #TODO: remove all external sources of randomness, make system deterministic
    parser.add_argument('--noise', action='store_true', default=True) 
    parser.add_argument('--render', action='store_true', default=True)

    parser.add_argument('--num_controlled', type=int, default=None)

    args = parser.parse_args()
    run(args)
