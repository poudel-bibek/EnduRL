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
        for i in range(args.num_rollouts):
            exp = Experiment(config_func(args, **kwargs))
            _ = exp.run(1, convert_to_csv=False)

    else:
        raise ValueError("Invalid Method")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--num_controlled', type=int, default=None)
    parser = update_arguments(parser)
    args = parser.parse_args()
    
    # If args.av_frac is None, then throw error
    if args.av_frac is None:
        raise ValueError("The 'av_frac' argument is required and must be a float.")

    run(args)
    