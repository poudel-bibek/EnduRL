import random 
import argparse
from flow.core.experiment import Experiment
from bcm_config import config_bcm

from flow.controllers import IDMController, BCMController

def run_bcm(args, **kwargs):

    # Add kwargs if necessary

    # To make random selection of ring length
    for i in range(args.num_rollouts):
        exp = Experiment(config_bcm(args, **kwargs))
        #print("RL action function = =", BCMController().get_accel(exp.env))
        _ = exp.run(1, convert_to_csv=False)
        #_ = exp.run(1, rl_actions(traditionalEnv), convert_to_csv=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_rollouts', type=int, default=1)
    # store_true gen_emission
    parser.add_argument('--gen_emission', action='store_true', default=False)

    parser.add_argument('--horizon', type=int, default=12000)
    parser.add_argument('--warmup', type=int, default=2500)
    parser.add_argument('--length', type=int, default=None)

    parser.add_argument('--shock', action='store_true', default=False)
    parser.add_argument('--shock_start_time', type=int, default=2500)
    parser.add_argument('--shock_end_time', type=int, default=2500)

    #TODO: remove all external sources of randomness, make system deterministic
    parser.add_argument('--noise', action='store_true', default=True) 
    parser.add_argument('--render', action='store_true', default=True)

    args = parser.parse_args()
    run_bcm(args)
