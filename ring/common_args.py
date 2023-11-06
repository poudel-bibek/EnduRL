"""
Test time arguments are shared by traditional and RL models
Update the argument parser for each using this file
"""
import argparse

def update_arguments(parser):
    # Test time arguments

    parser.add_argument('--length', type=int, default = None, help='Specifies the ring length.')
    parser.add_argument('--warmup', type=int, default = 2500, help='Specifies the warmup time.')
    parser.add_argument('--horizon', type=int, default= 15000, help='Specifies the horizon.')
    parser.add_argument('--noise', type=float, default=0.2)
    #TODO: remove all external sources of randomness, make system deterministic
    parser.add_argument('--no_noise', action='store_true', default=True)

    parser.add_argument('--stability', action='store_true', default=False)
    # store_true gen_emission
    parser.add_argument('--gen_emission', action='store_true', default=False)
    parser.add_argument('--num_rollouts', type=int,default=1, help='The number of rollouts to visualize.')

    # To shock, both shock and shock veh must be true as well as shock time 
    parser.add_argument('--shock', action='store_true', default=False)
    parser.add_argument('--shock_start_time', type=int, default=8000)
    parser.add_argument('--shock_end_time', type=int, default=11000)
    parser.add_argument('--shock_model', type=int, default= 2)
    parser.add_argument('--min_gap', type=float, default=0.1) # Small value to prevent collisions (Are collisions causing sim to stop?)
    parser.add_argument('--render', action='store_true', default=False)
    
    return parser