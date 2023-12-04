import argparse
"""
When to start the controllers:
    - After the congestion has formed
When to start the shock:
    - After they get some time to stabilize


"""
def update_arguments(parser):
    parser.add_argument('--num_rollouts', type=int,default=1, help='The number of rollouts to visualize.')
    parser.add_argument('--warmup', type=int, default = 800, help='Specifies the warmup time.') # Have to have some warmup
    parser.add_argument('--horizon', type=int, default= 3600, help='Specifies the horizon.') # 
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--stability', action='store_true', default=False)

    parser.add_argument('--gen_emission', action='store_true', default=False)

    # args.long_length
    parser.add_argument('--long_length', type=int, default=350) # How long the road segments are # Specific to intersection
    parser.add_argument('--short_length', type=int, default=350)

    parser.add_argument('--shock', action='store_true', default=False)
    parser.add_argument('--shock_start_time', type=int, default= 800) 
    parser.add_argument('--shock_end_time', type=int, default= 4400) 
    parser.add_argument('--shock_model', type=int, default= 2)

    parser.add_argument('--sim_step', type=float, default=0.1)
    parser.add_argument('--inflow', type=int, default= 1400) # For RL, at training time, the inflow value is not taken from here. See training config file
    parser.add_argument('--av_frac', type=float, default= 0.2) # Controls AV frac for all except RL at training time
    parser.add_argument('--render', action='store_true', default=False)
    return parser

