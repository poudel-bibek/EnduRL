"""
Be able to load a trained policy
Generate rollout data
Works for both Vinitsky and Ours
Specific to Bottleneck scenario
"""

import argparse
import gym
import numpy as np
import os
import sys
import time

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

import json 
from common_args import update_arguments
from flow.density_aware_util import get_shock_model, get_time_steps, get_time_steps_stability

EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""

def visualizer_rllib(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    # Grab the config and make modifications 
    flow_params_modify = json.loads(config["env_config"]["flow_params"])

    flow_params_modify["veh"][1]["acceleration_controller"] = ["ModifiedIDMController", {"noise": args.noise, # Just need to specify as string
                                                                                            "shock_vehicle": True}] 
    flow_params_modify["sim"]["sim_step"] = args.sim_step

    flow_params_modify["env"]["horizon"] = args.horizon
    flow_params_modify["env"]["warmup_steps"] = args.warmup
    # At test time, the inflow range is set to a constant value
    flow_params_modify["env"]["inflow_range"] = [args.inflow, args.inflow]

    # Dump our modifications to the config
    config["env_config"]["flow_params"] = json.dumps(flow_params_modify)

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # for hacks for old pkl files TODO: remove eventually
    if not hasattr(sim_params, 'use_ballistic'):
        sim_params.use_ballistic = False

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = True

    dir_path = os.path.dirname(os.path.realpath(__file__))
    rl_folder_name = f"{args.method}_stability" if args.stability else args.method
    emission_path = f"{dir_path}/test_time_rollout/{rl_folder_name}" #'{0}/test_time_rollout/'.format(dir_path)

    sim_params.emission_path = emission_path if args.gen_emission else None

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # check if the environment is a single or multiagent environment, and
    # get the right address accordingly
    # single_agent_envs = [env for env in dir(flow.envs)
    #                      if not env.startswith('__')]

    # if flow_params['env_name'] in single_agent_envs:
    #     env_loc = 'flow.envs'
    # else:
    #     env_loc = 'flow.envs.multiagent'

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)

    # if args.render_mode == 'sumo_gui':
    #     env.sim_params.render = True  # set to True after initializing agent and env
    if args.render: 
        env.sim_params.render = True
    else: 
        env.sim_params.render = False

    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn']
        for key in config['multiagent']['policies'].keys():
            rets[key] = []
    else:
        rets = []

    if config['model']['use_lstm']:
        use_lstm = True
        if multiagent:
            state_init = {}
            # map the agent id to its policy
            policy_map_fn = config['multiagent']['policy_mapping_fn']
            size = config['model']['lstm_cell_size']
            for key in config['multiagent']['policies'].keys():
                state_init[key] = [np.zeros(size, np.float32),
                                   np.zeros(size, np.float32)]
        else:
            state_init = [
                np.zeros(config['model']['lstm_cell_size'], np.float32),
                np.zeros(config['model']['lstm_cell_size'], np.float32)
            ]
    else:
        use_lstm = False

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params, render=sim_params.render)

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []

    # no need warmup offset, its already set to 4400
    shock_start_time = args.shock_start_time #- args.warmup
    shock_end_time = args.shock_end_time #- args.warmup

    for i in range(args.num_rollouts):
        vel = []
        state = env.reset()
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0

        # shock related
        shock_counter = 0
        current_duration_counter = 0

        shock_model_id = -1 if args.stability else args.shock_model
        if args.stability:
            pass 
        else:
            intensities, durations, frequency =  get_shock_model(shock_model_id, network_scaler=3, bidirectional=False, high_speed=False)
        shock_times = get_time_steps(durations, frequency, shock_start_time, shock_end_time)

        for step in range(env_params.horizon):
            vehicles = env.unwrapped.k.vehicle
            speeds = vehicles.get_speed(vehicles.get_ids())

            if args.shock and step >= shock_start_time and step <= shock_end_time:
                
                edges_allowed_list = ['3', '4_0', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7',  '4']#
                threshold_speed = 3.0
                sample_vehicles = 4

                if step == shock_times[0][0]: # This occurs only once
                    all_ids = env.k.vehicle.get_ids()

                    current_shockable_vehicle_ids = [i for i in all_ids if 'flow_00' not in i and env.unwrapped.k.vehicle.get_edge(i) in edges_allowed_list and env.unwrapped.k.vehicle.get_speed(i) > threshold_speed and env.unwrapped.k.vehicle.get_leader(i) is not None]

                    shock_ids = np.random.choice(current_shockable_vehicle_ids, sample_vehicles)
                    #print(f"\n\nShock ids: {shock_ids}\n\n")

                shock_counter, current_duration_counter, shock_ids = perform_shock(args, env, 
                            shock_times, 
                            step,
                            args.warmup,
                            shock_counter,
                            current_duration_counter,
                            intensities,
                            durations,
                            frequency, 
                            shock_ids,
                            edges_allowed_list,
                            threshold_speed,
                            sample_vehicles)

            # only include non-empty speeds
            if speeds:
                vel.append(np.mean(speeds))

            if multiagent:
                action = {}
                for agent_id in state.keys():
                    if use_lstm:
                        action[agent_id], state_init[agent_id], logits = \
                            agent.compute_action(
                            state[agent_id], state=state_init[agent_id],
                            policy_id=policy_map_fn(agent_id))
                    else:
                        action[agent_id] = agent.compute_action(
                            state[agent_id], policy_id=policy_map_fn(agent_id))
            else:
                action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            if multiagent:
                for actor, rew in reward.items():
                    ret[policy_map_fn(actor)][0] += rew
            else:
                ret += reward
            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

        if multiagent:
            for key in rets.keys():
                rets[key].append(ret[key])
        else:
            rets.append(ret)
        outflow = vehicles.get_outflow_rate(500)
        final_outflows.append(outflow)
        inflow = vehicles.get_inflow_rate(500)
        final_inflows.append(inflow)
        if np.all(np.array(final_inflows) > 1e-5):
            throughput_efficiency = [x / y for x, y in
                                     zip(final_outflows, final_inflows)]
        else:
            throughput_efficiency = [0] * len(final_inflows)
        mean_speed.append(np.mean(vel))
        std_speed.append(np.std(vel))
        if multiagent:
            for agent_id, rew in rets.items():
                print('Round {}, Return: {} for agent {}'.format(
                    i, ret, agent_id))
        else:
            print('Round {}, Return: {}'.format(i, ret))

    print('==== END ====')
    print("Return:")
    print(mean_speed)
    if multiagent:
        for agent_id, rew in rets.items():
            print('For agent', agent_id)
            print(rew)
            print('Average, std return: {}, {} for agent {}'.format(
                np.mean(rew), np.std(rew), agent_id))
    else:
        print(rets)
        print('Average, std: {}, {}'.format(
            np.mean(rets), np.std(rets)))

    print("\nSpeed, mean (m/s):")
    print(mean_speed)
    print('Average, std: {}, {}'.format(np.mean(mean_speed), np.std(
        mean_speed)))
    print("\nSpeed, std (m/s):")
    print(std_speed)
    print('Average, std: {}, {}'.format(np.mean(std_speed), np.std(
        std_speed)))

    # Compute arrival rate of vehicles in the last 500 sec of the run
    print("\nOutflows (veh/hr):")
    print(final_outflows)
    print('Average, std: {}, {}'.format(np.mean(final_outflows),
                                        np.std(final_outflows)))
    # Compute departure rate of vehicles in the last 500 sec of the run
    print("Inflows (veh/hr):")
    print(final_inflows)
    print('Average, std: {}, {}'.format(np.mean(final_inflows),
                                        np.std(final_inflows)))
    # Compute throughput efficiency in the last 500 sec of the
    print("Throughput efficiency (veh/hr):")
    print(throughput_efficiency)
    print('Average, std: {}, {}'.format(np.mean(throughput_efficiency),
                                        np.std(throughput_efficiency)))

    # terminate the environment
    env.unwrapped.terminate()

def perform_shock(args, env, 
                  shock_times, 
                  step, 
                  warmup, 
                  shock_counter, 
                  current_duration_counter, 
                  intensities, 
                  durations, 
                  frequency, 
                  shock_ids, 
                  edges_allowed_list,
                  threshold_speed,
                  sample_vehicles
                  ):
    """
    flow_00 is RL
    flow_10 is ModifiedIDM

    shock_times is the shock time steps
    """
    all_ids = env.k.vehicle.get_ids()

    controllers = [env.unwrapped.k.vehicle.get_acc_controller(i) for i in shock_ids]
    for controller in controllers:
        controller.set_shock_time(False)

    # Reset duration counter and increase shock counter, after completion of shock duration
    # The durations can be anywhere between 0.1 to 2.5 at intervals of 0.1 but the current duration counter does not increment that way (make it >=)
    if shock_counter < len(durations) and current_duration_counter >= durations[shock_counter]: 
        shock_counter += 1
        current_duration_counter = 0

        current_shockable_vehicle_ids = [i for i in all_ids if 'flow_00' not in i and env.unwrapped.k.vehicle.get_edge(i) in edges_allowed_list and env.unwrapped.k.vehicle.get_speed(i) > threshold_speed and env.unwrapped.k.vehicle.get_leader(i) is not None]
        shock_ids = np.random.choice(current_shockable_vehicle_ids, sample_vehicles)
        #print(f"\n\nShock ids: {shock_ids}\n\n")

    if shock_counter < frequency: # '<' because shock counter starts from zero
        if step >= shock_times[shock_counter][0] and step <= shock_times[shock_counter][1]:

            #print(f"Step = {step}, Shock params: {intensities[shock_counter], durations[shock_counter], frequency} applied to vehicle {shock_ids}\n")
            
            for controller in controllers:
                controller.set_shock_accel(intensities[shock_counter])
                controller.set_shock_time(True)
    #         # Change color to magenta
            for i in shock_ids:
                env.unwrapped.k.vehicle.set_color(i, (255,0,255))

            current_duration_counter += args.sim_step # 0.1 # increment current duration counter by one timestep seconds 

    return shock_counter, current_duration_counter, shock_ids

def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    
    parser.add_argument('--method',type=str,default=None, help='Method name, can be [vinitsky, ours]')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    parser = update_arguments(parser)
    args = parser.parse_args()

    if args.method is None:
        raise ValueError("Method name must be specified")

    # No policy mapping required here 

    ray.init(num_cpus=1)
    visualizer_rllib(args)
