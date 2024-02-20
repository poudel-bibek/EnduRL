"""
Velocity plot 1: Average velocity + Minimum velocity (optional) of a single environment (No shock)
Velocity plot 2: Average velocity + Minimum velocity (optional) of a single environment (Shock)
Velocity plot 3: Comparative velocity plot of two or more different environments
Space time plot:
# If we deal with 10 files, do we plot for every?

Use this file only to plot stability figure, all other figures are plotted through the eval_metrics file
"""
import os 
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
#sns.set_palette(palette='magma', n_colors = 15)
sns.set_style('darkgrid')

class Plotter:

    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.file = self.kwargs['files'][0]
        self.dataframe = pd.read_csv(self.file)
        self.vehicle_ids = self.dataframe['id'].unique()
        self.num_rollouts = len(self.kwargs['files'])

        self.method_name = self.file.split('/')[-1].split('_')[0]
        self.method_name = self.method_name[0].upper() + self.method_name[1:]

        self.warmup_time = self.args.warmup
        self.start_time = self.args.start_time
        self.end_time = self.args.end_time

        self.save_dir = kwargs['plots_dir'] #self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # pass args as well
    def plot_speeds(self, ):
        
        fig, ax = plt.subplots(figsize=(16, 5), dpi =100)
        # Collect average speed line across files 
        avg_speeds_collector = [] 
        for file in self.kwargs['files']:
            

            self.dataframe = pd.read_csv(file)
            self.vehicle_ids = self.dataframe['id'].unique()

            # Speed of all vehicles across time, for one file
            speeds_total = []
            for vehicle_id in self.vehicle_ids:
                speed = self.dataframe[self.dataframe['id'] == vehicle_id]['speed']
                speeds_total.append(speed)
            speeds_total = np.array(speeds_total)

            speeds_avg = np.mean(speeds_total, axis=0)
            #print(f"Speeds total: {speeds_total.shape}, avg =  {speeds_avg.shape}\n")
            avg_speeds_collector.append(speeds_avg)
            
        # Plot the average speed line across files and error bars
        avg_speeds_collector = np.array(avg_speeds_collector)
        avg_speeds = np.mean(avg_speeds_collector, axis=0)
        std_speeds = np.std(avg_speeds_collector, axis=0)
        ax.plot(avg_speeds, label='Average speed')
        ax.fill_between(np.arange(len(avg_speeds)), avg_speeds - std_speeds, avg_speeds + std_speeds, alpha=0.2)
        ax.set_title(f'{self.method_name}: Average speed of vehicles across {self.num_rollouts} rollouts')

        # Vertical lines at warming up and shock start and end times
        ax.axvline(x=self.args.warmup, color='r', linestyle='--', label='Warmup time')
        ax.axvline(x=self.args.start_time, color='g', linestyle='--', label='Shock start time')
        ax.axvline(x=self.args.end_time, color='g', linestyle='--', label='Shock end time')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_ylim(0,6.5)
        ax.legend()
        plt.savefig(self.save_dir + '/speeds.png')

        #plt.show()
    

    def plot_space_time(self, ):
        return 0

    def plot_time_headway_distribution(self, ):
        # Plot the headway distribution , x-axis is the headway, y-axis is the frequency
        # Scale this up for each vehicle?
        return 0

    def plot_ttc(self, ):
        # Plot the time to collision distribution , x-axis is the time to collision, y-axis is the frequency
        # Scale this up for each vehicle?
        return 0

    def plot_fuel_consumption(self, avg_mpg, std_mpg):
        print(f"MPG: {avg_mpg}, {std_mpg}")

        # fig, ax = plt.subplots(figsize=(16, 5), dpi =100)

        # ax.bar(np.arange(len(avg_mpg)), avg_mpg, yerr=std_mpg, align='center', alpha=0.5, ecolor='black', capsize=10)
        # ax.set_ylabel('MPG')
        # ax.set_xticks(np.arange(len(avg_mpg)))
        # ax.set_xticklabels(self.vehicle_ids)
        # ax.set_title(f'{self.method_name}: Average MPG across {self.num_rollouts} rollouts')
        # ax.yaxis.grid(True)
        # plt.tight_layout()

        # plt.savefig(self.save_dir + '/mpgs.png')
        

    def plot_throughput(self, ):
        # Plot the throughput distribution , x-axis is the throughput, y-axis is the frequency
        # Scale this up for each vehicle?
        return 0

    def plot_stability(self, ):
        """
        How to aggregate across rollouts?
        A single rollout is fine for now.
        Make use of the fact that we know the order is human_0... human_y, followed by controlled_0... controlled_x
        Leader of the controllers is human_0, followers are human_9, human_8
        """

        print("Generating stability plot.. (Make sure the files are correct)")
        i = 0 
        fontsize = 16
        dampening_ratio_mother= []

        # Generate for each rollout file that was found  
        for file in self.kwargs['files']:
            print(f"File: {file}")
            self.dataframe = pd.read_csv(file)
            self.vehicle_ids = self.dataframe['id'].unique()
            print(f"Vehicles: {self.vehicle_ids}")

            # add if human in id
            n_human = len([item for item in self.vehicle_ids if 'human' in item])
            print(f"Number of human vehicles: {n_human}")

            controlled = [item for item in self.vehicle_ids if 'human' not in item]
            n_controlled = len(controlled)
            print(f"Number of controlled vehicles: {n_controlled}")

            controlled_name = controlled[0].split('_')[0]
            print(f"Controlled vehicle name: {controlled_name}")
            
            if self.args.method == 'ours4x': 
                # human_3_0 is shocker (lead)
                sorted_ids = ['human_3_0'] + ['rl_3_0','rl_2_0','rl_1_0','rl_0_0']
                sorted_ids = sorted_ids + [f'human_3_{item}' for item in range(n_human-1, 0, -1)]
                end = n_controlled

            elif self.args.method == 'ours9x':
                sorted_ids = ['human_8_0'] + ['rl_8_0', 'rl_7_0', 'rl_6_0', 'rl_5_0', 'rl_4_0','rl_3_0', 'rl_2_0','rl_1_0','rl_0_0']
                sorted_ids = sorted_ids + [f'human_8_{item}' for item in range(n_human-1, 0, -1)]
                end = n_controlled

            else:
                if controlled_name == 'idm':
                    # idm_0 is shocker (lead)
                    sorted_ids = ['idm_0'] + [f'{controlled_name}_{item}' for item in range(1, n_controlled)]
                    end = n_controlled
                else: 
                    # human_0 is shocker (lead)
                    sorted_ids = ['human_0'] + [f'{controlled_name}_{item}' for item in range(n_controlled)]
                    sorted_ids = sorted_ids + [f'human_{item}' for item in range(n_human - 1, 0, -1)]
                    end = n_controlled + 1

            print(f"Sorted ids: {sorted_ids}")
            
            # Speed of all vehicles across time, for one file
            speeds_total = []
            
            # # The speed for RL also contains the warmup time, so we need to remove that
            # if args.method == 'rl':
            #     self.args.start_time = self.args.start_time - self.args.warmup
            #     self.args.end_time = self.args.end_time - self.args.warmup

            for vehicle_id in sorted_ids:
                speed = self.dataframe[self.dataframe['id'] == vehicle_id]['speed']

                #print(f"Speed: {speed.shape}")

                # speed in the time window
                # shock itself will last at most like 20 timesteps. 
                # It will take some time for that shock to propagate to the floower
                speed = speed[self.args.start_time : self.args.start_time + args.propogate_time]
                speeds_total.append(speed)

            speeds_total = np.array(speeds_total)
            print(f"Speeds total: {speeds_total.shape}\n")

            fig, ax = plt.subplots(figsize=(16, 5), dpi =100)

            # plot leader
            ax.plot(speeds_total[0], label='Leading human', color='black')
            
            # plot controlled 
            for j in range(1, end):
                ax.plot(speeds_total[j], color='slateblue') # , label=f'{controlled_name}_{j - 1}',

            if controlled_name != 'idm':
                # plot all followers
                # for k in range(n_controlled + 1, speeds_total.shape[0]):
                #     ax.plot(speeds_total[k], color='teal') #  label=f'Human_{k - n_controlled}'
                # ax.plot([], [], label=f'Follower human', color='teal') 
                ax.plot(speeds_total[n_controlled +1], label=f'Follower human', color='teal')
            # Add a label for each color black, slateblue, teal
            ax.plot([], [], label=f'{controlled_name}', color='slateblue')
            
            ax.legend()
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title(f'{self.method_name}: Stability plot for rollout {i}')
            ax.yaxis.grid(True)

            # Set font sizes
            ax.xaxis.label.set_size(fontsize)
            ax.yaxis.label.set_size(fontsize)
            ax.title.set_size(fontsize)

            # Set legend font size
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

            plt.tight_layout()
            plt.savefig(self.save_dir + f'/stability_{i}.png')
            i+=1 

            # # Calculate the metric (Damping/ Wave attenuation ratio) between human_0 and human_9
            # lowest_speed_lead = np.min(speeds_total[0])
            # print(f"Lowest speed of lead (index 0): {lowest_speed_lead}")
            # if controlled_name == 'idm':
            #     lowest_speed_follow = np.min(speeds_total[2])
            #     print(f"Lowest speed of follow (index 2): {lowest_speed_follow}")
            # else:
            #     lowest_speed_follow = np.min(speeds_total[end])
            #     print(f"Lowest speed of follow (index {end}): {lowest_speed_follow}")
            
            # WAR = 1 - (velocity drop of follower)/(Velocity drop of leader)
            lowest_speed_lead = np.min(speeds_total[0])
            highest_speed_lead = np.max(speeds_total[0])
            velocity_drop_lead = highest_speed_lead - lowest_speed_lead 
            print(f"Lead: Lowest speed: {lowest_speed_lead}\tHighest speed: {highest_speed_lead}\tVelocity drop: {velocity_drop_lead}")

            if controlled_name == 'idm':
                lowest_speed_follow = np.min(speeds_total[-1]) # This is the one that immediately follows the shocker
                highest_speed_follow = np.max(speeds_total[-1])
                velocity_drop_follow = highest_speed_follow - lowest_speed_follow
            else: 
                lowest_speed_follow = np.min(speeds_total[end])
                highest_speed_follow = np.max(speeds_total[end])
                velocity_drop_follow = highest_speed_follow - lowest_speed_follow

            print(f"Follow: Lowest speed: {lowest_speed_follow}\tHighest speed: {highest_speed_follow}\tVelocity drop: {velocity_drop_follow}")
            
            dampening_ratio = 1 - (velocity_drop_follow/velocity_drop_lead)
            dampening_ratio_mother.append(dampening_ratio)
            

        #############################
        print("\n####################")
        print(f'\nDampening ratio: {dampening_ratio_mother}\nAvg = {round(np.mean(dampening_ratio_mother),2)}\tStd = {round(np.std(dampening_ratio_mother),2)}')
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                        help='Path to emissions file')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--horizon', type=int, default=15000)
    parser.add_argument('--warmup', type=int, default=2500)

    parser.add_argument('--start_time', type=int, default=8000)
    parser.add_argument('--end_time', type=int, default=11500) # Warmup + Horizon

    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./metrics_plots')
    parser.add_argument('--propogate_time', type=int, default=100)

    args = parser.parse_args()
    if args.method is None or args.method not in ['bcm', 'idm', 'fs', 'pi', 'lacc', 'wu', 'ours', 'ours4x', 'ours9x']:
        raise ValueError("Please specify the method to evaluate metrics for\n Method can be [bcm, idm, fs, piws, lacc, wu, ours, ours4x, ours9x]")

    files = [f"{args.emissions_file_path}/{args.method}_stability/{item}" for item in os.listdir(f"{args.emissions_file_path}/{args.method}_stability") if item.endswith('.csv')]
    
    # Add more upon necessity
    kwargs = {'files': files,
                'plots_dir': f"{args.save_dir}/{args.method}/"
    }
   

    plotter = Plotter(args, **kwargs)
    plotter.plot_stability()

