"""
Mostly only for stability analysis
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
        self.start_time = self.args.start_time
        self.end_time = self.args.end_time

        self.method_name = self.file.split('/')[-1].split('_')[0]
        self.method_name = self.method_name[0].upper() + self.method_name[1:]

        self.save_dir = kwargs['plots_dir'] 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_stability(self, ):
        """ 
        Shock vehicle (leader): flow_00.4
        Follower: flow_10.1 # Controller
        Follower HV: flow_00.5

        WAR = 1- delta_follower/delta_leader
        delta_lead = velocity drop in leader
        delta_follower = velocity drop in follower # Controller
        """

        print("Generating stability plot.. (Make sure the files are correct)")
        i = 0 
        fontsize = 16
        dampening_ratio_mother= []

        for i in range(len(self.kwargs['files'])):

            file = self.kwargs['files'][i]
            print(f"File: {file}")
            self.dataframe = pd.read_csv(file)
            self.vehicle_ids = self.dataframe['id'].unique()
            #print(f"Vehicles: {self.vehicle_ids}")

            speeds_leader = []
            speeds_follower = [] # RV 
            speeds_follower_hv = [] # HV after RV

            # get the data around start_time for the 3 relevant vehicles    
            for vehicle in self.vehicle_ids:
                #print(f"Vehicle: {vehicle}")
                vehicle_df = self.dataframe[self.dataframe['id'] == vehicle]
                #print(vehicle_df)

                # Speeds start at 0 and then increment every timestep 
                if vehicle == 'flow_00.4':
                    speeds_leader = vehicle_df['speed'].values
                    #print("Leader: ", speeds_leader)
                    #print(f"Shape: {speeds_leader.shape}")
                    #print("Leader first", speeds_leader[0])
                          
                elif vehicle == 'flow_10.1':
                    speeds_follower = vehicle_df['speed'].values
                    #print("Follower: ", speeds_follower)
                    #print(f"Shape: {speeds_follower.shape}")

                elif vehicle == 'flow_00.5':
                    speeds_follower_hv = vehicle_df['speed'].values
                    #print("Follower HV: ", speeds_follower_hv)
                    #print(f"Shape: {speeds_follower_hv.shape}")
        
            fig, ax = plt.subplots(figsize=(16, 5), dpi =100)

            # plot speeds from start_time - propogate_time/2 to start_time + propogate_time/2
            strt = -150 + self.args.start_time - int(self.args.propogate_time/2)
            end = -150 + self.args.start_time + int(self.args.propogate_time/2)

            speeds_leader = np.asarray(speeds_leader[strt:end])
            speeds_follower = np.asarray(speeds_follower[strt:end])
            speeds_follower_hv = np.asarray(speeds_follower_hv[strt:end])

            #print(f"Leader: {speeds_leader.shape}")
            #print(f"Follower: {speeds_follower.shape}")
            #print(f"Follower HV: {speeds_follower_hv.shape}")

            ax.plot(speeds_leader, label = 'Leader')
            ax.plot(speeds_follower, label = 'Follower')
            ax.plot(speeds_follower_hv, label = 'Follower HV')

            ax.legend()
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title(f'{self.method_name}: Stability plot for rollout {i}')

            # Set font sizes
            ax.xaxis.label.set_size(fontsize)
            ax.yaxis.label.set_size(fontsize)
            ax.title.set_size(fontsize)
            
            plt.tight_layout()
            #plt.show()
            plt.savefig(self.save_dir + f'/stability_{i}.png')

            # Look at WAR only in the first 100 timesptes (look at plot)
            speeds_leader = speeds_leader[0:100]
            speeds_follower = speeds_follower[0:100]
            speeds_follower_hv = speeds_follower_hv[0:100]

            # Calculation of dampening ratio. In the given start: end window
            delta_leader = speeds_leader[0] - np.min(speeds_leader)  # Shocker vehicle
            delta_follower = speeds_follower[0] - np.min(speeds_follower) # Controller

            war = 1 - (delta_follower/ delta_leader)
            dampening_ratio_mother.append(war)
        
        print(f"WAR: {dampening_ratio_mother}")
        print(f"Mean WAR: {round(np.mean(dampening_ratio_mother),2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                        help='Path to emissions file')
    parser.add_argument('--method', type=str, default=None)

    parser.add_argument('--horizon', type=int, default = 3600)
    parser.add_argument('--warmup', type=int, default = 400)

    parser.add_argument('--start_time', type=int, default = 400)
    parser.add_argument('--end_time', type=int, default = 4000) # Warmup + Horizon

    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./metrics_plots')
    parser.add_argument('--propogate_time', type=int, default = 300)

    args = parser.parse_args()
    if args.method is None or args.method not in ['bcm', 'fs', 'piws', 'lacc', 'villarreal', 'ours']:
        raise ValueError("Please specify the method to evaluate metrics for\n Method can be [bcm, fs, piws, lacc, villarreall, ours]")

    files = [f"{args.emissions_file_path}/{args.method}_stability/{item}" for item in os.listdir(f"{args.emissions_file_path}/{args.method}_stability") if item.endswith('.csv')]
    
    # Add more upon necessity
    kwargs = {'files': files,
                'plots_dir': f"{args.save_dir}/{args.method}/"
    }
   

    plotter = Plotter(args, **kwargs)
    plotter.plot_stability()