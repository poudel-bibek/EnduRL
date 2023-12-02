"""
Specific to intersection but works for all classic and learning based controllers
In some situations (such as the middle of intersection), there may not exist certain values: 
    1. Acceleration
    2. x position (all set to -1001)
    3. Space headway can be 1000

The files collect data for all vehicles in the sim, for some metrics, we only need to look at certain flows
"""

import os 
import argparse

import numpy as np
import pandas as pd

class EvalMetrics():
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # Data is expressed as 360.2 instead of 3602 th timestep
        self.start_time = self.args.start_time*self.args.sim_step
        self.end_time = self.args.end_time*self.args.sim_step

    def efficiency(self, ):
        """
        Throughput and Fuel consumption

        FC: For the entire network, i.e., consider all vehicles
        Throughput: Measured in the East-West bound direction. Lanes top0_0_0 and bot0_1_0
        """

        # Collectors across rollouts
        mpgs_avg_mother = []

        for file in self.kwargs['files']:
            print(f"file: {file}")
            self.df = pd.read_csv(file)
            self.vehicle_ids = self.df['id'].unique()

            #############################
            print("####################")

            # Initially populated vehicles list
            initial_population = [f'{self.args.method}_{i}' for i in range(4)] + [f'human_{i}' for i in range(4)]

            fuel_total = [] 
            distances_travelled =[]
            for vehicle_id in self.vehicle_ids:
                # Ignore the initially populated vehicles
                if vehicle_id not in initial_population:
                    vehicle = self.df[self.df['id'] == vehicle_id]

                    # within the time that each vehicle has in the recorded data 
                    veh_time = vehicle['time'].values

                    # If there are values higher than the start time
                    if np.any(veh_time >= self.start_time):
                        # get the fuel 
                        fuel_vehicle = vehicle['fuel_consumption'].values

                        # Just get this in time greater than start time
                        # end time is not specified. because most vehicles exit the network before that
                        fuel_vehicle = fuel_vehicle[veh_time >= self.start_time]
        
                        reverse_ml_to_gallons = (1/0.000264172)
                        # In gallons
                        fuel_vehicle = fuel_vehicle*reverse_ml_to_gallons
                        
                        # correction for density. Sumo measurements are mg/s
                        fuel_vehicle = fuel_vehicle*(1/737)

                        # convert back to ml
                        fuel_vehicle = fuel_vehicle*0.000264172

                        # env step is used here to convert to gallons per time step (because we add the values every time step)
                        fuel_vehicle = fuel_vehicle*self.args.sim_step
                        print(f"fuel_vehicle shape: {fuel_vehicle.shape}")
                        # No need to sum it, it may have different lengths
                        fuel_total.append(fuel_vehicle.astype(object))
                        print(f"fuel: {np.sum(fuel_vehicle)}")
                        distance_vehicle = vehicle['distance_traveled'].values
                        # end time is not specified. because most vehicles exit the network before that
                        distance_vehicle = distance_vehicle[veh_time >= self.start_time] 
                        print(f"distance_vehicle shape: {distance_vehicle.shape}")
                        # now distance travelleed id the subtraction of last one with first one
                        distances_travelled.append(distance_vehicle[-1] - distance_vehicle[0])
                        print(f"distance: {0.000621371*(distances_travelled[-1] - - distance_vehicle[0])}")

                print("\n")
            fuel_total = np.asarray(fuel_total)
            fuel_total_sum = np.asarray([np.sum(x) for x in fuel_total])
            
            vmt = np.asarray(distances_travelled) * 0.000621371 # Meters to miles

            mpgs = vmt/fuel_total_sum
            avg_mpg = np.mean(mpgs)
            mpgs_avg_mother.append(avg_mpg)

        return mpgs_avg_mother

    def safety(self, ):
        """
        Time to Collision and Deceleration rate to avoid a crash
        For both, worst case taken
        Also for both, only control vehicles considered
        """

        pass

    def stability(self, ):
        """
        Only Controller acceleration variation here
        WAR has its own process to measure
        """

        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating metrics for the agent')
    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                    help='Path to emissions file')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--start_time', type=int, default= 800) 
    parser.add_argument('--end_time', type=int, default= 4400) 
    parser.add_argument('--sim_step', type=float, default=0.1)
    parser.add_argument('--num_rollouts', type=int, default=1)

    args = parser.parse_args()
    if args.method is None or args.method not in ['bcm', 'idm', 'fs', 'piws', 'lacc', 'villarreal', 'ours']:
        raise ValueError("Please specify the method to evaluate metrics for\n Method can be [bcm, idm, fs, pi, lacc, villarreal, ours]")
    
    files = [f"{args.emissions_file_path}/{args.method}/{item}" for item in os.listdir(f"{args.emissions_file_path}/{args.method}") \
        if item.endswith('.csv')]
    kwargs = {'files': files,}

    metrics = EvalMetrics(args, **kwargs)
    print(f"Calculating metrics for {args.num_rollouts} rollouts on files: \n{files}\n")

    mpgs = metrics.efficiency()
    metrics.safety()
    metrics.stability()

    print("####################")
    print("####################")
    print("\nFinal Aggregated Metrics (across all files):\n")
    
    print(f"MPG: {mpgs}")
    a = np.round(np.mean(mpgs), 2)
    