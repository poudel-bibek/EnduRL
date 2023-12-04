import os 
import argparse

import numpy as np
import pandas as pd

"""
Notes: 
There seems to be:
    -1001 in various positions (this is when the vehicles are in the zipper)
    1000 in various space ids
    No followers
    No accelerations
    Overlap of the x value existed because the edge starts in the network file was set differently.

Ignore the first 4 vehicles (they are there to populate the simulation)
Ids need to be either classic or flow

IDM exceptions:
    - All vehicles will have IDs flow_00
"""
class EvalMetrics():
    """
    Two metrics each (except WAR so total 5 metrics). Minimal implementation.
    """
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
        # Since the times exist as a fraction (in multiples of sim_time)
        self.start_time = self.args.start_time*self.args.sim_step
        self.end_time = self.args.end_time*self.args.sim_step

        if args.method == 'idm':
            self.reference_id = 'flow_00'
        elif args.method == 'vinitsky' or 'ours':
            self.reference_id = 'flow_10'
        else:
            self.reference_id = 'classic_00'

    def efficiency(self, ):
        """
        Throughput and Fuel consumption
        """

        # Collectors across rollouts
        mpgs_avg_mother = []
        throughput_mother = []

        for file in self.kwargs['files']:
            print(f"file: {file}")
            self.df = pd.read_csv(file)

            #print(f"DF head: \n{self.df.head()}")
            # Time increments in args.sim_step increments

            self.vehicle_ids = self.df['id'].unique()
            #print(f"Vehicle ids: {self.vehicle_ids}")

            #############################
            print("####################")
            # For miles per gallon, you need both distance travelled and fuel consumed

            fuel_total = [] 
            distances_travelled =[]

            for veh_id in self.vehicle_ids:

                # Get the vehicle data, only if vehicle is not `method_0`, `method_1`, `human_0`, `human_1`
                if veh_id not in [f'{self.args.method}_0', f'{self.args.method}_1', 'human_0', 'human_1']:

                    vehicle = self.df[self.df['id'] == veh_id]

                    # TODO: dont consider the shock times
                    #vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                    #not_shock_times = np.logical_not(vehicle_shock_times)

                    # within the time that each vehicle has in the recorded data 
                    veh_time = vehicle['time'].values

                    # Only consider the vehicle if the time lies within the start and end time, only then consider the vehicle data
                    #print(f"Vehicle time: {veh_time}")

                    # If there are values higher than the start time
                    if np.any(veh_time >= self.start_time):
                        #print(f"Vehicle id: {veh_id}")
                        #print(f"Vehicle time: {veh_time}")

                        # get the fuel 
                        fuel_vehicle = vehicle['fuel_consumption'].values
                        # Just get this in time greater than start time
                        fuel_vehicle = fuel_vehicle[veh_time >= self.start_time]

                        reverse_ml_to_gallons = (1/0.000264172)
                        fuel_vehicle = fuel_vehicle*reverse_ml_to_gallons
                        fuel_vehicle = fuel_vehicle*(1/737)
                        fuel_vehicle = fuel_vehicle*0.000264172
                        # env step is used here to convert to gallons per time step (because we add the values every time step)
                        fuel_vehicle = fuel_vehicle*self.args.sim_step # Consider only no_shock_times?

                        #print(f"Fuel vehicle: {fuel_vehicle}") # No need to sum it, it may have different lengths
                        fuel_total.append(fuel_vehicle.astype(object))

                        distance_vehicle = vehicle['distance_traveled'].values
                        # Just get the distance travelled since the time greater than start time
                        distance_vehicle = distance_vehicle[veh_time >= self.start_time]
                        # now distance travelleed id the subtraction of last one with first one
                        distances_travelled.append(distance_vehicle[-1] - distance_vehicle[0])

            fuel_total = np.asarray(fuel_total)

            #print(f"Fuel total before : {fuel_total.shape}")

            fuel_total_sum = np.asarray([np.sum(x) for x in fuel_total])
            vmt = np.asarray(distances_travelled) * 0.000621371 # Meters to miles
        
            #print(f"Fuel total after : {fuel_total_sum.shape}")
            #print(f"Distances travelled: {vmt.shape}")
            #print(f"VMT: {vmt}")
            # print(f"Fuel total: {fuel_total_sum}")

            mpgs = vmt/fuel_total_sum
            #print(f"\nMiles per gallon for each vehicle (mpg):\n{mpgs}\n")

            avg_mpg = np.mean(mpgs)
            #print(f"Average miles per gallon (mpg): {avg_mpg}")
            mpgs_avg_mother.append(avg_mpg)

            #############################
            print("####################")
            # Throughput, number of vehicles that make it to edge 5

            throughput = 0
            # In vehicle_x, there may be -1001 values
            for veh_id in self.vehicle_ids:

                # Get the vehicle data, only if vehicle is not `method_0`, `method_1`, `human_0`, `human_1`
                if veh_id not in [f'{self.args.method}_0', f'{self.args.method}_1', 'human_0', 'human_1']:

                    vehicle = self.df[self.df['id'] == veh_id]
                    veh_time = vehicle['time'].values

                    # If there exists a time greater
                    if np.any(veh_time >= self.start_time):
                        #print(f"Vehicle time: {veh_time}")
                        # Only consider the time greater
                        vehicle_x = vehicle['x'].values
                        #print(f"Length of vehicle_x: {len(vehicle_x)}")
                        vehicle_x = vehicle_x[int(self.start_time):] # Only consider subset
                        #print(f"Length of vehicle_x: {len(vehicle_x)}")

                        # Only consider vehicle_x where time is greater than start time
                        

                        # If vehicle_x is greater then the start of edge 5
                        # Count it once
                        if np.any(vehicle_x >= 980): #-1001 is not
                            #print(f"Vehicle id: {veh_id} made it to edge 5")
                            throughput += 1
                            #print(f"Vehicle id: {veh_id}, vehicle_x: {vehicle_x}")

            #print(f"First: Throughput (vehicles): {throughput}")
            #print(f"Number of vehicles that made it to edge 5: {throughput}")
            # First reverse the multiplication my sim_step (dont in init)
            interest_time = (self.end_time - self.start_time)/self.args.sim_step 
            # Now this is in timesteps, then convert to seconds
            interest_time = interest_time* self.args.sim_step # In seconds
            #print(f"Interest time (seconds): {interest_time}")
            # The way time is recorded here vs the ring, makes this claculation seem little different
            throughput = (throughput/interest_time) # Throughput per second
            #print(f"Second: Throughput (vehicles/ second): {throughput}")
            throughput= throughput*3600 # Throughput per hour
            #print(f"Last: Throughput (vehicles/ hour): {throughput}")
            #print(f"Throughput (vehicles/ hour): {throughput}")
            throughput_mother.append(throughput)

        return mpgs_avg_mother, throughput_mother
    
    def safety(self, ):
        """
        Time to Collision and Deceleration rate to avoid a crash
        For both, worst case taken. Also for both, only control vehicles considered
        """

        ttc_mother = []
        drac_mother = []

        # for file in self.kwargs['files']:
        #     self.df = pd.read_csv(file)
        #     self.vehicle_ids = self.df['id'].unique()
        #     print("####################")

        #     ttcs = [] 
        #     dracs= []
        #     for veh_id in self.vehicle_ids:
        #         if veh_id not in [f'{self.args.method}_0', f'{self.args.method}_1', 'human_0', 'human_1']:
        #             if self.reference_id in veh_id:
        #                 vehicle_subset = self.df[self.df['id'] == veh_id]
                        
        #                 # Further, for there to be a TTC, there must be a leader. The leader is too far then TTC is so large that it does not even matter
        #                 # Mostly at later times, there is a leader present but still perform this check since this is an open network
        #                 # If there is a leader, then only consider those timesteps
        #                 # Get rows where leader_id is not null and time is greater than or equal to start_time
        #                 # Right now, this does not have considerations of end time
        #                 valid_rows = vehicle_subset[vehicle_subset['leader_id'].notnull() & (vehicle_subset['time'] >= self.start_time)]

        #                 for index, row in valid_rows.iterrows():
        #                     time = row['time']
        #                     # Since the leader exists, at a time both leader and follower should have a recorded velocity
        #                     # Get the vehicle velocity at this timestep
        #                     vehicle_velocity = row['speed']
        #                     leader_id = row['leader_id']
                            
        #                     # Fetch leader's speed
        #                     leader_velocity = self.df[(self.df['id'] == leader_id) & (self.df['time'] == time)]['speed'].iloc[0]

        #                     # print(f"Time: {time}")
        #                     # print(f"Vehicle velocity at time {time}: {vehicle_velocity}")
        #                     # print(f"Leader velocity at time {time}: {leader_velocity}")

        #                     relative_positions = row['space_headway'] # when there is a leader, there is a space headway

        #                     # To avoid division by 0, if the relative velocities are 0, then substitute with a very small value
        #                     # In that case TTC will be very high and DRAC will be very small
                              #if np.abs(relative_velocities) < 0.001:
                                # replace but preserve sign
        #                       #relative_velocities = 0.001 if relative_velocities > 0.0 else -0.001

        #                     if relative_velocities < 0.0: # i.e., follower greater than leader
        #                         ttc = relative_positions/relative_velocities
        #                     else:
        #                         ## TTC is arbitrarily large value 
        #                         ttc = 1000.0 
                            
        #                     # Since we take leader - follower, we end up with negative values. Make them positive so that its easier to work with
        #                     ttcs.append(np.abs(ttc))
        #                     #print(f"TTC: {ttc}")


        #                     # for drac 
        #                     # Only if follower has velocity greater than leader i.e., when relative velocity is negative
        #                     if relative_velocities < 0.0: # i.e., follower greater than leader
        #                         relative_velocities_squared = np.square(relative_velocities)
        #                         drac = relative_velocities_squared/relative_positions
        #                     else: 
        #                         # Otherwise DRAC is 0.0 
        #                         drac = 0.0

        #                     #print(f"DRAC: {drac}")
        #                     dracs.append(drac)
            
        #     # Now take the worst case TTC (lowest)
        #     worst_ttc = np.min(ttcs)
        #     ttc_mother.append(worst_ttc)

        #     # Now take the worst case DRAC (highest)
        #     worst_drac = np.max(dracs)
        #     drac_mother.append(worst_drac)

        # ChatGPT optimized version 
        for file in self.kwargs['files']:
            self.df = pd.read_csv(file)
            self.vehicle_ids = self.df['id'].unique()
            print("####################")

            ttcs = [] 
            dracs= []
            
            # Precompute leader velocities for the entire dataframe once
            leader_velocities = self.df.set_index(['id', 'time'])['speed'].to_dict()
            
            for veh_id in self.vehicle_ids:
                if veh_id not in [f'{self.args.method}_0', f'{self.args.method}_1', 'human_0', 'human_1'] and self.reference_id in veh_id:
                    
                    valid_rows = self.df.loc[
                        (self.df['id'] == veh_id) & 
                        (self.df['leader_id'].notnull()) & 
                        (self.df['time'] >= self.start_time)
                    ]
                    
                    for _, row in valid_rows.iterrows():
                        time = row['time']
                        vehicle_velocity = row['speed']
                        leader_id = row['leader_id']
                        
                        # Use precomputed leader velocities
                        leader_velocity = leader_velocities.get((leader_id, time), 0.0) # defaulting to 0.0 if not found
                        #print(f"Time: {time}, Vehicle velocity: {vehicle_velocity}, Leader velocity: {leader_velocity}")
                        relative_positions = row['space_headway']
                        relative_velocities = leader_velocity - vehicle_velocity
                        if np.abs(relative_velocities) < 0.001:
                            # replace but preserve sign
                            relative_velocities = 0.001 if relative_velocities > 0.0 else -0.001

                        ttc = 1000.0  # arbitrarily value
                        if relative_velocities < 0.0: # i.e., follower greater than leader
                            ttc = relative_positions / relative_velocities
                        
                        ttcs.append(np.abs(ttc))

                        drac = 0.0  # default value
                        if relative_velocities < 0.0: # i.e., follower greater than leader
                            drac = np.square(relative_velocities) / relative_positions
                        dracs.append(drac)

            worst_ttc = np.min(ttcs)
            ttc_mother.append(worst_ttc)
            worst_drac = np.max(dracs)
            drac_mother.append(worst_drac)

        return ttc_mother, drac_mother

    def stability(self, ):
        """
        Only Controller acceleration variation here
        WAR has its own process to measure
        """
        
        cav_mother = [] # across rollouts 

        for file in self.kwargs['files']:
            #print(f"file: {file}")
            self.df = pd.read_csv(file)

            self.vehicle_ids = self.df['id'].unique()
            #print(f"Vehicle ids: {self.vehicle_ids}")

            #############################
            print("####################")

            # For controller acceleration variation, get all the controlled vehicles 

            cavs = [] 
            for veh_id in self.vehicle_ids:

                # Get the vehicle data, only if vehicle is not `method_0`, `method_1`, `human_0`, `human_1`
                if veh_id not in [f'{self.args.method}_0', f'{self.args.method}_1', 'human_0', 'human_1']:
                    
                    # Further, we consider CAV only for controller vehicles which have ids `classic_00` in them
                    if self.reference_id in veh_id:

                        vehicle = self.df[self.df['id'] == veh_id]

                        # within the time that each vehicle has in the recorded data 
                        veh_time = vehicle['time'].values

                        # Only consider the vehicle if the time lies within the start and end time, only then consider the vehicle data
                        #print(f"Vehicle time: {veh_time}")

                        # If there are values higher than the start time
                        if np.any(veh_time >= self.start_time):

                            # Only consider the time greater
                            vehicle_accel = vehicle['realized_accel'].values
                            vehicle_accel = vehicle_accel[veh_time >= self.start_time]

                            # Measure the standard deviation. This is always going to be positive
                            std_dev = np.std(vehicle_accel)
                            #print(f"Standard deviation of acceleration for vehicle {veh_id}: {std_dev}")

                            cavs.append(std_dev)

            worst_cav = np.max(cavs)
            #print(f"Worst CAV: {worst_cav}")
            # From a single rollout, take the worst case CAV
            cav_mother.append(worst_cav)

        return cav_mother
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating metrics for the agent')
    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                    help='Path to emissions file')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--horizon', type=int, default= 8000 ) # 1500
    parser.add_argument('--warmup', type=int, default= 3000) # 100
    parser.add_argument('--start_time', type=int, default= 4400) # 780
    parser.add_argument('--end_time', type=int, default= 8000) # 1500
    parser.add_argument('--num_rollouts', type=int, default=1)

    # This is going to be useful for Vinitsky
    parser.add_argument('--sim_step', type=float, default=0.1)

    args = parser.parse_args()
    if args.method is None or args.method not in ['bcm', 'idm', 'fs', 'pi', 'lacc', 'vinitsky', 'ours', 'ours4x','ours9x']:
        raise ValueError("Please specify the method to evaluate metrics for\n Method can be [bcm, idm, fs, pi, lacc, wu, ours]")
    
    files = [f"{args.emissions_file_path}/{args.method}/{item}" for item in os.listdir(f"{args.emissions_file_path}/{args.method}") \
        if item.endswith('.csv')]

    kwargs = {'files': files,}

    metrics = EvalMetrics(args, **kwargs)
    print(f"Calculating metrics for {args.num_rollouts} rollouts on files: \n{files}\n")

    mpgs_mother, throughput_mother = metrics.efficiency()
    ttc_mother, drac_mother = metrics.safety()
    cav_mother = metrics.stability()

    print("####################")
    print("####################")
    print("\nFinal Aggregated Metrics (across all files):\n")
    print("Safety:")
    print(f"Time to collision (TTC): \n{ttc_mother}\n\n")
    print(f"Deceleration rate to avoid crash (DRAC): \n{drac_mother}\n\n")
    print("Efficiency:")
    print(f"Miles per gallon (MPG): \n{mpgs_mother}\n\n")
    print(f"Throughput (vehicles/hour): \n{throughput_mother}\n\n")
    print("Stability:")
    print(f"Controller acceleration variation (CAV): \n{cav_mother}\n\n")

    # Probably have to take means of all across rollouts here
    # ttc, drac, fuel, throughput, cav
    a = round(np.mean(ttc_mother),2)
    b = round(np.mean(drac_mother),2)
    c = round(np.mean(mpgs_mother),2)
    d = round(np.mean(throughput_mother),2)
    e = round(np.mean(cav_mother),2)

    print(f"${a}$ & ${b}$ & ${c}$ & ${int(d)}$ & ${e}$ ")