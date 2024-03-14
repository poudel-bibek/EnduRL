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

        # Initially populated vehicles list. Exclusion list
        self.initial_population = [f'{self.args.method}_{i}' for i in range(4)] + [f'human_{i}' for i in range(4)]
        
        # Some metrics also require inclusion lists.
        # If classic but not idm them flow_30 and flow_10 is in list
        if self.args.method in ['fs', 'piws', 'bcm', 'lacc', 'villarreal']:
            self.inclusion_list = ['flow_30', 'flow_10'] # Control vehicle flow to include in ttc and drac

        elif self.args.method in ['idm']:
            # Define for idm. Everything? We still look at flow in the North South direction
            self.inclusion_list = ['flow_00', 'flow_10']

        # else: # villarreal, ours # The flow ids of vehicles that are controlled.
        #     self.inclusion_list = ['flow_00', 'flow_10', 'flow_20', 'flow_30']


    def efficiency(self, flag = 1):
        """
        Throughput and Fuel consumption

        FC: For the entire network, i.e., consider all vehicles
        Throughput: Measured in the East-West bound direction. Lanes top0_0_0 and bot0_1_0
        This is weird. The controller is stabilizing flow in the north-south direction. Why measure throughput in east-west direction as its performance?
        
        Since the controller is stabilizing traffic, 
        For RL, the reward was incentivizing the flow in east-west direction.

        # flag = 0 means throughput of all directions
        # flag = 1 means throughput of only East West
        # flag = 2 means throughput of North South
        """

        # Collectors across rollouts
        mpgs_avg_mother = []
        throughput_mother = []

        for file in self.kwargs['files']:
            print(f"file: {file}")
            self.df = pd.read_csv(file)
            self.vehicle_ids = self.df['id'].unique()

            #############################
            print("####################")

            fuel_total = [] 
            distances_travelled =[]
            for vehicle_id in self.vehicle_ids:
                # Ignore the initially populated vehicles
                if vehicle_id not in self.initial_population:
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
                        #print(f"fuel_vehicle shape: {fuel_vehicle.shape}")
                        # No need to sum it, it may have different lengths
                        fuel_total.append(fuel_vehicle.astype(object))
                        #print(f"fuel: {np.sum(fuel_vehicle)}")
                        distance_vehicle = vehicle['distance_traveled'].values
                        # end time is not specified. because most vehicles exit the network before that
                        distance_vehicle = distance_vehicle[veh_time >= self.start_time] 
                        #print(f"distance_vehicle shape: {distance_vehicle.shape}")
                        # now distance travelleed id the subtraction of last one with first one
                        distances_travelled.append(distance_vehicle[-1] - distance_vehicle[0])
                        #print(f"distance: {0.000621371*(distances_travelled[-1] - - distance_vehicle[0])}")

                #print("\n")
            fuel_total = np.asarray(fuel_total, dtype=object)
            fuel_total_sum = np.asarray([np.sum(x) for x in fuel_total])
            
            vmt = np.asarray(distances_travelled) * 0.000621371 # Meters to miles

            mpgs = vmt/fuel_total_sum
            avg_mpg = np.mean(mpgs)
            mpgs_avg_mother.append(avg_mpg)

            #############################
            print("####################")
            all_lanes = ['top0_0', 'bot0_1', 'right1_0', 'left0_0']
            east_west_lanes = ['top0_0', 'bot0_1']
            north_south_lanes = ['right1_0', 'left0_0']
        
            if flag == 0:
                lanes = all_lanes
            elif flag == 1:
                lanes = east_west_lanes
            elif flag == 2:
                lanes = north_south_lanes

            # If time is greater than start time and vehicles are in those lanes count them once    
            throughput = 0
            for vehicle_id in self.vehicle_ids:
                # exclusion list 
                if vehicle_id not in self.initial_population:

                    vehicle = self.df[self.df['id'] == vehicle_id]
                    veh_time = vehicle['time'].values

                    # If there are values higher than the start time
                    if np.any(veh_time >= self.start_time):

                        # Just get this in time greater than start time
                        veh_edge = vehicle['edge'].values
                        veh_edge = veh_edge[veh_time >= self.start_time]

                        # If the vehicle is in the lanes of interest
                        if any(item in veh_edge for item in lanes):
                            throughput += 1

            # Convert to vehicles per hour
            interest_time = (self.end_time - self.start_time)/self.args.sim_step 
            # Now this is in timesteps, then convert to seconds
            interest_time = interest_time* self.args.sim_step # In seconds
            throughput = (throughput/interest_time) # Throughput per second
            throughput= throughput*3600 # Throughput per hour 

            #print(f"Throughput: {throughput}")
            throughput_mother.append(throughput)

        return mpgs_avg_mother, throughput_mother

    def safety(self, ):
        """
        Time to Collision and Deceleration rate to avoid a crash
        For both, worst case taken. Also for both, only control vehicles considered.

        Space Headway (Relative position) can have values 1000. When?
        Its crazy how compute heavy the code below is
        """

        ttc_mother = []
        drac_mother = []
        for file in self.kwargs['files']:
            self.df = pd.read_csv(file)
            vehicle_ids = self.df['id'].unique()
            print("####################")

            ttcs = [] 
            dracs= []
            
            # Precompute velocities (all vehicle types) for the entire dataframe once
            velocities = self.df.set_index(['id', 'time'])['speed'].to_dict()
            headways = self.df.set_index(['id', 'time'])['space_headway'].to_dict()

            for veh_id in vehicle_ids:

                # Just look at the control vehicles. #If any item in self.inclusion_list is in vehicle id
                if any(item in veh_id for item in self.inclusion_list):
                    
                    # This is vehicle df
                    vehicle = self.df[self.df['id'] == veh_id]
                    leader_id = vehicle['leader_id'].values[0] # Since there are no lane changes, will always have a unique value
                    #print(f"leader_id: {leader_id}")
                    
                    # This is leader df
                    leader = self.df[self.df['id'] == leader_id]

                    # Just select the speeds after the start time
                    veh_time = vehicle['time'].values
                    veh_time = veh_time[veh_time >= self.start_time]

                    # Ignore all the times where the space headway is 1000
                    veh_time = np.array([i for i in veh_time if headways.get((veh_id, i), 0) != 1000])

                    veh_speed = np.array([velocities.get((veh_id, i), 0) for i in veh_time])

                    # Since leader will exit the network before the vehicle, replace the nan with the last value
                    last_speed = leader['speed'].values[-1]
                    leader_speed = np.array([velocities.get((leader_id, i), last_speed ) for i in veh_time])
                    
                    #print(f"veh_speed: {veh_speed.shape}")
                    #print(f"leader_speed: {leader_speed.shape}")

                    # relative speeds across all time steps
                    rel_speeds = veh_speed - leader_speed
                    
                    # print(f"rel_speeds: {rel_speeds}")
                    # print(f"rel_speeds: {rel_speeds.shape, type(rel_speeds)}")
                    
                    new_rel_speeds = []
                    for speed in rel_speeds:
                        if np.abs(speed) < 0.001:
                            new_speed = 0.001 if speed > 0.0 else -0.001
                            new_rel_speeds.append(new_speed)
                        else: 
                            new_rel_speeds.append(speed)

                    new_rel_speeds = np.array(new_rel_speeds)
                    
                    # Times do not contain 1000. Keep None as replacement. Ideally this should not be a problem
                    rel_positions = np.array([headways.get((veh_id, i), None) for i in veh_time])
                    
                    # print(f"rel_speeds: {new_rel_speeds.shape}")
                    # print(f"rel_positions: {rel_positions.shape}")
                    
                    # New rel speed less than zero means, follower speed is less than leader speed and crash is NOT possible 
                    # Hence the > sign 
                    # 1000.0 # arbitrarily large number
                    ttc = np.array([ np.abs(rel_positions[i]/ new_rel_speeds[i]) if new_rel_speeds[i] > 0.0 else 1000.0 for i in range(len(new_rel_speeds))])
                    # print(f"ttc: {ttc.shape}")
                    # print(f"ttc: {ttc}")
                    ttcs.extend(ttc) # For all vehicles collect first

                    # default small value 0.0
                    drac = np.array([ np.square(new_rel_speeds[i]) / rel_positions[i] if new_rel_speeds[i] > 0.0 else 0.0 for i in range(len(new_rel_speeds))])

                    # print vehicle id and drac if it is higher than 20 
                    # if np.max(drac) > 20.0:
                    #     print(f"veh_id: {veh_id}")
                    #     print(f"drac: {np.max(drac)}")
                    
                    #     # At what timesteps is the drac higher than 20 and what are the other conditions
                    #     print(f" timestep where drac is higher than 20: {veh_time[drac > 20.0]}")
                    #     print(f" speeds at those timesteps: {veh_speed[drac > 20.0]}")
                    #     print(f" leader speeds at those timesteps: {leader_speed[drac > 20.0]}")
                    #     print(f" rel_speeds at those timesteps: {new_rel_speeds[drac > 20.0]}")
                    #     print(f" rel_positions at those timesteps: {rel_positions[drac > 20.0]}")
                    #     print(f" ttc at those timesteps: {ttc[drac > 20.0]}")
                    #     print(f" drac at those timesteps: {drac[drac > 20.0]}")
                       
                   
                    dracs.extend(drac)
            

            # Every file has one worst ttc and drac
            # worst ttcs and worst dracs
            ttc_mother.append(np.min(ttcs))
            drac_mother.append(np.max(dracs))
            
            # print(f"ttc_mother: {ttc_mother}")
            # print(f"drac_mother: {drac_mother}")

        return ttc_mother, drac_mother
        
    def stability(self, ):
        """
        Only Controller acceleration variation here
        WAR has its own process to measure
        """
        cav_mother = [] 

        for file in self.kwargs['files']:
            self.df = pd.read_csv(file)
            vehicle_ids = self.df['id'].unique()
            print("####################")

            cavs = [] 
            for veh_id in vehicle_ids:
                # Just look at the control vehicles. #If any item in self.inclusion_list is in vehicle id
                if any(item in veh_id for item in self.inclusion_list):

                    vehicle = self.df[self.df['id'] == veh_id]
                    veh_time = vehicle['time'].values

                    # If there are values higher than the start time
                    if np.any(veh_time >= self.start_time):

                        # Only consider the time greater
                        vehicle_accel = vehicle['realized_accel'].values
                        #print(f"vehicle_accel: {vehicle_accel.shape}")
                        vehicle_accel = vehicle_accel[veh_time >= self.start_time]

                        # Measure the standard deviation. This is always going to be positive
                        std_dev = np.std(vehicle_accel)
                        cavs.append(std_dev)
            
            worst_cav = np.max(cavs)
            cav_mother.append(worst_cav)

        return cav_mother

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

    mpgs, throughputs = metrics.efficiency(flag = 1)
    ttcs, dracs = metrics.safety()
    cavs = metrics.stability()

    print("####################")
    #print("####################")
    print("\nFinal Aggregated Metrics (across all files):\n")
    
    print(f"MPG: {mpgs}")
    print(f"Throughput: {throughputs}")
    print(f"TTCS: {ttcs}")
    print(f"DRACS: {dracs}")
    print(f"CAVS: {cavs}")
    

    # Take mean across rollouts
    # ttc & drac & mpgs & throughput & cav
    a = np.round(np.mean(ttcs), 2)
    b = np.round(np.mean(dracs), 2)
    c = np.round(np.mean(mpgs), 2)
    d = int(np.round(np.mean(throughputs), 2))
    e = np.round(np.mean(cavs), 2)

    print("####################")
    print("\n")
    print(f"& ${a}$ & ${b}$ & ${c}$ & ${d}$ & ${e}$")
    