"""
Three metrics each for Safety, Efficiency and Stability

"""
import os 
import argparse

import numpy as np
import pandas as pd

from eval_plots import Plotter

class EvalMetrics():
    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.emissions_file_path = self.args.emissions_file_path
        self.horizon = self.args.horizon
        self.warmup = self.args.warmup

        # Set the start and end time for the evaluation
        self.start_time = self.args.start_time
        self.end_time = self.args.end_time

        self.file = self.kwargs['files'][0]
        self.dataframe = pd.read_csv(self.file)
        self.vehicle_ids = self.dataframe['id'].unique()
        print(f"Vehicle ids: {self.vehicle_ids}\n")

        self.plotter = Plotter(args, **kwargs)
        
        # only required for plots (Maybe later required to save the eval metrics data)
        # self.save_dir = kwargs['plots_dir'] #self.args.save_dir
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)


    def safety(self, ):
        """
        1. Time to Collision: 
        Only measured for controlled vehicles, if the leader happens to apply shock, that data is discarded? NO, its controllers responsibility
        2. Time Headway 
        3. Variation of Acceleration/ Deceleration rate during shocks
        """
        
        ttc_worst_mother = []
        ttc_best_mother = []
        ttc_avg_mother = []
        ttc_std_mother = []
        
        drac_worst_mother = []

        for file in self.kwargs['files']:
            self.dataframe = pd.read_csv(file)
            
            #filter for each vehicle
            self.vehicle_ids = self.dataframe['id'].unique()

            time_to_collision_total = []

            for vehicle_id in self.vehicle_ids:
                
                if args.method == "idm":
                    if vehicle_id=="idm_0":
                        # Get the dataframe for each vehicle
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]
                        
                        # Since the vehicles travel in a single lane, leader wont change
                        # Get the leader dataframe
                        leader_id = vehicle['leader_id'].unique()
                        leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]
                        #print(leader.shape)

                        # Omit: check whether the leader or follower produced a shockwave 
                        # -1 and -2 are to check for errors, 0 = no shock_time, 1 = shock_time
                        # If for any vehicle if -1 or -2 is present, it was not a HV that could produce a shockwave
                        # from the column shock_time, we can get the time when the shock was produced
                        # To reduce calculation, we can just look at shock times
                        #leader_shock_times = leader['shock_time'].values[self.start_time:self.end_time]
                        #print(leader_shock_times.shape, leader_shock_times)
                        #vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                        #print(vehicle_shock_times.shape, vehicle_shock_times)

                        # take NOR of the two arrays, shock time will have False
                        #not_shock_times = np.logical_not(np.logical_or(leader_shock_times, vehicle_shock_times))

                        # Instead of doing gymnastics with the positions, we can just use the space headway
                        relative_positions = vehicle['space_headway'].values
                        relative_positions = relative_positions[self.start_time:self.end_time] #[not_shock_times]


                        # leader velocity
                        leader_velocities = leader['speed'].values
                        # current vehicle velocity
                        vehicle_velocities = vehicle['speed'].values

                        assert vehicle_velocities.shape == leader_velocities.shape

                        # Split
                        vehicle_velocities = vehicle_velocities[self.start_time:self.end_time] #[not_shock_times]
                        leader_velocities = leader_velocities[self.start_time:self.end_time] #[not_shock_times]

                        #print(vehicle_velocities.shape, leader_velocities.shape)

                        # relative velocity difference
                        relative_velocities = leader_velocities - vehicle_velocities 

                        # set positive values to 0
                        #relative_velocities[relative_velocities > 0] = 0.01 # to avoid division by 0
                        #print(relative_velocities.shape, relative_velocities)

                        # time to collision
                        time_to_collision = relative_positions/relative_velocities
                        #print(f"Before clip:: {time_to_collision.shape}")

                        # only consider negative values, positive wont collide
                        time_to_collision = time_to_collision[time_to_collision < 0.0]
                        #print(f"After clip:: {time_to_collision.shape}\n")

                        time_to_collision_total.append(time_to_collision.astype(object))
                else:
                    # only for controlled vehicles
                    if "human" not in vehicle_id:
                        # Get the dataframe for each vehicle
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]
                        
                        # Since the vehicles travel in a single lane, leader wont change
                        # Get the leader dataframe
                        leader_id = vehicle['leader_id'].unique()
                        leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]
                        #print(leader.shape)

                        # Omit: check whether the leader or follower produced a shockwave 
                        # -1 and -2 are to check for errors, 0 = no shock_time, 1 = shock_time
                        # If for any vehicle if -1 or -2 is present, it was not a HV that could produce a shockwave
                        # from the column shock_time, we can get the time when the shock was produced
                        # To reduce calculation, we can just look at shock times
                        #leader_shock_times = leader['shock_time'].values[self.start_time:self.end_time]
                        #print(leader_shock_times.shape, leader_shock_times)
                        #vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                        #print(vehicle_shock_times.shape, vehicle_shock_times)

                        # take NOR of the two arrays, shock time will have False
                        #not_shock_times = np.logical_not(np.logical_or(leader_shock_times, vehicle_shock_times))

                        # Instead of doing gymnastics with the positions, we can just use the space headway
                        relative_positions = vehicle['space_headway'].values # This is gap
                        relative_positions = relative_positions[self.start_time:self.end_time] #[not_shock_times]


                        # leader velocity
                        leader_velocities = leader['speed'].values
                        # current vehicle velocity
                        vehicle_velocities = vehicle['speed'].values

                        assert vehicle_velocities.shape == leader_velocities.shape

                        # Split
                        vehicle_velocities = vehicle_velocities[self.start_time:self.end_time] #[not_shock_times]
                        leader_velocities = leader_velocities[self.start_time:self.end_time] #[not_shock_times]

                        #print(vehicle_velocities.shape, leader_velocities.shape)

                        # relative velocity difference
                        relative_velocities = leader_velocities - vehicle_velocities 

                        # set positive values to 0
                        #relative_velocities[relative_velocities > 0] = 0.01 # to avoid division by 0
                        #print(relative_velocities.shape, relative_velocities)

                        # time to collision
                        time_to_collision = relative_positions/relative_velocities
                        #print(f"Before clip:: {time_to_collision.shape}")

                        # only consider negative values, positive wont collide
                        time_to_collision = time_to_collision[time_to_collision < 0.0] # Since we take leader - follower, negative values mean collision. If opposite, postive values mean collision
                    
                        #print(f"After clip:: {time_to_collision.shape}\n")

                        time_to_collision_total.append(time_to_collision.astype(object))
                
            time_to_collision_total = np.asarray(time_to_collision_total)
            print(time_to_collision_total.shape)

            # Aggregations (since we had variable length arrays they are objects)

            # 1. Worst case time to collision for each vehicle
            time_to_collision_worst = np.asarray([np.max(x) for x in time_to_collision_total])
            print(f"Worst case time to collision for each vehicle (s): \n{time_to_collision_worst}\n")

            # 2. Best case time to collision for each vehicle
            time_to_collision_best = np.asarray([np.min(x) for x in time_to_collision_total])
            print(f"Best case time to collision for each vehicle (s): \n{time_to_collision_best}\n")

            # 3. Average time to collision for each vehicle
            time_to_collision_avg = np.asarray([np.mean(x) for x in time_to_collision_total])
            print(f"Average time to collision for each vehicle (s): \n{time_to_collision_avg}\n")

            # 4. Standard deviation of time to collision for each vehicle
            time_to_collision_std = np.asarray([np.std(x) for x in time_to_collision_total])
            print(f"Standard deviation of time to collision for each vehicle (s): \n{time_to_collision_std}\n")

            # A single value 
            worst_ttc = np.max(time_to_collision_worst) # worst of all vehicles (average is doone at multiple rollouts ), this is negative
            best_ttc = np.min(time_to_collision_best) #  best of all vehicles
            avg_ttc = np.mean(time_to_collision_avg) # Average of average of all vehicles
            std_ttc = np.mean(time_to_collision_std) # Average of std of all vehicles

            print(f"Time to Collision (s):\n\tWorst= {round(worst_ttc,2)}\n\tBest= {round(best_ttc,2)}\n\tAverage= {round(avg_ttc,2)}\n\tStd= {round(std_ttc,2)}\n")

            ttc_worst_mother.append(worst_ttc)
            ttc_best_mother.append(best_ttc)
            ttc_avg_mother.append(avg_ttc)
            ttc_std_mother.append(std_ttc)
            
            #############################
            print("####################")

            

            drac_total = []

            for vehicle_id in self.vehicle_ids:

                if args.method == "idm":
                    if vehicle_id=="idm_0":
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                        leader_id = vehicle['leader_id'].unique()
                        leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]

                        leader_velocities = leader['speed'].values
                        # current vehicle velocity
                        vehicle_velocities = vehicle['speed'].values

                        assert vehicle_velocities.shape == leader_velocities.shape

                        # only take relative velocity for vehicle_velocities> leader_velocities # Here we take follower - leader 
                        relative_velocities = vehicle_velocities - leader_velocities
                        relative_positions = vehicle['space_headway'].values # This is gap
                        
                        # First apply start time and end time
                        relative_positions = relative_positions[self.start_time:self.end_time]
                        relative_velocities = relative_velocities[self.start_time:self.end_time]

                        relative_velocities[relative_velocities < 0] = 0.0 

                        # find the indices where relative velocity is less than 0
                        indices = np.where(relative_velocities > 0.0)[0]
                            
                        # apply indices 
                        relative_positions = relative_positions[indices]
                        relative_velocities = relative_velocities[indices]

                        assert relative_velocities.shape == relative_positions.shape
                        
                        # square relative velocities
                        relative_velocities_squared = np.square(relative_velocities)
                        
                        drac = relative_velocities_squared/relative_positions
                        
                        #print(drac.shape)
                        drac_total.append(drac.astype(object)) # contains all drac
                else: 
                    if "human" not in vehicle_id:
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                        leader_id = vehicle['leader_id'].unique()
                        leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]

                        leader_velocities = leader['speed'].values
                        # current vehicle velocity
                        vehicle_velocities = vehicle['speed'].values

                        assert vehicle_velocities.shape == leader_velocities.shape

                        # only take relative velocity for vehicle_velocities> leader_velocities # Here we take follower - leader 
                        relative_velocities = vehicle_velocities - leader_velocities
                        relative_positions = vehicle['space_headway'].values # This is gap
                        
                        # First apply start time and end time
                        relative_positions = relative_positions[self.start_time:self.end_time]
                        relative_velocities = relative_velocities[self.start_time:self.end_time]

                        relative_velocities[relative_velocities < 0] = 0.0 

                        # find the indices where relative velocity is less than 0
                        indices = np.where(relative_velocities > 0.0)[0]
                            
                        # apply indices 
                        relative_positions = relative_positions[indices]
                        relative_velocities = relative_velocities[indices]

                        assert relative_velocities.shape == relative_positions.shape
                        
                        # square relative velocities
                        relative_velocities_squared = np.square(relative_velocities)
                        
                        drac = relative_velocities_squared/relative_positions
                        
                        #print(drac.shape)
                        drac_total.append(drac.astype(object)) # contains all drac
                
            drac_total = np.asarray(drac_total)
            print(drac_total.shape)

            # Aggregations, worst case is the max 
            worst_case_drac = np.asarray([np.max(x) for x in drac_total]) # For each controlled vehicle
            print(f"Worst case drac for each vehicle (m/s^2): \n{worst_case_drac}\n")

            # A singe value
            worst_drac = np.max(worst_case_drac)
            print(f"DRAC (m/s^2):\n\tWorst= {round(worst_drac,2)}\n")

            drac_worst_mother.append(worst_drac)

        return ttc_worst_mother, ttc_best_mother, ttc_avg_mother, ttc_std_mother, drac_worst_mother

    def efficiency(self, ):
        """
        1. Fuel Economy during shocks: Average fuel consumption by the 22 vehicles (Miles per gallon).
        2. Average Speed/Velocity during shocks: After the warmup period, average speed of the 22 vehicles during the shocks.
        3. Throughput as measure by flow (vehicles per hour) in a reference point. A

        """
        # Collectors across rollouts
        mpgs_avg_mother = []
        mpgs_std_mother = []
        speeds_avg_mother = []
        speeds_std_mother = []
        flows_mother = []

        for file in self.kwargs['files']:
            self.dataframe = pd.read_csv(file)
            
            #filter for each vehicle
            self.vehicle_ids = self.dataframe['id'].unique()
            #print(f"Vehicle ids: {self.vehicle_ids}\n")

            #############################
            print("####################")

            # Fuel consumption:
            fuel_total= []
            distances_travelled =[]

            for vehicle_id in self.vehicle_ids:
            
                # Flow converts to gallons per second from sumo default (at the time) ml/ second
                vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                not_shock_times = np.logical_not(vehicle_shock_times)
                
                fuel_vehicle = vehicle['fuel_consumption'].values
                
                # Filter for the start and end time, this is already in gallons per second
                # TODO: Get conversion factor 0.1 from env params
                # The times in which shocks were applied can be excluded
                # Sumo version 1.15 will return mg/s instead of ml /s
                # Flow does ml to gallons conversion
                # First reverse that
                reverse_ml_to_gallons = (1/0.000264172)
                fuel_vehicle = fuel_vehicle*reverse_ml_to_gallons
                # Now this is mg/s 
                # Convert to ml/s per second
                # 1 ml of petrol is 0.737 gram, that is in milligrams 737 mg
                # so 1 mg/s of petrol is 1/737 ml/s
                fuel_vehicle = fuel_vehicle*(1/737)
                # Now this is ml/s
                # Convert to gallons per second
                fuel_vehicle = fuel_vehicle*0.000264172
                fuel_vehicle = fuel_vehicle[self.start_time:self.end_time][not_shock_times]*0.1 # This is env step (0.1)
                #print(vehicle_id, fuel_vehicle.shape, fuel_vehicle)

                # Append the fuel consumed by each vehicle
                fuel_total.append(fuel_vehicle.astype(object))

                distance_vehicle = vehicle['distance_traveled'].values
                distances_during_shock = distance_vehicle[self.start_time:self.end_time][vehicle_shock_times == 1] # Just get distances during shock times
                if distances_during_shock.shape[0] == 0:
                    distances_during_shock = np.array([0, 0])
                distance_travelled_during_shock = distances_during_shock[-1] - distances_during_shock[0]
                #print(vehicle_id, distance_travelled_during_shock)
                #print(vehicle_id, distance_vehicle.shape, distance_vehicle[self.end_time], distance_vehicle[self.start_time])

                # Get the distance traveled by vehicles from start to end (meters)
                distances_travelled_veh = (distance_vehicle[self.end_time] - distance_vehicle[self.start_time]) - distance_travelled_during_shock
                distances_travelled.append(distances_travelled_veh)
                
            fuel_total = np.asarray(fuel_total)
            print(fuel_total.shape)

            # Fuel consumed from start to end (gallons)
            # Since instantaneous average measurements are not required for fuel, we take a sum over the time period
            fuel_total_sum = np.asarray([np.sum(x) for x in fuel_total])
            #print(f"Fuel consumed by each vehicle (gallons): \n{fuel_total_sum}\n")
            
            # vehicle miles traveled (VMT) (Miles)
            vmt = np.asarray(distances_travelled) * 0.000621371
            #print(f"Vehicle miles travelled by each vehicle (miles): \n{vmt}\n")

            # MPG is fuel mileage (Miles per gallon) = VMT / fuel consumed
            mpgs = vmt / fuel_total_sum
            print(f"\nMiles per gallon for each vehicle (mpg):\n{mpgs}\n")

            avg_mpg = np.mean(mpgs)
            std_mpg = np.std(mpgs)
            print(f"Miles per gallon (mpg): Average= {round(avg_mpg,2)}, std = {round( std_mpg,2)}\n")

            mpgs_avg_mother.append(avg_mpg)
            mpgs_std_mother.append(std_mpg)

            #############################
            print("####################")

            # Average speed:
            speeds_total = []
            
            for vehicle_id in self.vehicle_ids: 
                vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                not_shock_times = np.logical_not(vehicle_shock_times)

                # Get the  speed of each vehicle during time of interest
                speed = vehicle['speed'].values[self.start_time:self.end_time][not_shock_times]
                #print(vehicle_id, speed.shape, avg_speed)

                speeds_total.append(speed.astype(object))
            
            speeds_total = np.asarray(speeds_total)
            print(speeds_total.shape)

            # Average of each vehicle first
            speeds_total_avg = np.asarray([np.mean(x) for x in speeds_total]) #np.mean(speeds_total, axis=1)
            print(f"\nAverage speed of each vehicle (m/s): \n{speeds_total_avg}\n")

            speeds_total_std = np.asarray([np.std(x) for x in speeds_total]) #np.std(speeds_total, axis=1)
            print(f"Standard deviation of speed of each vehicle (m/s): \n{speeds_total_std}\n")

            # A single value
            avg_speed = np.mean(speeds_total_avg)
            std_speed = np.mean(speeds_total_std)
            print(f"Speed of all vehicles (m/s): Average= {round(avg_speed,2)}, std={round(std_speed,2)}\n")

            speeds_avg_mother.append(avg_speed)
            speeds_std_mother.append(std_speed)

            #############################
            print("####################")

            # This is a sum 
            # After the shock period started, how many vehicles pass the zero point?
            throughput_total = 0

            for vehicle_id in self.vehicle_ids:
                vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                # Get the position in the time of interest
                position = vehicle['x'].values[self.start_time:self.end_time]

                # How many times does the position cross the zero point?
                # How to identify if it crossed a zero? If position at t-1 is larger than position at t
                zero_crossings = np.sum([1 for i in range(1, len(position)) if position[i-1] > position[i]])
                #print(vehicle_id, zero_crossings)

                throughput_total+= zero_crossings

            print(f"Total number of vehicles that crossed the zero point: {throughput_total}\n")
            
            # TODO: Get the 10 from env_params
            interest_time = (self.end_time - self.start_time) / 10 # seconds
            flow = (throughput_total / interest_time) * 3600 # vehicles/hr

            print(f"Flow (veh/hour): {round(flow,2)}\n")
            flows_mother.append(flow)

        if self.args.save_plots:
            self.plotter.plot_speeds() # Speeds are plotted over entire horizon
            self.plotter.plot_fuel_consumption(np.asarray(mpgs_avg_mother), np.asarray(mpgs_std_mother)) # Fuel is plotted between start and end times

        return mpgs_avg_mother, mpgs_std_mother, speeds_avg_mother, speeds_std_mother, flows_mother


    def stability(self, ):
        """
        1. Time to stabilize : Time interval between the controller activation and the vehicles stabilizing.
                                If the average velocity standard deviation is less than the IDM noise, system = Stable
        2. Minimum number of vehicles required to stabilize: Obtain this from the config files, no need to obtain it from the data
        3. Dampening ratio
        """

        #############################
        print("####################")

        # Measure time to stabilize (tts) in seconds
        # Should only consider the time before the shocks start

        time_headways_worst_mother = []
        time_headways_avg_mother = []
        time_headways_std_mother = []
        tts_mother = []
        cav_mother = []

        for file in self.kwargs['files']:
            print(f"File: {file}")
            self.dataframe = pd.read_csv(file)
            
            #filter for each vehicle
            self.vehicle_ids = self.dataframe['id'].unique()

            # Time Headway (Average and standard deviation)
            time_headway_total = []

            for vehicle_id in self.vehicle_ids:
                # For IDM check for all vehicles
                if args.method == "idm":
                    if vehicle_id == "idm_0":
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                        # Shock times omit code in short 
                        leader_id = vehicle['leader_id'].unique()
                        leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]
                        
                        #leader_shock_times = leader['shock_time'].values[self.start_time:self.end_time]
                        #vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                        #not_shock_times = np.logical_not(np.logical_or(leader_shock_times, vehicle_shock_times))

                        # meter
                        space_headway = vehicle['space_headway'].values[self.start_time:self.end_time] #[not_shock_times]
                        #print(vehicle_id, space_headway.shape, space_headway)
                        
                        # meter per second
                        velocity = vehicle['speed'].values[self.start_time:self.end_time] #[not_shock_times]
                        #print(vehicle_id, velocity.shape, velocity)
                        #print(np.max(velocity), np.min(velocity))

                        # To avoid a divide by zero error and very high time headways at low velocities
                        # If a velocity is less than 0.01 m/s, set it to 0.01 m/s (One centimeter per second),
                        velocity = np.where(velocity < 0.01, 0.01, velocity)
                        #print(np.max(velocity), np.min(velocity))

                        time_headway = space_headway / velocity
                        #print(vehicle_id, time_headway.shape, time_headway)
                        #print(np.max(time_headway), np.min(time_headway))
                        
                        time_headway_total.append(time_headway.astype(object))

                else: 
                    # only for controlled vehicles
                    if "human" not in vehicle_id: # nice
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                        # Shock times omit code in short 
                        leader_id = vehicle['leader_id'].unique()
                        leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]
                        
                        #leader_shock_times = leader['shock_time'].values[self.start_time:self.end_time]
                        #vehicle_shock_times = vehicle['shock_time'].values[self.start_time:self.end_time]
                        #not_shock_times = np.logical_not(np.logical_or(leader_shock_times, vehicle_shock_times))

                        # meter
                        space_headway = vehicle['space_headway'].values[self.start_time:self.end_time] #[not_shock_times]
                        #print(vehicle_id, space_headway.shape, space_headway)
                        
                        # meter per second
                        velocity = vehicle['speed'].values[self.start_time:self.end_time] #[not_shock_times]
                        #print(vehicle_id, velocity.shape, velocity)
                        #print(np.max(velocity), np.min(velocity))

                        # To avoid a divide by zero error and very high time headways at low velocities
                        # If a velocity is less than 0.01 m/s, set it to 0.01 m/s (One centimeter per second),
                        velocity = np.where(velocity < 0.01, 0.01, velocity)
                        #print(np.max(velocity), np.min(velocity))

                        time_headway = space_headway / velocity
                        #print(vehicle_id, time_headway.shape, time_headway)
                        #print(np.max(time_headway), np.min(time_headway))
                        
                        time_headway_total.append(time_headway.astype(object))

            time_headway_total = np.asarray(time_headway_total)
            print(time_headway_total.shape)

            # newly added
            time_headway_worst = np.asarray([np.min(x) for x in time_headway_total])
            print(f"Worst case time headway of controller(s): \n{time_headway_worst}\n")

            # Time headway average for each vehicle 
            time_headway_avg = np.asarray([np.mean(x) for x in time_headway_total]) #np.mean(time_headway_total, axis=1)
            print(f"\nAverage time headway of controller(s): \n{time_headway_avg}\n")

            # Time headway standard deviation for each vehicle 
            time_headway_std = np.asarray([np.std(x) for x in time_headway_total]) #np.std(time_headway_total, axis=1)
            print(f"Standard deviation of time headway of controller(s): \n{time_headway_std}\n")

            # A single value
            worst_time_headway = np.min(time_headway_worst) # worst among all controlled vehicle
            avg_time_headway = np.mean(time_headway_avg)
            std_time_headway = np.mean(time_headway_std) #avg of std of all controlled vehicles
            print(f"Time headway (s): Worst = {round(worst_time_headway, 2)}, Avg= {round(avg_time_headway,2)}, std= {round(std_time_headway,2)}\n")

            time_headways_worst_mother.append(worst_time_headway)
            time_headways_avg_mother.append(avg_time_headway)
            time_headways_std_mother.append(std_time_headway)

            # #############################
            print("####################")

            speed_total= []

            for vehicle_id in self.vehicle_ids:
                vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                # Get the speed (start from warmup end to shock start)
                speed = vehicle['speed'].values[self.warmup:]
                speed_total.append(speed)
            # Calculate average speed of all vehicles 
            speeds_total = np.asarray(speed_total)
            speeds_total_avg = np.mean(speeds_total, axis=0) # average speed of all vehicles in each timestep
            
            # Use standard deviation with bessel's correction
            std_error = []
            for i in range(len(speeds_total_avg)): # for each timestep
                sum = 0
                for j in range(len(speeds_total)): # for each vehicle
                    timestep_mean = speeds_total_avg[i]
                    vehicle_speed = speeds_total[j][i]
                    error = (vehicle_speed - timestep_mean)**2
                    sum += error
                std_error.append(sum/(len(speeds_total)-1)) # This is 0 or 1 # Manually perform bessel's correction
            std_error = np.asarray(std_error)
            std_error = np.sqrt(std_error)
            #print("One", std_error)

            #speeds_total_std = np.std(speeds_total, axis=0)
            #print("Two", speeds_total_std)
            speeds_total_std = std_error

            #print(f"Speed of all vehicles (m/s): {speeds_total_avg.shape}\n")
            #print(f"Std of average speeds{speeds_total_std}")
            
            if args.method == 'fs': # For unstable percentages put names here. ;bcm
                tts = -1
            else: 
                # Where was the first instance of the std being less than the IDM noise?
                # For 10 consecutive
                try:
                    for i in range(len(speeds_total_std)):
                        if speeds_total_std[i] <= self.args.idm_noise:
                            # Check the next 10 steps
                            if np.all(speeds_total_std[i:i+100] < self.args.idm_noise):
                                # TODO: Get the 10 from env_params
                                tts = i/10
                                break

                except:
                    tts = self.horizon/10
                    print(f"Could not stabilize within this time, set shock start time further right")
                
            #print(f"Time to stabilize (s), (time elapsed after warmup ends): {tts}\n")
            tts_mother.append(tts)

            #############################
            print("####################")

            # Since its worst case measurement, it can fall into safety
            # Worst case controller vehicles acceleration variation (standard deviation), no need to omit data during shocks
            # TURN OFF BESSEL CORRECTION
            cav_total = []
            for vehicle_id in self.vehicle_ids:

                if args.method == "idm":
                    if vehicle_id == "idm_0": # This acts as a controller
                        #print(vehicle_id)
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]
                        # For each controller vehicle
                        
                        acceleration = vehicle['realized_accel'].values[self.start_time:self.end_time]

                        # Every now and then this can have empty values, so omit them
                        acceleration = acceleration[~np.isnan(acceleration)]
                        
                        # Each controller has its own acceleration variation
                        # calculate std of acceleration, turn off bessel correction
                        cav= np.std(acceleration, ddof=1)
                        #cav = np.std(acceleration)
                        
                        cav_total.append(cav)
                else: 
                    # Only for controller vehicles, they dont have a shock time
                    if "human" not in vehicle_id:
                        #print(vehicle_id)
                        vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]
                        # For each controller vehicle
                        acceleration = vehicle['realized_accel'].values[self.start_time:self.end_time]

                        # Every now and then this can have empty values, so omit them
                        acceleration = acceleration[~np.isnan(acceleration)] #except nan
                        
                        # Each controller has its own acceleration variation
                        cav = np.std(acceleration, ddof=1)
                        #print("CAV: ", cav, "\n")
                        cav_total.append(cav)

            # Worst from those standard deviations so that we have a single value
            cav_worst = np.max(cav_total)
            print(f"Acceleration variation (m/s^2): {cav_mother}\n")

            cav_mother.append(np.max(cav_worst))
            

        # return tts_mother, time_headways_worst_mother, time_headways_avg_mother, time_headways_std_mother

        return tts_mother, time_headways_worst_mother, cav_mother

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating metrics for the agent')

    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                        help='Path to emissions file')
    parser.add_argument('--method', type=str, default=None)
    #parser.add_argument('--metric', type=str, default=None)

    parser.add_argument('--horizon', type=int, default=15000)
    parser.add_argument('--warmup', type=int, default=2500)

    parser.add_argument('--start_time', type=int, default=8000)
    parser.add_argument('--end_time', type=int, default=11600) # 3600 timesteps = 6 minutes of evaluation window

    parser.add_argument('--num_rollouts', type=int, default=1)
    
    parser.add_argument('--save_plots', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default='./metrics_plots')
    
    # The value of IDM noise that was added to the vehicles, typically 0.2
    parser.add_argument('--idm_noise', type=float, default=0.2)

    args = parser.parse_args()

    if args.method is None or args.method not in ['bcm', 'idm', 'fs', 'pi', 'lacc', 'wu', 'ours', 'ours4x', 'ours9x', 'ours13x']:
        raise ValueError("Please specify the method to evaluate metrics for\n Method can be [bcm, idm, fs, pi, lacc, wu, ours, , ours4x, ours9x, ours13x]")

    #if args.metric is None:
        #raise ValueError("Please specify the metric to evaluate\n Metric can be [Stability, safety, efficiency]")

    files = [f"{args.emissions_file_path}/{args.method}/{item}" for item in os.listdir(f"{args.emissions_file_path}/{args.method}") \
        if item.endswith('.csv')]
    
    # Add more upon necessity
    kwargs = {'files': files,
              'plots_dir': f"{args.save_dir}/{args.method}/"
    }
    print(f"Calculating metrics for {args.num_rollouts} rollouts on files: \n{files}\n")
    metrics = EvalMetrics(args, **kwargs)
    ttc_worst_mother, ttc_best_mother, ttc_avg_mother, ttc_std_mother, drac_worst_mother = metrics.safety()

    mpgs_avg_mother, mpgs_std_mother, speeds_avg_mother, speeds_std_mother, flows_mother = metrics.efficiency()

    #tts_mother, time_headways_worst_mother, time_headways_avg_mother, time_headways_std_mother = metrics.stability()
    tts_mother, time_headways_worst_mother, cav_mother  = metrics.stability()

    print("####################")
    print("####################")
    print("\nFinal Aggregated Safety Metrics (across all files):\n")
    a = round(np.mean(ttc_worst_mother),2)
    b = round(np.std(ttc_worst_mother),2)

    c = round(np.mean(cav_mother),2)
    d = round(np.std(cav_mother),2)

    e = round(np.mean(mpgs_avg_mother),2)
    f = round(np.mean(mpgs_std_mother),2)

    g = round(np.mean(flows_mother),2)
    h = round(np.std(flows_mother),2)

    x = round(np.mean(drac_worst_mother),2)
    #i = round(np.mean(time_headways_avg_mother),2)
    #j = round(np.mean(time_headways_std_mother),2) # This is not the std of the average values across rollouts.. this is similar to CAV
    

    print(f"Time to Collision across rollouts (s):\n\tWorst= {ttc_worst_mother} \n\tAvg= {a}, std= {b}\n\n\tBest: Avg= {round(np.mean(ttc_best_mother),2)}, \
        std= {round(np.std(ttc_best_mother),2)}\n\tAverage: Avg= {round(np.mean(ttc_avg_mother),2)}, std= {round(np.std(ttc_avg_mother),2)}\n\tStd: Avg= {round(np.mean(ttc_std_mother),2)}, std= {round(np.std(ttc_std_mother),2)}\n")
    
    print(f"DRAC across rollouts (m/s^2):\n\tWorst= {drac_worst_mother} \n\tAvg= {x}, std= {round(np.std(drac_worst_mother),2)}\n")
    # print(f"Time headway across rollouts (s): \n\tWorst={time_headways_worst_mother}\n\tAvg= {round(np.mean(time_headways_worst_mother),2)}, std= {round(np.std(time_headways_worst_mother),2)} \
    #       \n\nAverage: {time_headways_avg_mother}\n\t Avg={i}, std= {round(np.std(time_headways_avg_mother),2)}\nstd= {time_headways_std_mother}\n\tAvg={j}, std= {round(np.std(time_headways_std_mother),2)}")
    
    print(f"CAV across rollouts (veh/hr): {cav_mother} \n\tAvg= {c}, std= {d}\n")

    print("####################")
    print("\nFinal Aggregated Efficiency Metrics (across all files):\n")
    print(f"MPG across rollouts (miles/gallon): {mpgs_avg_mother} \n\tAvg= {e}, std= {f}\n")
    print(f"Speed across rollouts (m/s): {speeds_avg_mother} \n\tAvg= {round(np.mean(speeds_avg_mother),2)}, std= {round(np.mean(speeds_std_mother),2)}\n")
    print(f"Throughput/ Flow across rollouts (veh/hr): {flows_mother} \n\tAvg= {g}, std= {h}\n")
    
    print("####################")
    tts_avg = np.mean(tts_mother)
    tts_std = np.std(tts_mother)
    print("\nFinal Aggregated Stability Metrics (across all files):\n")
    print(f"Time to stabilize (s), (time elapsed after warmup ends): {tts_mother} \n\tAvg= {round(tts_avg,2)}, std= {round(tts_std,2)}\n")

    print(f"${-1*a}$ & ${x}$ & ${e}$ & ${int(g)}$ & ${c}$ ")
    # TODO: Controlled vehicles and human vehicle have separate stats?
