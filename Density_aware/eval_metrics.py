"""
Three metrics each for Safety, Efficiency and Stability
"""
import os 

import argparse
import numpy as np
import pandas as pd

from eval_plots import plot_speeds

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

        

    def safety(self, ):
        """
        PENDING HOW TO AGGREGATE.
        1. Time to Collision: Time interval between two vechicles before collision if same velocity difference is maintained.
                           A measure of how much time does a driver have to react before a potential collision.
        
        2. Variation of Acceleration/ Deceleration rate during shocks: How aggressive is the driving.
        3. Emergency braking count during shocks: A high number indicates that the agent is not able to predict potential collision scenarios.
        """
        pass 

    def efficiency(self, ):
        """
        1. Fuel Economy during shocks: Average fuel consumption by the 22 vehicles (Miles per gallon).
        2. Average Speed/Velocity during shocks: After the warmup period, average speed of the 22 vehicles during the shocks.

        PENDING HOW TO AGGREGATE.
        3. Time Headway: Time interval between two consecutive vehicles passing a reference point in the road. 
                        A measure of how closely consecutive vehicles are following each other. Time headway is a measure of traffic density (capacity utiliztion) Also related to space headway.
        """

        
        file = self.kwargs['files'][0]
        self.dataframe = pd.read_csv(file)
        
        #filter for each vehicle
        vehicle_ids = self.dataframe['id'].unique()

        ###############
        print("####################")
        # Fuel consumption:
        fuel_total= []
        distances_travelled =[]

        print("Vehicle ids: ", vehicle_ids)

        for vehicle_id in vehicle_ids:
            # Get the fuel consumed by vehicles from start to end 
            # For newer sumo version:
            # The unit is mg/s or ml/s (used rather interchangably in sumo) i.e., this is fuel consumed per second
            # But the reading itself is of one timestep i.e., 0.1 seconds
            # Convert that to fuel consumed per time step, ml/step = ml/s * 0.1 
            # SEE: https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getFuelConsumption

            # Since we use version 1.1.0, we need to use the following:
            # This is the fuel consumed in the last time step (0.1 seconds) in ml

            # Also there was a mismatch between documentation and API
            # See: https://github.com/eclipse/sumo/issues/5659
            # See: https://www.eclipse.org/lists/sumo-user/msg04350.html

            # So this is in ml/s
            vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]
            fuel_vehicle = vehicle['fuel_consumption'].values
            
            # Filter for the start and end time, convert to ml/step
            # TODO: Get conversion factor 0.1 from env params
            fuel_vehicle = fuel_vehicle[self.start_time:self.end_time] *0.1 
            #print(vehicle_id, fuel_vehicle.shape, fuel_vehicle)

            # Append the fuel consumed by each vehicle
            fuel_total.append(fuel_vehicle)

            distance_vehicle = vehicle['distance_traveled'].values
            #print(vehicle_id, distance_vehicle.shape, distance_vehicle[self.end_time], distance_vehicle[self.start_time])

            # Get the distance traveled by vehicles from start to end (meters)
            distances_travelled.append(distance_vehicle[self.end_time] - distance_vehicle[self.start_time])
            
        fuel_total = np.asarray(fuel_total)
        print(fuel_total.shape)

        # Fuel consumed from start to end (Milli liters)
        fuel_total_sum = np.sum(fuel_total, axis=1)
        #print(f"Total fuel consumed by each vehicle (milli liters): \n{fuel_total_sum}\n")

        # Convert to gallons
        fuel_total_sum = fuel_total_sum #* 0.000264172 # The data does not make sense with this conversion (it must already be in gallons)
        print(f"Fuel consumed by each vehicle (gallons): \n{fuel_total_sum}\n")
        
        # vehicle miles traveled (VMT) (Miles)
        vmt = np.asarray(distances_travelled) * 0.000621371
        print(f"Vehicle miles travelled by each vehicle (miles): \n{vmt}\n")

        # MPG is fuel mileage (Miles per gallon) = VMT / fuel mile
        mpgs = vmt / fuel_total_sum
        print(f"Miles per gallon for each vehicle (mpg):\n{mpgs}\n")

        ###############
        print("####################")
        # Average speed:
        # Average speed of the 22 vehicles during the shocks.

        speeds_total = []
        # Get the average speed of each vehicle
        for vehicle_id in vehicle_ids: 
            vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

            # Get the average speed of each vehicle during shock time
            avg_speed = vehicle['speed'].values[self.start_time:self.end_time]
            #print(vehicle_id, avg_speed.shape, avg_speed)

            speeds_total.append(avg_speed)
        
        speeds_total = np.asarray(speeds_total)
        print(speeds_total.shape)

        # If eval_plots is True, pass this to the plot function
        if self.args.eval_plots:
            plot_speeds(speeds_total, vehicle_ids)

        speeds_total_avg = np.mean(speeds_total, axis=1)
        print(f"\nAverage speed of each vehicle (m/s): \n{speeds_total_avg}\n")

        ###############
        # Time Headway
        

    def stability(self, ):
        """
        1. Time to stabilize after shock: Time interval between the last shock and the vehicles stabilizing.

        PENDING HOW TO AGGREGATE.
        2. Variation in Space Headway: How well is the agent able to maintain a constant distance to the leader/ follower.
                                        During shocks, RL agent should be able to maintain a constant space headway.        
        3. String stability: A traffic flow is said to be string stable if small perturbations in the speed or position of a vehicle do not propagate through the traffic stream.
                            There is no single definitive way to measure this (We use PENDING: XXX, YYY). We have to make sure the disturbances do not come from sources other than the ones we provide.
                            A combination of metrics may be the best way to measure this i.e., provide a complete picture. 

        4. Minimum number of vehicles required to stabilize
        """
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating metrics for the agent')

    parser.add_argument('--emissions_file_path', type=str, default='./test_time_rollout',
                        help='Path to emissions file')

    parser.add_argument('--horizon', type=int, default=6000)
    parser.add_argument('--warmup', type=int, default=2500)

    parser.add_argument('--start_time', type=int, default=8000)
    parser.add_argument('--end_time', type=int, default=11500) # Warmup + Horizon

    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--eval_plots', type=bool, default=True)

    args = parser.parse_args()
    files = [args.emissions_file_path + '/' + item for item in os.listdir(args.emissions_file_path)]
    
    # Add more upon necessity
    kwargs = {'files': files,
    }
    print(f"Calculating metrics for {args.num_rollouts} rollouts on files: \n{files}\n")

    metrics = EvalMetrics(args, **kwargs)
    metrics.safety()
    metrics.efficiency()
    metrics.stability()


    # TODO: Controlled vehicles and human vehicle have separate stats
    #
    #
    #