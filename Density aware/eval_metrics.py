"""
Three metrics each for Safety, Efficiency and Stability
"""
import argparse
import numpy as np
import pandas as pd


class EvalMetrics():
    def __init__(self, args):
        self.args = args
        self.emissions_file_path = self.args.emissions_file_path
        self.horizon = self.args.horizon
        self.warmup = self.args.warmup

        # Set the start and end time for the evaluation
        self.start_time = self.args.start_time
        self.end_time = self.args.end_time

        self.dataframe = pd.read_csv(self.emissions_file_path)
        
        #print(self.dataframe.head())
        #print(self.dataframe.columns)

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

        # This is fuel consumed
        fuel_consumption_column = self.dataframe['fuel_consumption']
        
        #filter for each vehicle
        vehicle_ids = self.dataframe['id'].unique()
        fuel_total= []
        distances_travelled =[]

        for vehicle_id in vehicle_ids:
            fuel_vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]['fuel_consumption'].values
            fuel_vehicle = fuel_vehicle[self.start_time:self.end_time]
            fuel_total.append(fuel_vehicle)

            distance_vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]['distance_traveled'].values
            #print(vehicle_id, distance_vehicle.shape, distance_vehicle[self.end_time], distance_vehicle[self.start_time])

            # Get the distance traveled by vehicles from start to end (meters)
            distances_travelled.append(distance_vehicle[self.end_time] - distance_vehicle[self.start_time])
            
        fuel_total = np.asarray(fuel_total)
        #print(fuel_total.shape)

        # Fuel consumed from start to end (Milli liters)
        fuel_total_sum = np.sum(fuel_total, axis=1)
        # Convert to gallons
        fuel_total_sum = fuel_total_sum * 0.000264172

        #print(fuel_total_sum)
        
        # vehicle miles traveled (VMT) (Miles)
        vmt = np.asarray(distances_travelled) * 0.000621371
        #print(vmt)

        mpgs = vmt / fuel_total_sum
        print(mpgs)

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

    parser.add_argument('--emissions_file_path', type=str, 
                        help='Path to emissions file')

    parser.add_argument('--horizon', type=int, default=6000)
    parser.add_argument('--warmup', type=int, default=2500)

    parser.add_argument('--start_time', type=int, default=2500)
    parser.add_argument('--end_time', type=int, default=8500) # Warmup + Horizon

    metrics = EvalMetrics(parser.parse_args())

    metrics.safety()
    metrics.efficiency()
    metrics.stability()