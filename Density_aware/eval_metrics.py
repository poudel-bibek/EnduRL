"""
Three metrics each for Safety, Efficiency and Stability

Across 10 rollouts, mean and standard deviation. 
# If we deal with 10 files, do we plot for every?

Q. 
For safety metrics, do we exclude the data where we introduced the shocks?

"""
import os 

import argparse
import numpy as np
import pandas as pd

from eval_plots import plot_speeds, plot_time_headway_distribution, plot_space_time, plot_ttc

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

    def safety(self, ):
        """
        
        1. Time to Collision: Time interval between two vechicles before collision if same velocity difference is maintained.
                           A measure of how much time does a driver have to react before a potential collision.
                           Although one vehicle shocks the system for a duration, the dynamics of the entire system is affected. So calculate this for all vehicles
                           Aggregation (Since this falls under safety, we also include the wost and best cases):
                            1. Worst case: Minimum time to collision (low negative value) e.g., -1000 seconds to collide
                            2. Best case: Maximum time to collision (high negative value) e.g., -10 seconds to collide
                            3. Average case: Average time to collision
                            4. Standard deviation: Standard deviation of time to collision 
            Note: Collision can only occur if the velocity of the follower is higher than that of a leader. So all positive values are set to 0.
                    This is akin to saying, "if there were to be a collision, what would be it like?" So exclude cases where there would be no collision.
            Note: Position x is measured at the tip of each vehicles front bumper for all vehicles. 
            Note: Space headway measures distance between two vehicles (rear of the leader to the front of the follower) 
            Another approach: Since we are dealing with the autonomous vehicles, we can also calculate the time to collision for the autonomous vehicles only.??

            TTC calculation: https://arxiv.org/abs/2203.04749 (relative velocity should be calculated as leader - follower)
            A negative value of TTC indicates that they are on a collision course. 
            If the speed of follower is greater than the leader, then the TTC is negative. Any value less than 0 indicates a collision.
            Higher negative value is worse
`           See also: https://www.sciencedirect.com/science/article/pii/S0001457500000191
            Assumption in this TTC calculation is that at this time-step (instant), the velocity of the vehicles is constant (acceleration is 0)
            A small TTC value indicates that accident can only be avoided by taking an action (one of the vehicle has to either accelrate or brake).
            TTC is higher than or equal to time headway. 

            According to the book, Values in the range 0 < τTTC ≤ 4 s are generally considered as critical.
        ******
        By default, in sumo:
            a vehicle may only drive slower than the speed that is deemed safe by the car following model
            a vehicle may not exceed the bounds of acceleration and deceleration 
            there are more rules regarding traffic light and intersection behavior

        We do not want the vehicles to regard safe speed. But we do not want the vehicle that produces a shock to go and collide.
        But we want others humans to collide and perform emergency brakes?

        We do not want the controllers themselves to go out of these bounds.
        ******

        PENDING HOW TO AGGREGATE.
        2. Variation of Acceleration/ Deceleration rate during shocks: How aggressive is the driving? Jerk?

        3. Emergency braking count during shocks: A high number indicates that the agent is not able to predict potential collision scenarios.
        # What makes a braking event an emergency one?

        4. Minimum time-headway: 
        Time headway: time elapsed between front of a leader vehicle passing a point in the road and front of following vehicle passing the same point.
        A shorter time headway indicates a higher risk of collision. 
        The literature mentions that headway and ttc are independent of each other in car following siatuations. 
        see: https://www.sciencedirect.com/science/article/pii/S0001457502000222
        This is more intuitively understandable and if a higher TH is maintained, dangerous TTC values will not form at all.
        """
        
        time_to_collision_total = []

        for vehicle_id in self.vehicle_ids:
            # Get the dataframe for each vehicle
            vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

            # Instead of doing gymnastics with the positions, we can just use the space headway
            relative_positions = vehicle['space_headway'].values
            relative_positions = relative_positions[self.start_time:self.end_time]

            # Since the vehicles travel in a single lane, leader wont change
            # Get the leader dataframe
            leader_id = vehicle['leader_id'].unique()
            leader = self.dataframe.loc[self.dataframe['id'] == leader_id[0]]
            #print(leader.shape)

            # leader velocity
            leader_velocities = leader['speed'].values
            # current vehicle velocity
            vehicle_velocities = vehicle['speed'].values

            assert vehicle_velocities.shape == leader_velocities.shape

            # Split
            vehicle_velocities = vehicle_velocities[self.start_time:self.end_time]
            leader_velocities = leader_velocities[self.start_time:self.end_time]

            # relative velocity difference
            relative_velocities = leader_velocities - vehicle_velocities 

            # set positive values to 0
            #relative_velocities[relative_velocities > 0] = 0.01 # to avoid division by 0
            #print(relative_velocities.shape, relative_velocities)

            # time to collision
            time_to_collision = relative_positions/relative_velocities
            #print(f"Before clip:: {time_to_collision.shape}")

            # only consider negative values
            time_to_collision = time_to_collision[time_to_collision < 0.0]
            #print(f"After clip:: {time_to_collision.shape}\n")

            time_to_collision_total.append(time_to_collision.astype(object))
            
        time_to_collision_total = np.asarray(time_to_collision_total, dtype= object)
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

        # Plot the distribution of time to collision
        if self.args.eval_plots:
            plot_ttc(time_to_collision_total, self.vehicle_ids)

        # A single value 
        worst_ttc = np.mean(time_to_collision_worst) # Average of worst of all vehicles 
        best_ttc = np.mean(time_to_collision_best) # Average of best of all vehicles
        avg_ttc = np.mean(time_to_collision_avg) # Average of average of all vehicles
        std_ttc = np.mean(time_to_collision_std) # Average of std of all vehicles

        print(f"Time to Collision (s):\n\tWorst= {round(worst_ttc,2)}\n\tBest= {round(best_ttc,2)}\n\tAverage= {round(avg_ttc,2)}\n\tStd= {round(std_ttc,2)}\n")

        #############################
        print("####################")



    def efficiency(self, ):
        """
        Efficiency of each vehicle
        Effenciency of the system of vehicles (in terms of capacity utilization and throughput)

        1. Fuel Economy during shocks: Average fuel consumption by the 22 vehicles (Miles per gallon).
        2. Average Speed/Velocity during shocks: After the warmup period, average speed of the 22 vehicles during the shocks.
        3. Through put as measure by flow (vehicles per hour) in a reference point. All controllers experience shocks for same duration, which one maximizes flow?
        
        3. Time Headway: Time interval between two consecutive vehicles passing a reference point in the road. 
                        A measure of how closely consecutive vehicles are following each other. 
                        Time headway is a measure of traffic density (capacity utiliztion) Also related to space headway.
                        Calculate the average time headway for the entire system (during the time period of interest).
                        A shorter time headway indicates lower capacity utilization by the traffic flow.
        """
        # Collectors across rollouts
        mpgs_avg_mother = []
        mpgs_std_mother = []
        speeds_avg_mother = []
        speeds_std_mother = []
        time_headways_avg_mother = []
        time_headways_std_mother = []

        for file in self.kwargs['files']:
            self.dataframe = pd.read_csv(file)
            
            #filter for each vehicle
            self.vehicle_ids = self.dataframe['id'].unique()
            #print(f"Vehicle ids: {self.vehicle_ids}\n")

            #############################

            # Fuel consumption:
            fuel_total= []
            distances_travelled =[]

            for vehicle_id in self.vehicle_ids:
            
                # Flow converts to gallons per second from sumo default (at the time) ml/ second
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

            # Fuel consumed from start to end (gallons)
            # Since instantaneous average measurements are not required for fuel, we take a sum over the time period
            fuel_total_sum = np.sum(fuel_total, axis=1)
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

                # Get the average speed of each vehicle during time of interest
                avg_speed = vehicle['speed'].values[self.start_time:self.end_time]
                #print(vehicle_id, avg_speed.shape, avg_speed)

                speeds_total.append(avg_speed)
            
            speeds_total = np.asarray(speeds_total)
            print(speeds_total.shape)

            speeds_total_avg = np.mean(speeds_total, axis=1)
            print(f"\nAverage speed of each vehicle (m/s): \n{speeds_total_avg}\n")

            speeds_total_std = np.std(speeds_total, axis=1)
            print(f"Standard deviation of speed of each vehicle (m/s): \n{speeds_total_std}\n")

            # If eval_plots is True, pass this to the plot function
            if self.args.eval_plots:
                plot_speeds(speeds_total, self.vehicle_ids)

            # A single value
            avg_speed = np.mean(speeds_total_avg)
            std_speed = np.mean(speeds_total_std)
            print(f"Speed of all vehicles (m/s): Average= {round(avg_speed,2)}, std={round(std_speed,2)}\n")

            speeds_avg_mother.append(avg_speed)
            speeds_std_mother.append(std_speed)

            #############################
            print("####################")

            # Time Headway (Average and standard deviation)
            time_headway_total = []

            for vehicle_id in self.vehicle_ids:
                vehicle = self.dataframe.loc[self.dataframe['id'] == vehicle_id]

                # meter
                space_headway = vehicle['space_headway'].values[self.start_time:self.end_time]
                #print(vehicle_id, space_headway.shape, space_headway)
                
                # meter per second
                velocity = vehicle['speed'].values[self.start_time:self.end_time]
                #print(vehicle_id, velocity.shape, velocity)
                #print(np.max(velocity), np.min(velocity))

                # To avoid a divide by zero error and very high time headways at low velocities
                # If a velocity is less than 0.01 m/s, set it to 0.01 m/s (One centimeter per second),
                velocity = np.where(velocity < 0.01, 0.01, velocity)
                #print(np.max(velocity), np.min(velocity))

                time_headway = space_headway / velocity
                #print(vehicle_id, time_headway.shape, time_headway)
                #print(np.max(time_headway), np.min(time_headway))

                time_headway_total.append(time_headway)

            time_headway_total = np.asarray(time_headway_total)
            print(time_headway_total.shape)

            # Time headway average for each vehicle 
            time_headway_avg = np.mean(time_headway_total, axis=1)
            print(f"\nAverage time headway for each vehicle (s): \n{time_headway_avg}\n")

            # Time headway standard deviation for each vehicle 
            time_headway_std = np.std(time_headway_total, axis=1)
            print(f"Standard deviation of time headway for each vehicle (s): \n{time_headway_std}\n")

            # If eval_plots is True, pass this to the plot function
            if self.args.eval_plots:
                plot_time_headway_distribution(time_headway_total, self.vehicle_ids)

            # A single value
            avg_time_headway = np.mean(time_headway_avg)
            std_time_headway = np.mean(time_headway_std)
            print(f"Time headway of all vehicles (s): Avg= {round(avg_time_headway,2)}, std= {round(std_time_headway,2)}\n")
            
            time_headways_avg_mother.append(avg_time_headway)
            time_headways_std_mother.append(std_time_headway)

            #############################
            print("####################")

        #############################
        print("####################")
        print("\nFinal Aggregated Efficiency Metrics (across all files):\n")
        print(f"MPG across rollouts (miles/gallon): {mpgs_avg_mother} \n\tAvg= {round(np.mean(mpgs_avg_mother),2)}, std= {round(np.mean(mpgs_std_mother),2)}\n")
        print(f"Speed across rollouts (m/s): {speeds_avg_mother} \n\tAvg= {round(np.mean(speeds_avg_mother),2)}, std= {round(np.mean(speeds_std_mother),2)}\n")
        print(f"Time headway across rollouts (s): {time_headways_avg_mother} \n\tAvg= {round(np.mean(time_headways_avg_mother),2)}, std= {round(np.mean(time_headways_std_mother),2)}\n")


    def stability(self, ):
        """
        1. Time to stabilize after shock: Time interval between the (last shock) or (warmup end)? and the vehicles stabilizing.
                                            If the average velocity standard deviation is less than the IDM noise, system = Stable

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
    files = [args.emissions_file_path + '/' + item for item in os.listdir(args.emissions_file_path) if item.endswith('.csv')]
    
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