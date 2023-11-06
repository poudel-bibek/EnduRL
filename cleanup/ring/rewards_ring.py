"""
################################################
TIMEGAP CONTROLLER REWARDS
################################################
"""

## Reward 1. Cathy's code reward

def compute_reward(self, rl_actions, **kwargs):
    """ 
    Cathy's original reward (in code)
    """
    # for warmup 
    if rl_actions is None: 
        return 0 

    vel = np.array([
        self.k.vehicle.get_speed(veh_id)
        for veh_id in self.k.vehicle.get_ids()
    ])
    
    # Fail the current episode if these
    if any(vel <-100) or kwargs['fail']:
        return 0 
    
    rl_id = self.k.vehicle.get_rl_ids()[0]

    # In her case this is not realized acceleration, rather desired acceleration (control action)
    rl_accel = self.k.vehicle.get_realized_accel(rl_id) 
    
    magnitude = np.abs(rl_accel)
    sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

    print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

    # The acceleration penalty is only for the magnitude
    reward = 0.2*np.mean(vel) - 4*magnitude
    print(f"Reward: {reward}")
    
    return reward

##############################
## Reward 2: My Original reward

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        The TSE based reward 
        The action is to control time-gap, the reward is based on acceleration behavior of agent
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        # During data collection for TSE, assign a random value. For a brief time horizon has to run. 
        # Comment for RL
        #self.tse_output = [0] 

        # Detect collision, does not seem to work
        #print("Collision: ", self.k.simulation.check_collision())

        # TSE based reward 
        rl_id = self.k.vehicle.get_rl_ids()[0]
        rl_accel = self.k.vehicle.get_realized_accel(rl_id)
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        reward = 0
        penalty_scalar = -10

        # Leaving
        if self.tse_output[0] == 0:
            if sign < 0:
                reward += penalty_scalar*magnitude # If congestion is leaving, penalize deceleration
            print(f"Leaving: {reward}")

        # Forming
        elif self.tse_output[0] == 1:
            if sign > 0:
                reward += penalty_scalar*magnitude # If congestion is fomring, penalize acceleration
            print(f"Forming: {reward}")

        # Free Flow
        elif self.tse_output[0] == 2:
            # Penalize acceleration/deceleration magnitude
            reward += penalty_scalar*magnitude
            print(f"Free Flow: {reward}")
    
        # Congested
        elif self.tse_output[0] == 3:
            # Penalize acceleration/deceleration magnitude
            reward += penalty_scalar*magnitude
            print(f"Congested: {reward}")

        # Undefined
        elif self.tse_output[0] == 4:
            reward += 0 
            print(f"Undefined: {reward}")

        # No vehicle in front
        elif self.tse_output[0] == 5:
            reward += 0 
            print(f"No vehicle in front: {reward}")

        # Generally high reward for high average velocity and low acceleration
        general_reward = 2*np.mean(vel) - 0.5*magnitude 

        # Penalize high action changes?
        # action change = rl_actions - rl_actions_prev

        total = general_reward + reward
        print(f"Reward total: {total}, first term: {reward} , second term: {general_reward} \n")
        return total

############# 
## Reward 3: Cathy's reward from paper

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 

        # Cathy's reward (in effect)
        # Reward high average velocity and penalize for high accleration
        rl_id = self.k.vehicle.get_rl_ids()[0]
        rl_accel = self.k.vehicle.get_realized_accel(rl_id)
        first = np.mean(vel)
        second = 0.1*np.abs(np.array(rl_accel))
        reward = first - second
        print(f"\nReward: {reward}, first:{first}, second:{second}")

        return reward

############# 
## Reward 4: Incorporate Safety, Efficiency, Stability in the original reward.


    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 

        
        rl_id = self.k.vehicle.get_rl_ids()[0]
        rl_accel = self.k.vehicle.get_realized_accel(rl_id)
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0
        print(f"RL accel: {round(rl_accel,2)}, magnitude: {round(magnitude,2)}, sign: {sign}")

        current_length = self.k.network.length()

        # Shaping reward: TSE based reward 
        shaping_reward = 0
        penalty_scalar = -5
        fixed_penalty = -10

        # Leaving
        if self.tse_output[0] == 0:
            if sign < 0:
                #shaping_reward += penalty_scalar*magnitude # If congestion is leaving, penalize deceleration
                shaping_reward += fixed_penalty
            print(f"Leaving")

        # Forming
        elif self.tse_output[0] == 1:
            if sign > 0:
                #shaping_reward += penalty_scalar*magnitude # If congestion is fomring, penalize acceleration
                shaping_reward += fixed_penalty
            print(f"Forming")

        # Free Flow
        elif self.tse_output[0] == 2:
            # Penalize acceleration/deceleration magnitude
            #shaping_reward += penalty_scalar*magnitude
            shaping_reward += fixed_penalty
            print(f"Free Flow")
    
        # Congested
        elif self.tse_output[0] == 3:
            # Penalize acceleration/deceleration magnitude
            shaping_reward += penalty_scalar*magnitude
            print(f"Congested")

        # Undefined
        elif self.tse_output[0] == 4:
            shaping_reward += 0 
            print(f"Undefined")

        # No vehicle in front
        elif self.tse_output[0] == 5:
            shaping_reward += 0 
            print(f"No vehicle in front")
        
        print(f"Shaping reward: {round(shaping_reward,2)}")

        # Stability reward
        vehicles_back = self.sort_vehicle_list(self.k.vehicle.get_veh_list_local_zone(rl_id, 
                                                                                         current_length, 
                                                                                         self.LOCAL_ZONE,
                                                                                         direction = 'back' ))

        # standard deviation of the velocities in back (this promotes stability, low fuel consumption)
        velocities_back = [self.k.vehicle.get_speed(veh_id) for veh_id in vehicles_back]

        stability_reward = -2*np.std(velocities_back) # lower the better
        print(f"Stability reward: {round(stability_reward,2)}")

        # Safety reward 
        # based on TTC i.e., a SSM of rear-end collision (only valid when follower has a velocity higher than leader)
        # Only for RL
        ttc = 0 # Default
        leader_id = self.k.vehicle.get_leader(rl_id)
        leader_speed = self.k.vehicle.get_speed(leader_id)
        rl_speed = self.k.vehicle.get_speed(rl_id)
        if rl_speed > leader_speed:
            relative_speed = rl_speed - leader_speed
            rl_pos = self.k.vehicle.get_x_by_id(rl_id)
            relative_position = (self.k.vehicle.get_x_by_id(leader_id) - rl_pos) % current_length
            ttc = relative_position/relative_speed

        safety_reward = 2*np.log(max(0.1,ttc)/4) # higher the better, at 4 0, at less than 4 negative, at more than 4 positive
        print(f"Safety reward: {round(safety_reward,2)}")

        # Efficiency reward 
        # road utilization and fuel consumption efficiency
        all_ids = self.k.vehicle.get_ids()
        avg_fuel_consumption =  np.mean([self.k.vehicle.get_fuel_consumption(id) for id in all_ids]) # In gallons/s
        avg_velocity = np.mean(vel)

        efficiency_reward = 2*avg_velocity - 15*avg_fuel_consumption # first term is higher the better, second term is lower the better
        print(f"Efficiency reward: {round(efficiency_reward,2)}, avg velocity: {round(avg_velocity,2)}, avg fuel consumption: {round(avg_fuel_consumption,2)}")

        # Control action penalty
        action_penalty = 0#-0.5*magnitude 
        print(f"Action penalty: {round(action_penalty,2)}")

        reward = shaping_reward + stability_reward + safety_reward + efficiency_reward + action_penalty
        print(f"Reward: {round(reward,2)}")

        return reward



############# 
## Reward 5: Shaping removed, only stability, safety, efficiency and action penalty

def compute_reward(self, rl_actions, **kwargs):
    """ 

    """
    # for warmup 
    if rl_actions is None: 
        return 0 

    vel = np.array([
        self.k.vehicle.get_speed(veh_id)
        for veh_id in self.k.vehicle.get_ids()
    ])
    
    # Fail the current episode if these
    if any(vel <-100) or kwargs['fail']:
        return 0 
    
    # During data collection for TSE, assign a random value. For a brief time horizon has to run. 
    # Comment for RL
    #self.tse_output = [0] 
    The TSE based reward 
    The action is to control time-gap, the reward is based on acceleration behavior of agentized_accel(rl_id)
    magnitude = np.abs(rl_accel)
    sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

    print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

    reward = 0
    penalty_scalar = -10

    # Forming
    if self.tse_output[0] == 1:
        if sign > 0:
            reward += penalty_scalar*magnitude # If congestion is fomring, penalize acceleration
        print(f"Forming: {reward}")

    # Generally high reward for high average velocity and low acceleration
    general_reward = 2*np.mean(vel) - 2*magnitude 

    total = general_reward + reward
    print(f"Reward total: {total}, first term: {reward} , second term: {general_reward} \n")
    return total




##############
## Reward 6: Modified Cathy's reward plus forming shaping

def compute_reward(self, rl_actions, **kwargs):
    """ 
    
    """
    # for warmup 
    if rl_actions is None: 
        return 0 

    vel = np.array([
        self.k.vehicle.get_speed(veh_id)
        for veh_id in self.k.vehicle.get_ids()
    ])

    # Fail the current episode if these
    if any(vel <-100) or kwargs['fail']:
        return 0 

    # During data collection for TSE, assign a random value. For a brief time horizon has to run. 
    # Comment for RL
    #self.tse_output = [0] 

    # Detect collision, does not seem to work
    #print("Collision: ", self.k.simulation.check_collision())

    # TSE based reward 
    rl_id = self.k.vehicle.get_rl_ids()[0]
    rl_accel = self.k.vehicle.get_realized_accel(rl_id)
    magnitude = np.abs(rl_accel)
    sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

    print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

    reward = 0
    penalty_scalar = -10

    # Forming
    if self.tse_output[0] == 1:
        if sign > 0:
            reward += penalty_scalar*magnitude # If congestion is fomring, penalize acceleration
        print(f"Forming: {reward}")

    # Generally high reward for high average velocity and low acceleration
    general_reward = 2*np.mean(vel) - 2*magnitude 

    total = general_reward + reward
    print(f"Reward total: {total}, first term: {reward} , second term: {general_reward} \n")
    return total


########
## Reward 7: Cathy's original reward (in code) with desired acceleration plus Forming shaping

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 

        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # cannot replace realized acceleration with desired acceleration 
        # as long as the RL controller is not acceleration controller
        rl_accel = self.k.vehicle.get_realized_accel(rl_id) 

        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        reward = 0
        penalty_scalar = -10

        # It looks fairly complicated because I want to print the reward terms
        # If the agent violates the forming penalty, it does not get positive average velocity reward as well
        # Forming
        if self.tse_output[0] == 1:
            if sign > 0:
                first = penalty_scalar*magnitude
            else: 
                first = 0
            print(f"Forming: {first}")
            second = 0

        else:
            # Generally high reward for high average velocity
            first = 0 
            second = 0.2*np.mean(vel) 

        third = -4*magnitude # Acceleration penalty
        # 0.2 and 4 are obtained from Cathy's code
        
        total = first + second + third
        print(f"Reward total: {total}, first term: {first} , second term: {second}, third term: {third} \n")

        return total

########
## Reward 11: Cathy's original reward (in code) with desired acceleration plus Forming shaping
    def compute_reward(self, rl_actions, **kwargs):
        """ 
        Cathy's original reward (in code)
        There is a better way to write our forming penalty

        if self.tse_output[0] == 1:
            if sign>=0:
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                forming_penalty = min(-1, penalty_scalar*magnitude) 
                reward += forming_penalty

        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # In her case this is not realized acceleration, rather desired acceleration (control action)
        rl_accel = self.k.vehicle.get_realized_accel(rl_id) 
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.2*np.mean(vel) - 4*magnitude
        print(f"First Reward: {reward}")
        
        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration


        print(f"Last Reward: {reward}")
        return reward


"""
################################################
ACCELERATION CONTROLLER REWARDS
################################################
"""

## Reward 1: Reward for the obvious approach (Cathy's accel controller) with Forming shaping

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        Cathy's original reward (in code)
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.2*np.mean(vel) - 4*magnitude
        print(f"First Reward: {reward}")
        
        # Forming shaping 
        penalty_scalar = -10
        if self.tse_output[0] == 1:
            if sign > 0:
                forming_penalty = penalty_scalar*magnitude
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        print(f"Last Reward: {reward}")
        return reward

########
## Reward 2: Reward 1 with corrected Forming shaping

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.2*np.mean(vel) - 4*magnitude
        print(f"First Reward: {reward}")
        
        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        print(f"Last Reward: {reward}")
        return reward


"""
################################################
MULTIAGENT RING: ACCELERATION CONTROLLER REWARDS
################################################
"""
########
## Reward 1: Followers are penalized for thier own mean accelerations

def compute_reward(self, rl_actions, **kwargs):
    """
    Lead gets rewarded the same as the singleagent version 
    For maximum stability: others get rewarded for?
    """
    # in the warmup steps
    if rl_actions is None:
        return 0

    vel = np.array([
        self.k.vehicle.get_speed(veh_id)
        for veh_id in self.k.vehicle.get_ids()
    ])

    if any(vel < -100) or kwargs['fail']:
        return 0.
    
    rew = {}

    # reward for leader
    lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
    lead_accel = rl_actions[lead_rl_id].item()
    magnitude = np.abs(lead_accel)
    sign = np.sign(lead_accel)
    print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

    reward_leader = 0.2*np.mean(vel) - 4*magnitude

    # Forming shaping 
    penalty_scalar = -10
    fixed_penalty = -1
    if self.tse_output[0] == 1:
        if sign>=0:
            forming_penalty = penalty_scalar*magnitude
            # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
            # min is correct bacause values are -ve
            forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
            print(f"Forming: {forming_penalty}")
            reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
    rew.update({lead_rl_id : reward_leader})

    # reward for all followers
    follower_ids = self.k.vehicle.get_rl_ids()[:-1]
    follower_actions = [rl_actions[id] for id in follower_ids]
    mean_actions_followers = np.mean(np.abs(follower_actions))

    reward_followers = 0.2*np.mean(vel) - 4*mean_actions_followers

    for follower_id in follower_ids:
        rew.update({follower_id : reward_followers})
    
    print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
    return rew
########
## Reward 2: Followers are penalized for difference in mean accelerations with the leader

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions))

        # Penalize the difference of (mean actions of the follower RL vehicles with the lead RL)
        #Instead of penalizing the mean action of follower RL vehicles themselves (mean_actions_followers)
        action_difference =  np.abs(lead_accel - mean_actions_followers)
        reward_followers = 0.2*np.mean(vel) - 4*action_difference 

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 3: Followers are penalized for difference in mean accelerations with the leader, and the standard deviation of accelerations
# In addition, the leader has more emphasis on average speed

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.4*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions))

        # Penalize the difference of (mean actions of the follower RL vehicles with the lead RL)
        #Instead of penalizing the mean action of follower RL vehicles themselves (mean_actions_followers)
        action_difference =  np.abs(lead_accel - mean_actions_followers)

        # Add standard deviation penalty 
        std_dev = np.std(np.abs(follower_actions))

        reward_followers = 0.2*np.mean(vel) - 2*action_difference - 2*std_dev

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 4: Followers are penalized time headway

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.4*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = self.k.vehicle.get_headway(vehicle)
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        # Option 1 Follow closely
        reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers -2*mean_time_headway  -2*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew


########
## Reward 5: Corrections to 4

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = self.k.vehicle.get_headway(vehicle)
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        # Option 1 Follow closely
        reward_followers = -2*mean_actions_followers -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 6: Corrections to 5

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*np.mean(vel) -2*mean_actions_followers -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 7: Corrections to 6

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.4*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*np.mean(vel) -2*mean_actions_followers -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew


########
## Reward 8: Corrections to 7

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.8*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*np.mean(vel) -2*mean_actions_followers -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 9: Corrections to 8

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.8*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*followers_avg_speed -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew


########
## Reward 10: Corrections to 9

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.6*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -5
        fixed_penalty = -0.5
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*followers_avg_speed -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

# The observation has also changed from below. 
# Prior observations

    def get_state(self):
        """See class definition."""
        

        # Get TSE ouput
        # This is the way lead is assigned
        self.lead_rl_id = f"{self.k.vehicle.get_rl_ids()[-1]}"
        rl_pos = self.k.vehicle.get_x_by_id(self.lead_rl_id)
        current_length = self.k.network.length()

        # Get the list of all vehicles in the local zone (sorted from farthest to closest)
        vehicles_in_zone = self.sort_vehicle_list(self.k.vehicle.get_veh_list_local_zone(self.lead_rl_id, 
                                                                                         current_length, 
                                                                                         self.LOCAL_ZONE )) # Direction i front by default


        observation_tse = np.full((10, 2), -1.0)
        num_vehicle_in_zone = len(vehicles_in_zone)
        distances = []
        if num_vehicle_in_zone > 0:
            for i in range(len(vehicles_in_zone)):
                # Distance is measured center to center between the two vehicles (if -5 present, distance if bumper to bumper)
                rel_pos = (self.k.vehicle.get_x_by_id(vehicles_in_zone[i]) - rl_pos) % current_length
                norm_pos = rel_pos / self.LOCAL_ZONE # This is actually the normalized distance
                distances.append(norm_pos)

                vel = self.k.vehicle.get_speed(vehicles_in_zone[i])
                norm_vel = vel / self.MAX_SPEED

                observation_tse[i] = [norm_pos, norm_vel]
                
        observation_tse = np.array(observation_tse, dtype=np.float32)
        #print("Observation TSE: ", observation_tse)

        self.tse_output = self.get_tse_output(observation_tse)
        self.tse_output_encoded = np.zeros(6) 
        self.tse_output_encoded[self.tse_output] = 1

        print(f"\nTSE output: {self.tse_output}, one hot encoded: {self.tse_output_encoded}, meaning: {self.label_meaning[self.tse_output[0]]}")

        # For RL agent
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])

            # Only lead gets the full observation
            if rl_id == self.lead_rl_id:
                observation = np.append(observation, self.tse_output_encoded)
            else: 
                observation = np.append(observation, np.zeros(6)) # Dummy, zeros (because observation space is fixed)
                
            obs.update({rl_id: observation})
            print(f"RL_ID: {rl_id.split('_')[1]}, observation: {observation}, shape: {observation.shape}")

        print("\n")
        #print(f"Observations new: {obs} \n")
        return obs

########
## Reward 11: Corrections to 10

    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        lead_accel = rl_actions[lead_rl_id].item()
        magnitude = np.abs(lead_accel)
        sign = np.sign(lead_accel)
        print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        reward_leader = 0.5*np.mean(vel) - 4*magnitude

        # Forming shaping 
        penalty_scalar = -4
        fixed_penalty = -0.4
        if self.tse_output[0] == 1:
            if sign>=0:
                forming_penalty = penalty_scalar*magnitude
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward_leader += forming_penalty # If congestion is fomring, penalize acceleration
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*followers_avg_speed -0.5*mean_time_headway  -0.5*std_time_headway 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 12: Minor Corrections to 11 (2 varieties tested)


########
## Reward 13: Radically different approach, the leader is no more trained
def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        # lead_accel = rl_actions[lead_rl_id].item()
        # magnitude = np.abs(lead_accel)
        # sign = np.sign(lead_accel)
        # print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        # reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        # penalty_scalar = -8 #4
        # fixed_penalty = -0.8 #0.4
        # if self.tse_output[0] == 1:
        #     if sign>=0:
        #         forming_penalty = penalty_scalar*magnitude
        #         # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
        #         # min is correct bacause values are -ve
        #         forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
        #         print(f"Forming: {forming_penalty}")
        #         reward_leader += forming_penalty # If congestion is fomring, penalize acceleration

        reward_leader = 0.0
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}, mean actions: {mean_actions_followers}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
 	reward_followers = 0.2*followers_avg_speed -0.5*mean_time_headway  -0.5*std_time_headway
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 14: Corrections to 13, followers have high variation
def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        # lead_accel = rl_actions[lead_rl_id].item()
        # magnitude = np.abs(lead_accel)
        # sign = np.sign(lead_accel)
        # print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        # reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        # penalty_scalar = -8 #4
        # fixed_penalty = -0.8 #0.4
        # if self.tse_output[0] == 1:
        #     if sign>=0:
        #         forming_penalty = penalty_scalar*magnitude
        #         # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
        #         # min is correct bacause values are -ve
        #         forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
        #         print(f"Forming: {forming_penalty}")
        #         reward_leader += forming_penalty # If congestion is fomring, penalize acceleration

        reward_leader = 0.0
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        follower_actions = [rl_actions[id] for id in follower_ids]
        mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}, mean actions: {mean_actions_followers}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*followers_avg_speed -0.5*mean_time_headway  -4*mean_actions_followers 
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 15: Corrections to 15, followers get action penalties independently
    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        # lead_accel = rl_actions[lead_rl_id].item()
        # magnitude = np.abs(lead_accel)
        # sign = np.sign(lead_accel)
        # print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        # reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        # penalty_scalar = -8 #4
        # fixed_penalty = -0.8 #0.4
        # if self.tse_output[0] == 1:
        #     if sign>=0:
        #         forming_penalty = penalty_scalar*magnitude
        #         # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
        #         # min is correct bacause values are -ve
        #         forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
        #         print(f"Forming: {forming_penalty}")
        #         reward_leader += forming_penalty # If congestion is fomring, penalize acceleration

        reward_leader = 0.0
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        #follower_actions = [rl_actions[id] for id in follower_ids]
        #mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        time_headways = []
        for vehicle in follower_ids:
            lead_id = self.k.vehicle.get_leader(vehicle)
            # prevent division by zero
            front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
            front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
            time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
            print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        mean_time_headway = np.mean(time_headways)
        std_time_headway = np.std(time_headways)
        print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*followers_avg_speed -0.5*mean_time_headway  #-4*mean_actions_followers # This is 9* in Ours9x
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            reward_follow = reward_followers - 4*np.abs(rl_actions[follower_id].item())
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 16: Corrections to 15, followers get action penalties independently
    def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]
        # lead_accel = rl_actions[lead_rl_id].item()
        # magnitude = np.abs(lead_accel)
        # sign = np.sign(lead_accel)
        # print(f"Lead accel: {lead_accel}, magnitude: {magnitude}, sign: {sign}")

        # reward_leader = 0.2*np.mean(vel) - 4*magnitude

        # Forming shaping 
        # penalty_scalar = -8 #4
        # fixed_penalty = -0.8 #0.4
        # if self.tse_output[0] == 1:
        #     if sign>=0:
        #         forming_penalty = penalty_scalar*magnitude
        #         # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
        #         # min is correct bacause values are -ve
        #         forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
        #         print(f"Forming: {forming_penalty}")
        #         reward_leader += forming_penalty # If congestion is fomring, penalize acceleration

        reward_leader = 0.0
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]
        #follower_actions = [rl_actions[id] for id in follower_ids]
        #mean_actions_followers = np.mean(np.abs(follower_actions)) # penalize control action
        #std_actions_followers = np.std(np.abs(follower_actions))

        # Penalize the time headway difference to thier respective leader (both mean and std) because the same reward is shared 
        # time_headways = []
        # for vehicle in follower_ids:
        #     lead_id = self.k.vehicle.get_leader(vehicle)
        #     # prevent division by zero
        #     front_speed = max(0.01, self.k.vehicle.get_speed(self.k.vehicle.get_leader(vehicle)))
        #     front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(vehicle)) % self.k.network.length()
        #     time_headways.append(front_distance/front_speed)  # close approximation, assume zero instantaneous acceleration
        #     print(f"ID: {vehicle} Front speed: {front_speed}, front distance: {front_distance}, time headway: {front_distance/front_speed}")
        
        # mean_time_headway = np.mean(time_headways)
        # std_time_headway = np.std(time_headways)
        # print(f"Mean time headway: {mean_time_headway}, std time headway: {std_time_headway}")

        followers_avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in follower_ids])

        # Option 1 Follow closely
        # Include average velocity to precent them from stopping (to maximize the rest of the reward)
        reward_followers = 0.2*followers_avg_speed #-0.5*mean_time_headway  #-4*mean_actions_followers # This is 9* in Ours9x
        
        # Option 2 Follow closely but maintain a minimum
        # reward_followers = 0.2*np.mean(vel) - 2*mean_actions_followers  -2*std_time_headway 
        # headway_threshold = 1.5 # s
        # if mean_time_headway < headway_threshold:
        #     reward_followers -= 20*(headway_threshold - mean_time_headway) 
        # else:
        #     reward_followers -= mean_time_headway

        for follower_id in follower_ids:
            reward_follow = reward_followers - 4*np.abs(rl_actions[follower_id].item())
            rew.update({follower_id : reward_followers})
        
        print(f"Leader reward: {reward_leader}, Follower reward: {reward_followers}")
        return rew

########
## Reward 17: Corrections to 16, simplifications

def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]

        reward_leader = 0.0
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]


        for follower_id in follower_ids:
            
            # Speed of the follower
            follower_speed = self.k.vehicle.get_speed(follower_id)

            # Time headway of the follower
            # lead_id = self.k.vehicle.get_leader(follower_id)
            # # prevent division by zero
            # front_speed = max(0.01, self.k.vehicle.get_speed(lead_id))
            # front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(follower_id)) % self.k.network.length()
            # time_headway = front_distance/front_speed  # close approximation, assume zero instantaneous acceleration

            # Acceleration of the follower
            follower_accel_magnitude = np.abs(rl_actions[follower_id].item())

            reward_follow =  0.2 * follower_speed - 4 * follower_accel_magnitude
            print(f"follower_id: {follower_id}, reward_follow: {reward_follow} \n")
            rew.update({follower_id : reward_follow})
        
        return rew

"""
################################################
SINGLEAGENT: EFFICIENCY
################################################
"""
# Reward 1: Make the agent more like follower stopper
    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.2*np.mean(vel) - 5*magnitude
        print(f"First Reward: {reward}")
        
        # Congested and undefined shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 3 or self.tse_output[0] == 4:
            if sign>=0:
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        elif self.tse_output[0] == 2: # Free flow
            # Penalize acceleration/deceleration magnitude
            free_flow_penalty = -5*magnitude
            print(f"Free flow: {free_flow_penalty}")
            reward += free_flow_penalty

        print(f"Last Reward: {reward}")
        return reward

########
# Reward 2: 1) with corrections

# Did not put here 

########
# Reward 3: 2) with corrections
    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.8*np.mean(vel) - 3*magnitude # The general acceleration penalty
        print(f"First Reward: {reward}")
        
        # Congested, fomring and undefined shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 3 or self.tse_output[0] == 4 or self.tse_output[0] == 1:
            if sign>=0:
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        elif self.tse_output[0] == 2: # Free flow
            # Penalize acceleration/deceleration magnitude
            free_flow_penalty = -5*magnitude
            print(f"Free flow: {free_flow_penalty}")
            reward += free_flow_penalty

        # Leaving shaping 
        elif self.tse_output[0] == 0:
            # We want the acceleration to be positive
            if sign<0:
                leaving_penalty = -2*magnitude
                print(f"Leaving: {leaving_penalty}")
                reward += leaving_penalty

        print(f"Last Reward: {reward}")
        return reward

########
# Reward 4: 3) with corrections

    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = 0.7*np.mean(vel) - 4*magnitude # The general acceleration penalty
        print(f"First Reward: {reward}")
        
        # Congested, fomring and undefined shaping 
        penalty_scalar = -10
        fixed_penalty = -1
        if self.tse_output[0] == 3 or self.tse_output[0] == 4 or self.tse_output[0] == 1:
            if sign>0: # This equal to sign must be removed. To make more like FS sign=0 has to be enabled
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        elif self.tse_output[0] == 2: # Free flow
            # Penalize acceleration/deceleration magnitude
            free_flow_penalty = -5*magnitude
            print(f"Free flow: {free_flow_penalty}")
            reward += free_flow_penalty

        # Leaving shaping 
        elif self.tse_output[0] == 0:
            # We want the acceleration to be positive
            if sign<0:
                leaving_penalty = -2*magnitude
                print(f"Leaving: {leaving_penalty}")
                reward += leaving_penalty

        print(f"Last Reward: {reward}")
        return reward

########
# Reward 5: 4) with corrections
    def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = np.mean(vel) - 2*magnitude # The general acceleration penalty
        print(f"First Reward: {reward}")
        
        # Congested, forming and undefined shaping 
        penalty_scalar = -10
        #penalty_scalar_2 = -5
        penalty_scalar_3 = -10
        fixed_penalty = -1

        # Maintaining velocity is fine
        if self.tse_output[0] == 3 or self.tse_output[0] == 4 or self.tse_output[0] == 1:
            if sign>0: # This equal to sign must be removed. To make more like FS sign=0 has to be enabled
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        # Acheivement of free flow itself should not induce any penalty. Or it will keep oscillating
        # elif self.tse_output[0] == 2: # Free flow
        #     # Penalize acceleration/deceleration magnitude
        #     free_flow_penalty = -penalty_scalar_2*magnitude
        #     print(f"Free flow: {free_flow_penalty}")
        #     reward += free_flow_penalty

        # Leaving shaping 
        elif self.tse_output[0] == 0:
            # We want the acceleration to be positive
            if sign<0:
                leaving_penalty = min(fixed_penalty, penalty_scalar_3*magnitude)
                print(f"Leaving: {leaving_penalty}")
                reward += leaving_penalty

        print(f"Last Reward: {reward}")
        return reward

########
# Reward 6: 5) with corrections

def compute_reward(self, rl_actions, **kwargs):
        """ 
        """
        # for warmup 
        if rl_actions is None: 
            return 0 

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])
        
        # Fail the current episode if these
        if any(vel <-100) or kwargs['fail']:
            return 0 
        
        rl_id = self.k.vehicle.get_rl_ids()[0]

        # Desired acceleration (control action)
        rl_accel = rl_actions[0]
        
        magnitude = np.abs(rl_accel)
        sign = np.sign(rl_accel) # Sign can be 0.0 as well as -1.0 or 1.0

        print(f"RL accel: {rl_accel}, magnitude: {magnitude}, sign: {sign}")

        # The acceleration penalty is only for the magnitude
        reward = np.mean(vel) - 3*magnitude # The general acceleration penalty
        print(f"First Reward: {reward}")
        
        # Congested, forming and undefined shaping 
        penalty_scalar = -10
        penalty_scalar_3 = -10
        fixed_penalty = -1

        # Maintaining velocity is fine
        if self.tse_output[0] == 3 or self.tse_output[0] == 4 or self.tse_output[0] == 1:
            if sign>0: # This equal to sign must be removed. To make more like FS sign=0 has to be enabled
                # Fixed penalty of -1, to prevent agent from cheating the system when sign= 0 
                # min is correct bacause values are -ve
                forming_penalty = min(fixed_penalty, penalty_scalar*magnitude) 
                print(f"Forming: {forming_penalty}")
                reward += forming_penalty # If congestion is fomring, penalize acceleration

        # Leaving shaping 
        elif self.tse_output[0] == 0:
            # We dont want acceleration to be negative
            if sign<0:
                leaving_penalty = min(fixed_penalty, penalty_scalar_3*magnitude)
                print(f"Leaving: {leaving_penalty}")
                reward += leaving_penalty

        print(f"Last Reward: {reward}")
        return reward


"""
################################################
MULTIAGNET: EFFICIENCY
################################################
"""

# 1 and 2 were a little different with the coefficients
########
# Reward 3: 2) with corrections
def compute_reward(self, rl_actions, **kwargs):
        """
        Lead gets rewarded the same as the singleagent version 
        For maximum stability: others get rewarded for?
        """
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.
        
        rew = {}

        # reward for leader
        lead_rl_id = self.k.vehicle.get_rl_ids()[-1]

        reward_leader = 0.0
        rew.update({lead_rl_id : reward_leader})

        # reward for all followers
        follower_ids = self.k.vehicle.get_rl_ids()[:-1]


        for follower_id in follower_ids:
            
            # Speed of the follower
            follower_speed = self.k.vehicle.get_speed(follower_id)

            # Time headway of the follower
            #lead_id = self.k.vehicle.get_leader(follower_id)
            # prevent division by zero
            #front_speed = max(0.01, self.k.vehicle.get_speed(lead_id))
            #front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(follower_id)) % self.k.network.length()
            #time_headway = front_distance/front_speed  # close approximation, assume zero instantaneous acceleration

            # Acceleration of the follower
            follower_accel_magnitude = np.abs(rl_actions[follower_id].item())

            # For safety, stability 
            #reward_follow =  0.2 * follower_speed - 4 * follower_accel_magnitude 

            # For efficiency 
            reward_follow = follower_speed - 4 * follower_accel_magnitude

            print(f"follower_id: {follower_id}, reward_follow: {reward_follow} \n")
            rew.update({follower_id : reward_follow})
        
        return rew

# Reward 4: 3) with corrections

    def compute_reward(self, rl_actions, **kwargs):
            """
            Lead gets rewarded the same as the singleagent version 
            For maximum stability: others get rewarded for?
            """
            # in the warmup steps
            if rl_actions is None:
                return 0

            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in self.k.vehicle.get_ids()
            ])

            if any(vel < -100) or kwargs['fail']:
                return 0.
            
            rew = {}

            # reward for leader
            lead_rl_id = self.k.vehicle.get_rl_ids()[-1]

            reward_leader = 0.0
            rew.update({lead_rl_id : reward_leader})

            # reward for all followers
            follower_ids = self.k.vehicle.get_rl_ids()[:-1]


            for follower_id in follower_ids:
                
                # Speed of the follower
                follower_speed = self.k.vehicle.get_speed(follower_id)

                # Time headway of the follower
                #lead_id = self.k.vehicle.get_leader(follower_id)
                # prevent division by zero
                #front_speed = max(0.01, self.k.vehicle.get_speed(lead_id))
                #front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(follower_id)) % self.k.network.length()
                #time_headway = front_distance/front_speed  # close approximation, assume zero instantaneous acceleration

                # Acceleration of the follower
                follower_accel_magnitude = np.abs(rl_actions[follower_id].item())

                # For safety, stability 
                #reward_follow =  0.2 * follower_speed - 4 * follower_accel_magnitude 

                # For efficiency 
                reward_follow = 2.0 * follower_speed - 4 * follower_accel_magnitude

                print(f"follower_id: {follower_id}, reward_follow: {reward_follow} \n")
                rew.update({follower_id : reward_follow})
            
            return rew

########
# Reward 6: 5) with corrections

def compute_reward(self, rl_actions, **kwargs):
            """
            Lead gets rewarded the same as the singleagent version 
            For maximum stability: others get rewarded for?
            """
            # in the warmup steps
            if rl_actions is None:
                return 0

            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in self.k.vehicle.get_ids()
            ])

            if any(vel < -100) or kwargs['fail']:
                return 0.
            
            rew = {}

            # reward for leader
            lead_rl_id = self.k.vehicle.get_rl_ids()[-1]

            reward_leader = 0.0
            rew.update({lead_rl_id : reward_leader})

            # reward for all followers
            follower_ids = self.k.vehicle.get_rl_ids()[:-1]


            for follower_id in follower_ids:
                
                # Speed of the follower
                follower_speed = self.k.vehicle.get_speed(follower_id)

                # Time headway of the follower
                #lead_id = self.k.vehicle.get_leader(follower_id)
                # prevent division by zero
                #front_speed = max(0.01, self.k.vehicle.get_speed(lead_id))
                #front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(follower_id)) % self.k.network.length()
                #time_headway = front_distance/front_speed  # close approximation, assume zero instantaneous acceleration

                # Acceleration of the follower
                follower_accel_magnitude = np.abs(rl_actions[follower_id].item())

                # For safety, stability 
                #reward_follow =  0.2 * follower_speed - 4 * follower_accel_magnitude 

                # For efficiency 
                follower_follower = self.k.vehicle.get_follower(follower_id)
                follower_follower_speed = self.k.vehicle.get_speed(follower_follower)
                average_speed = (follower_speed + follower_follower_speed) / 2

                reward_follow = 2.5 * average_speed - 4 * follower_accel_magnitude

                print(f"follower_id: {follower_id}, reward_follow: {reward_follow} \n")
                rew.update({follower_id : reward_follow})
            
            return rew
            
########
# Reward 7: 6) with corrections
    def compute_reward(self, rl_actions, **kwargs):
            """
            Lead gets rewarded the same as the singleagent version 
            For maximum stability: others get rewarded for?
            """
            # in the warmup steps
            if rl_actions is None:
                return 0

            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in self.k.vehicle.get_ids()
            ])

            if any(vel < -100) or kwargs['fail']:
                return 0.
            
            rew = {}

            # reward for leader
            lead_rl_id = self.k.vehicle.get_rl_ids()[-1]

            reward_leader = 0.0
            rew.update({lead_rl_id : reward_leader})

            # reward for all followers
            follower_ids = self.k.vehicle.get_rl_ids()[:-1]


            for follower_id in follower_ids:
                
                # Speed of the follower
                follower_speed = self.k.vehicle.get_speed(follower_id)

                # Acceleration of the follower
                follower_accel_magnitude = np.abs(rl_actions[follower_id].item())

                # For safety, stability 
                #reward_follow =  0.2 * follower_speed - 4 * follower_accel_magnitude 

                # For efficiency 
                # For Ours 4x
                #reward_follow = 2.0 * follower_speed - 4 * follower_accel_magnitude

                # For Ours 9x
                lead_id = self.k.vehicle.get_leader(follower_id)
                front_speed = max(0.01, self.k.vehicle.get_speed(lead_id))
                front_distance = (self.k.vehicle.get_x_by_id(lead_id) - self.k.vehicle.get_x_by_id(follower_id)) % self.k.network.length()
                time_headway = front_distance/front_speed  # close approximation, assume zero instantaneous acceleration
                
                reward_follow = 2* follower_speed - 0.5* time_headway

                print(f"follower_id: {follower_id}, reward_follow: {reward_follow} \n")
                rew.update({follower_id : reward_follow})
            
            return rew