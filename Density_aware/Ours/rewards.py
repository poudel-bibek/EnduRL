 ## Round_3_1
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
        
        intensity = np.abs(rl_actions[0]) # absolute value of acceleration
        direction = np.sign(intensity) # direction 1.0 = acceleration, -1.0 = deceleration
        #print("Intensity, Direction: ", intensity, direction)

        #rl_id = self.k.vehicle.get_rl_ids()[0]
        #rl_speed = self.k.vehicle.get_speed(rl_id)
        
        response_scaler = 4

        reward = 0
        penalty = -2

        # Reward 
        if self.approching_congestion:

            if direction == -1.0:
                reward += response_scaler * (self.MAX_DECEL - intensity)
                #print("One")
            else:
                reward -= penalty
                #print("Two")
            
        elif self.leaving_congestion:
            if direction == 1.0:
                reward += response_scaler * intensity
                #print("Three")
            else:
                reward -= penalty
                #print("Four")
        else:
            reward += np.mean(vel)
            #print("Five")
        
        # Penalize too much control actions regardless of the situation
        reward -= np.mean(np.abs(rl_actions))

        #print("Reward: ", reward)
        return float(reward)

## Round 3_2

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
        
        intensity = np.abs(rl_actions[0]) # absolute value of acceleration
        direction = np.sign(intensity) # direction 1.0 = acceleration, -1.0 = deceleration
        #print("Intensity, Direction: ", intensity, direction)

        #rl_id = self.k.vehicle.get_rl_ids()[0]
        #rl_speed = self.k.vehicle.get_speed(rl_id)
        
        response_scaler = 4

        reward = 0
        penalty = -2

        # Reward 
        if self.approching_congestion:

            if direction == -1.0:
                reward += response_scaler * (self.MAX_DECEL - intensity)
                #print("One")
            else:
                reward -= penalty
                #print("Two")
            
        elif self.leaving_congestion:
            if direction == 1.0:
                reward += response_scaler * intensity
                #print("Three")
            else:
                reward -= penalty
                #print("Four")
        else:
            reward += np.mean(vel)
            #print("Five")
        
        # Reward a high average speed regardless of the siatuation
        reward += np.mean(vel)

        # Penalize too much control actions regardless of the siatuation
        reward -= 4*np.mean(np.abs(rl_actions))

        #print("Reward: ", reward)
        return float(reward)