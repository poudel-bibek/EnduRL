"""
Controllers written for Density Aware RL agent
IDM controllers that can also provide shock
"""
import math 
import numpy as np
from flow.controllers.base_controller import BaseController

class ModifiedIDMController(BaseController):
    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None,

                 shock_vehicle = False,):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

        self.shock_vehicle = shock_vehicle
        self.shock_acceleration = 0.0 # Default
        self.shock_time = False # Per time step decision on whether to shock or not

    def get_accel(self, env):
        """
        it will automatically call this for each vehicle
        At shock times, we have to return the shock acceleration
        """

        # If the vehicle is a registered shock vehicle and shock model says shock now
        if self.shock_vehicle and self.shock_time:
            return self.get_shock_accel()
        else: 
            return self.get_idm_accel(env)

    def set_shock_accel(self, accel):
        self.shock_acceleration = accel
        #print(f"\nFrom the controller: {self.veh_id, self.shock_acceleration}\n")
        #return accel
    
    def get_shock_accel(self):
        #print("Shock")
        return self.shock_acceleration

    def set_shock_time(self, shock_time):
        self.shock_time = shock_time

    def get_shock_time(self):
        return self.shock_time

    def get_idm_accel(self, env):
        """
        it will automatically call this for each vehicle
        At shock times, we have to return the shock acceleration
        """
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        #print("IDM")
        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)