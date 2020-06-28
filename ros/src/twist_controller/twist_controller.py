import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
from math import *

##
#    Throttle PID
##
Kp_v = 1.25      #0.03      # 0.08      # 0.05
Kd_v = 1.     #0.01      # 0.02      # 0.02
Ki_v = 0.1    #.002     # 0.005     # 0.001

##
#    Steering PID
##
Kp_s = 1.0
Kd_s = 0.6
Ki_s = 0.0005


##
#    Vehicle throttle control setting
## 
GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_THROTTLE = -2.0
MAX_THROTTLE = 1.0

MIN_INTEGRAL = -10.  # this will max the integral error to only contribute 0.2 to the throttle
MAX_INTEGRAL = 10.

class Controller(object):
    # TODO: Implement
    def __init__(self,vehicle_mass,fuel_capacity,brake_deadband,decel_limit,accel_limit,wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):
        self.target_v = 0.0
        self.target_w = 0.0
        self.current_v = 0.0
        self.error_v  = 0.0
        self.min_speed = 0.0
        self.highest_v = 0

        self.dbw_status = False
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.vehicle_mass = vehicle_mass
        self.wheel_radius  = wheel_radius

        self.steering = YawController(self.wheel_base, self.steer_ratio, self.min_speed, self.max_lat_accel, self.max_steer_angle)
        self.steer_pid = PID(Kp_s, Ki_s, Kd_s, mn=-self.max_steer_angle, mx=self.max_steer_angle, min_i=-self.max_steer_angle, max_i=self.max_steer_angle)
        self.steer_filter = LowPassFilter(14, 1)  # use only 10%

        self.throttle = PID(Kp_v, Ki_v, Kd_v, mn=MIN_THROTTLE, mx=MAX_THROTTLE, min_i=MIN_INTEGRAL, max_i=MAX_INTEGRAL)
        self.vel_filter = LowPassFilter(29, 1)  # use only 14.29% of latest error
		
        self.last_time = rospy.get_time()

    def control(self,target_v,target_w,current_v,current_w,dbw_status):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # Throttle := [0,1], Brake := N*m, Steering := Radian      
        if not dbw_status:
            self.throttle.reset()
            self.steer_pid.reset()
            return 0., 0., 0. 

        # linear speed error in x
        v_current = self.vel_filter.filt(abs(current_v.x))
        v_target  = abs(target_v.x)
        v_error   = v_target - v_current

        self.highest_v = max(self.highest_v, current_v.x)

        throttle_cmd = 0.    # Throttle command value
        brake_cmd = 0.     # Throttle command value

        d_t = 2.0  # desire braking time: reduce to the target speed within d_t
        # over the limit, then we need to apply brake

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle_cmd = self.throttle.step(v_error,sample_time)
        brake = 0

        # Throttle and brake control
        if(current_v.x < 2.) and (target_v.x < 1.):
            throttle_cmd = 0.
            self.throttle.reset()
            brake_cmd = 400.  #Nm

            # allow a small margin of error before applying the brakes
        elif (throttle_cmd < 0.):
            gain = abs(throttle_cmd) * current_v.x / 10.  
            Bf = self.vehicle_mass #* 1. * 9.81
            brake_cmd = fabs(Bf * self.wheel_radius * gain)              

            throttle_cmd = max(throttle_cmd, 0.0)

        # Steering angle
        w_target  = target_w.z
        steering_error = w_target 
        steering_cmd = self.steering.get_steering(v_target, steering_error, v_current)
        steering_cmd = self.steer_pid.step(steering_cmd, sample_time)
       
        # Return throttle, brake, steer in this order
        return throttle_cmd, brake_cmd, steering_cmd
