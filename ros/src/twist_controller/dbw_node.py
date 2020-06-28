#!/usr/bin/env python

import rospy
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # publish message to the vehicle 
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # TODO: Create `Controller` object
        self.controller = Controller(vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                                     wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle)

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped,
                         self.current_velocity_cb)  # car current velocity (from sim/carla)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)  # car planned velocity (from waypoint)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # subscribe variables
        self.current_vel = None  # current linear/angular velocity
        self.twist_cmd = None  # target linear/angular velocity
        self.dbw_enabled = False  # dbw enable
        self.current_velocity_init = False  # first connection established(in sync with publish loop)
        self.twist_cmd_init = False  # first connection established(in sync with publish loop)
        # publish loop
        self.loop()

    def current_velocity_cb(self, msg):
        if not self.current_velocity_init:
            self.current_velocity_init = True
        self.current_vel = msg

    def twist_cmd_cb(self, msg):
        if not self.twist_cmd_init:
            self.twist_cmd_init = True
        self.twist_cmd = msg

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg

    def connection_estblish(self):
        return self.current_velocity_init & self.twist_cmd_init

    def loop(self):
        # publishing freq.
        rate = rospy.Rate(50)
        # using estimate time loop for now (try message time stamp?)
        dt = 1. / 50
        # first connection
        throttle = 0.0
        brake = 0.0
        steering = 0.0

        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled

            # check subscribe connections if established            
            if self.current_velocity_init and self.twist_cmd_init:
                # feedback control
                target_v = self.twist_cmd.twist.linear
                target_w = self.twist_cmd.twist.angular
                current_v = self.current_vel.twist.linear
                current_W = self.current_vel.twist.angular

                # publish commands only when dbw is enable
                if self.dbw_enabled:
                    rospy.loginfo("DBW is enable......ON ")
                    throttle, brake, steering = self.controller.control(target_v, target_w, current_v, current_W,
                                                                        self.dbw_enabled)

                    rospy.loginfo("DBW node throttle: %f ... brake: %f ... Steer: %f", throttle, brake, steering)

                    self.publish(throttle, brake, steering)
                else:
                    rospy.loginfo("DBW is disable.....OFF! ")
                    self.publish(0., 0., 0.)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
