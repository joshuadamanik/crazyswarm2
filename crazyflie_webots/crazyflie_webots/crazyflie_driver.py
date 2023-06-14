#!/usr/bin/env python3

import rclpy
import numpy as np
import math
from rclpy.node import Node
import importlib

from crazyflie_interfaces.srv import Takeoff, Land, GoTo
from crazyflie_interfaces.srv import UploadTrajectory, StartTrajectory, NotifySetpointsStop
from crazyflie_interfaces.msg import Hover, FullState
from std_msgs.msg import Float64MultiArray
import rowan

from functools import partial

# import BackendRviz from .backend_rviz
# from .backend import *
# from .backend.none import BackendNone
from .sim_data_types import State, Action

class CrazyflieDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.__timestep = int(self.__robot.getBasicTimeStep())

        self.__m1_motor = self.__robot.getDevice('m1_motor')
        self.__m2_motor = self.__robot.getDevice('m2_motor')
        self.__m3_motor = self.__robot.getDevice('m3_motor')
        self.__m4_motor = self.__robot.getDevice('m4_motor')

        self.__imu = self.__robot.getDevice('inertial unit')
        self.__gps = self.__robot.getDevice('gps')
        self.__gyro = self.__robot.getDevice('gyro')
        self.__camera = self.__robot.getDevice('camera')
        self.__range_front = self.__robot.getDevice('range_front')
        self.__range_left = self.__robot.getDevice('range_left')
        self.__range_right = self.__robot.getDevice('range_right')
        self.__range_back = self.__robot.getDevice('range_back')

        self.__m1_motor.setPosition(float('inf'))
        self.__m2_motor.setPosition(float('inf'))
        self.__m3_motor.setPosition(float('inf'))
        self.__m4_motor.setPosition(float('inf'))

        self.__m1_motor.setVelocity(-1)
        self.__m2_motor.setVelocity(1)
        self.__m3_motor.setVelocity(-1)
        self.__m4_motor.setVelocity(1)

        self.__imu.enable(self.__timestep)
        self.__gps.enable(self.__timestep)
        self.__gyro.enable(self.__timestep)
        self.__camera.enable(self.__timestep)
        self.__range_front.enable(self.__timestep)
        self.__range_left.enable(self.__timestep)
        self.__range_right.enable(self.__timestep)
        self.__range_back.enable(self.__timestep)

        self.__past_time = self.__robot.getTime()

        self.__past_pos_x = 0
        self.__past_pos_y = 0
        self.__past_pos_z = 0

        rclpy.init(args=None)

        self.__node = rclpy.create_node('crazyflie_driver')
        self.__state_publisher = self.__node.create_publisher(FullState, 'full_state', 1)
        self.__motor_cmd_subscriber = self.__node.create_subscription(Float64MultiArray, 'motor_cmd', self.__motor_cmd_callback, 1)
        self.__timer = self.__node.create_timer(0.001, self.__publish_state)

    def __publish_state(self):
        curr_time = self.__robot.getTime()
        dt = curr_time - self.__past_time

        roll = self.__imu.getRollPitchYaw()[0]
        pitch = self.__imu.getRollPitchYaw()[1]
        yaw = self.__imu.getRollPitchYaw()[2]

        # Convert to quaternion
        quat = rowan.from_euler(roll, pitch, yaw, 'xyz')

        pos_x = 1. * self.__gps.getValues()[0]
        pos_y = 1. * self.__gps.getValues()[1]
        pos_z = 1. * self.__gps.getValues()[2]

        vel_x = (pos_x - self.__past_pos_x) / dt
        vel_y = (pos_y - self.__past_pos_y) / dt
        vel_z = (pos_z - self.__past_pos_z) / dt

        full_state = FullState()
        full_state.header.frame_id = 'webots'
        full_state.header.stamp = rclpy.time.Time(seconds=curr_time).to_msg()

        full_state.pose.position.x = pos_x
        full_state.pose.position.y = pos_y
        full_state.pose.position.z = pos_z

        full_state.pose.orientation.w = quat[0]
        full_state.pose.orientation.x = quat[1]
        full_state.pose.orientation.y = quat[2]
        full_state.pose.orientation.z = quat[3]

        full_state.twist.linear.x = vel_x
        full_state.twist.linear.y = vel_y
        full_state.twist.linear.z = vel_z

        full_state.twist.angular.x = self.__gyro.getValues()[0]
        full_state.twist.angular.y = self.__gyro.getValues()[1]
        full_state.twist.angular.z = self.__gyro.getValues()[2]

        self.__state_publisher.publish(full_state)

        self.__past_time = curr_time
        self.__past_pos_x = pos_x
        self.__past_pos_y = pos_y
        self.__past_pos_z = pos_z

        # Note: we assume here that our control is forces
        arm_length = 0.046 # m
        arm = 0.707106781 * arm_length
        kt = 4e-5 # thrust coefficient
        kq = 2.4e-6 # torque coefficient
        # t2t = 0.006 # thrust-to-torque ratio
        t2t = 0.06 # thrust-to-torque ratio
        self.__B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
        ])
        self.__B1 = np.array([
            [kt, kt, kt, kt],
            [-kt * arm, -kt * arm, kt * arm, kt * arm],
            [-kt * arm, kt * arm, kt * arm, -kt * arm],
            [-kq, kq, -kq, kq]
        ])
        self.__B1_inv = np.linalg.inv(self.__B1)

    def __motor_cmd_callback(self, msg):
        self.__m1_motor.setVelocity(-1. * msg.data[0])
        self.__m2_motor.setVelocity(1. * msg.data[1])
        self.__m3_motor.setVelocity(-1. * msg.data[2])
        self.__m4_motor.setVelocity(1. * msg.data[3])

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
