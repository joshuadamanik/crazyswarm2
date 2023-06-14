from __future__ import annotations

from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rclpy.time import Time
from ..sim_data_types import State, Action
from crazyflie_interfaces.msg import Hover, FullState
from std_msgs.msg import Float64MultiArray
import numpy as np


class Backend:
    """Tracks the desired state perfectly (no physics simulation)"""

    def __init__(self, node: Node, names: list[str], states: list[State]):
        self.node = node
        self.state_subscriber = node.create_subscription(FullState, 'full_state', self.state_callback, 10)
        self.motor_cmd_publisher = node.create_publisher(Float64MultiArray, 'motor_cmd', 10)
        self.state = states[0]

        self.names = names
        self.clock_publisher = node.create_publisher(Clock, 'clock', 10)
        self.t = 0
        arm_length = 0.046 # m
        arm = 0.707106781 * arm_length
        kt = 4e-05
        kq = 2.4e-07
        t2t = kq / kt

        arm_length = 0.046 # m
        arm = 0.707106781 * arm_length
        t2t = 0.006 # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
        ])
        self.B1 = np.array([
            [kt, kt, kt, kt],
            [-kt*arm, -kt*arm, kt*arm, kt*arm],
            [-kt*arm, kt*arm, kt*arm, -kt*arm],
            [-kq, kq, -kq, kq]
        ])
        self.inv_B1 = np.linalg.pinv(self.B1)

    def state_callback(self, msg: FullState):
        self.t = msg.header.stamp.sec
        self.state.pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.state.vel = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        self.state.quat = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        self.state.omega = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]

    def time(self) -> float:
        return self.t

    def step(self, states_desired: list[State], actions: list[Action]) -> list[State]:
        # advance the time

        # publish the current clock
        clock_message = Clock()
        clock_message.clock = Time(seconds=self.time()).to_msg()
        self.clock_publisher.publish(clock_message)

        # publish the motor commands
        action = actions[0]
        def rpm_to_force(rpm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [2.55077341e-08, -4.92422570e-05, -1.51910248e-01]
            force_in_grams = np.polyval(p, rpm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        force = rpm_to_force(action.rpm)
        eta = np.sqrt(np.dot(self.inv_B1, np.dot(self.B0, force)))

        motor_cmd = Float64MultiArray()
        motor_cmd.data = [
            1. * eta[0],
            1. * eta[1],
            1. * eta[2],
            1. * eta[3],
        ]
        self.motor_cmd_publisher.publish(motor_cmd)

        # pretend we were able to follow desired states perfectly
        return [self.state,]

    def shutdown(self):
        pass

