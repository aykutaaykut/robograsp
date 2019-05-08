#!/usr/bin/env python

import sys
import math
import random
import numpy as np
import rospy
import moveit_commander
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from gym import spaces
from tf import TransformListener
from panda_manipulation.MyObject import MyObject, Sphere, Box, Cylinder, Duck, Bunny

class RobotEnv():
    def __init__(self):
        rospy.init_node('robot_env')
        rospy.sleep(1)

        self.robot = moveit_commander.RobotCommander()

        self.arm = moveit_commander.move_group.MoveGroupCommander('panda_arm')
        self.arm.set_end_effector_link('hand')
        self.arm.set_pose_reference_frame('panda_link0')

        self.hand = moveit_commander.move_group.MoveGroupCommander('hand')

        self.arm_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.arm_curr_joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.arm_pre_joint_values = self.arm_curr_joint_values

        self.hand_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        self.hand_curr_joint_values = [0.0, 0.0]
        self.hand_pre_joint_values = self.hand_curr_joint_values

        self.arm_joint_limits = {'panda_joint1' : [-2.8973, 2.8973], 'panda_joint2' : [-1.7628, 1.7628], 'panda_joint3' : [-2.8973, 2.8973],\
        'panda_joint4' : [-3.0718, 0.0175], 'panda_joint5' : [-2.8973, 2.8973], 'panda_joint6' : [-0.0175, 3.7525],'panda_joint7' : [-2.8973, 2.8973]}
        self.hand_joint_limits = {'panda_finger_joint1' : [-0.001, 0.04], 'panda_finger_joint2' : [-0.001, 0.04]}

        self.observation_limit = np.array([100]*15)
        self.action_limit = np.array([0.1] * (len(self.arm_curr_joint_values) + len(self.hand_curr_joint_values)))
        self.observation_space = spaces.Box(low = -self.observation_limit, high = self.observation_limit)
        self.action_space = spaces.Box(low = -self.action_limit, high = self.action_limit)

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.discount = 1.0
        self.joint_exec_duration = 0.01
        self.exec_time = [joint_num * self.joint_exec_duration for joint_num in range(1, 10)]

        self.distance_threshold = 0.1
        self.object_offset = np.array([0.0, 0.0, 0.5])
        self.object_move_threshold = 0.05

        self.transform_listener = TransformListener()
        self.joint_states_sub = rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, self.joint_states_buffer)
        self.object = Box()

    def joint_states_buffer(self, joint_states_msg):
        joint_dict = dict(zip(joint_states_msg.name, joint_states_msg.position))
        self.arm_pre_joint_values = [joint_dict[joint] for joint in self.arm_joint_names]
        self.hand_pre_joint_values = [joint_dict[joint] for joint in self.hand_joint_names]

    def get_arm_joint_lower_limits(self):
        return [self.arm_joint_limits[k][0] for k in self.arm_joint_names]

    def get_hand_joint_lower_limits(self):
        return [self.hand_joint_limits[k][0] for k in self.hand_joint_names]

    def get_arm_joint_upper_limits(self):
        return [self.arm_joint_limits[k][1] for k in self.arm_joint_names]

    def get_hand_joint_upper_limits(self):
        return [self.hand_joint_limits[k][1] for k in self.hand_joint_names]

    def is_in_motion(self):
        arm_joint_diff = np.abs(np.array(self.arm_curr_joint_values) - np.array(self.arm_pre_joint_values))
        hand_joint_diff = np.abs(np.array(self.hand_curr_joint_values) - np.array(self.hand_pre_joint_values))
        return any(arm_joint_diff > 0.1) or any(hand_joint_diff > 0.1)

    def get_arm_pose(self):
        return self.arm.get_current_pose()

    def get_hand_pose(self):
        return self.arm.get_current_pose()

    def get_gripper_position(self, reference):
        self.transform_listener.waitForTransform(reference, '/panda_hand', rospy.Time(), rospy.Duration(10))
        tf = self.transform_listener.getLatestCommonTime(reference, '/panda_hand')
        position, quaternion = self.transform_listener.lookupTransform(reference, '/panda_hand', tf)
        return np.array([position[0], position[1], position[2]])

    def get_object_position(self):
        object_pos = self.object.get_position()
        return np.array([object_pos.position.x, object_pos.position.y, object_pos.position.z])

    def calculate_distance(self):
        gripper_curr_position = self.get_gripper_position('/world')
        object_position = self.get_object_position()
        return np.linalg.norm(gripper_curr_position - object_position - self.object_offset)

    def initialize_arm_joint_values(self):
        return np.random.uniform(self.get_arm_joint_lower_limits(), self.get_arm_joint_upper_limits(), len(self.arm_curr_joint_values)).tolist()

    def initialize_hand_joint_values(self):
        return np.random.uniform(self.get_hand_joint_lower_limits(), self.get_hand_joint_upper_limits(), len(self.hand_curr_joint_values)).tolist()

    def wait(self, step, threshold):
        waiting_time = 0.0
        while self.is_in_motion():
            rospy.sleep(step)
            waiting_time += 0.01
            if waiting_time >= threshold:
                print "Time is up!"
                raise
    def execute(self):
        arm_plan = self.arm.plan(self.arm_curr_joint_values)
        self.arm.execute(arm_plan)
        hand_plan = self.hand.plan(self.hand_curr_joint_values)
        self.hand.execute(hand_plan)

    def reset(self):
        self.arm_curr_joint_values = self.initialize_arm_joint_values()
        self.hand_curr_joint_values = self.initialize_hand_joint_values()
        self.execute()
        self.wait(0.01, 500)
        object_pose = geometry_msgs.msg.Pose()
        object_pose.position.x = 0.15
        object_pose.position.y = 0.0
        object_pose.position.z = 0.8
        self.object.set_position(object_pose)
        self.object.place_on_table()
        rospy.sleep(1)
        self.object_initial_position = self.get_object_position()
        return np.concatenate((np.concatenate((self.arm_curr_joint_values, self.hand_curr_joint_values), axis=0), self.get_gripper_position('/world'), self.object_initial_position), axis=0).tolist()

    def step(self, action):
        done = False
        info = {}
        arm_action = action[:7]
        hand_action = action[7:]
        curr_distance = self.calculate_distance()
        self.arm_curr_joint_values = np.clip(self.arm_curr_joint_values + arm_action, self.get_arm_joint_lower_limits(), self.get_arm_joint_upper_limits())
        self.hand_curr_joint_values = np.clip(self.hand_curr_joint_values + hand_action, self.get_hand_joint_lower_limits(), self.get_hand_joint_upper_limits())
        self.execute()
        self.wait(0.01, 500)
        next_distance = self.calculate_distance()
        reward = -(next_distance * next_distance) + (0.5 * (next_distance - curr_distance) * (next_distance - curr_distance))
        if next_distance <= self.distance_threshold:
            reward += 100
            done = True
        elif self.get_object_position()[2] < 0.5:
            reward -= 10
            done = True
        elif np.linalg.norm(self.get_object_position() - self.object_initial_position) >= self.object_move_threshold:
            reward -= 10
            done = True
        return np.concatenate((np.concatenate((self.arm_curr_joint_values, self.hand_curr_joint_values), axis=0), self.get_gripper_position('world'), self.get_object_position()), axis=0).tolist(), reward, done, {}

    def done(self):
        self.joint_states_sub.unregister()
        rospy.signal_shutdown("done")
