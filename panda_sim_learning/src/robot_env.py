#!/usr/bin/env python
# roscore
# roslaunch panda_sim_learning world.launch
# rosrun pc_segmentation def_loop -rt /camera/points -v 0 -dt 0.5 -ct 5 -t 5 -e 0.1
# gzclient
# rostopic echo /joint_states

import sys
import math
from itertools import product
from operator import add
import random
import numpy as np
import rospy
from pc_segmentation.msg import PcFeatures
import moveit_commander
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from gym import spaces
from panda_manipulation.MyObject import MyObject, Sphere, Box, Cylinder, Duck, Bunny
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, Pose

class RobotEnv():
    def __init__(self):
        self.robot = moveit_commander.RobotCommander()

        self.arm = moveit_commander.move_group.MoveGroupCommander('panda_arm')
        self.arm.set_end_effector_link('panda_hand')
        self.arm.set_pose_reference_frame('panda_link0')

        self.hand = moveit_commander.move_group.MoveGroupCommander('hand')

        self.arm_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.arm_curr_joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.arm_pre_joint_values = self.arm_curr_joint_values

        self.hand_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        self.hand_curr_joint_values = [0.0, 0.0]
        self.hand_pre_joint_values = self.hand_curr_joint_values

        self.arm_joint_limits = {'panda_joint1' : [-1.4, 1.4], 'panda_joint2' : [-1.0, 1.2], 'panda_joint3' : [-1.4, 1.4],\
        'panda_joint4' : [-3.0, 0.0], 'panda_joint5' : [-2.8, 0], 'panda_joint6' : [0, 3.6], 'panda_joint7' : [-2.8, 2.8]}
        self.hand_joint_limits = {'panda_finger_joint1' : [-0.001, 0.04], 'panda_finger_joint2' : [-0.001, 0.04]}

        # State and action space definition
        self.observation_limit = np.array([100]*2)
        self.observation_space = spaces.Box(low = -self.observation_limit, high = self.observation_limit)
        self.action_step_size = 0.1
#        self.action_space = spaces.Discrete(2**len(self.arm_joint_names))
#        self.actions = {k : v for k, v in zip(range(self.action_space.n), [list(a) for a in list(product([-self.action_step_size, +self.action_step_size], repeat = len(self.arm_joint_names)))])}
        self.action_space = spaces.Discrete(4)
        self.actions = {0 : [0.0, +self.action_step_size, 0.0, +self.action_step_size, 0.0, 0.0, 0.0], 1 : [0.0, -self.action_step_size, 0.0, -self.action_step_size, 0.0, 0.0, 0.0], 2 : [0.0, +self.action_step_size, 0.0, -self.action_step_size, 0.0, 0.0, 0.0], 3 : [0.0, -self.action_step_size, 0.0, +self.action_step_size, 0.0, 0.0, 0.0]}

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.n
#        self.discount = 1.0
#        self.joint_exec_duration = 0.01
#        self.exec_time = [joint_num * self.joint_exec_duration for joint_num in range(1, 10)]

        self.distance_threshold = 0.1
        self.object_offset = np.array([0.0, 0.0, 0.05])
        self.object_move_threshold = 0.05

        self.joint_states_sub = rospy.Subscriber('/joint_states', sensor_msgs.msg.JointState, self.joint_states_buffer)
        self.object = Box()
        rospy.sleep(1)
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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
        return any(arm_joint_diff > (self.action_step_size/4.0)) or any(hand_joint_diff > (self.action_step_size/4.0))

    def get_arm_pose(self):
        return self.arm.get_current_pose()

    def get_hand_pose(self):
        return self.hand.get_current_pose()

    def get_gripper_position(self, reference):
        tf = self.tf_buffer.lookup_transform(reference, 'panda_hand', rospy.Time(0), rospy.Duration(10.0))
        return np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])

    def get_object_position(self):
        object_pos = self.object.get_position()
        return np.array([object_pos.position.x, object_pos.position.y, object_pos.position.z])

    def calculate_distance(self):
        gripper_curr_position = self.get_gripper_position('world')
        object_position = self.get_object_position()
        return np.linalg.norm(gripper_curr_position - (object_position + self.object_offset))

    def initialize_arm_joint_values(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def initialize_hand_joint_values(self):
        return [0.02, 0.02]

    def random_initialize_arm_joint_values(self):
        return np.random.uniform(self.get_arm_joint_lower_limits(), self.get_arm_joint_upper_limits(), len(self.arm_curr_joint_values)).tolist()

    def random_initialize_hand_joint_values(self):
        return np.random.uniform(self.get_hand_joint_lower_limits(), self.get_hand_joint_upper_limits(), len(self.hand_curr_joint_values)).tolist()

    def wait(self, step, threshold):
        waiting_time = 0.0
        while self.is_in_motion():
            rospy.sleep(step)
            waiting_time += 0.01
            if waiting_time >= threshold:
                print "Time is up!"
                break

    def execute(self):
        arm_plan = self.arm.plan(self.arm_curr_joint_values)
        hand_plan = self.hand.plan(self.hand_curr_joint_values)
        if (len(arm_plan.joint_trajectory.points) == 0) or (len(hand_plan.joint_trajectory.points) == 0):
            return False
        else:
            self.arm.execute(arm_plan)
            self.hand.execute(hand_plan)
            return True

    def reset(self):
        self.arm_curr_joint_values = self.initialize_arm_joint_values()
        self.hand_curr_joint_values = self.initialize_hand_joint_values()
        self.execute()
#        self.wait(0.01, 300)
        object_pose = geometry_msgs.msg.Pose()
        object_pose.position.x = 0.3
        object_pose.position.y = 0.0
        object_pose.position.z = 0.8
        self.object.set_position(object_pose)
        self.object.place_on_table()
        rospy.sleep(1)
        self.object_initial_position = self.get_object_position()
        return [self.arm_curr_joint_values[1]] + [self.arm_curr_joint_values[3]]
#        return np.concatenate((np.concatenate((self.arm_curr_joint_values, self.hand_curr_joint_values), axis=0), self.get_gripper_position('world'), self.object_initial_position), axis=0).tolist()

    def transform(self, coordinates):
        transform = self.tf_buffer.lookup_transform('panda_link0', 'camera_depth_optical_frame', rospy.Time(0), rospy.Duration(1.0))

        kinect_object = PointStamped()
        kinect_object.point.x = coordinates.x
        kinect_object.point.y = coordinates.y
        kinect_object.point.z = coordinates.z
        kinect_object.header = transform.header

        robot_object = tf2_geometry_msgs.do_transform_point(kinect_object, transform)

        pos = Pose()
        pos.position.x = robot_object.point.x
        pos.position.y = robot_object.point.y
        pos.position.z = robot_object.point.z
#        pos.orientation = action.orientation
        return pos

    def step(self, action):
#        before_data = rospy.wait_for_message('/baris/features', PcFeatures)
#        pose_before = self.transform(before_data.bb_center)
        done = False
        info = {}
        arm_action = self.actions[action]
#        hand_action = action
        curr_distance = self.calculate_distance()
        self.arm_curr_joint_values = np.clip(list(map(add, self.arm_curr_joint_values, arm_action)), self.get_arm_joint_lower_limits(), self.get_arm_joint_upper_limits())
#        self.hand_curr_joint_values = np.clip(self.hand_curr_joint_values + hand_action, self.get_hand_joint_lower_limits(), self.get_hand_joint_upper_limits())
        if self.execute():
            successful_planning = True
        else:
            successful_planning = False

#        self.wait(0.01, 300)
        next_distance = self.calculate_distance()
        # Reward definition
#        reward = -(next_distance**2) + (0.5 * ((next_distance - curr_distance)**2))
        reward = -next_distance
        if not successful_planning:
            reward -= 10
            done = True
        elif next_distance <= self.distance_threshold:
            reward += 100
            done = True
        elif self.get_object_position()[2] < 0.5:
            reward -= 10
            done = True
        elif np.linalg.norm(self.get_object_position() - self.object_initial_position) >= self.object_move_threshold:
            reward -= 10
            done = True
        elif self.get_gripper_position("world")[2] < 0.7:
            reward -= 100
            done = True
        next_state = [self.arm_curr_joint_values[1]] + [self.arm_curr_joint_values[3]]
#        next_state = np.concatenate((np.concatenate((self.arm_curr_joint_values, self.hand_curr_joint_values), axis=0), self.get_gripper_position('world'), self.get_object_position()), axis=0).tolist()
#        after_data = rospy.wait_for_message('/baris/features', PcFeatures)
#        pose_after = self.transform(after_data.bb_center)
#        reward = pose_after.position.z - pose_before.position.z
        # pose_after.data yi 16 boyuta cevir
        return  next_state, reward, done, info, next_distance

    def done(self):
        self.joint_states_sub.unregister()
        rospy.signal_shutdown("done")
