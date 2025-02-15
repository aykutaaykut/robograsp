#!/usr/bin/env python
# roscore
# roslaunch panda_sim_learning world.launch
# rosrun pc_segmentation def_loop -rt /camera/points -v 0 -dt 0.5 -ct 5 -t 5 -e 0.1
# gzclient
# rostopic echo /joint_states

import sys
import math
import random
import numpy as np
import rospy
import moveit_commander
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import tf2_ros
import tf2_geometry_msgs
import tf
from math import pi
from itertools import product
from operator import add
from pc_segmentation.msg import PcFeatures
from gym import spaces
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import Grasp
from panda_manipulation.MyObject import MyObject, Sphere, Box, Cylinder, Duck, Bunny

class RobotEnv():
    def __init__(self):
        self.object = None
        self.number_of_objects = 4
        self.arm_joint_indices_to_use = [1, 3, 5]
        self.action_step_size = 0.025
        self.distance_threshold = 0.03
        self.object_move_threshold = 0.025
        self.object_grasp_threshold = 0.05

        self.scene = moveit_commander.PlanningSceneInterface()

        self.robot = moveit_commander.RobotCommander()

        self.arm = self.robot.get_group('panda_arm')
        self.arm.set_end_effector_link('panda_hand')
        self.arm.allow_looking(True)
        self.arm.allow_replanning(True)

        self.hand = self.robot.get_group('hand')
        self.hand.allow_looking(True)
        self.hand.allow_replanning(True)

        self.arm_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.arm_joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.arm_joint_limits = {'panda_joint1' : [-1.4, 1.4],
                                 'panda_joint2' : [-0.6, 1.2],
                                 'panda_joint3' : [-1.4, 1.4],
                                 'panda_joint4' : [-2.8, 0.0],
                                 'panda_joint5' : [-2.8, 0.0],
                                 'panda_joint6' : [0.0, 3.6],
                                 'panda_joint7' : [-2.8, 2.8]}
        self.hand_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        self.hand_joint_values = [0.0, 0.0]
        self.hand_joint_limits = {'panda_finger_joint1' : [0.0, 0.04],
                                  'panda_finger_joint2' : [0.0, 0.04]}

        self.state_dim = (308, len(self.arm_joint_indices_to_use))
        self.action_dim = 2**len(self.arm_joint_indices_to_use) # 2*len(self.arm_joint_indices_to_use)
        self.action_space = spaces.Discrete(self.action_dim)

        self.actions = {}
        # for action_no in range(2*len(self.arm_joint_indices_to_use)):
        #     if action_no % 2 == 0:
        #         coefficient = +self.action_step_size
        #     else:
        #         coefficient = -self.action_step_size
        #     joint_index = self.arm_joint_indices_to_use[int(action_no/2)]
        #     action_list = np.zeros(7).tolist()
        #     action_list[joint_index] = coefficient
        #     self.actions[action_no] = action_list
        # print self.actions

        for action_no in range(self.action_dim):
            binary_action_no = np.array(self.change_base(action_no, 2, len(self.arm_joint_indices_to_use)))
            binary_action_no = 1 - 2 * binary_action_no
            action_list = [0.0] * 7
            for index, coefficient in list(zip(self.arm_joint_indices_to_use, binary_action_no.tolist())):
                action_list[index] = coefficient * self.action_step_size
            self.actions[action_no] = action_list
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def change_base(self, number, new_base, length):
        result = []
        while number > 0:
            number, digit = divmod(number, new_base)
            result.append(digit)
        for i in range(length - len(result)):
            result.append(0)
        return result[::-1]

    def get_arm_joint_lower_limits(self):
        return [self.arm_joint_limits[k][0] for k in self.arm_joint_names]

    def get_hand_joint_lower_limits(self):
        return [self.hand_joint_limits[k][0] for k in self.hand_joint_names]

    def get_arm_joint_upper_limits(self):
        return [self.arm_joint_limits[k][1] for k in self.arm_joint_names]

    def get_hand_joint_upper_limits(self):
        return [self.hand_joint_limits[k][1] for k in self.hand_joint_names]

    # def get_arm_pose(self):
    #     return self.arm.get_current_pose()
    #
    # def get_hand_pose(self):
    #     return self.hand.get_current_pose()

    def get_gripper_position(self):
        tf = self.tf_buffer.lookup_transform('world', 'panda_rightfinger', rospy.Time(0), rospy.Duration(10.0))
        return np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])

    def get_object_position(self):
        object_pos = self.object.get_position()
        return np.array([object_pos.position.x, object_pos.position.y, object_pos.position.z])

    def get_object_shape(self):
        return self.object_shape

    def get_distance_between_gripper_and_object(self):
        gripper_position = self.get_gripper_position()
        object_position = self.get_object_position()
        return np.linalg.norm(gripper_position - (object_position + self.object_offset))

#    def is_gripper_over_object(self):
#        gripper_position = self.get_gripper_position()
#        object_position = self.get_object_position()
#        object_shape = self.get_object_shape()
#        return (abs(gripper_position[0] - object_position[0]) < (object_shape[0]/2.0)) and (abs(gripper_position[1] - object_position[1]) < (object_shape[1]/2.0))

    def initialize_arm_joint_values(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def initialize_hand_joint_values(self):
        return [0.0, 0.0]

#    def random_initialize_arm_joint_values(self):
#        return np.random.uniform(self.get_arm_joint_lower_limits(), self.get_arm_joint_upper_limits(), len(self.arm_joint_values)).tolist()
#    
#    def random_initialize_hand_joint_values(self):
#        return np.random.uniform(self.get_hand_joint_lower_limits(), self.get_hand_joint_upper_limits(), len(self.hand_joint_values)).tolist()
#    
#    def wait(self, step, threshold):
#        waiting_time = 0.0
#        while self.is_in_motion():
#            rospy.sleep(step)
#            waiting_time += 0.01
#            if waiting_time >= threshold:
#                print "Time is up!"
#                break
    
    def plan(self, arm_new_joint_values, hand_new_joint_values):
        arm_plan = self.arm.plan(arm_new_joint_values)
        hand_plan = self.hand.plan(hand_new_joint_values)
        if (len(arm_plan.joint_trajectory.points) == 0):
            arm_plan = None
        if (len(hand_plan.joint_trajectory.points) == 0):
            hand_plan = None
        return arm_plan, hand_plan

    def arm_execute(self, arm_plan):
        self.arm.execute(arm_plan)

    def hand_execute(self, hand_plan):
        self.hand.execute(hand_plan)

    def plan_and_execute(self, arm_new_joint_values, hand_new_joint_values):
        arm_plan, hand_plan = self.plan(arm_new_joint_values, hand_new_joint_values)
        if (arm_plan is not None):
            self.arm_execute(arm_plan)
            self.arm_joint_values = arm_new_joint_values

        if (hand_plan is not None):
            self.hand_execute(hand_plan)
            self.hand_joint_values = hand_new_joint_values
        return arm_plan, hand_plan

    def reset(self):
        if self.object is not None:
            self.object.delete()
            rospy.sleep(5)
            self.scene.remove_world_object(name = 'object_id')
            self.scene.remove_attached_object('panda_hand')
        select_object = random.random()*self.number_of_objects
        if select_object <= 1: #Box
            self.object_type = 0
            self.object_shape = [0.2, 0.04, 0.04]
            self.object = Box()
            self.object_offset = np.array([0.0, 0.0, self.object_shape[2]/2.0])
            self.object_position = [0.3, -0.15, 0.73 + self.object_shape[2]/2] #fixed z=0.73 (table_height + table_thickness + object_shape/2)
        elif select_object <= 2: #Sphere
            self.object_type = 2
            self.object_shape = [0.04, 0.04, 0.04]
            self.object = Sphere()
            self.object_offset = np.array([0.0, 0.0, self.object_shape[2]/2.0])
            self.object_position = [0.3, -0.15, 0.73 + self.object_shape[2]/2] #fixed z=0.73 (table_height + table_thickness + object_shape/2)
        elif select_object <= 3: #Bunny
            self.object_type = 3
            self.object_shape = [0.0472, 0.08, 0.08]
            self.object = Bunny()
            self.object_offset = np.array([0.0, 0.0, self.object_shape[2]/2.0])
            self.object_position = [0.3, -0.15, 0.66 + self.object_shape[2]/2] #fixed z=0.73 (table_height + table_thickness + object_shape/2)
        elif select_object <= 4: #Cylinder
            self.object_type = 1
            self.object_shape = [0.06, 0.06, 0.15]
            self.object = Cylinder('v')
            self.object_offset = np.array([0.0, 0.0, self.object_shape[2]/2.0])
            self.object_position = [0.3, -0.15, 0.66 + self.object_shape[2]/2] #fixed z=0.73 (table_height + table_thickness + object_shape/2)
        else:
            print 'NO OBJECT FOUND!'
            raise
            
        arm_new_joint_values = self.initialize_arm_joint_values()
        hand_new_joint_values = self.initialize_hand_joint_values()
        self.plan_and_execute(arm_new_joint_values, hand_new_joint_values)
        
        object_pose = geometry_msgs.msg.Pose()
        object_pose.position.x = self.object_position[0]
        object_pose.position.y = self.object_position[1]
        object_pose.position.z = self.object_position[2]
        self.object.set_position(object_pose)
        self.object.place_on_table()
        rospy.sleep(2)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.create_planning_scene()
        rospy.sleep(2)
        self.object_initial_position = self.get_object_position()

        #        return np.concatenate((np.concatenate((self.arm_joint_values, self.hand_joint_values), axis=0), self.get_gripper_position('world'), self.object_initial_position), axis=0).tolist()
        state = [self.arm_joint_values[i] for i in self.arm_joint_indices_to_use]
#        object_state = [0, 0, 0, 0]
#        object_state[self.object_type] = 1
        object_data = rospy.wait_for_message('/baris/features', PcFeatures)
        object_state = object_data.data
        state = (object_state, state)
        return state

    def transform(self, coordinates):
        transform = self.tf_buffer.lookup_transform('world', 'camera_depth_optical_frame', rospy.Time(0), rospy.Duration(1.0))

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
        return pos

    def step(self, action):
#        before_data = rospy.wait_for_message('/baris/features', PcFeatures)
#        pose_before = self.transform(before_data.bb_center)
        done = False
        info = {}
        arm_action = self.actions[action]
        # hand_action = action
        curr_distance = self.get_distance_between_gripper_and_object()

        arm_new_joint_values = np.clip(list(map(add, self.arm_joint_values, arm_action)), self.get_arm_joint_lower_limits(), self.get_arm_joint_upper_limits())
        hand_new_joint_values = self.hand_joint_values

        arm_check_plan, hand_check_plan = self.plan_and_execute(arm_new_joint_values, hand_new_joint_values)

#        self.hand_joint_values = np.clip(self.hand_joint_values + hand_action, self.get_hand_joint_lower_limits(), self.get_hand_joint_upper_limits())

        if (arm_check_plan is not None) and (hand_check_plan is not None):
            successful_planning = True
        else:
            successful_planning = False

        next_distance = self.get_distance_between_gripper_and_object()
        # Reward definition
        reward = 10*(curr_distance - next_distance) - 1

        successful_grasping = False

        if next_distance <= self.distance_threshold:
            self.grasp()
            reward += 200
            done = True
            if self.get_object_position()[2] > self.object_initial_position[2] + self.object_grasp_threshold:
                reward += 200
                successful_grasping = True
        elif not successful_planning:
            reward -= 5
        elif self.get_object_position()[2] < 0.5:
            reward -= 100
            done = True
        elif self.get_gripper_position()[2] < 0.7:
            reward -= 100
            done = True
        elif np.linalg.norm(self.get_object_position() - self.object_initial_position) >= self.object_move_threshold:
            reward -= 5
        next_state = [self.arm_joint_values[i] for i in self.arm_joint_indices_to_use]
#        object_state = [0, 0, 0, 0]
#        object_state[self.object_type] = 1
#        next_state = next_state + object_state
        object_data = rospy.wait_for_message('/baris/features', PcFeatures)
        object_state = object_data.data
        next_state = (object_state, next_state)
#        pose_after = self.transform(after_data.bb_center)
#        reward = pose_after.position.z - pose_before.position.z
        # pose_after.data yi 16 boyuta cevir
        return  next_state, reward, done, info, next_distance, successful_grasping

    def create_planning_scene(self):
        object_pose = PoseStamped()
        object_pose.header.frame_id = self.robot.get_planning_frame()
        object_position = self.get_object_position()
        if self.object_type == 0:
            object_pose.pose.position.x = self.get_object_position()[0]
            object_pose.pose.position.y = self.get_object_position()[1]
            object_pose.pose.position.z = self.get_object_position()[2] - 0.1
            object_size = (self.get_object_shape()[0], self.get_object_shape()[1], self.get_object_shape()[2])
            self.scene.add_box('object_id', object_pose, size = object_size)
        elif self.object_type == 1:
            object_pose.pose.position.x = self.get_object_position()[0]
            object_pose.pose.position.y = self.get_object_position()[1]
            object_pose.pose.position.z = self.get_object_position()[2] - 0.1
            object_size = (self.get_object_shape()[0], self.get_object_shape()[1], self.get_object_shape()[2])
            self.scene.add_box('object_id', object_pose, size = object_size)
        elif self.object_type == 2:
            object_pose.pose.position.x = self.get_object_position()[0]
            object_pose.pose.position.y = self.get_object_position()[1]
            object_pose.pose.position.z = self.get_object_position()[2] - 0.1
            radius = self.get_object_shape()[0]/2
            self.scene.add_sphere('object_id', object_pose, radius = radius)
        elif self.object_type == 3:
            object_pose.pose.position.x = self.get_object_position()[0]
            object_pose.pose.position.y = self.get_object_position()[1]
            object_pose.pose.position.z = self.get_object_position()[2] - 0.05
            object_size = (self.get_object_shape()[0], self.get_object_shape()[1], self.get_object_shape()[2])
            self.scene.add_box('object_id', object_pose, size = object_size)

    def open_gripper(self, pre_grasp_posture):
        pre_grasp_posture.header.frame_id = self.robot.get_planning_frame()
        pre_grasp_posture.joint_names = self.hand_joint_names
        pre_grasp_posture.points = [JointTrajectoryPoint()]
        pre_grasp_posture.points[0].positions = self.get_hand_joint_upper_limits()
        pre_grasp_posture.points[0].time_from_start = rospy.Duration(0.5)

    def close_gripper(self, grasp_posture, target1, target2):
        grasp_posture.header.frame_id = self.robot.get_planning_frame()
        grasp_posture.joint_names = self.hand_joint_names
        grasp_posture.points = [JointTrajectoryPoint()]
        grasp_posture.points[0].positions = [target1 - 0.0001, target2 - 0.0001]
        grasp_posture.points[0].time_from_start = rospy.Duration(0.5)

    def grasp(self):
        grasp_msg = Grasp()

        #the pose of panda_hand
        #grasp_pose
        object_position = self.get_object_position()
        grasp_msg.grasp_pose.header.frame_id = self.robot.get_planning_frame()
        if self.object_type == 0 or self.object_type == 2:
            grasp_msg.grasp_pose.pose.position.x = object_position[0]
            grasp_msg.grasp_pose.pose.position.y = object_position[1]
            grasp_msg.grasp_pose.pose.position.z = object_position[2] - 0.1 + 0.058 + (self.get_object_shape()[2]/2.0) + 0.02
            orientation = tf.transformations.quaternion_from_euler(0, -pi, 0)
            grasp_msg.grasp_pose.pose.orientation.x = orientation[0]
            grasp_msg.grasp_pose.pose.orientation.y = orientation[1]
            grasp_msg.grasp_pose.pose.orientation.z = orientation[2]
            grasp_msg.grasp_pose.pose.orientation.w = orientation[3]
            
            #the approach direction to take before picking an object
            #pre_grasp_approach
            grasp_msg.pre_grasp_approach.direction.header.frame_id = self.robot.get_planning_frame()
            grasp_msg.pre_grasp_approach.direction.vector.x = 0.0
            grasp_msg.pre_grasp_approach.direction.vector.y = 0.0
            grasp_msg.pre_grasp_approach.direction.vector.z = -0.4
            grasp_msg.pre_grasp_approach.min_distance = 0.095
            grasp_msg.pre_grasp_approach.desired_distance = 0.115
            
            #grasp_posture with close_gripper
            self.close_gripper(grasp_msg.grasp_posture, self.get_object_shape()[1]/2.0, self.get_object_shape()[1]/2.0)
        elif self.object_type == 1:
            grasp_msg.grasp_pose.pose.position.x = object_position[0] + 0.058 + (self.get_object_shape()[0]/2.0) + 0.02
            grasp_msg.grasp_pose.pose.position.y = object_position[1]
            grasp_msg.grasp_pose.pose.position.z = object_position[2] - 0.1
            orientation = tf.transformations.quaternion_from_euler(0, -pi/2, 0)
            grasp_msg.grasp_pose.pose.orientation.x = orientation[0]
            grasp_msg.grasp_pose.pose.orientation.y = orientation[1]
            grasp_msg.grasp_pose.pose.orientation.z = orientation[2]
            grasp_msg.grasp_pose.pose.orientation.w = orientation[3]
            
            #the approach direction to take before picking an object
            #pre_grasp_approach
            grasp_msg.pre_grasp_approach.direction.header.frame_id = self.robot.get_planning_frame()
            grasp_msg.pre_grasp_approach.direction.vector.x = -0.4
            grasp_msg.pre_grasp_approach.direction.vector.y = 0.0
            grasp_msg.pre_grasp_approach.direction.vector.z = 0.0
            grasp_msg.pre_grasp_approach.min_distance = 0.095
            grasp_msg.pre_grasp_approach.desired_distance = 0.115
            
            #grasp_posture with close_gripper
            self.close_gripper(grasp_msg.grasp_posture, self.get_object_shape()[1]/2.0 - 0.00008182548, self.get_object_shape()[1]/2.0 - 0.00008182548)
        elif self.object_type == 3:
            grasp_msg.grasp_pose.pose.position.x = object_position[0] - 0.008
            grasp_msg.grasp_pose.pose.position.y = object_position[1] + 0.058 + (self.get_object_shape()[1]/2.0) + 0.02
            grasp_msg.grasp_pose.pose.position.z = object_position[2] - 0.04
            orientation = tf.transformations.quaternion_from_euler(0, -pi/2, pi/2)
            grasp_msg.grasp_pose.pose.orientation.x = orientation[0]
            grasp_msg.grasp_pose.pose.orientation.y = orientation[1]
            grasp_msg.grasp_pose.pose.orientation.z = orientation[2]
            grasp_msg.grasp_pose.pose.orientation.w = orientation[3]
            
            #the approach direction to take before picking an object
            #pre_grasp_approach
            grasp_msg.pre_grasp_approach.direction.header.frame_id = self.robot.get_planning_frame()
            grasp_msg.pre_grasp_approach.direction.vector.x = 0.0
            grasp_msg.pre_grasp_approach.direction.vector.y = -0.4
            grasp_msg.pre_grasp_approach.direction.vector.z = 0.0
            grasp_msg.pre_grasp_approach.min_distance = 0.095
            grasp_msg.pre_grasp_approach.desired_distance = 0.115
            
            #grasp_posture with close_gripper
            self.close_gripper(grasp_msg.grasp_posture, self.get_object_shape()[0]/2.0, self.get_object_shape()[0]/2.0)

        #the retreat direction to take after a grasp has been completed (object is attached)
        #post_grasp_retreat
        grasp_msg.post_grasp_retreat.direction.header.frame_id = self.robot.get_planning_frame()
        grasp_msg.post_grasp_retreat.direction.vector.x = 0.0
        grasp_msg.post_grasp_retreat.direction.vector.y = 0.0
        grasp_msg.post_grasp_retreat.direction.vector.z = 1.0
        grasp_msg.post_grasp_retreat.min_distance = 0.1
        grasp_msg.post_grasp_retreat.desired_distance = 0.25

        #pre_grasp_posture with open_gripper
        self.open_gripper(grasp_msg.pre_grasp_posture)

        self.arm.set_support_surface_name('table_top')

        self.arm.pick('object_id', [grasp_msg])
        
        if self.object_type == 1:
            rospy.sleep(20)
        else:
            rospy.sleep(10)
        # self = RobotEnv()
        # self.reset()
