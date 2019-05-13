#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import tf
from math import pi
from robot_env import RobotEnv
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import Grasp
from geometry_msgs.msg import PoseStamped
from moveit_commander import PlanningSceneInterface

def open_gripper(pre_grasp_posture):
    pre_grasp_posture.joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
    pre_grasp_posture.points = [JointTrajectoryPoint()]
    pre_grasp_posture.points[0].positions = [0.04, 0.04]
    pre_grasp_posture.points[0].time_from_start = rospy.Duration(0.5)

def close_gripper(grasp_posture, target1, target2):
    grasp_posture.joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
    grasp_posture.points = [JointTrajectoryPoint()]
    grasp_posture.points[0].positions = [target1, target2]
    grasp_posture.points[0].time_from_start = rospy.Duration(0.5)
    
def pick(env):
    grasp_msg = Grasp()
    
    #the pose of panda_link8
    #grasp_pose
    grasp_msg.grasp_pose.header.frame_id = env.robot.get_planning_frame()
    grasp_msg.grasp_pose.pose.position.x = env.get_object_position()[0]
    grasp_msg.grasp_pose.pose.position.y = env.get_object_position()[1]
    grasp_msg.grasp_pose.pose.position.z = env.get_object_position()[2] + 0.02 + 0.058 + 0.1
    orientation = tf.transformations.quaternion_from_euler(-pi, 0, 0)
    grasp_msg.grasp_pose.pose.orientation.x = orientation[0]
    grasp_msg.grasp_pose.pose.orientation.y = orientation[1]
    grasp_msg.grasp_pose.pose.orientation.z = orientation[2]
    grasp_msg.grasp_pose.pose.orientation.w = orientation[3]
    
    #the approach direction to take before picking an object
    #pre_grasp_approach
    grasp_msg.pre_grasp_approach.direction.header.frame_id = env.robot.get_planning_frame()
    grasp_msg.pre_grasp_approach.direction.vector.x = 0.0
    grasp_msg.pre_grasp_approach.direction.vector.y = 0.0
    grasp_msg.pre_grasp_approach.direction.vector.z = -1.0
    grasp_msg.pre_grasp_approach.min_distance = 0.095
    grasp_msg.pre_grasp_approach.desired_distance = 0.115
    
    #the retreat direction to take after a grasp has been completed (object is attached)
    #post_grasp_retreat
    grasp_msg.post_grasp_retreat.direction.header.frame_id = env.robot.get_planning_frame()
    grasp_msg.post_grasp_retreat.direction.vector.x = 0.0
    grasp_msg.post_grasp_retreat.direction.vector.y = 0.0
    grasp_msg.post_grasp_retreat.direction.vector.z = 1.0
    grasp_msg.post_grasp_retreat.min_distance = 0.1
    grasp_msg.post_grasp_retreat.desired_distance = 0.25
    
    #pre_grasp_posture with open_gripper
    open_gripper(grasp_msg.pre_grasp_posture)
    
    #grasp_posture with close_gripper
    close_gripper(grasp_msg.grasp_posture, 0.02, 0.02)
    
    env.arm.set_support_surface_name('table_top')
    
    env.arm.pick('box_id', [grasp_msg])

def create_planning_scene(env):
    scene = PlanningSceneInterface()
    
    rospy.sleep(2)
    
    box_pose = PoseStamped()
    box_pose.header.frame_id = env.robot.get_planning_frame()
    box_pose.pose.position.x = env.get_object_position()[0]
    box_pose.pose.position.y = env.get_object_position()[1]
    box_pose.pose.position.z = env.get_object_position()[2]
    scene.add_box('box_id', box_pose, size = (0.2, 0.04, 0.04))

if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('pick_py', anonymous = True)
    rospy.sleep(1)
    
    env = RobotEnv()
    env.reset()
    
    create_planning_scene(env)
    
    pick(env)





