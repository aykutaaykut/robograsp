#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import tf
from math import pi
from robot_env import RobotEnv
import moveit_commander
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import Grasp
from geometry_msgs.msg import PoseStamped

if __name__ == '__main__':
    #simple test for pick
    
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('pick_py', anonymous = True)
    rospy.sleep(1)
    
    env = RobotEnv()
    env.reset()
    env.grasp()





