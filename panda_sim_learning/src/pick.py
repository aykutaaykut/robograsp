#!/usr/bin/env python

import sys
import rospy
import moveit_commander
from robot_env import RobotEnv

if __name__ == '__main__':
    #simple test for pick
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('pick_py', anonymous = True)
    rospy.sleep(1)

    env = RobotEnv()
    env.reset()
    env.step(2)
#    env.grasp()
#    while True:
#        env.grasp()
#        env.reset()
#        next_distance = env.get_distance_between_gripper_and_object()
#        if next_distance <= 0.07:
#            env.grasp()
#            env.reset()
#        env.step(2)
