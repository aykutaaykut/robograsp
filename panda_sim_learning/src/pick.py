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
    while True:
        env.reset()
        env.step(2)
        env.step(2)
        env.step(2)
        env.step(2)
        env.step(2)
        env.grasp()
