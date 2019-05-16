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

    env = RobotEnv(object_type = 1)
    env.reset()
    env.grasp()
#    while True:
#        next_distance = env.get_distance_between_gripper_and_object()
#        if next_distance <= 0.03:
#            env.grasp()
#            rospy.sleep(10)
#            env.reset()
#        env.step(2)
#        env.step(2)
#        env.step(2)
#        env.step(2)
#        env.step(2)
#    env.grasp()
