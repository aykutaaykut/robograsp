#!/usr/bin/env python

import sys
import random
import rospy
import moveit_commander
from robot_env import RobotEnv

if __name__ == '__main__':
    rospy.init_node('robot_env')
#    rospy.sleep(1)

    env = RobotEnv()
    env.reset()


#    panda = moveit_commander.RobotCommander()
#    panda_arm = moveit_commander.move_group.MoveGroupCommander('panda_arm')
#    panda_hand = moveit_commander.move_group.MoveGroupCommander('hand')

#    joint_upper_limits = [2.8973, 1.7628, 2.8973, 0.0175, 2.8973, 3.7525, 2.8973, 0.04, 0.04]
#    joint_lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.001, -0.001]

#    for e in range(1000):
#        goal = []
#        for i in range(9):
#            r = random.uniform(joint_lower_limits[i], joint_upper_limits[i])
#            goal.append(r)

#        panda_arm_plan = panda_arm.plan(goal[:7])
#        panda_hand_plan = panda_hand.plan(goal[7:])

#        panda_arm.execute(panda_arm_plan)
#        panda_hand.execute(panda_hand_plan)

    while True:
        s, r, d, i = env.step(env.action_space.sample())
