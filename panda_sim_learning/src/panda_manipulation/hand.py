import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from math import pi
import numpy as np

class Hand:

	def __init__(self):
		self.client = actionlib.SimpleActionClient('panda/franka_gripper_node/gripper_action', GripperCommandAction)
		print "Waiting for gripper server..."
		self.client.wait_for_server()
		print "Connected to gripper server"
		self.goal = GripperCommandGoal()

		self.joint_names = ['panda_finger_joint1', 'panda_finger_joint2']

	def open_gripper(self):
		return self.command(0.075, 1.0)

	def close_gripper(self):
		return  self.command(0.0, 1.0)

	def command(self, position, max_effort):
		self.goal.command.position = position
		self.goal.command.max_effort = max_effort
		self.client.send_goal(self.goal)

	def stop(self):
		self.client.cancel_goal()
