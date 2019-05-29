#!/usr/bin/env python
import rospy
from hand import Hand
from panda_manipulation.msg import GripperExecuteTrajectory

class GripperTrajectoryNode:
	def __init__(self):
		self.gripper = Hand()

	def run(self):
		rospy.Subscriber('gripper_trajectory_node/execute', GripperExecuteTrajectory, self.execute)

	def execute(self, req):
		waypoints = req.waypoints
		durations = req.durations
		success = self.gripper.execute_trajectory(waypoints, durations)

def main():
    rospy.init_node('gripper_trajectory_node', anonymous=True) #arm_trajectory_node  ??
    atn = GripperTrajectoryNode()
    atn.run()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
		pass