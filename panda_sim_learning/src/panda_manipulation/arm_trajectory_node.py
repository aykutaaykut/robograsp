#!/usr/bin/env python
import rospy
from arm import Arm
from panda_manipulation.msg import ArmExecuteTrajectory

class ArmTrajectoryNode:
	def __init__(self):
		self.arm = Arm()

	def run(self):
		rospy.Subscriber('arm_trajectory_node/execute', ArmExecuteTrajectory, self.execute)

	def execute(self, req):
		waypoints = req.waypoints
		durations = req.durations
		success = self.arm.execute_trajectory(waypoints, durations)

def main():
    rospy.init_node('arm_trajectory_node', anonymous=True)
    atn = ArmTrajectoryNode()
    atn.run()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
		pass