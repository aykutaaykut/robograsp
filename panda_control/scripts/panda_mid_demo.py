import sys
import rospy
from moveit_commander import RobotCommander
from moveit_commander import roscpp_initialize
from moveit_commander import roscpp_shutdown
from moveit_msgs.msg import RobotState

def pregrasp_top():
	return [0.0, 0.5424, 0.0, -2.3625, 0.0, 2.9446, 0.6219]

def pregrasp_behind():
	return [0.0, 1.4722, 0.0, -0.6966, 0.0, 0.8111, 0.6219]

def pregrasp_left():
	return [0.4457, 1.0848, -0.0318, -1.7028, -1.1462, 1.3704, 0.4457]

def pregrasp_right():
	return [-0.1592, 1.1042, -0.2865, -1.7522, 1.4009, 1.2046, 1.0507]

def default():
	return [0.0, 0.0, 0.0, -0.0698, 0.0, 3.2346, 0.0]

def run(mode):
	roscpp_initialize(sys.argv)
	rospy.init_node('panda_mid_demo', anonymous=True)

	panda = RobotCommander()
	rospy.sleep(1)

	panda_arm = panda.panda_arm
	print 'Start state:'
	print panda_arm.get_current_joint_values()
	
	if mode == 'top':
		goal = pregrasp_top()
	elif mode == 'behind':
		goal = pregrasp_behind()
	elif mode == 'left':
		goal = pregrasp_left()
	elif mode == 'right':
		goal = pregrasp_right()
	elif mode == 'default':
		goal = default()

	print 'Goal State:'
	print goal
	plan = panda_arm.plan(goal)
	print 'Plan:'
	print plan
	panda_arm.execute(plan)

	roscpp_shutdown()

if __name__ == '__main__':
	run(sys.argv[1])
