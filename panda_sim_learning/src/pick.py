import rospy
from moveit_python import *
from moveit_msgs.msg import Grasp, PlaceLocation

rospy.init_node("moveit_py")
# provide arm group and gripper group names
# also takes a third parameter "plan_only" which defaults to False
p = PickPlaceInterface("panda_arm", "panda_hand")

g = Grasp()
# fill in g
# setup object named object_name using PlanningSceneInterface
p.pickup("box_link", [g, ], support_name = "table_top")
