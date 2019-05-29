#!/usr/bin/env python
import roslib
import rospy
from arm import Arm
from hand import Hand
from manipulator import Manipulator
from MyAction import Push
from geometry_msgs.msg import PoseStamped, PointStamped, Pose
import moveit_commander
import sys

def main():
    sys.argv.append("robot_description:=panda/robot_description")
    sys.argv.append("joint_states:=panda/joint_states")
    sys.argv.append("move_group:=panda/move_group")
    sys.argv.append("pickup:=panda/pickup")
    sys.argv.append("place:=panda/place")
    sys.argv.append("execute_trajectory:=panda/execute_trajectory")
    #sys.argv.append("trajectory_execution_event:=panda/trajectory_execution_event")
    #sys.argv.append("collision_object:=panda/collision_object")
    #sys.argv.append("attached_collision_object:=panda/attached_collision_object")
    #sys.argv.append("error_recovery:=panda/error_recovery")
    #sys.argv.append("franka_gripper_node:=panda/franka_gripper_node")
    #sys.argv.append("joint_state_controller:=panda/franka_state_controller")
    #sys.argv.append("planning_scene:=panda/planning_scene")
    #sys.argv.append("planning_scene_world:=panda/planning_scene_world")
    #sys.argv.append("joint_trajectory_controller:=panda/position_joint_trajectory_controller")
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_shell', anonymous=True)
    manipulator = Manipulator()
    push_action = Push(manipulator)

    while True:
        raw = raw_input('panda@shell:$ ')
        if len(raw) == 0:
            continue
        cmd_input = raw.split(' ')
        cmd = cmd_input[0]
        if cmd == 'exit':
            break
        elif cmd == 'get_pose':
            print manipulator.arm.get_pose()
        elif cmd == 'get_rotation':
            print manipulator.arm.get_rotation()
        elif cmd == 'change_arm_joint_angles':
            args = cmd_input[1]
            angles = [float(num) for num in args.split(',')]
            manipulator.arm.change_joint_angles(angles)
            manipulator.arm.ang_cmd(angles, 5.0)
        elif cmd == 'test':
            manipulator.arm.test_joint_plan()
        elif cmd == 'get_joints':
            print manipulator.arm.get_joints(ik=False)
        elif cmd == 'go_home':
            manipulator.arm.go_home()
        elif cmd == 'execute_trajectory':
            args = cmd_input[1]
            angles = [float(num) for num in args.split(',')]
            manipulator.arm.execute_trajectory(angles, [1.0])
        elif cmd == 'close_gripper':
            manipulator.hand.close_gripper()
        elif cmd == 'open_gripper':
            manipulator.hand.open_gripper()
        elif cmd == 'get_state':
            print manipulator.arm.get_state()
        elif cmd == 'test_poses_plan':
            p1 = [0.109784330654, -0.201825250008, 0.927634606574,
                  0.953892131709, -0.259908849954, 0.112249454584, 0.0996857598901]
            manipulator.arm.plan_poses([p1])
        elif cmd == 'test_jt':
            # EXERCISE WITH CAUTION
            manipulator.arm.execute_trajectory(
            [[-0.0029068310484290124, -0.5389024951798576, -0.32719744423457553, -1.0994862112317767, -0.46650070729425974, 0.8834752980981554, 0.6584624329124178],
            [0.6360157456696034, -0.1287396351950509, -0.854471571615764, -1.1812223290034702, -0.2326772375021662, 1.8844526242188044, 0.6579868964978627],
            [0.6102343267330101, 0.15414944928033011, -0.8639917395796095, -0.8486092205047607, -0.23309919845206398, 1.9201719971043723, 0.6168578206987253],
            [0, 0, 0, 0, 0, 0, 0.75]],
            [2, 4, 6, 9])
        elif cmd == 'get_fk':
            print manipulator.arm.get_FK()
        elif cmd == 'get_ik':
            print manipulator.arm.get_joints(ik=True)
        elif cmd == 'home':
            manipulator.arm.execute_trajectory([[0,0,0,0,0,0,0.75]], [5])
        elif cmd == 'gc':
            manipulator.arm.gravity_compensation()
        elif cmd == 'exit_gc':
            manipulator.arm.exit_gravity_compensation()
        elif cmd == 'close_gripper':
            manipulator.hand.close_gripper()
        elif cmd == 'open_gripper':
            manipulator.hand.open_gripper()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
