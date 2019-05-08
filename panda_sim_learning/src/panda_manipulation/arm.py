#!/usr/bin/env python
import roslib
import utils
import rospy
import moveit_msgs.msg
import moveit_msgs.srv
import moveit_commander
import std_msgs.msg
import geometry_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import actionlib
from controller_manager_msgs.srv import SwitchController, LoadController, UnloadController
import numpy as np

class Arm:
    def __init__(self):
        self.root = 'panda_link0'
        self.ee_link = 'panda_hand'

        self.joint_trajectory_controller = 'position_joint_trajectory_controller'
        self.gc_controller = 'gravity_compensation_controller'
        self.joint_trj_service = 'panda/{}/follow_joint_trajectory'.format(self.joint_trajectory_controller)

        self.client = actionlib.SimpleActionClient(self.joint_trj_service, FollowJointTrajectoryAction)
        print "Waiting for arm server..."
        self.client.wait_for_server()
        print "Connected to arm server"

        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3',
        'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander("panda_arm_hand")
        self.group.set_end_effector_link("panda_hand")
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_pose_reference_frame(self.root)

    @property
    def num_joints(self):
        return len(self.joint_names)

    def get_state(self):
        return self.robot.get_current_state()

    def get_joints(self, ik=False, group=False):
        if not ik:
            return self.robot.get_current_state().joint_state.position[:self.num_joints]
        else:
            return self.get_IK(self.get_pose(group=group))

    def get_pose(self, group=False):
        if group:
            return self.group.get_current_pose()
        else:
            return self.get_FK()

    def get_IK(self, pose):
        rospy.wait_for_service('panda/compute_ik')
        compute_ik = rospy.ServiceProxy('panda/compute_ik', moveit_msgs.srv.GetPositionIK)
        robot_state = self.get_state()

        wkPose = pose

        msgs_request = moveit_msgs.msg.PositionIKRequest()
        msgs_request.group_name = self.group.get_name()
        msgs_request.robot_state = robot_state
        msgs_request.pose_stamped = wkPose
        msgs_request.timeout.secs = 2
        msgs_request.avoid_collisions = False

        try:
            reply = compute_ik(msgs_request)
            return reply.solution.joint_state.position[:self.num_joints]
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def get_FK(self):
        rospy.wait_for_service('panda/compute_fk')
        compute_fk = rospy.ServiceProxy('panda/compute_fk', moveit_msgs.srv.GetPositionFK)

        header = std_msgs.msg.Header()
        header.frame_id = self.root
        fk_link_names = [self.ee_link]
        robot_state = self.get_state()

        try:
            reply = compute_fk(header, fk_link_names, robot_state)
            return reply.pose_stamped[0]
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def get_rotation(self):
        return self.group.get_current_rpy()

    def test_joint_plan(self):
        group_variable_values = self.group.get_current_joint_values()
        print "Joint values", group_variable_values
        group_variable_values[0] = 1.0
        self.group.set_joint_value_target(group_variable_values)
        plan = self.group.plan()
        self.group.execute(plan)

    def plan_joint(self, angles, execute=True):
        self.group.set_joint_value_target(angles)
        plan = self.group.plan()
        if execute:
            self.group.execute(plan)

    def plan_pose(self, pose, execute=True):
        self.group.set_pose_target(pose)
        plan = self.group.plan()
        if execute:
            self.group.execute(plan)

    def plan_poses(self, pose, execute=True):
        self.group.set_pose_targets(pose)
        plan = self.group.plan()
        if execute:
            self.group.execute(plan)

    def go_to_pose_cartesian(self, waypoints):
        trial = 0
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0.0, False)
        while fraction != 1.0:
            if trial > 5:
                return -1
            rospy.loginfo("Path computed with %f fraction. Retrying..." % fraction)
            (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0.0, False)
            trial += 1
        rospy.loginfo("Path computed successfully with %f fraction. Moving the arm." % fraction)
        self.group.execute(plan)
        return 1

    def execute_trajectory(self, waypoints, durations, velocities=None):
        '''
        Waypoints and velocities are 2D arrays
        If velocities is None velocities are calculated by (waypoints[i+1] - waypoints[i] / (durations[i+1]-durations[i]))
        durations are time from start
        '''
        g = FollowJointTrajectoryGoal()
        g.trajectory = JointTrajectory()
        g.trajectory.joint_names = self.joint_names
        g.trajectory.points = []
        calc_vel = True if velocities is None else False
        # Convert to np for broadcating
        waypoints = np.array(waypoints)

        for i, joints in enumerate(waypoints):
            if calc_vel:
                if i == len(waypoints) - 1:
                    velocity = [0] * self.num_joints
                else:
                    velocity = (waypoints[i+1] - waypoints[i]) / (durations[i+1]-durations[i])
            else:
                velocity = velocities[i]
            g.trajectory.points.append(JointTrajectoryPoint(positions=joints, velocities=velocity,
                                        time_from_start=rospy.Duration(durations[i])))

        # g.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)
        g.trajectory.header.frame_id = format(self.root)
        self.client.send_goal(g)

    def switch_controller(self, c1=None, c2=None): # c1 is stoping c2 is stating
        print "Waiting for switch service"
        rospy.wait_for_service('/panda/controller_manager/switch_controller')
        try:
            switch_controller_srv = rospy.ServiceProxy(
                       '/panda/controller_manager/switch_controller', SwitchController)
            if c2 is None:
                ret = switch_controller_srv([], [c1], 2) #Stops
            elif c1 is None:
                ret = switch_controller_srv([c2], [], 2) #Starts
            else:
                ret = switch_controller_srv([c2], [c1], 2)#Switch
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def unload_controller(self, c):
        print "Waiting for unload service"
        rospy.wait_for_service('/panda/controller_manager/unload_controller')
        try:
            uc_src = rospy.ServiceProxy(
                       '/panda/controller_manager/unload_controller', UnloadController)
            ret = uc_src(c)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def load_controller(self, c):
        print "Waiting for load service"
        rospy.wait_for_service('/panda/controller_manager/load_controller')
        try:
            lc_srv = rospy.ServiceProxy(
                       '/panda/controller_manager/load_controller', LoadController)
            ret = lc_srv(c)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def gravity_compensation(self):
        self.switch_controller(c1=self.joint_trajectory_controller) # Stop joint
        self.unload_controller(self.joint_trajectory_controller) # Unload joint
        self.load_controller(self.gc_controller) # Load gc
        self.switch_controller(c2=self.gc_controller) # Start gc

    def exit_gravity_compensation(self):
        self.switch_controller(c1=self.gc_controller) # Stop gc
        self.unload_controller(self.gc_controller) # Unload gc
        self.load_controller(self.joint_trajectory_controller) # Load joint
        self.switch_controller(c2=self.joint_trajectory_controller) # Start joint
