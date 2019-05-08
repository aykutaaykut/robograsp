import rospy
import numpy as np
import tf2_ros
from solver import *
from mid_frame_broadcaster import orient_towards_object, orient_towards_object_double
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from geometry_msgs.msg import Pose, PoseStamped
from panda_manipulation.msg import ArmExecuteTrajectory, GripperExecuteTrajectory, FloatList

def pose_dist(pose1, pose2):
    return np.sqrt((pose1.position.x-pose2.position.x)**2+(pose1.position.y-pose2.position.y)**2+(pose1.position.z-pose2.position.z)**2)

class Trajectory_Generator:

    def __init__(self):
        self.arm_traj_pub = rospy.Publisher('arm_trajectory_node/execute', ArmExecuteTrajectory, queue_size=28)
        self.hand_traj_pub = rospy.Publisher('gripper_trajectory_node/execute', GripperExecuteTrajectory, queue_size=28)
        #self.solver = kdlSolver("/home/dkebude/catkin_ws/src/panda_sim/panda_sim_description/urdf/panda_arm_hand.urdf","panda_link0","panda_hand", 10000, 125, 1e-4)
        #self.num_jnts = self.solver.getNumJoints()
        #self.verbose_mode = True
        #self.jnt_dist_thr = 0.5

    def gen_arm_traj(self, waypoints, durations):
        arm_traj_rq = ArmExecuteTrajectory()
        wp = []
        for i in range(len(waypoints)):
            float_list = FloatList()
            float_list.elements = [float(x) for x in waypoints[i]]
            wp.append(float_list)
        arm_traj_rq.waypoints = wp
        arm_traj_rq.durations = durations
        self.arm_traj_pub.publish(arm_traj_rq)

    def gen_hand_traj(self, waypoints, durations):
        hand_traj_rq = GripperExecuteTrajectory()
        wp = []
        for i in range(len(waypoints)):
            float_list = FloatList()
            float_list.elements = [float(x) for x in waypoints[i]]
            wp.append(float_list)
        hand_traj_rq.waypoints = wp
        hand_traj_rq.durations = durations
        self.hand_traj_pub.publish(hand_traj_rq)

    def gen_traj(self, arm_wp, arm_dur, hand_wp, hand_dur):
        arm_sc = self.gen_arm_traj(arm_wp, arm_dur)
        hand_sc = self.gen_hand_traj(hand_wp, hand_dur)

    def cart_trj_to_jnt_trj_kdl_solver(self, arm, trj, targ_obj):
        angle_set = []
        pose_set = []

        if(targ_obj == 1):
            m2b_frame_tf = arm.tfBuffer.lookup_transform('panda_link0', 'blue_frame', rospy.Time(0))
        elif(targ_obj == 2):
            m2b_frame_tf = arm.tfBuffer.lookup_transform('panda_link0', 'red_frame', rospy.Time(0))

        b_pose = Pose()
        b_pose.position = m2b_frame_tf.transform.translation
        b_pose.orientation = m2b_frame_tf.transform.rotation
        
        q_targ = DoubleVector(self.num_jnts)
        q_init = DoubleVector(self.num_jnts)
        p_targ = DoubleVector(7)                # 7; because x,y,z + quaternion angles
        p_init = DoubleVector(7)

        robot_state = arm.robot.get_current_state()
        q_init[0] = robot_state.joint_state.position[0]
        q_init[1] = robot_state.joint_state.position[1]
        q_init[2] = robot_state.joint_state.position[2]
        q_init[3] = robot_state.joint_state.position[3]
        q_init[4] = robot_state.joint_state.position[4]
        q_init[5] = robot_state.joint_state.position[5]
        q_init[6] = robot_state.joint_state.position[6]

        ee_pose = arm.get_current_pose()
        p_init[0] = ee_pose.position.x
        p_init[1] = ee_pose.position.y
        p_init[2] = ee_pose.position.z
        p_init[3] = ee_pose.orientation.x
        p_init[4] = ee_pose.orientation.y
        p_init[5] = ee_pose.orientation.z
        p_init[6] = ee_pose.orientation.w
        normalizeQuaternion(p_init)

        angle_set.append(q_init)
        pose_set.append(p_init)

        p_targ_orientation = orient_towards_object_double(arm.tfBuffer, p_init, b_pose)

        p_targ[0] = trj[0,0]
        p_targ[1] = trj[0,1]
        p_targ[2] = trj[0,2]
        p_targ[3] = p_targ_orientation[0]
        p_targ[4] = p_targ_orientation[1]
        p_targ[5] = p_targ_orientation[2]
        p_targ[6] = p_targ_orientation[3]
        normalizeQuaternion(p_targ)
    
        self.solver.solvePoseIk(q_init, q_targ, p_targ, self.verbose_mode)

        angle_set.append(q_targ)
        pose_set.append(p_targ)

        q_init = q_targ
        p_init = p_targ

        for i in range(1,len(trj)):
            p_targ_orientation = orient_towards_object_double(arm.tfBuffer, p_init, b_pose)

            p_targ[0] = trj[i,0]
            p_targ[1] = trj[i,1]
            p_targ[2] = trj[i,2]
            p_targ[3] = p_targ_orientation[0]
            p_targ[4] = p_targ_orientation[1]
            p_targ[5] = p_targ_orientation[2]
            p_targ[6] = p_targ_orientation[3]
            normalizeQuaternion(p_targ)
            
            self.solver.solveHybridIk(q_init, q_targ, p_targ, self.jnt_dist_thr, self.verbose_mode)

            angle_set.append(q_targ)
            pose_set.append(p_targ)
            
            q_init = q_targ
            p_init = p_targ

        return angle_set, pose_set

    def cart_trj_to_jnt_trj_tracik_solver(self, arm, trj, targ_obj):
        rospy.wait_for_service('panda/compute_ik')
        compute_ik = rospy.ServiceProxy('panda/compute_ik', GetPositionIK)
        angle_set = []
        pose_set = []

        robot_state = arm.robot.get_current_state()

        if(targ_obj == 1):
            m2b_frame_tf = arm.tfBuffer.lookup_transform('panda_link0', 'blue_frame', rospy.Time(0))
        elif(targ_obj == 2):
            m2b_frame_tf = arm.tfBuffer.lookup_transform('panda_link0', 'red_frame', rospy.Time(0))

        b_pose = Pose()
        b_pose.position = m2b_frame_tf.transform.translation
        b_pose.orientation = m2b_frame_tf.transform.rotation

        p_init = arm.get_pose().pose
        q_init = robot_state.joint_state.position[0:len(arm.joint_names)]
        p_targ = p_init

        p_targ_stamped = PoseStamped()
        p_targ_stamped.header.frame_id = arm.root
        p_targ_stamped.header.stamp = rospy.Time.now()
        p_targ_stamped.pose = p_targ

        msg_req = PositionIKRequest()
        msg_req.group_name = arm.group.get_name()
        msg_req.robot_state = robot_state
        msg_req.pose_stamped = p_targ_stamped
        msg_req.timeout.secs = 2
        msg_req.avoid_collisions = False

        jointAngle=compute_ik(msg_req)
        q_targ=list(jointAngle.solution.joint_state.position[0:len(arm.joint_names)])
        if jointAngle.error_code.val == -31:
            arm.write_to_csv(p_init, q_init, p_targ)
            print 'No IK solution found at initial pose'
            return None

        angle_set.append(q_targ)
        pose_set.append(p_targ)

        q_init = list(q_targ)
        p_init = p_targ

        for i in range(0, len(trj)):
            p_targ.position.x = trj[i,0]
            p_targ.position.y = trj[i,1]
            p_targ.position.z = trj[i,2]
            p_targ.orientation = orient_towards_object(arm.tfBuffer, p_init, b_pose)

            if(pose_dist(b_pose, p_targ) <= 0.18):
                break

            q_init_tmp = list(q_init)
            q_init_tmp.append(0.0)
            q_init_tmp.append(0.0)
            
            ik_joint_state = JointState()
            ik_joint_state.header.frame_id = robot_state.joint_state.header.frame_id
            ik_joint_state.header.stamp = rospy.Time.now()
            ik_joint_state.name = robot_state.joint_state.name
            ik_joint_state.position = q_init_tmp

            ik_robot_state = RobotState()
            ik_robot_state.joint_state = ik_joint_state

            p_targ_stamped = PoseStamped()
            p_targ_stamped.header.frame_id = arm.root
            p_targ_stamped.header.stamp = rospy.Time.now()
            p_targ_stamped.pose = p_targ
            
            msg_req = PositionIKRequest()
            msg_req.group_name = arm.group.get_name()
            msg_req.robot_state = ik_robot_state
            msg_req.pose_stamped = p_targ_stamped
            msg_req.timeout.secs = 2
            msg_req.avoid_collisions = False
            
            jointAngle=compute_ik(msg_req)
            q_targ=list(jointAngle.solution.joint_state.position[0:len(arm.joint_names)])
            if jointAngle.error_code.val == -31:
                arm.write_to_csv(p_init, q_init, p_targ)
                print 'No IK solution found at trajectory point %d'%i
                return None
            
            angle_set.append(q_targ)
            pose_set.append(p_targ)
            
            q_init = list(q_targ)
            p_init = p_targ

        return angle_set, pose_set