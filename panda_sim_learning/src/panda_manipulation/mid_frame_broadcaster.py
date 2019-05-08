#!/usr/bin/python
import rospy
import math
import tf2_ros
import tf2_msgs.msg
import numpy as np
import tf_conversions
import PyKDL
from geometry_msgs.msg import TransformStamped, PointStamped, Pose, Quaternion

base_frame = 'panda_link0'

def orient_towards_object(tfBuffer, ee_pose, object_pose):
	w2w = tfBuffer.lookup_transform(base_frame, base_frame, rospy.Time(0))

	w2wPose = Pose()
	w2wPose.position.x = w2w.transform.translation.x
	w2wPose.position.y = w2w.transform.translation.y
	w2wPose.position.z = w2w.transform.translation.z
	w2wPose.orientation = w2w.transform.rotation

	w2i_kdl = tf_conversions.fromMsg(ee_pose)
	w2i_x = w2i_kdl.M.UnitX()
	w2i_pos = w2i_kdl.p

	w2e_kdl = tf_conversions.fromMsg(object_pose)
	w2e_x = w2e_kdl.M.UnitX()
	w2e_pos = w2e_kdl.p

	w2w_kdl = tf_conversions.fromMsg(w2wPose)
	w2w_z = w2w_kdl.M.UnitZ()
	w2w_pos = w2w_kdl.p

	z_rot_wp = (w2e_pos-w2i_pos)/((w2e_pos-w2i_pos).Norm())
	y_rot_wp = -w2w_z*z_rot_wp
	x_rot_wp = -z_rot_wp*y_rot_wp
	wp_pos = w2i_pos

	wp_M = PyKDL.Rotation(x_rot_wp, y_rot_wp, z_rot_wp)

	wp_kdl = PyKDL.Frame(wp_M, wp_pos)

	wp_pose = tf_conversions.toMsg(wp_kdl)

	recipNorm = 1/math.sqrt(wp_pose.orientation.x**2+
					   		wp_pose.orientation.y**2+
					   		wp_pose.orientation.z**2+
					   		wp_pose.orientation.w**2)
	wp_pose.orientation.x = wp_pose.orientation.x*recipNorm
	wp_pose.orientation.y = wp_pose.orientation.y*recipNorm 
	wp_pose.orientation.z = wp_pose.orientation.z*recipNorm 
	wp_pose.orientation.w = wp_pose.orientation.w*recipNorm 

	return wp_pose.orientation

def orient_towards_object_double(tfBuffer, ee_vec, object_pose):
	w2w = tfBuffer.lookup_transform(base_frame, base_frame, rospy.Time(0))

	w2wPose = Pose()
	w2wPose.position.x = w2w.transform.translation.x
	w2wPose.position.y = w2w.transform.translation.y
	w2wPose.position.z = w2w.transform.translation.z
	w2wPose.orientation = w2w.transform.rotation

	ee_pose = Pose()
	ee_pose.position.x = ee_vec[0]
	ee_pose.position.y = ee_vec[1]
	ee_pose.position.z = ee_vec[2]
	ee_pose.orientation.x = ee_vec[3]
	ee_pose.orientation.y = ee_vec[4]
	ee_pose.orientation.z = ee_vec[5]
	ee_pose.orientation.w = ee_vec[6]

	w2i_kdl = tf_conversions.fromMsg(ee_pose)
	w2i_x = w2i_kdl.M.UnitX()
	w2i_pos = w2i_kdl.p

	w2e_kdl = tf_conversions.fromMsg(object_pose)
	w2e_x = w2e_kdl.M.UnitX()
	w2e_pos = w2e_kdl.p

	w2w_kdl = tf_conversions.fromMsg(w2wPose)
	w2w_z = w2w_kdl.M.UnitZ()
	w2w_pos = w2w_kdl.p

	z_rot_wp = (w2e_pos-w2i_pos)/((w2e_pos-w2i_pos).Norm())
	y_rot_wp = -w2w_z*z_rot_wp
	x_rot_wp = -z_rot_wp*y_rot_wp
	wp_pos = w2i_pos

	wp_M = PyKDL.Rotation(x_rot_wp, y_rot_wp, z_rot_wp)

	wp_kdl = PyKDL.Frame(wp_M, wp_pos)

	wp_pose = tf_conversions.toMsg(wp_kdl)

	recipNorm = 1/math.sqrt(wp_pose.orientation.x**2+
					   		wp_pose.orientation.y**2+
					   		wp_pose.orientation.z**2+
					   		wp_pose.orientation.w**2)
	wp_pose.orientation.x = wp_pose.orientation.x*recipNorm
	wp_pose.orientation.y = wp_pose.orientation.y*recipNorm 
	wp_pose.orientation.z = wp_pose.orientation.z*recipNorm 
	wp_pose.orientation.w = wp_pose.orientation.w*recipNorm 

	wp_orientation = []
	wp_orientation.append(wp_pose.orientation.x)
	wp_orientation.append(wp_pose.orientation.y)
	wp_orientation.append(wp_pose.orientation.z)
	wp_orientation.append(wp_pose.orientation.w)

	return wp_orientation

if __name__ == '__main__':
		rospy.init_node('mid_frame_broadcaster')
		base_frame = 'panda_link0'
		init_frame = 'panda_hand'
		end_frame1 = 'blue_frame'
		end_frame2 = 'red_frame'
		pub_tf1 = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
		pub_tf2 = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
		tfBuffer = tf2_ros.Buffer()
		listener = tf2_ros.TransformListener(tfBuffer)
		
		rate = rospy.Rate(5.0)

		while not rospy.is_shutdown():
			try:
				w2i  = tfBuffer.lookup_transform(base_frame, init_frame, rospy.Time(0))
				w2e1 = tfBuffer.lookup_transform(base_frame, end_frame1, rospy.Time(0))
				w2e2 = tfBuffer.lookup_transform(base_frame, end_frame2, rospy.Time(0))
				w2w  = tfBuffer.lookup_transform(base_frame, base_frame, rospy.Time(0))
				
				w2iPose = Pose()
				w2iPose.position.x = w2i.transform.translation.x
				w2iPose.position.y = w2i.transform.translation.y
				w2iPose.position.z = w2i.transform.translation.z
				w2iPose.orientation = w2i.transform.rotation

				w2e1Pose = Pose()
				w2e1Pose.position.x  = w2e1.transform.translation.x
				w2e1Pose.position.y  = w2e1.transform.translation.y
				w2e1Pose.position.z  = w2e1.transform.translation.z
				w2e1Pose.orientation = w2e1.transform.rotation

				w2e2Pose = Pose()
				w2e2Pose.position.x  = w2e2.transform.translation.x
				w2e2Pose.position.y  = w2e2.transform.translation.y
				w2e2Pose.position.z  = w2e2.transform.translation.z
				w2e2Pose.orientation = w2e2.transform.rotation

				w2wPose = Pose()
				w2wPose.position.x = w2w.transform.translation.x
				w2wPose.position.y = w2w.transform.translation.y
				w2wPose.position.z = w2w.transform.translation.z
				w2wPose.orientation = w2w.transform.rotation

				w2i_kdl = tf_conversions.fromMsg(w2iPose)
				w2i_x = w2i_kdl.M.UnitX()
				w2i_pos = w2i_kdl.p

				w2e1_kdl = tf_conversions.fromMsg(w2e1Pose)
				w2e1_x = w2e1_kdl.M.UnitX()
				w2e1_pos = w2e1_kdl.p

				w2e2_kdl = tf_conversions.fromMsg(w2e2Pose)
				w2e2_x = w2e2_kdl.M.UnitX()
				w2e2_pos = w2e2_kdl.p

				w2w_kdl = tf_conversions.fromMsg(w2wPose)
				w2w_z = w2w_kdl.M.UnitZ()
				w2w_pos = w2w_kdl.p
				
				x_rot_wp1 = (w2e1_pos-w2i_pos)/((w2e1_pos-w2i_pos).Norm())
				y_rot_wp1 = w2w_z*x_rot_wp1
				z_rot_wp1 = x_rot_wp1*y_rot_wp1
				wp1_pos = (w2i_pos+w2e1_pos)/2

				x_rot_wp2 = (w2e2_pos-w2i_pos)/((w2e2_pos-w2i_pos).Norm())
				y_rot_wp2 = w2w_z*x_rot_wp2
				z_rot_wp2 = x_rot_wp2*y_rot_wp2
				wp2_pos = (w2i_pos+w2e2_pos)/2

				wp1_M = PyKDL.Rotation(x_rot_wp1, y_rot_wp1, z_rot_wp1)
				wp2_M = PyKDL.Rotation(x_rot_wp2, y_rot_wp2, z_rot_wp2)

				wp1_kdl = PyKDL.Frame(wp1_M, wp1_pos)
				wp2_kdl = PyKDL.Frame(wp2_M, wp2_pos)

				wp1_pose = tf_conversions.toMsg(wp1_kdl)
				wp2_pose = tf_conversions.toMsg(wp2_kdl)

				w2mid1 = TransformStamped()
				w2mid1.header.frame_id = base_frame
				w2mid1.header.stamp = rospy.Time.now()
				w2mid1.child_frame_id = 'wp1_frame'

				w2mid1.transform.translation.x = wp1_pose.position.x
				w2mid1.transform.translation.y = wp1_pose.position.y
				w2mid1.transform.translation.z = wp1_pose.position.z
				recipNorm1 = 1/math.sqrt(wp1_pose.orientation.x**2+
								   		 wp1_pose.orientation.y**2+
								   		 wp1_pose.orientation.z**2+
								   		 wp1_pose.orientation.w**2)
				w2mid1.transform.rotation.x = wp1_pose.orientation.x*recipNorm1
				w2mid1.transform.rotation.y = wp1_pose.orientation.y*recipNorm1 
				w2mid1.transform.rotation.z = wp1_pose.orientation.z*recipNorm1 
				w2mid1.transform.rotation.w = wp1_pose.orientation.w*recipNorm1

				w2mid2 = TransformStamped()
				w2mid2.header.frame_id = base_frame
				w2mid2.header.stamp = rospy.Time.now()
				w2mid2.child_frame_id = 'wp2_frame'

				w2mid2.transform.translation.x = wp2_pose.position.x
				w2mid2.transform.translation.y = wp2_pose.position.y
				w2mid2.transform.translation.z = wp2_pose.position.z
				recipNorm2 = 1/math.sqrt(wp2_pose.orientation.x**2+
								   		 wp2_pose.orientation.y**2+
								   		 wp2_pose.orientation.z**2+
								   		 wp2_pose.orientation.w**2)
				w2mid2.transform.rotation.x = wp2_pose.orientation.x*recipNorm2
				w2mid2.transform.rotation.y = wp2_pose.orientation.y*recipNorm2 
				w2mid2.transform.rotation.z = wp2_pose.orientation.z*recipNorm2 
				w2mid2.transform.rotation.w = wp2_pose.orientation.w*recipNorm2

				tfmsg1 = tf2_msgs.msg.TFMessage([w2mid1])
				tfmsg2 = tf2_msgs.msg.TFMessage([w2mid2])
				pub_tf1.publish(tfmsg1)
				pub_tf2.publish(tfmsg2)

			except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
				rate.sleep()
				continue