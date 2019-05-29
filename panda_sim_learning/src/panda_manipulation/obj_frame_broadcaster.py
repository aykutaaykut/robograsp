#!/usr/bin/env python
from geometry_msgs.msg import TransformStamped, PointStamped, Pose, Quaternion
from pc_segmentation.msg import PcObjects
from tf2_msgs.msg import TFMessage
import rospy
import PyKDL
import tf2_ros
import tf2_geometry_msgs
import tf_conversions

base_frame   = 'panda_link0'
camera_frame = 'camera_depth_optical_frame'
eps  = 10
blue = 235
red  = 355

def get_bbox(topic):
	blue_obj, red_obj = None, None
	
	objects =  rospy.wait_for_message(topic, PcObjects, timeout=2)

	for box in objects.boxes:
		if blue-eps <= box.hue and box.hue <= blue+eps:
			blue_obj = box
		elif red-eps <= box.hue and box.hue <= red+eps:
			red_obj = box

	assert blue_obj is not None and red_obj is not None, 'Objects are not in the scene!'

	return blue_obj, red_obj

def get_obj_frames(blue_obj, red_obj, base_tf, base2kinect_tf):
	base_pose = Pose()
	base_pose.position    = base_tf.transform.translation
	base_pose.orientation = base_tf.transform.rotation
	base_kdl = tf_conversions.fromMsg(base_pose)
	base_unitX = base_kdl.M.UnitX()
	base_unitY = base_kdl.M.UnitY()
	base_unitZ = base_kdl.M.UnitZ()

	### Frame for Blue Object

	blue_center_kinect = PointStamped()
	blue_center_kinect.header = base2kinect_tf.header
	blue_center_kinect.point  = blue_obj.bb_center
	blue_center = tf2_geometry_msgs.do_transform_point(blue_center_kinect, base2kinect_tf)

	blue_pose = Pose()
	blue_pose.position = blue_center.point
	blue_pose.position.z = blue_pose.position.z - blue_obj.bb_dims.z/2
	blue_pose.orientation = base_tf.transform.rotation
	blue_kdl = tf_conversions.fromMsg(blue_pose)
	blue_pos = blue_kdl.p
	blue_rot = PyKDL.Rotation(-base_unitX, -base_unitY, base_unitZ)
	blue_kdl = PyKDL.Frame(blue_rot, blue_pos)
	blue_pose = tf_conversions.toMsg(blue_kdl)

	blue_frame = TransformStamped()
	blue_frame.header.frame_id = base_frame
	blue_frame.header.stamp = rospy.Time.now()
	blue_frame.child_frame_id = 'blue_frame'
	blue_frame.transform.translation = blue_pose.position
	blue_frame.transform.rotation    = blue_pose.orientation

	### Frame for Red Object

	red_center_kinect = PointStamped()
	red_center_kinect.header = base2kinect_tf.header
	red_center_kinect.point  = red_obj.bb_center
	red_center = tf2_geometry_msgs.do_transform_point(red_center_kinect, base2kinect_tf)

	red_pose = Pose()
	red_pose.position = red_center.point
	red_pose.position.z = red_pose.position.z - red_obj.bb_dims.z/2
	red_pose.orientation = base_tf.transform.rotation
	red_kdl = tf_conversions.fromMsg(red_pose)
	red_pos = red_kdl.p
	red_rot = PyKDL.Rotation(-base_unitX, -base_unitY, base_unitZ)
	red_kdl = PyKDL.Frame(red_rot, red_pos)
	red_pose = tf_conversions.toMsg(red_kdl)

	red_frame = TransformStamped()
	red_frame.header.frame_id = base_frame
	red_frame.header.stamp = rospy.Time.now()
	red_frame.child_frame_id = 'red_frame'
	red_frame.transform.translation = red_pose.position
	red_frame.transform.rotation    = red_pose.orientation

	return blue_frame, red_frame

if __name__ == '__main__':
	rospy.init_node('obj_frame_broadcaster')
	
	pub_blue_tf = rospy.Publisher('/tf', TFMessage, queue_size=1)
	pub_red_tf  = rospy.Publisher('/tf', TFMessage, queue_size=1)
	tfBuffer    = tf2_ros.Buffer()
	listener    = tf2_ros.TransformListener(tfBuffer)
		
	rate = rospy.Rate(5.0)

	blue_obj, red_obj = get_bbox('/baris/objects')

	while not rospy.is_shutdown():
		try:
			base_tf = tfBuffer.lookup_transform(base_frame, base_frame, rospy.Time(0))
			base2kinect_tf = tfBuffer.lookup_transform(base_frame, camera_frame, rospy.Time(0))

			blue_frame, red_frame = get_obj_frames(blue_obj, red_obj, base_tf, base2kinect_tf)
			
			blue_msg = TFMessage([blue_frame])
			red_msg  = TFMessage([red_frame])
			pub_blue_tf.publish(blue_msg)
			pub_red_tf.publish(red_msg)

		except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			rate.sleep()
			continue