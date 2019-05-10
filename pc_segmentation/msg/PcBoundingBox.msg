# Contains the width, height, x, y, z and color information of a single
# bounding box. (Dogancan, just for you <3)

#################################
#Bounding box

#position wrt sensor and angle wrt table normal
geometry_msgs/Vector3 bb_center
float64 bb_angle

# Bounding box dimensions
geometry_msgs/Vector3 bb_dims

#################################
# Color related

std_msgs/ColorRGBA rgba_color
float64 hue

#################################
