#!/usr/bin/python
import math
import numpy as np

def RotX(vector, angle):

	rot_x = np.array([[1, 0, 0], [0, math.cos(angle), math.sin(angle)], [0, -math.sin(angle), math.cos(angle)]])
	vector_rot = vector.dot(rot_x)

	return vector_rot

def RotY(vector, angle):

	rot_y = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
	vector_rot = vector.dot(rot_y)

	return vector_rot

def RotZ(vector, angle):

	rot_z = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
	vector_rot = vector.dot(rot_z)

	return vector_rot