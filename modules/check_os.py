import os
import platform

def check_operating_system():

	system = ["Linux","macOS"]

	op_system = platform.platform()[:5]
	if(op_system == system[0]):

		root = "/home"
		media = "/media/gerardomendo"

	elif(op_system == system[1]):

		root = "/Users"
		media = "/Volumes"

	return root, media
