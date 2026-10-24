import os
import json
from pathlib import Path
import numpy as np

def return_fileNames(file,system=None):

	#System must be defined, if not an assertion error is raised
	assert (system != None), "Error in system name"

	#Loading file names JSON file
	with open(f'/home/gerardomendo/Documents/DAS/toServer/SVRDAS/systems.json','r',encoding='utf-8') as f:
		sysNames = json.load(f)

	#Retrieve name from input
	sysName = sysNames[system]

	#Retrieve first section
	head = sysName.split("_")[0]

	#Splitting file variable
	fsplit = file.split("_")

	#Split file variable
	headFile = fsplit[8]

	#Assertion of head and fileHead
	assert head == headFile, "No match"

	#Identify the head in the file variable
	if(system == "Sanriku2"):
		name_original = f'{fsplit[8]}_{fsplit[9]}_{fsplit[10]}_{fsplit[11]}.h5'
		name_det = fsplit[11]
	elif(system == "Sanriku1"):
		name_original = f'{fsplit[8]}_{fsplit[9]}_{fsplit[10]}.h5'
		name_det = fsplit[10]
	else:
		name_original = f'{fsplit[8]}_{fsplit[9]}_{fsplit[10]}'
		#print(fsplit[10][:-3])
		name_det = fsplit[10][:-3]

	return name_original, name_det


