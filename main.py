import sys
sys.path.append('/path/to/svrdas/modules')

from dict_params import params_tempMatch, params_picking, paths_TM
from TempMatchDAS import *
from NewPickingModule import *

########################################################################################################################################################

#Main function
def main(params_tempMatch,params_picking,detection=True,picking=True,download=False):

	if(detection):

		print(f"Running template matching...")
		TemplateMatchingDAS(params_tempMatch,download=False)

	if(picking):

		print(f"Running SVRDAS for picking...")
		Picking(params_tempMatch,params_picking,download=False)

########################################################################################################################################################

if __name__ == '__main__':

	# Calling main
	main(params_tempMatch,params_picking,detection=False,picking=True,download=False)