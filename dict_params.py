from check_os import *

#Retrieving root and media
root, media = check_operating_system()

#Main paths
path2TM = f"{root}/path/to/svrdas/data/TempMatching"
path2Pick = f"{root}/path/to/svrdas/data/SeisPicking"

#DAS system (type your DAS system name here - must match the name of the files)
system = None

#Length of DAS files (integer)
length = 60

#Data structure
#Sanriku1 -> [NSAMPLES,NCHANNELS]
#Sanriku2 -> [NSAMPLES,NCHANNELS]
#Nnet     -> [NSAMPLES,NCHANNELS]

# Parameters TemplateMatching 
params_tempMatch = {

	#DAS system
	"system":system,
	"path_wav":f"{media}/IOdata-16TB/N-net_2024-08-02",

	#Path to Template Matching Dir
	"main":path2TM,

	#Temporal path to store downloaded data - uncomment if downloading data
	#"path_wav":f"/home/gerardomendo/Documents/DAS/toServer/SVRDAS/temp",

	#Name info data 
	"info_name":"InfoFile_From_TempMatchResults_info.txt",

	#Path output template matching
	"path_temp":f"{path2TM}/Input/Templates",
	"path_det":f"{path2TM}/Output/Detections/{system}",
	"path_coeff":f"{path2TM}/Output/Coefficients/{system}",
	"path_tr":f"{path2TM}/Output/Traces/{system}",

	#Flags for envelope usage and storing info
	"env_flag":True,
	"save_flag":True,

	#Filter type	
	"filter_type":"band",

	#Overlap (0 - 0% and 1 - 100%) and threshold (0 to 1) values 
	"overlap":0.5,
	"threshold":0.8,

}

# Parameters Picking - N-net
params_picking = {
	
	#Paths to store outputs
	"main":path2Pick,
	"img_labels": f"{path2Pick}/img_labels/",
	"corrected": f"{path2Pick}/corrected/",
	"csv_methods": f"{path2Pick}/csv_methods/",
	"csv": f"{path2Pick}/csv/",
	"imagesDAS": f"{path2Pick}/images_DAS/",
	"boxplots": f"{path2Pick}/boxplots/",
	"picks": f"{path2Pick}/picks/",
	"initial": f"{path2Pick}/initial/",

	#Length DAS files
	"T":60,

	#Sampling rate
	"fs":100,

	#Concatenation (if searching in 180 min data using 60 min DAS files)
	"concat":False,

	#Coefficients to convert from optical phase to strain rate units
	"lamb":4e-9,
	"nc":1.457,
	"eps":0.78,
	"vp":3500,

	#Method used to obtain data windows for picking - centered or percentiles
	"typeData":"centered",

	#Decimation factor on channel axis
	"decCh":1,

	#Decimation factor on time axis
	"decTime":2,

	#Number of channels
	"nelements":10000,

	#Filter type
	"filter_type":"band"

}