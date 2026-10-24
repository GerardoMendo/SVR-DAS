import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import obspy
from obspy import read
from obspy import UTCDateTime
import obspy.signal.filter as filt
from obspy.signal.detrend import spline
import obspy.signal.cross_correlation as correlation
import obspy.signal.trigger as trigger

import scipy.signal
from scipy import fftpack
import scipy.signal as signal
import scipy.stats as stats
from scipy.stats import norm

import sys
sys.path.append('/path/to/svrdas/modules')

from timeit import default_timer as timer
from detect_peaks import *
from numba import jit
from utils import *
from check_os import *
from plotting_functions_DAS import *
from TempMatchDAS_func import *
from data_utils import *
from pick_paramsnnet import *
from obtainTraces import *
from runMethods import *
from obtainInitialPicks import *

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

#Retrieving root and media
root, media = check_operating_system()

def Picking(dict_TM,dict_picking,download=False):

	#Setting the variables - Paths

	system = dict_TM["system"]
	main_path = dict_picking["main"]
	path_data = dict_TM["path_det"]
	path_ascii = dict_TM["path_coeff"]
	path_original = dict_TM["path_wav"]
	path_images_all = dict_picking["img_labels"]
	path_corrected = dict_picking["corrected"]
	path_csv_methods = dict_picking["csv_methods"]
	path_csv = dict_picking["csv"]
	path_images = dict_picking["imagesDAS"]
	path_boxplots = dict_picking["boxplots"]
	path_picks = dict_picking["picks"]
	path_initial = dict_picking["initial"]
	path2TM = dict_TM["main"]

	#Concatenate files flag
	concat = dict_picking["concat"]

	################### Conversion of DAS data from optical phase to strain/s ############################################################################################
	#Parameters 

	#Sampling rate
	fs = dict_picking["fs"]

	#Length DAS data
	length = dict_picking['T']

	#Wavelength of user laser ligth in a vacuum [m]
	lamb = dict_picking["lamb"]

	#Refraction index of a fiber                                                                              
	nc = dict_picking["nc"]

	#Optical-elastic coefficient for the fiber direction in isotropic media                                                                                                                                                                  
	eps = dict_picking["eps"]

	#Plane wave velocity [m/s]                                         	                                  
	vp = dict_picking["vp"]

	#centered or percentiles																				  
	type_data = dict_picking["typeData"]

	#Decimation factors                                                                    
	decChannels = dict_picking["decCh"]
	decfac = dict_picking["decTime"]

	#Other options  
	nelements = dict_picking["nelements"]
	filter_type = dict_picking["filter_type"]
	infoName = dict_TM['info_name']

	################### Plotting STA/LTA - AR parameters #################################################################################################################

	#Path to save STA/LTA and AR files
	path_stalta = f"{path_images_all}/STALTA"
	path_ar = f"{path_images_all}/AR"

	#Checking directories to store STALTA and AR labels
	check_directory(path_stalta)
	check_directory(path_ar)

	######################################################################################################################################################################

	#Create directories if not existing
	check_directories(main_path,[path_images_all,path_corrected,path_csv,path_images,path_boxplots,path_picks])

	#Retrieve number of files
	os.chdir(path_data)
	detections = os.listdir()
	list_filter = glob.glob("*NH8*")
	list_original = glob.glob("*nofilter*")

	os.chdir(path_ascii)
	dascii_files = os.listdir()
	ascii_files = remove_hidden_files(path_data,dascii_files)
	ascii_files.sort()
	print(path_ascii)

	###############################################################################################################################################################################################################
	#Loading data

	path2info = f"{path2TM}/Output/Traces/{system}"

	#Sanriku
	#infoData = np.loadtxt(f"{path2info}/TempMatchDAS_TDAS_Template_2023-07-26T122703_O5500_7500_C800_info.txt",dtype=str)

	#Nnet
	infoData = np.loadtxt(f'{path2info}/{infoName}',dtype=str)

	names = infoData[:,0]
	freq = infoData[:,1].astype(int)
	stds = infoData[:,2].astype(float)
	cumfreq = infoData[:,3].astype(int)

	#SNR dict
	snr_dict = {}

	#SNR dict
	model_dict = {}

	for idx,file in enumerate(ascii_files):

		#Loading data
		st_data, old_data, points, pch_idx, st_idx, params = load_data(file,length,{'original':path_original,'data':path_data,'ascii':path_ascii},concat=concat,system=system,download=download)

		#Filtering data
		if(filter_type == "band"):
			c2, c1 = signal.butter(4,(5.0,10.0),filter_type,fs=params[-1])
		if(filter_type == "high"):
			c2, c1 = signal.butter(2,8.0,filter_type,fs=params[-1])
		filtered = signal.filtfilt(c2,c1,st_data,axis=1)
		filtered = signal.filtfilt(c2,c1,filtered,axis=1)

		#Retrieving parameters
		L,dx,starttime,fs = params[0], params[1], params[2], params[3]

		#St_sec
		st_sec = st_idx/fs

		#Desired window size
		len_window = 40

		#Setting low point estimation
		if(concat == True):
			lp = np.mean(points[0])-1000
		else:
			lp = None

		#Obtaining iqr_sminp
		nchp, nchs, newp, news, snr_total, max_snr, iqr_snr, iqr_sminp, low_point, picks = obtainIniPicks(name_det,pch_idx,st_data,filtered,nelements,dx,fs,len_window,path_stalta,path_ar,path_initial,concat,type_data,plotPicks=True,plotIni=True,lp=lp)
		print(f"IQR S-P: {iqr_sminp}")
		lows = np.ones(filtered.shape[0])*(np.mean(low_point)) 
		ups = np.ones(filtered.shape[0])*(np.mean(low_point) + 4000)

		#Estimating SNR from a subset of filtered elements
		nevsnr = estimation_snr(filtered,np.arange(filtered.shape[0]),lows,ups)

		#Storing in a dictionary
		snr_dict[name_det] = iqr_snr

		#Picks
		ppicks, spicks, chns = picks[0], picks[1], picks[2]

		############################################# SVR fitting ####################################################################################################################################################

		#Setting kernel
		kernelp,kernels = obtain_kernel(nchp,nchs,dx,iqr_sminp,dthres=50000,pthres=10,system=system)

		if(kernelp == "linear" and kernels == "linear"):

			nchp, nchs, newp, news = retrieveBuriedTraces(nchp,nchs,newp,news,dx,value=60000)

		if(len(newp) > 20 and len(news) > 20):

			if(iqr_sminp < 20):

				print(f"IQR s-p: {iqr_sminp}, len P picks: {len(newp)}, len S picks: {len(news)}")

				start = timer()
				anewch, anewp, anews = obtain_pickModel(newp,news,[nchp,nchs],low_point,nelements,np.amax(snr_total),10,1e-6,addlowpoints=False,kernelp=kernelp,kernels=kernels)
				end = timer()
				print(f"Elasped time - obtain initial model - {end - start} seconds")

				name_allpicks = f"{path_picks}/{name_det}/DAS_{name_det}_all_picks.csv"
				name_pickModel = f"{path_picks}/{name_det}/DAS_{name_det}_picks_model.csv"
				name_ppicks = f"{path_picks}/{name_det}/DAS_{name_det}_Pini_picks.csv"
				name_spicks = f"{path_picks}/{name_det}/DAS_{name_det}_Sini_picks.csv"

				check_directory(f"{path_picks}/{name_det}")
				writing2csvAll(name_allpicks,ppicks,spicks,chns[1],name_det,low_point)
				writing2csvPick(name_ppicks,nchp,newp,name_det)
				writing2csvPick(name_spicks,nchs,news,name_det)
				writing2csvModel(name_pickModel,anewch,anewp,anews,nevsnr,name_det)

				#Refining models

				#Parameters for refining using spec
				params_spec = {

					"fs":fs,
					"nperseg":128,
					"noverlap":120

					}

				anewch, newss, newps = refine_models(filtered,nelements,anewch,anewp,anews,params_spec,0.2,0.8)

				#Path to the Model directory
				path_model = f"{path_images_all}/Model"

				start = timer()
				chps, ps = obtain_pickModelOne(newps,anewch,low_point,nelements,addlowpoints=False,kernel=kernelp)
				chps, sss = obtain_pickModelOne(newss,anewch,low_point,nelements,addlowpoints=False,kernel=kernels)
				end = timer()
				print(f"Elasped time - obtain refined model - {end - start} seconds")

				#Storing into csv
				name_pickModelRef = f"{path_picks}/{name_det}/DAS_{name_det}_picks_modelRef.csv"
				writing2csvModel(name_pickModelRef,chps,ps,sss,nevsnr,name_det)

				#Writing into csv files
				name_csv = f"PS_picks_{name_det}_Final.csv"
				writing2csv(path_csv,name_csv,st_data,[anewch,anewch],newps,newss,fs,f"{name_det}",iqr_snr,decChannels)

				#Plotting 
				plottingDASpicks(path_images,[dx,fs],st_data.T,name_det,[anewch],ps,sss,max_snr,low_point,nevsnr,extra_name="SVRDAS_Picking",type="png",chidx=False,add_lowpoints=False)
				model_dict[name_det] = 1

		else:

			print(f"Not enough data to create model for {name_det}")
		# 	model_dict[name_det] = 0

		if(download == True and "temp" in path_original):

			os.chdir(path_original)
			os.system(f'rm -r *')

		
