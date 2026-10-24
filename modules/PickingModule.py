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
sys.path.append('/home/gerardomendo/Documents/DAS/SVRDAS/Modules')

from timeit import default_timer as timer
#from ATMC_functions_DAS import *
#from ATMC_functions_2 import *
from detect_peaks import *
from numba import jit
from utils import *
from check_os import *
from plotting_functions_DAS import *
from TempMatchDAS_func import *
from data_utils import *
from parameters import *

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

#Retrieving root and media
root, media = check_operating_system()

######################################################################################################################################################################

def Picking(paths_TM,paths_pick,parameters):

	#Setting the variables - Paths

	main_path = paths_pick["main"]
	path_data = paths_TM["path_det"]
	path_ascii = paths_TM["path_coeff"]
	path_original = paths_TM["path_wav"]
	path_images_all = paths_pick["img_labels"]
	path_corrected = paths_pick["corrected"]
	path_csv_methods = paths_pick["csv_methods"]
	path_csv = paths_pick["csv"]
	path_images = paths_pick["imagesDAS"]
	path_boxplots = paths_pick["boxplots"]
	path_picks = paths_pick["picks"]

	#Params

	#Concatenate files flag
	concat = parameters["concat"]

	#Conversion of DAS data from optical phase to strain/s
	#Parameters 

	#Sampling rate
	fs = parameters["fs"]

	lamb = parameters["lamb"]                                                                                #Wavelength of user laser ligth in a vacuum [m]
	nc = parameters["nc"]                                                                                 #Refraction index of a fiber                                                                                  
	eps = parameters["eps"]                                         	                                        #Optical-elastic coefficient for the fiber direction in isotropic media
	vp = parameters["vp"]																				   #Plane wave velocity [m/s]
	type_data = parameters["typeData"]                                                                     #centered or percentiles
	decChannels = parameters["decCh"]
	decfac = parameters["decTime"]  
	nelements = parameters["nelements"]
	filter_type = parameters["filter_type"]

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

	###############################################################################################################################################################################################################
	#Loading data

	#SNR dict
	snr_dict = {}

	#SNR dict
	model_dict = {}

	for idx,file in enumerate(ascii_files):

		if(idx > 0):
			break
	 
		idx = 1; file = ascii_files[idx]

		#Date in UTC Time

		#For Sanriku files
		#name_det = file[-33:-15]
		
		#For N-net files
		name_det = file[-30:-12]
		#print(f"File: {file}, Name file: {name_det}")

		#Loading data
		#start = timer()
		st_data, old_data, points, pch_idx, st_idx, params = load_data(file,{'original':path_original,'data':path_data,'ascii':path_ascii},concat=concat,system="Sanriku")
		print(f'st data shape: {st_data.shape}')

		#plotting_test(st_data,params)

		#Estimating standard deviation
		ccmatrix = create_correlation_matrix(path_data,file)
		print(f"std {name_det}: {np.std(ccmatrix.flatten())}")

		#end = timer()
		#print(f"Elasped time - load data - {end - start} seconds")

		#Retrieving parameters
		L,dx,starttime,fs = params[0], params[1], params[2], params[3]

		#St_sec
		st_sec = st_idx/fs

		#Retrieving centers
		start = timer()
		data, new_original, low_point, up_point, maxs, ch_idx = retrieve_limits(pch_idx,st_data,fs,system="Sanriku")
		end = timer()

		# print(f"data: {data.shape}")
		# print(f"low point: {low_point.shape}")
		# print(f"up point: {up_point.shape}")
		# print(f"ch idx: {ch_idx}")
		# print(f"maxs: {maxs}")
		print(f'params: {params}')

		b, a = scipy.signal.butter(4,(5.0,15.0),"band",fs=fs)
		ftrace = signal.filtfilt(b,a,st_data,axis=1)

		plt.ion()
		fig,ax = plt.subplots(1,1,figsize=(10,10))
		c = ax.imshow(ftrace.T,aspect="auto",interpolation='nearest',cmap="seismic",origin="upper",vmin=-np.percentile(ftrace,97),vmax=np.percentile(ftrace,97))

		#Estimate RMS - original
		snr_total = estimate_snr(new_original)
		max_snr = np.amax(snr_total)
		mean_snr = np.mean(snr_total)
		sort_snr = np.sort(snr_total)
		iqr_snr = stats.iqr(sort_snr)
		snr_data = estimate_snr(data); snr_olddata = estimate_snr(old_data)

		# ########################################################################################################################################################################################################

		# #Retrieving centers
		# start = timer()
		# data, new_original, low_point, up_point = retrieve_centered_traces(ch_idx,st_data,fs)
		# end = timer()

		# #Estimate RMS - original
		# snr_total = estimate_snr(new_original)
		# max_snr = np.amax(snr_total)
		# mean_snr = np.mean(snr_total)
		# sort_snr = np.sort(snr_total)
		# iqr_snr = stats.iqr(sort_snr)
		# snr_data = estimate_snr(data); snr_olddata = estimate_snr(old_data)

		# ########################################################################################################################################################################################################

		# #Renaming
		# if(type_data == 'percentiles'):

		# 	#Retrieve sliced data using SNR percentile
		# 	start = timer()
		# 	data2, filtered, ev_snr2, newchsnr2, low_point2, up_point2 = retrieve_data_percentiles(st_data,60,hf_dur=2500)
		# 	end = timer()
		# 	print(f"Elasped time - retrieve data percentiles - {end - start} seconds")

		# 	#UNCOMMENT IF USING NEW RETRIEVE CENTERED TRACES
		# 	#data = data2; ev_snr = ev_snr2; newchsnr = newchsnr2; low_point = low_point2; up_point = up_point2
		# 	#params_sta_lta, params_rms, params_kurtosis, params_skew, params_ar = obtain_parametersLowSNR(fs)


		# #If using retrieve centered traces
		# else:
		# 	if(max_snr > -0.1):
		# 		print(f"data shape: {data.shape}")
		# 		data = data; ev_snr = snr_data; newchsnr = list(ch_idx)
		# 		params_sta_lta, params_rms, params_kurtosis, params_skew, params_ar = obtain_parametersHighSNR(fs)
		# 	else:
		# 		#print(f"data shape: {data.shape}")
		# 		data = old_data; ev_snr = snr_olddata; newchsnr = list(ch_idx); low_point = points[0]; up_point = points[1]
		# 		params_sta_lta, params_rms, params_kurtosis, params_skew, params_ar = obtain_parametersLowSNR(fs)

		# ########################################################################################################################################################################################################

		# #Parameters of pickers

		# # #Obtaining labels using stalta
		# stalta_p, stalta_s, ch_stalta = labeling_stalta(data,newchsnr,params_sta_lta)

		# #Path to save STA/LTA files
		# path_stalta = f"{path_images_all}/STALTA"

		# #Checking directory to store STALTA labels
		# check_directory(path_stalta)
		# plottingDASpicks(path_stalta,[dx,fs],st_data.T,name_det,[ch_stalta],stalta_p,stalta_s,mean_snr,low_point,ev_snr,extra_name=f"STALTA_picks_{type_data}",type="pdf",chidx=False,add_lowpoints=True)

		# ########################################################################################################################################################################################################

		# # #Obtaining labels using AR - AIC
		# ar_p, ar_s, ar_ch = labeling_ar(data,newchsnr,params_ar)

		# #Path to save AR files
		# path_ar = f"{path_images_all}/AR"

		# #Checking directory to store AR labels
		# check_directory(path_ar)
		# plottingDASpicks(path_ar,[dx,fs],st_data,name_det,[ar_ch],ar_p,ar_s,mean_snr,low_point,ev_snr,extra_name=f"AR_picks_{type_data}",type="pdf",chidx=False,add_lowpoints=True)

		# ########################################################################################################################################################################################################

		# #Plotting all
		# ppicks = (stalta_p,ar_p)
		# spicks = (stalta_s,ar_s)
		# chns = (ch_stalta,ar_ch)

		# #Filtering data
		# if(filter_type == "band"):
		# 	c2, c1 = signal.butter(4,(5.0,15.0),filter_type,fs=fs)
		# if(filter_type == "high"):
		# 	c2, c1 = signal.butter(4,8.0,filter_type,fs=fs)
		# filtered = signal.filtfilt(c2,c1,st_data,axis=1)

		# #Estimating SNR from a subset of filtered elements
		# nevsnr = estimate_snr(filtered[:nelements,:])

		# ch_p, ch_s, p_picks, s_picks = comparison_picks(ppicks,spicks,chns[1],low_point,c_thres = 500)
		# nchp, nchs, newp, news = correct_picks([ch_p,ch_s],p_picks,s_picks)

		# #Obtaining interquartile range of S minus P
		# iqr_sminp = obtain_iqr(nchp,nchs,newp,news)

		# print(f"iqr snr: {iqr_snr}, iqr sminp: {iqr_sminp}")
		# print(f"Min. SNR filtered: {np.amin(nevsnr)}, Max. SNR filtered: {np.amax(nevsnr)}, mean SNR filtered: {np.mean(nevsnr)}, iqr SNR filtered: {stats.iqr(nevsnr)}")

		# snr_dict[name_det] = iqr_snr

		# #Setting kernel
		# kernelp,kernels = obtain_kernel(nchp,nchs,dx,iqr_sminp,dthres=50000,pthres=10)
		# #kernelp,kernels = "rbf","rbf"

		# if(len(newp) > 50 and len(news) > 50):

		# 	if(iqr_sminp < 5.0):

		# 		print(f"IQR s-p: {iqr_sminp}, len P picks: {len(newp)}, len S picks: {len(news)}")

		# 		start = timer()
		# 		anewch, anewp, anews = obtain_pickModel(newp,news,[nchp,nchs],low_point,nelements,np.amax(snr_total),addlowpoints=False,kernelp=kernelp,kernels=kernels)
		# 		end = timer()
		# 		print(f"Elasped time - obtain initial model - {end - start} seconds")

		# 		name_allpicks = f"{path_picks}/{name_det}/DAS_{name_det}_all_picks.csv"
		# 		name_pickModel = f"{path_picks}/{name_det}/DAS_{name_det}_picks_model.csv"
		# 		name_ppicks = f"{path_picks}/{name_det}/DAS_{name_det}_Pini_picks.csv"
		# 		name_spicks = f"{path_picks}/{name_det}/DAS_{name_det}_Sini_picks.csv"

		# 		check_directory(f"{path_picks}/{name_det}")
		# 		writing2csvAll(name_allpicks,ppicks,spicks,chns[1],name_det,low_point)
		# 		writing2csvPick(name_ppicks,nchp,newp,name_det)
		# 		writing2csvPick(name_spicks,nchs,news,name_det)
		# 		writing2csvModel(name_pickModel,anewch,anewp,anews,nevsnr,name_det)

		# 		#Refining models

		# 		#Parameters for refining using spec
		# 		params_spec = {

		# 			"fs":fs,
		# 			"nperseg":128,
		# 			"noverlap":120

		# 			}

		# 		anewch, newss, newps = refine_models(filtered,nelements,anewch,anewp,anews,params_spec)

		# 		#Path to the Model directory
		# 		path_model = f"{path_images_all}/Model"

		# 		start = timer()
		# 		chps, ps = obtain_pickModelOne(newps,anewch,low_point,nelements,addlowpoints=False,kernel=kernelp)
		# 		chps, sss = obtain_pickModelOne(newss,anewch,low_point,nelements,addlowpoints=False,kernel=kernels)
		# 		end = timer()
		# 		print(f"Elasped time - obtain refined model - {end - start} seconds")

		# 		#Storing into csv
		# 		name_pickModelRef = f"{path_picks}/{name_det}/DAS_{name_det}_picks_modelRef.csv"
		# 		writing2csvModel(name_pickModelRef,chps,ps,sss,nevsnr,name_det)

		# 		#Writing into csv files
		# 		name_csv = f"PS_picks_{name_det}_Final.csv"
		# 		writing2csv(path_csv,name_csv,st_data,[anewch,anewch],newps,newss,fs,f"{name_det}",iqr_snr,decChannels)

		# 		#Plotting 
		# 		plottingDASpicks(path_images,[dx,fs],st_data,name_det,[anewch],ps,sss,mean_snr,low_point,nevsnr,extra_name="SVRDAS_Picking",type="png",chidx=False,add_lowpoints=False)
		# 		model_dict[name_det] = 1

		# else:

		# 	print(f"Not enough data to create model for {name_det}")
		# 	model_dict[name_det] = 0

		# #Plotting all picks
		# plottingDASpicks(path_images,[dx,fs],st_data,name_det,[anewch],anewp,anews,mean_snr,low_point,nevsnr,extra_name="Model_new",type="pdf",chidx=False,add_lowpoints=False)
		# plottingDASpicks(path_images,[dx,fs],st_data,name_det,[anewch],ps,sss,mean_snr,low_point,nevsnr,extra_name="Model2_new",type="pdf",chidx=False,add_lowpoints=False)
		# plottingDASpicks(path_images,[dx,fs],st_data,name_det,[anewch],newps,newss,mean_snr,low_point,nevsnr,extra_name="Model3_new",type="pdf",chidx=False,add_lowpoints=False)
		# plotting2D(path_images,[dx,fs],st_data,name_det,[nchp,nchs], np.array([newp[ch] for ch in newp.keys()])/fs, np.array([news[ch] for ch in news.keys()])/fs, mean_snr,extra_name="AfterCorr_new",type="pdf",chidx=False)
		
		# #Further plotting

		# plotting_label_traces(path_model,filtered[:2000,:],anewch,anewp,anews,fs,name_det,method="SP_SVR",zoom=[40,80])
		# plottingDASpicks(main_path,[dx,fs],st_data,name_det,[ch_p,ch_s],p_picks,s_picks,mean_snr,low_point,ev_snr,extra_name="BeforeCorr_new",type="png",chidx=False,add_lowpoints=False)
		# plotting2D(main_path,[dx,fs],st_data,name_det,[ch_p,ch_s], np.array([p_picks[ch] for ch in p_picks.keys()])/fs, np.array([s_picks[ch] for ch in s_picks.keys()])/fs, mean_snr,extra_name="BeforeCorr_new",type="pdf",chidx=False)
		
