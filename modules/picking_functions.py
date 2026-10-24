import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import obspy
import obspy.core
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

from timeit import default_timer as timer
#from ATMC_functions_DAS import *
#from ATMC_functions_2 import *
from detect_peaks import *
from numba import jit
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from map_kurt_skew import *

import h5py
import os

#################################################################################################################

def obtain_parameters(fs):

	#STA/LTA
	params_sta_lta = {

		#"STA":int(0.35*fs),
		#"LTA":int(3.5*fs),
		#"scale_th":6

		"STA":int(0.5*fs),
		"LTA":int(3.0*fs),
		"scale_th":6

		}

	#RMS
	params_rms = {

		"fs":fs,
		"length":5.0,
		"overlap":0.25,
		"threshold":0.8,
		"envelope":False,
		"sf_length":10,
		"sf_order":3

		}

	#RMS and Kurtosis

	params_kurtosis = {

		"fs":fs,
		"length_rms":0.1,
		"overlap_rms":0.9,
		"length_kurt":0.1,
		"overlap_kurt":0.9,
		"threshold":0.2,
		"envelope_rms":False,
		"envelope_kurt":False,
		"sf_length":30,
		"sf_order":3

		}

	params_skew = {

		"fs":fs,
		"length_kurt":0.5,
		"overlap_kurt":0.25,
		"th_kurt":0.6,
		"length_skew":0.5,
		"overlap_skew":0.25,
		"th_skew":0.6

		}

	params_ar = {

	"fs":fs,
	"f1":1.0,
	"f2":15.0,
	"lta_p":3.0,
	"sta_p":0.5,
	"lta_s":5.0,
	"sta_s":1.0,
	"a":2,
	"b":8,
	#"std_p":0.1,
	"std_p":5.0,
	"std_s":0.1

	}

	# params_ar = {

	# "fs":fs,
	# "f1":10.0,
	# "f2":20.0,
	# "lta_p":1.0,
	# "sta_p":0.1,
	# "lta_s":2.5,
	# "sta_s":0.5,
	# "a":2,
	# "b":8,
	# "std_p":0.1,
	# "std_s":0.1

	# }

	return params_sta_lta, params_rms, params_kurtosis, params_skew, params_ar

#################################################################################################################
#######################RMS calculation###########################################################################

def RMS(array):
	
	suma = 0
	N = array.shape[0]
	for i in range(0,N,1):

		#print(f"i: {i}, N: {N}")

		square_comp = array[i]**2
		suma = suma + square_comp
	
	value = suma/N
	rms = np.sqrt(value)
	return rms

##############################################################################################################

def envelope(input_signal):

	#Calculate envelope
	analytic_signal = hilbert(input_signal)
	envelope = np.abs(analytic_signal)
	smooth = savgol_filter(envelope,30,3)

	return smooth

##############################################################################################################

def skew_detector(input_signal,fs,window_size,overlap):

	#Number of windows in seconds
	Nwin = int(input_signal.shape[0]/((window_size*fs)*overlap))
	#print(f"Nwin: {Nwin}")

	#Initial lower and upper limit
	inc = int(window_size*fs)
	li = 0; ls = inc

	#Overlap percentage where window starts 
	rov = 1 - overlap  

	#RMS array
	rarray = np.zeros(Nwin)
	#print(Nwin)

	#Estimate number of windows
	for i in range(0,Nwin):

		a = input_signal[li:ls]

		#print(f"Nwin:{Nwin}, Number of window: {i}")
		#print(f" Window size: {a.shape[0]}")
		#print(f" Lower limit: {li}, upper limit: {ls}")
		
		#Obtain rms
		rarray[i] = stats.skew(a)	
		#print(f'N:{i},li:{li},ls:{ls},N:{a.shape[0]},RMS:{rarray[i]}')
		li = int((i*inc)*overlap); ls = li + inc

		if(ls > input_signal.shape[0]):

			#print(f"Breaks in window {i}")

			break

	#return savgol_filter(rarray,30,3)
	return rarray

##############################################################################################################

def kurtosis_detector(input_signal,fs,window_size,overlap):

	#Number of windows in seconds
	Nwin = int(input_signal.shape[0]/((window_size*fs)*overlap))
	#print(f"Nwin: {Nwin}")

	#Initial lower and upper limit
	inc = int(window_size*fs)
	li = 0; ls = inc

	#Overlap percentage where window starts 
	rov = 1 - overlap  

	#RMS array
	rarray = np.zeros(Nwin)
	#print(Nwin)

	#Estimate number of windows
	for i in range(0,Nwin):

		a = input_signal[li:ls]

		#print(f"Nwin:{Nwin}, Number of window: {i}")
		#print(f" Window size: {a.shape[0]}")
		#print(f" Lower limit: {li}, upper limit: {ls}")
		
		#Obtain rms
		rarray[i] = stats.kurtosis(a)	
		#print(f'N:{i},li:{li},ls:{ls},N:{a.shape[0]},RMS:{rarray[i]}')
		li = int((i*inc)*overlap); ls = li + inc

		if(ls > input_signal.shape[0]):

			#print(f"Breaks in window {i}")

			break

	#return savgol_filter(rarray,30,3)
	return rarray

##########################################################################################################

def rms_detector(input_signal,fs,window_size,overlap):

	#Number of windows in seconds
	Nwin = int(input_signal.shape[0]/((window_size*fs)*overlap))
	#print(f"Nwin: {Nwin}")

	#Initial lower and upper limit
	inc = int(window_size*fs)
	li = 0; ls = inc

	#Overlap percentage where window starts 
	rov = 1 - overlap  

	#RMS array
	rarray = np.zeros(Nwin)
	#print(Nwin)

	#Estimate number of windows
	for i in range(0,Nwin):

		a = input_signal[li:ls]

		#print(f"Nwin:{Nwin}, Number of window: {i}")
		#print(f" Window size: {a.shape[0]}")
		#print(f" Lower limit: {li}, upper limit: {ls}")
		
		#Obtain rms
		rarray[i] = RMS(a)	
		#print(f'N:{i},li:{li},ls:{ls},N:{a.shape[0]},RMS:{rarray[i]}')
		li = int((i*inc)*overlap); ls = li + inc

		if(ls > input_signal.shape[0]):

			#print(f"Breaks in window {i}")

			break

	#return savgol_filter(rarray,30,3)
	return rarray

##########################################################################################################

def obtain_picks(trace,rms,threshold):

	#Obtaining the half
	#half = int(rms.shape[0]/2)

	#Setting the list of indexes
	index = []

	#Obtaining the gradient

	#Flag to only save first point
	only_one = False

	for i in range(0,rms.shape[0]-1):

		a = rms[i]/np.amax(rms); b = rms[i+1]/np.amax(rms)
		diff = (b-a)
		#print(f"i: {i}, i+1:{i+1}, diff={diff}")
		if(diff > 1e-2 and only_one == False):
			index.append(i+1)
			only_one = True

		if(diff < threshold):
			only_one = False

	return index

##########################################################################################################

def plotting_all(path_images,name_file,channels,data,all_rmsk,all_kurt,p_stalta,p_rms,p_rk,p_ar):

	#print(f"Plotting...")

	for channel in channels:

		#Index
		trace = data[channel,:]
		time = np.arange(0,trace.shape[0])/100.0

		#Picks STA/LTA
		stlabel_p = p_stalta[0][str(channel)]; stlabel_s = p_stalta[1][str(channel)]; stlabel_n = p_stalta[2][str(channel)]

		#Picks RMS
		rlabel_p = p_rms[0][str(channel)]; rlabel_s = p_rms[1][str(channel)]; rlabel_n = p_rms[2][str(channel)]

		#Picks RMS-Kurtosis
		rklabel_p = p_rk[0][str(channel)]; rklabel_s = p_rk[1][str(channel)]; rklabel_n = p_rk[2][str(channel)]

		#Picks AR - AIC
		arlabel_p = p_ar[0][str(channel)]; arlabel_s = p_ar[1][str(channel)]; arlabel_n = p_ar[2][str(channel)]

		#evaluate_peaks()

		#Picks average
		avlabel_p = np.median(np.array([stlabel_p,rlabel_p,rklabel_p,arlabel_p]),axis=0)
		avlabel_s = np.median(np.array([stlabel_s,rlabel_s,rklabel_s,arlabel_s]),axis=0)

		name_image = f"All_Phases_{name_file}_Channel_{channel}.png"

		plot = True
		if(plot):

			#Plotting
			fig, ax = plt.subplots(2,1,figsize=(20,10))
			ax[0].plot(time,trace*(1e6),linewidth=0.7,c="k")
			ax[0].set_xlim(time[0],time[-1])
			ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
			ax[0].tick_params(axis="x",color="k",labelsize=14.0)
			ax[0].tick_params(axis="y",color="k",labelsize=14.0)
			
			ax[1].plot(time,stlabel_p,linewidth=0.7,c="r",label="P_sta_lta")
			ax[1].plot(time,stlabel_s,linewidth=0.7,c="b",label="S_sta_lta")
			ax[1].set_xlim(time[0],time[-1])
			
			ax[1].plot(time,rlabel_p,linewidth=0.7,c="r",label="P_rms",linestyle="dashed")
			ax[1].plot(time,rlabel_s,linewidth=0.7,c="b",label="S_rms",linestyle="dashed")

			ax[1].plot(time,rklabel_p,linewidth=0.7,c="r",label="P_rms_kurt",linestyle="dotted")
			ax[1].plot(time,rklabel_s,linewidth=0.7,c="b",label="S_rms_kurt",linestyle="dotted")

			ax[1].plot(time,arlabel_p,linewidth=0.7,c="r",label="P_rms_AR",linestyle="dashdot")
			ax[1].plot(time,arlabel_s,linewidth=0.7,c="b",label="S_rms_AR",linestyle="dashdot")

			#ax[1].plot(time,avlabel_p,linewidth=2.0,c="r",label="AV",linestyle="solid")
			#ax[1].plot(time,avlabel_s,linewidth=2.0,c="b",label="AV",linestyle="solid")

			ax[1].set_xlim(time[0],time[-1])
			ax[1].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
			ax[1].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
			ax[1].tick_params(axis="x",color="k",labelsize=14.0)
			ax[1].tick_params(axis="y",color="k",labelsize=14.0)

			ax[1].legend(loc="upper right")
			plt.savefig(f"{path_images}{name_image}")
			plt.close()


##########################################################################################################

def plotting_rmsk(path_images,channels,data,all_rms,all_kurt,labels_p,labels_s,labels_n):

	#print(f"Plotting...")

	for channel in channels:

		#Index
		trace = data[channel,:]
		time = np.arange(0,trace.shape[0])/100.0
		rms = all_rms[channel]
		kurt = all_kurt[channel]
		time_rms = np.arange(0,rms.shape[0])*(0.25*0.23)
		label_p = labels_p[str(channel)]
		label_s = labels_s[str(channel)]
		label_n = labels_n[str(channel)]

		name_image = f"RMS_Phases_Channel_{channel}.png"

		plot = True
		if(plot):

			#Plotting
			fig, ax = plt.subplots(4,1,figsize=(20,10))
			ax[0].plot(time,trace,linewidth=0.7,c="k")
			ax[1].plot(rms,linewidth=0.7,c="k")
			ax[2].plot(kurt,linewidth=0.7,c="k")
			ax[3].plot(time,label_p,linewidth=0.7,c="r")
			ax[3].plot(time,label_s,linewidth=0.7,c="b")
			plt.savefig(f"{path_images}{name_image}")
			plt.close()


##########################################################################################################

def plotting_rms(path_images,channels,data,all_rms,indexes,labels_p,labels_s,labels_n):

	#print(f"Plotting...")

	for channel in channels:

		#Index
		trace = data[channel,:]
		time = np.arange(0,trace.shape[0])/100.0
		rms = all_rms[channel]
		time_rms = np.arange(0,rms.shape[0])*(0.25*0.23)
		index = indexes[channel]
		label_p = labels_p[str(channel)]
		label_s = labels_s[str(channel)]
		label_n = labels_n[str(channel)]

		name_image = f"RMS_Phases_Channel_{channel}.png"

		plot = True
		if(plot):

			#Plotting
			fig, ax = plt.subplots(3,1,figsize=(20,10))
			ax[0].plot(time,trace,linewidth=0.7,c="k")
			ax[1].scatter(np.arange(0,rms.shape[0]),rms[:],s=20,c="b")


			for ix in index:
				ax[1].scatter(ix,rms[ix],s=20,c="r")

			ax[2].plot(time,label_p,linewidth=0.7,c="r")
			ax[2].plot(time,label_s,linewidth=0.7,c="b")
			plt.savefig(f"{path_images}{name_image}")
			plt.close()


##########################################################################################################

def plotting_stalta(path_images,channels,data,labels_p,labels_s,labels_n):

	#print(f"Plotting...")

	for channel in channels:

		#Index
		trace = data[channel,:]
		time = np.arange(0,trace.shape[0])/100.0
		label_p = labels_p[str(channel)]
		label_s = labels_s[str(channel)]
		label_n = labels_n[str(channel)]

		name_image = f"STALTA_Phases_Channel_{channel}.png"

		plot = True
		if(plot):

			#Plotting
			fig, ax = plt.subplots(2,1,figsize=(20,10))
			ax[0].plot(time,trace,linewidth=0.7,c="k")
			ax[1].plot(time,label_p,linewidth=0.7,c="r")
			ax[1].plot(time,label_s,linewidth=0.7,c="b")
			ax[1].plot(time,label_n,linewidth=0.7,c="g")
			plt.savefig(f"{path_images}{name_image}")
			plt.close()

##########################################################################################################

def creating_psn_labels(peaks,length,n):

	#Setting the P phase
	max_index_p = peaks[0]
	#Setting the S phase
	max_index_s = peaks[1]
	#print(f"P:{max_index_p}, S:{max_index_s}")

	#Obtaining the normal distribution
	norm_dist_p = norm.pdf(np.arange(max_index_p-n,max_index_p+n),loc=max_index_p,scale=10)
	norm_dist_s = norm.pdf(np.arange(max_index_s-n,max_index_s+n),loc=max_index_s,scale=10)

	#Normalizing the normal distribution
	norm_dist_p /= np.amax(norm_dist_p)
	norm_dist_s /= np.amax(norm_dist_s)

	#Creating the array P label
	label_p = np.zeros(length)
	#Creating the array S label
	label_s = np.zeros(length)
	#Creating the noise label array
	label_n = np.ones(length)

	#Putting the normal distribution for the P label array
	if(max_index_p + n < length):
		if(max_index_p > 0 and max_index_p - n > 0):
			try:
				label_p[max_index_p-n:max_index_p+n] = norm_dist_p
			except Exception:
				if((max_index_p-n) < 0):
					print("n is to high")
					#print(f"left: {max_index_p-n}, center: {max_index_p}, right:{max_index_p+n}")
		else:
			label_p = label_p
	else:
		max_limit = (length-1) - (max_index_p - n)
		#print(f"Length: {length}, max p: {max_index_p}, Max limit:{max_limit}")
		label_p[max_index_p-n:length-1] = norm_dist_p[:max_limit]

	#Putting the normal distribution in the S label array
	if(max_index_s + n < length):
		if(max_index_s > 0 and max_index_p - n > 0):
			label_s[max_index_s-n:max_index_s+n] = norm_dist_s

		else:
			label_s = label_s
	else:
		max_limit = (length-1) - (max_index_s - n)
		label_s[max_index_s-n:length-1] = norm_dist_s[:max_limit]

	# #Putting the normal distribution in the noise label array
	# noise_dist1 = norm_dist_p*(-1) + 1
	# noise_dist2 = norm_dist_s*(-1) + 1

	# if(max_index_p + n < length):

	# 	if(max_index_p > 0 and max_index_s > 0 and max_index_p - n > 0 and max_index_s - n > 0):

	# 		try:
	# 			label_n[max_index_p-n:max_index_p+n] = noise_dist1
	# 			label_n[max_index_s-n:max_index_s+n] = noise_dist2
	# 		except Exception:
	# 			print("Here")
	# 			print(f"left: {max_index_p-n}, center: {max_index_p}, right:{max_index_p+n}")
	# 			print(f"left: {max_index_s-n}, center: {max_index_s}, right:{max_index_s+n}")
			
	# 	else:
	# 		label_n = label_n
	# else:
	# 	max_limit_p = (length-1) - (max_index_p - n)
	# 	max_limit_s = (length-1) - (max_index_s - n)
	# 	label_n[max_index_p-n:length-1] = noise_dist1[:max_limit_p]
	# 	label_n[max_index_s-n:length-1] = noise_dist2[:max_limit_s]

	label_n = 1 - (label_p + label_s)

	return label_p, label_s, label_n

##########################################################################################################

def creating_sn_labels(peaks,length,n):

	#Setting the S phase
	max_index_s = peaks[0]

	#Obtaining the normal distribution
	norm_dist_s = norm.pdf(np.arange(max_index_s-n,max_index_s+n),loc=max_index_s,scale=10)

	#Normalizing the normal distribution
	norm_dist_s /= np.amax(norm_dist_s)

	#Creating the array P label
	label_p = np.zeros(length)
	#Creating the array S label
	label_s = np.zeros(length)
	#Creating the noise label array
	label_n = np.ones(length)

	#Putting the normal distribution in the S label array
	if(max_index_s + n < length):
		if(max_index_s > 0):
			label_s[max_index_s-n:max_index_s+n] = norm_dist_s
		else:
			label_s = label_s
	else:
		max_limit = (length-1) - (max_index_s - n)
		label_s[max_index_s-n:length-1] = norm_dist_s[:max_limit]

	#Putting the normal distribution in the noise label array
	noise_dist2 = norm_dist_s*(-1) + 1

	if(max_index_s + n < length):

		if(max_index_s > 0):

			label_n[max_index_s-n:max_index_s+n] = noise_dist2

		else:
			label_n = label_n
	else:
		max_limit_s = (length-1) - (max_index_s - n)
		label_n[max_index_s-n:length-1] = noise_dist2[:max_limit_s]

	return label_p, label_s, label_n

##########################################################################################################

def obtain_p_s_difference(channels,fs,labels_p,labels_s):

	ps_diff = []
	ps_p = []
	ps_s = []

	for channel in channels:

		label_p = labels_p[str(channel)]
		label_s = labels_s[str(channel)]
		maxp = np.amax(label_p); maxs = np.amax(label_s)

		ixp = np.where(label_p == maxp)[0][0]; ixs = np.where(label_s == maxs)[0][0]
		diff = (ixs/fs) - (ixp/fs)

		ps_diff.append(diff)
		ps_p.append(ixp/fs)
		ps_s.append(ixs/fs)

	return np.array(ps_diff), np.array(ps_p), np.array(ps_s)

##########################################################################################################

def labeling_stalta(data,channels,parameters):

	#Flag to obtain peaks
	available = False

	try:
		assert data.shape[0] == len(channels)
	except AssertionError:
		print(f"Mismatch between number of traces and number of channels")
		return;

	#Number of traces, length of each trace
	n_traces, length = data.shape[0], data.shape[1]

	#Size of the normal distribution
	n = 100

	#List of "labels"
	labels = []
	labels_p = {}
	labels_s = {}
	labels_n = {}

	#Parameters
	sta_length = parameters["STA"]
	lta_length = parameters["LTA"]
	scale_th = parameters["scale_th"]

	#New channels
	new_ch = []

	#Loop for create data labels
	for i,trace in enumerate(data):

		#Parameters for using HIGHPASS FILTERED DATA
		stalta = trigger.recursive_sta_lta(trace,sta_length,lta_length)

		#NORMALIZING WITH RESPECT TO THE MAXIMUM
		nstalta = stalta/np.amax(stalta)

		#Obtaining detect_peaks from P and S phases - FILTER DATA
		peaks = detect_peaks(nstalta,scale_th*stats.median_abs_deviation(nstalta),300)	

		#Appending into list of labels

		try:
			assert peaks.shape[0] <= 1
			available = True
		except AssertionError:

			#print(f"i: {i}, channels length: {len(channels)}")
			labels_p[str(channels[i])] = peaks[0]
			labels_s[str(channels[i])] = peaks[1]
			available = False

		if(available):

			try:
				labels_p[str(channels[i])] = peaks[0]
			except Exception:
				labels_p[str(channels[i])] = 0

			try:
				labels_s[str(channels[i])] = peaks[1]
			except Exception:
				labels_s[str(channels[i])] = 0

		new_ch.append(channels[i])
		#print(f"STALTA picking, reading {channels[i]} channel")

	return labels_p,labels_s,new_ch

#################################################################################################
#################################################################################################

def labeling_rms(data,channels,parameters):

	#print("Performing picking with RMS")

	#Parameters
	fs = parameters["fs"]
	l = parameters["length"]
	overlap = parameters["overlap"]
	th = parameters["threshold"]

	scale = fs*l*overlap

	if(parameters["envelope"]):

		sf_length = parameters["sf_length"]
		sf_order = parameters["sf_order"]

	#Number of traces, length of each trace
	n_traces, length = data.shape[0], data.shape[1]

	#Size of the normal distribution
	n = 50

	#List of "labels"
	all_rms = []
	indexes = []
	labels_p = {}
	labels_s = {}
	labels_n = {}

	#Waveforms where no RMS where present
	strange = {}

	#Flag
	obtained = True

	#Counter where RMS cannot be applied
	c_nposs = 0
	nposs = {}

	#Counter and list of successful labeling
	c_good = 0
	new_ch = []

	for i,trace in enumerate(data):

		#Estimating the RMS envelope
		if(parameters["envelope"]):
			rms = savgol_filter(rms_detector(trace,fs,l,overlap),sf_length,sf_order); all_rms.append(rms)
		else:
			rms = rms_detector(trace,fs,l,overlap); all_rms.append(rms)

		#Retrieving the obtain_picks
		index = np.array(obtain_picks(trace,rms,th)); indexes.append(index)

		if(index.shape[0] != 0):
			obtained = True
		else:
			obtained = False

		if(obtained):
			values = np.zeros(index.shape[0])

			for ch in range(0,index.shape[0]):

				values[ch] = rms[index[ch]]

			#Retrieving P and S picks
			p_pick = index[0]
			s_pick = np.where(rms == np.amax(rms))[0][0]
			
			if(p_pick != s_pick):

				try:
					labels_p[str(channels[i])] = int(p_pick*scale)
					labels_s[str(channels[i])] = int(s_pick*scale)

				except Exception:
					
					try:

						labels_p[str(channels[i])] = 0 
						labels_s[str(channels[i])] = int(s_pick*scale)

					except Exception:
						
						label_p[str(channels[i])] = 0
						label_s[str(channels[i])] = 0

			else:

				peaks = [s_pick*int(p_pick*scale)]
				try:

					labels_p[str(channels[i])] = 0
					labels_s[str(channels[i])] = int(s_pick*scale)

				except Exception:

					labels_p[str(channels[i])] = 0
					labels_s[str(channels[i])] = 0


		if not obtained:

			print("Waveform not readable")

			strange[str(channels[i])] = trace
			labels_p[str(channels[i])]= 0
			labels_s[str(channels[i])]= 0

		new_ch.append(channels[i])

	return labels_p,labels_s,new_ch

#################################################################################################

def labeling_kurtosis_skewness(data,channels,parameters):

	#Parameters Kurtosis
	fs = parameters["fs"]

	#Number of traces, length of each trace
	n_traces, length = data.shape[0], data.shape[1]

	#Size of the normal distribution
	n = 100

	#Threshold
	th = 0.20

	#List of "labels"
	all_srms = []
	all_kurt = []
	labels_p = {}
	labels_s = {}
	labels_n = {}

	#Waveforms where no RMS where present
	strange = {}

	#Counter where RMS cannot be applied
	c_nposs = 0
	nposs = {}

	#Counter and list of successful labeling
	c_good = 0
	new_ch = []

	#Flag
	available = False

	for i,trace in enumerate(data):

		#Estimating Kurtosis and Skewness
		#rms = rms_detector(trace,fs,l_rms,overlap_rms)
		skew = skew_detector(trace,fs,0.5,0.9)
		kurt = kurtosis_detector(trace,fs,0.5,0.9)
		kurt -= np.mean(kurt); kurt /= np.mean(kurt); np.nan_to_num(kurt,copy=False)
		peaks = detect_peaks(kurt,0.4,100)
		
		#Appending into list of labels

		try:
			assert peaks.shape[0] != 0
			available = True
		except AssertionError:

			#print(f"i: {i}, channels length: {len(channels)}")
			print(f"peaks: {peaks}")
			labels_p[str(channels[i])] = 0
			labels_s[str(channels[i])] = 0
			available = False

		if(available):

			try:
				labels_p[str(channels[i])] = int(peaks[0]*fs*0.5*0.9)
			except Exception:
				labels_p[str(channels[i])] = 0

			try:
				labels_s[str(channels[i])] = int(peaks[1]*fs*0.5*0.9)
			except Exception:
				labels_s[str(channels[i])] = 0

		new_ch.append(channels[i])

	return labels_p,labels_s,new_ch

#################################################################################################

def labeling_kurt_skew(data,parameters):

	#print("Performing picking with RMS/Kurtosis")

	#Parameters

	#Sampling rate
	fs = parameters["fs"]	

	#Kurtosis
	l_kurt = parameters["length_kurt"]
	overlap_kurt = parameters["overlap_kurt"]
	th_kurt = parameters["th_kurt"]

	#Skewness
	l_skew = parameters["length_skew"]
	overlap_skew = parameters["overlap_skew"]
	th_skew = parameters["th_skew"]

	scale_kurt = fs*l_kurt*overlap_kurt
	scale_skew = fs*l_skew*overlap_skew

	#Parameters Kurtosis

	#Number of traces, length of each trace
	n_traces, length = data.shape[0], data.shape[1]

	#Size of the normal distribution
	n = 100

	#Threshold
	th = 0.60

	#List of "labels"
	all_skew = []
	all_kurts = []
	labels_p = {}
	labels_s = {}
	labels_n = {}

	#Waveforms where no RMS where present
	strange = {}

	#Counter where RMS cannot be applied
	c_nposs = 0
	nposs = {}

	#Counter and list of successful labeling
	c_good = 0
	channels = []

	for i,trace in enumerate(data):

		#print(f"Channel no. {i}")

		#Estimating the autocorrelation - NO NEEDED
		#ac = scipy.signal.correlate(trace,trace)
		#nenv = np.abs(ac)

		#For filtering kurtosis and skewness
		bk, ak = scipy.signal.butter(2,5.0,"high",fs=100.0)

		#Estimating skewness 
		skew = skew_detector(trace,fs,l_skew,overlap_skew)

		#Taper for Skew CF
		tuk = scipy.signal.windows.tukey(skew.shape[0],alpha=0.1)
		fskew = scipy.signal.filtfilt(bk,ak,skew,axis=0)
		abs_skew = abs(fskew*tuk)/np.amax(abs(fskew*tuk))
		all_skew.append(abs_skew)

		#Estimating kurtosis
		kurt = kurtosis_detector(trace,fs,l_kurt,overlap_kurt)
		tuk = scipy.signal.windows.tukey(kurt.shape[0],alpha=0.1)
		fkurt = scipy.signal.filtfilt(bk,ak,kurt,axis=0)
		abs_kurt = abs(fkurt*tuk)/np.amax(abs(fkurt*tuk))
		all_kurts.append(abs_kurt)

		#Obtaining peaks
		peak_kurt = detect_peaks(abs_kurt,np.amax(abs_kurt)*0.3,100); #print(f"Channel: {i}, Peaks: {peak}")
		peak_skew = detect_peaks(abs_kurt,np.amax(abs_kurt)*0.3,100);
		#print(f"Kurt max: {kurt}")

		#Retrieving P and S picks

		#p_pick = np.where(kurt == np.amax(kurt))[0][0]
		#s_pick = np.where(srms == np.amax(srms))[0][0]

		if(peak_kurt.shape[0] == peak_skew.shape[0]):

			diff_p = peak_kurt[0] - peak_skew[0]
			diff_s = peak_kurt[1] - peak_skew[1]

		elif(peak_kurt.shape[0] == 1 or peak_kurt.shape[0] == 1):

			diff_s = peak_kurt[0] - peak_skew[0]

		else:

			pass


		#if()

		try:

			p_pick = peak[0]
			s_pick = np.where(srms == np.amax(srms))[0][0]

		except Exception:

			p_pick = peak[0]
			s_pick = np.where(srms == np.amax(srms))[0][0]
			
		if(p_pick != s_pick):

			peaks = [int(p_pick*scale_kurt),int(s_pick*scale_rms)]
			#print(f"Peaks: {peaks}")
			try:
				label_p, label_s, label_n = creating_psn_labels(peaks,length,n)

			except Exception:
				print("Not possible this")
				c_nposs += 1
				nposs[str(i)] = trace
				label_p, label_s, label_n = np.zeros(trace.shape[0]), np.zeros(trace.shape[0]), np.ones(trace.shape[0])

			labels_p[str(i)]=label_p
			labels_s[str(i)]=label_s
			labels_n[str(i)]=label_n
			c_good += 1
			channels.append(i)

		else:

			peaks = [s_pick*int(p_pick*scale_rms)]
			try:
				label_p, label_s, label_n = creating_sn_labels(peaks,length,n)
			except Exception:
				print("Not possible that")
				c_nposs += 1
				nposs[str(i)] = trace
				label_p, label_s, label_n = np.zeros(trace.shape[0]), np.zeros(trace.shape[0]), np.ones(trace.shape[0])
			
			labels_p[str(i)]=label_p
			labels_s[str(i)]=label_s
			labels_n[str(i)]=label_n
			c_good += 1
			channels.append(i)

				#print("Waveform not readable")

				#strange[str(i)] = trace
				#labels_p[str(i)]=np.zeros(trace.shape[0])
				#labels_s[str(i)]=np.zeros(trace.shape[0])
				#labels_n[str(i)]=np.ones(trace.shape[0])

	#print("Done")

	return labels_p, labels_s, labels_n, channels, all_srms, all_kurt	

##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################

def obtain_ar_picks(trace,fs,f1,f2,lta_p,sta_p,lta_s,sta_s,a,b,std_p,std_s):

	st = obspy.core.stream.Stream()

	p_pick, s_pick = trigger.ar_pick(trace,trace,trace,fs,f1,f2,lta_p,sta_p,lta_s,sta_s,a,b,std_p,std_s)

	return p_pick,s_pick

##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################

def labeling_ar(data,channels,parameters):

	#print("Performing picking with AR - AIC")

	#Number of traces, length of each trace
	n_traces, length = data.shape[0], data.shape[1]

	#Size of the normal distribution
	n = 100

	#List of "labels"
	labels = []
	labels_p = {}
	labels_s = {}
	labels_n = {}

	#Parameters

	fs = parameters["fs"]
	f1 = parameters["f1"]
	f2 = parameters["f2"]
	lta_p = parameters["lta_p"]
	sta_p = parameters["sta_p"]
	lta_s = parameters["lta_s"]
	sta_s = parameters["sta_s"]
	a = parameters["a"]
	b = parameters["b"]
	std_p = parameters["std_p"]
	std_s = parameters["std_s"]

	new_ch = []

	#Loop for create data labels
	for i,trace in enumerate(data):

		p_pick, s_pick = obtain_ar_picks(trace,fs,f1,f2,lta_p,sta_p,lta_s,sta_s,a,b,std_p,std_s)
		#print(f"P pick: {(p_pick)}, S pick: {(s_pick)}")

		#Appending into list of labels

		try:

			assert p_pick != 0
			available_p = True

		except AssertionError:

			#print(f"i: {i}, channels length: {len(channels)}")
			labels_p[str(channels[i])] = 0
			available_p = False

		try:

			assert s_pick != 0
			available_s = True

		except AssertionError:

			labels_s[str(channels[i])] = 0
			available_s = False

		if(available_p):

			try:
				labels_p[str(channels[i])] = int(p_pick*fs)
			except Exception:
				print(f"i: {i}, ")
				labels_p[str(channels[i])] = 0

		if(available_s):

			try:
				labels_s[str(channels[i])] = int(s_pick*fs)
			except Exception:
				labels_s[str(channels[i])] = 0

		new_ch.append(channels[i])

		available_p = False
		available_s = False

	return labels_p,labels_s,new_ch

##################################################################################################################################################################################################################################################

# def labeling_ar(data,parameters):

# 	#print("Performing picking with AR - AIC")

# 	#Number of traces, length of each trace
# 	n_traces, length = data.shape[0], data.shape[1]

# 	#Size of the normal distribution
# 	n = 100

# 	#Channels list
# 	channels = []

# 	#List of "labels"
# 	labels = []
# 	labels_p = {}
# 	labels_s = {}
# 	labels_n = {}

# 	#Parameters

# 	fs = parameters["fs"]
# 	f1 = parameters["f1"]
# 	f2 = parameters["f2"]
# 	lta_p = parameters["lta_p"]
# 	sta_p = parameters["sta_p"]
# 	lta_s = parameters["lta_s"]
# 	sta_s = parameters["sta_s"]
# 	a = parameters["a"]
# 	b = parameters["b"]
# 	std_p = parameters["std_p"]
# 	std_s = parameters["std_s"]

# 	#Loop for create data labels
# 	for i,trace in enumerate(data):

# 		p_pick, s_pick = obtain_ar_picks(trace,fs,f1,f2,lta_p,sta_p,lta_s,sta_s,a,b,std_p,std_s)
# 		#print(f"P pick: {(p_pick)}, S pick: {(s_pick)}")

# 		if(p_pick != 0 and s_pick != 0):

# 			label_p, label_s, label_n = creating_psn_labels([int(p_pick*fs),int(s_pick*fs)],length,n)
# 			channels.append(i)	

# 		else:

# 			label_p = np.zeros(length)
# 			label_s = np.zeros(length)
# 			label_n = np.ones(length)
# 			channels.append(i)		

# 		#Appending into list of labels
# 		labels_p[str(i)] = label_p
# 		labels_s[str(i)] = label_s
# 		labels_n[str(i)] = label_n

# 	#print("Done")

# 	return labels_p,labels_s,labels_n, channels

##################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################

def labeling_kurtSkew(data,parameters):

	#print("Performing picking with AR - AIC")

	#Number of traces, length of each trace
	n_traces, length = data.shape[0], data.shape[1]

	#Size of the normal distribution
	n = 100

	#Channels list
	channels = []

	#List of "labels"
	labels = []
	labels_p = {}
	labels_s = {}
	labels_n = {}

	#Parameters

	fs = parameters["fs"]
	f1 = parameters["f1"]
	f2 = parameters["f2"]
	lta_p = parameters["lta_p"]
	sta_p = parameters["sta_p"]
	lta_s = parameters["lta_s"]
	sta_s = parameters["sta_s"]
	a = parameters["a"]
	b = parameters["b"]
	std_p = parameters["std_p"]
	std_s = parameters["std_s"]

	#Loop for create data labels
	for i,trace in enumerate(data):

		p_pick, s_pick = obtain_ar_picks(trace,fs,f1,f2,lta_p,sta_p,lta_s,sta_s,a,b,std_p,std_s)
		#print(f"P pick: {(p_pick)}, S pick: {(s_pick)}")

		if(p_pick != 0 and s_pick != 0):

			label_p, label_s, label_n = creating_psn_labels([int(p_pick*fs),int(s_pick*fs)],length,n)
			channels.append(i)	

		else:

			label_p = np.zeros(length)
			label_s = np.zeros(length)
			label_n = np.ones(length)
			channels.append(i)		

		#Appending into list of labels
		labels_p[str(i)] = label_p
		labels_s[str(i)] = label_s
		labels_n[str(i)] = label_n

	#print("Done")

	return labels_p,labels_s,labels_n, channels

################################################################################################################

def evaluate_peaks(list_picks_p,list_picks_s,fs,length,n,th=100):

	#Evaluation for P
	n_methods = len(list_picks_p)
	methods = ["stalta","rms","kurtosis","ar-aic"]
	#methods = ["stalta","rms","ar-aic"]

	#Order of inputs: stalta, rms, kurtosis, skewness, ar-aic
	n_peaksp = {}
	for n in range(n_methods):

		method = list_picks_p[n]
		n_pos = np.where(method == 1.0)[0]
		if(n_pos.shape[0] > 0):

			n_peaksp[methods[n]] = n_pos[0]

	#Evaluation for S
	#Order of inputs: stalta, rms, kurtosis, skewness, ar-aic
	n_peakss = {}
	for n in range(n_methods):

		method = list_picks_s[n]
		n_pos = np.where(method == 1.0)[0]
		if(n_pos.shape[0] > 0):

			n_peakss[methods[n]] = n_pos[0]

	if(len(n_peaksp) > 0):

		if(len(n_peaksp) == 1 and len(n_peakss) == 1):

			keyp = [key for key in n_peaksp.keys()]; posp = n_peaksp[keyp[0]]
			keys = [key for key in n_peakss.keys()]; poss = n_peakss[keys[0]]

			#Retrieving creating label
			label_p, label_s, label_n = creating_psn_labels([int(posp),int(poss)],length,n)
			#print(f"Posp in s: {int(posp*fs)}, Label p mean: {np.mean(label_p)}")

		elif(len(n_peaksp) > 1 and len(n_peakss) == 1):

			keyp = [key for key in n_peaksp.keys()];
			n_values = np.zeros(len(keyp))

			for idx,key in enumerate(keyp):

				n_values[idx] = n_peaksp[key]

			diff = abs(np.diff(n_values))
			#print(f"val: {n_values}, diff: {diff}, mean:{abs(np.mean(diff))}")
			if(np.mean(diff) < th):
				
				posp = int(np.mean(n_values))
				#print(f"P > 1; posp: {posp}")

			else:

				keyp = [key for key in n_peaksp.keys()];
				posp = n_peaksp[keyp[0]]

			keys = [key for key in n_peakss.keys()]; poss = n_peakss[keys[0]]

			#Retrieving creating label
			label_p, label_s, label_n = creating_psn_labels([int(posp),int(poss)],length,n)
			#print(f"Posp in s: {int(posp*fs)}, Label p mean: {np.mean(label_p)}")

		elif(len(n_peaksp) == 1 and len(n_peakss) > 1):

			keys = [key for key in n_peakss.keys()];
			n_values = np.zeros(len(keys))

			for idx,key in enumerate(keys):

				n_values[idx] = n_peakss[key]

			diff = abs(np.diff(n_values))
			#print(f"val: {n_values}, diff: {diff}, mean:{abs(np.mean(diff))}")
			if(np.mean(diff) < th):

				poss = int(np.mean(n_values))
				#print(f"S > 1; poss: {poss}")

			else:

				keys = [key for key in n_peakss.keys()];
				poss = n_peakss[keys[0]]

			keys = [key for key in n_peaksp.keys()]; posp = n_peaksp[keys[0]]

			#Retrieving creating label
			label_p, label_s, label_n = creating_psn_labels([int(posp),int(poss)],length,n)
			#print(f"Posp in s: {int(posp*fs)}, Label p mean: {np.mean(label_p)}")

		elif(len(n_peaksp) > 1 and len(n_peakss) > 1):

			keys = [key for key in n_peakss.keys()];
			n_values = np.zeros(len(keys))

			for idx,key in enumerate(keys):

				n_values[idx] = n_peakss[key]

			diff = abs(np.diff(n_values))
			#print(f"val: {n_values}, diff: {diff}, mean:{abs(np.mean(diff))}")
			if(np.mean(diff) < th):

				poss = int(np.mean(n_values))
				#print(f"P > 1 y S > 1; poss: {poss}")

			else:

				keys = [key for key in n_peakss.keys()];
				poss = n_peakss[keys[0]]

			keyp = [key for key in n_peaksp.keys()];
			n_values = np.zeros(len(keyp))

			for idx,key in enumerate(keyp):

				n_values[idx] = n_peaksp[key]

			diff = abs(np.diff(n_values))
			#print(f"val: {n_values}, diff: {diff}, mean:{abs(np.mean(diff))}")
			if(np.mean(diff) < th):

				posp = int(np.mean(n_values))
				#print(f"P > 1 y S > 1; posp: {posp}")

			else:

				keyp = [key for key in n_peaksp.keys()];
				posp = n_peaksp[keyp[0]]

			#Retrieving creating label
			#print(f"posp: {posp}, poss: {poss}")
			label_p, label_s, label_n = creating_psn_labels([int(posp),int(poss)],length,n)
			#print(f"Posp in s: {int(posp)}, Label p mean: {np.mean(label_p)}")
		
	if(len(n_peakss) == 0 or len(n_peaksp) == 0):

		label_p = np.zeros(length)
		label_s = np.zeros(length)
		label_n = np.ones(length)

	#print(f"Len p: {len(n_peaksp)} Len s: {len(n_peakss)}")
	#print(f"labelp: {label_p}")
	
	return label_p,label_s,label_n

#################################################################################################################
######################################################################################################################################

def plotting_select_two(path_images,path_csv,name_file,name_csv,channels,data,all_rmsk,all_kurt,p_stalta,p_rms,p_rk,p_ar,fs,length,array_seconds,ch_idx,idx,date,n):

	#print(f"Plotting...")
	plot = True

	#f = open(f"{path_csv}/{name_csv}","a")

	ch = []
	p_picks = []
	s_picks = []

	for channel in channels:

		#Index
		trace = data[channel,:]
		time = np.arange(0,trace.shape[0])/fs

		#Picks STA/LTA
		stlabel_p = p_stalta[0][str(channel)]; stlabel_s = p_stalta[1][str(channel)]; stlabel_n = p_stalta[2][str(channel)]

		#Picks RMS
		try:
			rlabel_p = p_rms[0][str(channel)]; rlabel_s = p_rms[1][str(channel)]; rlabel_n = p_rms[2][str(channel)]
		except KeyError:
			#print(f"Picks RMS: {p_rms}")
			rlabel_p = 0; rlabel_s = 0; rlabel_n = 0

		#Picks RMS-Kurtosis
		rklabel_p = p_rk[0][str(channel)]; rklabel_s = p_rk[1][str(channel)]; rklabel_n = p_rk[2][str(channel)]

		#Picks AR - AIC
		arlabel_p = p_ar[0][str(channel)]; arlabel_s = p_ar[1][str(channel)]; arlabel_n = p_ar[2][str(channel)]

		label_p,label_s,label_n = evaluate_peaks([stlabel_p,rlabel_p,rklabel_p,arlabel_p],[stlabel_s,rlabel_s,rklabel_s,arlabel_s],fs,length,n)
		#label_p,label_s,label_n = evaluate_peaks([stlabel_p,rlabel_p,arlabel_p],[stlabel_s,rlabel_s,arlabel_s],fs,length,n)

		try:
			posp = np.where(stlabel_p == 1.0)[0][0]; poss = np.where(stlabel_s == 1.0)[0][0]
			ch.append(int(ch_idx[channel]))
			p_picks.append(posp/fs + array_seconds[channel])
			s_picks.append(poss/fs + array_seconds[channel])
		except Exception:
			#print(f"Max p:{np.amax(label_p)}")
			posp = 0
			poss = 0

		#Picks average

		name_image = f"Mean_Phases_{name_file}_Channel_{channel}_{idx}.png"

		p_pick = int((posp/fs + array_seconds[channel])*fs)
		s_pick = int((poss/fs + array_seconds[channel])*fs)

		#DateTime Channel p_pick(s) s_pick(s) p_pick(sample) s_pick(sample) p_std s_std
		#f.write(f"{date} {channel} {round(posp/fs + array_seconds[channel],3)} {round(poss/fs + array_seconds[channel],3)} {p_pick} {s_pick} {round((posp/fs + array_seconds[channel])*0.1,3)} {round((poss/fs + array_seconds[channel])*0.1,3)}\n")

		if(np.mean(label_p) > 0 and np.mean(label_s) > 0):
		
			plot = True
		
		if(plot):

			#Plotting
			fig, ax = plt.subplots(4,1,figsize=(20,10))
			ax[0].plot(time,trace,linewidth=0.7,c="k")
			ax[0].set_xlim(time[0],time[-1])
			#ax[0].set_xlim(10,15)
			ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
			ax[0].tick_params(axis="x",color="k",labelsize=14.0)
			ax[0].tick_params(axis="y",color="k",labelsize=14.0)

			f,t,s = signal.spectrogram(trace,fs,nperseg=64,noverlap=60)
			c=ax[2].imshow(10*np.log10(s),origin="lower",aspect="auto",interpolation="lanczos",cmap="jet",extent=(t[0],t[-1],f[0],f[-1]),vmin=-225,vmax=-125)
			#fig.colorbar(c,ax=ax[1],location="right")

			envelope = (abs(np.gradient(np.sum(s,axis=0)))*1e7)*2
			times_env = np.arange(0,envelope.shape[0])*(t[1]-t[0])
			mean_envelope = np.mean(envelope)
			mad_envelope = stats.median_abs_deviation(envelope)
			peaks = detect_peaks(envelope,2*mad_envelope,200)
			#print(f"times: {times_env[0]}, {times_env[-1]}")
			#print(f"Peaks: {peaks}")
			mean_arr = np.ones(t.shape[0])*mean_envelope
			mad_arr = np.ones(t.shape[0])*mad_envelope

			try:
				new_idx = np.where(envelope > mad_envelope)[0][0]
				ax[0].plot([times_env[peaks[0]],times_env[peaks[0]]],[-2e-6,2e-6],lw=2.0,c="g")
				ax[0].plot([times_env[peaks[1]],times_env[peaks[1]]],[-2e-6,2e-6],lw=2.0,c="g")
				ax[0].plot([times_env[peaks[2]],times_env[peaks[2]]],[-2e-6,2e-6],lw=2.0,c="g")
			except Exception:
				print("No MAD value")


			ax[1].plot(t,envelope,lw=0.7,c="b")
			ax[1].plot(t,mean_arr,lw=0.7,c="r",linestyle="dashed")
			ax[1].plot(t,2*mad_arr,lw=0.7,c="g",linestyle="dashed")
			ax[1].set_xlim(t[0],t[-1])
			#ax[1].set_xlim(10,15)
			
			#ax[3].plot(time,label_p,linewidth=0.7,c="r",label="P")
			#ax[3].plot(time,label_s,linewidth=0.7,c="b",label="S")

			ax[3].plot(time,stlabel_p,linewidth=0.7,c="r",label="P_STALTA",linestyle="dotted")
			ax[3].plot(time,stlabel_s,linewidth=0.7,c="b",label="S_STALTA",linestyle="dotted")

			ax[3].plot(time,rklabel_p,linewidth=0.7,c="r",label="KS",linestyle="dashed")
			ax[3].plot(time,rklabel_s,linewidth=0.7,c="b",label="KS",linestyle="dashed")

			#ax[3].plot(time,rlabel_p,linewidth=0.7,c="r",label="P_RMS",linestyle="dashed")
			#ax[3].plot(time,rlabel_s,linewidth=0.7,c="b",label="S_RMS",linestyle="dashed")

			ax[3].plot(time,arlabel_p,linewidth=0.7,c="r",label="P_AR",linestyle="dashdot")
			ax[3].plot(time,arlabel_s,linewidth=0.7,c="b",label="S_AR",linestyle="dashdot")


			ax[3].set_xlim(time[0],time[-1])

			#ax[1].plot(time,avlabel_p,linewidth=2.0,c="r",label="AV",linestyle="solid")
			#ax[1].plot(time,avlabel_s,linewidth=2.0,c="b",label="AV",linestyle="solid")

			ax[3].set_xlim(time[0],time[-1])
			ax[3].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
			ax[3].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
			ax[3].tick_params(axis="x",color="k",labelsize=14.0)
			ax[3].tick_params(axis="y",color="k",labelsize=14.0)

			ax[3].legend(loc="upper right")

			try:
				plt.savefig(f"{path_images}/{date}/{name_image}")
			except FileNotFoundError:
				os.mkdir(f"{path_images}/{date}")
				plt.savefig(f"{path_images}/{date}/{name_image}")

			plt.close()

	#f.close()

	return ch, p_picks, s_picks

######################################################################################################################################

def plotting_select(path_images,path_csv,name_file,name_csv,channels,data,p_stalta,p_rms,p_ar,fs,length,array_seconds,ch_idx,idx,date,n):

	#print(f"Plotting...")
	plot = False

	#f = open(f"{path_csv}/{name_csv}","a")

	ch = []
	p_picks = []
	s_picks = []

	for channel in channels:

		#Index
		trace = data[channel,:]
		time = np.arange(0,trace.shape[0])/fs

		#Picks STA/LTA
		stlabel_p = p_stalta[0][str(channel)]; stlabel_s = p_stalta[1][str(channel)]; stlabel_n = p_stalta[2][str(channel)]

		#Picks RMS
		rlabel_p = p_rms[0][str(channel)]; rlabel_s = p_rms[1][str(channel)]; rlabel_n = p_rms[2][str(channel)]

		#Picks AR - AIC
		arlabel_p = p_ar[0][str(channel)]; arlabel_s = p_ar[1][str(channel)]; arlabel_n = p_ar[2][str(channel)]

		#label_p,label_s,label_n = evaluate_peaks([stlabel_p,rlabel_p,rklabel_p,arlabel_p],[stlabel_s,rlabel_s,rklabel_s,arlabel_s],fs,length,n)
		label_p,label_s,label_n = evaluate_peaks([stlabel_p,rlabel_p,arlabel_p],[stlabel_s,rlabel_s,arlabel_s],fs,length,n)

		try:
			posp = np.where(stlabel_p == 1.0)[0][0]; poss = np.where(stlabel_s == 1.0)[0][0]
			ch.append(int(ch_idx[channel]))
			p_picks.append(posp/fs + array_seconds[channel])
			s_picks.append(poss/fs + array_seconds[channel])
		except Exception:
			#print(f"Max p:{np.amax(label_p)}")
			posp = 0
			poss = 0

		#Picks average

		name_image = f"Mean_Phases_{name_file}_Channel_{channel}_{idx}.pdf"

		p_pick = int((posp/fs + array_seconds[channel])*fs)
		s_pick = int((poss/fs + array_seconds[channel])*fs)

		#DateTime Channel p_pick(s) s_pick(s) p_pick(sample) s_pick(sample) p_std s_std
		#f.write(f"{date} {channel} {round(posp/fs + array_seconds[channel],3)} {round(poss/fs + array_seconds[channel],3)} {p_pick} {s_pick} {round((posp/fs + array_seconds[channel])*0.1,3)} {round((poss/fs + array_seconds[channel])*0.1,3)}\n")

		if(np.mean(label_p) > 0 and np.mean(label_s) > 0):
		
			plot = False
		
		if(plot):

			#Plotting
			fig, ax = plt.subplots(2,1,figsize=(20,10))
			ax[0].plot(time,trace,linewidth=0.7,c="k")
			ax[0].set_xlim(time[0],time[-1])
			ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
			ax[0].tick_params(axis="x",color="k",labelsize=14.0)
			ax[0].tick_params(axis="y",color="k",labelsize=14.0)
			
			ax[1].plot(time,label_p,linewidth=0.7,c="r",label="P")
			ax[1].plot(time,label_s,linewidth=0.7,c="b",label="S")

			ax[1].plot(time,stlabel_p,linewidth=0.7,c="r",label="P_STALTA",linestyle="dotted")
			ax[1].plot(time,stlabel_s,linewidth=0.7,c="b",label="S_STALTA",linestyle="dotted")

			ax[1].plot(time,rlabel_p,linewidth=0.7,c="r",label="P_RMS",linestyle="dashed")
			ax[1].plot(time,rlabel_s,linewidth=0.7,c="b",label="S_RMS",linestyle="dashed")

			ax[1].plot(time,stlabel_p,linewidth=0.7,c="r",label="P_AR",linestyle="dashdot")
			ax[1].plot(time,stlabel_s,linewidth=0.7,c="b",label="S_AR",linestyle="dashdot")


			ax[1].set_xlim(time[0],time[-1])

			#ax[1].plot(time,avlabel_p,linewidth=2.0,c="r",label="AV",linestyle="solid")
			#ax[1].plot(time,avlabel_s,linewidth=2.0,c="b",label="AV",linestyle="solid")

			ax[1].set_xlim(time[0],time[-1])
			ax[1].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
			ax[1].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
			ax[1].tick_params(axis="x",color="k",labelsize=14.0)
			ax[1].tick_params(axis="y",color="k",labelsize=14.0)

			ax[1].legend(loc="upper right")

			try:
				plt.savefig(f"{path_images}/{date}/{name_image}")
			except FileNotFoundError:
				os.mkdir(f"{path_images}/{date}")
				plt.savefig(f"{path_images}/{date}/{name_image}")

			plt.close()

	#f.close()

	return ch, p_picks, s_picks

##########################################################################################################

def plotDASchannels(path,list_parameters,data,name,snr,extra_name=None):

	c2, c1 = signal.butter(8,(5.0,15.0),"band",fs=100.0)
	fdata = signal.filtfilt(c2,c1,data,axis=1)

	dx,fs = list_parameters[0],list_parameters[1]

	nx, nt = fdata.shape[0], fdata.shape[1]
	x = np.arange(nx) * 16.33
	t = np.arange(nt) / fs

	fdata = fdata*1e6

	fig, ax = plt.subplots(1,1, figsize=(20,10))
	c = ax.imshow(fdata.T, cmap="seismic", vmin=-np.percentile(fdata,97), vmax=np.percentile(fdata,97),aspect="auto", interpolation="none",extent=[x[0],x[-1],t[-1],t[0]],origin="upper")
	#c = ax.imshow(new_das, cmap="seismic", vmin=-np.percentile(new_das,99), vmax=np.percentile(new_das,99),aspect="auto", interpolation="none",origin="upper")
	cbar = plt.colorbar(c)
	cbar.set_label("Strain rate (μstrain/s)", fontsize=16.0)
	ax.set_title(f"Kamaishi - {UTCDateTime(name)} SNR: {snr}")
	ax.set_xlabel("Length (m)",fontsize=16.0)
	ax.set_ylabel("Time (s)",fontsize=16.0)
	if(extra_name is not None):
		plt.savefig(f"{path}/{name}_{extra_name}_img.pdf")
	else:
		plt.savefig(f"{path}/{name}_img.png")
	plt.tight_layout()
	plt.close()

##########################################################################################################

def plotting2D(path,list_parameters,data,name,channel_list,p_picks,s_picks,snr,extra_name=None,type="pdf",chidx=False):

	if(len(channel_list) == 1):
		channelp = channel_list[0]
		channels = channel_list[0]
	else:
		channelp = channel_list[0]
		channels = channel_list[1]

	st_data = data.T

	c2, c1 = signal.butter(2,(5.0,15.0),"band",fs=100.0)
	fdata = signal.filtfilt(c2,c1,st_data,axis=0)
	fdata = signal.filtfilt(c2,c1,fdata,axis=0)

	dx,fs = list_parameters[0],list_parameters[1]
	nx, nt = st_data.shape[1], st_data.shape[0]

	#print(f"nx: {nx}, nt: {nt}")
	#print(f"x: {x[-1]}, t: {t[-1]}")
	if(chidx):
		x = np.arange(nx)
		channelp = np.array(channelp)
		channels = np.array(channels)

	else:
		x = np.arange(nx) * dx	
		channelp = np.array(channelp)*dx
		channels = np.array(channels)*dx

	t = np.arange(nt) / fs
		
	fdata = fdata*1e6

	fig, ax = plt.subplots(1,1, figsize=(20,10))
	c = ax.imshow(fdata, cmap="seismic", vmin=-np.percentile(fdata,97), vmax=np.percentile(fdata,97),aspect="auto", interpolation="none",extent=[x[0],x[-1],t[-1],t[0]],origin="upper")
	#c = ax.imshow(new_das, cmap="seismic", vmin=-np.percentile(new_das,99), vmax=np.percentile(new_das,99),aspect="auto", interpolation="none",origin="upper")
	ax.scatter(channelp,p_picks,s=20,c="r")
	ax.scatter(channels,s_picks,s=20,c="b")
	cbar = plt.colorbar(c)
	cbar.set_label("Strain rate (μstrain/s)", fontsize=16.0)
	ax.set_title(f"Kamaishi - {UTCDateTime(name)} SNR: {snr}")
	ax.set_xlabel("Length (m)",fontsize=16.0)
	ax.set_ylabel("Time (s)",fontsize=16.0)
	#ax.set_ylim(100,0)

	#ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
	#ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
	if(extra_name is not None):
		plt.savefig(f"{path}/{name}_{extra_name}_img.{type}")
	else:
		plt.savefig(f"{path}/{name}_img.{type}")
	plt.tight_layout()
	plt.close()

##########################################################################################################

def estimate_snr(data):

	#print(f"Original shape: {original.shape}")
	original = data
	
	c2, c1 = signal.butter(2,(5.0,15.0),"band",fs=100.0)
	filtered = signal.filtfilt(c2,c1,original,axis=1)

	noise = original - filtered

	psignal = np.sum(filtered**2,axis=1)
	pnoise = np.sum(noise**2,axis=1)

	snr = 10*np.log10(psignal/pnoise)

	return snr

##########################################################################################################

def add_data(original_data,detections,start_index,end_index,channel_index,fs,size=10):

	stidx = start_index
	eidx = end_index
	chidx = channel_index
	samples = int(size*fs)

	data_size,data_sh = detections.shape[0],detections.shape[1]
	#print(f"Size: {data_size}, Shape: {data_sh}")

	new_data = np.zeros((data_size,data_sh+samples))
	#print(f"New data shape: {new_data.shape}")

	#print(f"Channel size: {chidx.shape}")

	if(chidx.shape[0] > data_size):

		chidx = chidx[:data_size]

	for idx,channel in enumerate(chidx):

		startsample = stidx[idx]
		endsample = eidx[idx]

		try:
			new_trace = original_data[channel,startsample-samples:endsample]
		except Exception:
			new_trace = original_data[channel,startsample:endsample+samples]

		new_data[idx,:] = new_trace

	return new_data

##########################################################################################################

def retrieve_centered_traces(channel_index,original_data,fs):

	#RMS
	fs = 100.0
	l = 0.5
	ov = 0.1
	th_rms = 0.6
	sf_length = 30
	sf_ord = 3
	scale = fs*l*ov

	oszie, oshape = original_data.shape[0],original_data.shape[1]
	#print(f"odata.shape: {oshape}")

	#Lengths to retrieve (right and left)
	len_right = 7; slen_right = int(len_right*fs)
	len_left = 23; slen_left = int(len_left*fs) 
	hlen = len_right + len_left
	hlen_samples = int(hlen*fs)

	#ch_rms = np.zeros((channel_index.shape[0],2))

	c2, c1 = signal.butter(2,(5.0,15.0),"band",fs=100.0)
	data = signal.filtfilt(c2,c1,original_data,axis=1)

	low_point = []
	up_point = []

	new_traces = np.zeros((channel_index.shape[0],hlen_samples))
	new_otraces = np.zeros((channel_index.shape[0],hlen_samples))

	for idx,ch in enumerate(channel_index):

		trace = data[ch,:]
		otrace = original_data[ch,:]		
		rms = rms_detector(trace,fs,l,ov)
		max_rms = np.where(rms == np.amax(rms))[0][0]

		#Storing maximum RMS
		#ch_rms[idx,0] = ch; ch_rms[idx,1] = max_rms

		#Maximum RMS sample position  
		max_rms_sample = int(max_rms*scale)

		try:
			new_traces[idx,:] = trace[max_rms_sample-slen_left:max_rms_sample+slen_right]
			new_otraces[idx,:] = otrace[max_rms_sample-slen_left:max_rms_sample+slen_right]
			low_point.append(max_rms_sample-slen_left)
			up_point.append(max_rms_sample+slen_right)
		except Exception:
			if(max_rms_sample-slen_left < 0):

				#print(f"Resetting half length to the right: {max_rms_sample-slen_left}, max_rms_sample: {max_rms_sample}, new traces shape: {slen_left+slen_right}")
				#print(f"New shape: {trace[max_rms_sample:max_rms_sample+(slen_left+slen_right)].shape}")

				new_traces[idx,:] = trace[max_rms_sample:abs(max_rms_sample+slen_left+slen_right)]
				new_otraces[idx,:] = otrace[max_rms_sample:abs(max_rms_sample+slen_left+slen_right)]
				low_point.append(max_rms_sample)
				up_point.append(max_rms_sample+slen_left+slen_right)

			elif(max_rms_sample+slen_right > oshape):
				#print(f"Resetting half length to the left: {max_rms_sample+slen_right},slenright: {slen_right} max_rms_sample: {max_rms_sample}")
				excedent = (max_rms_sample+slen_right) - oshape
				remanent = oshape - max_rms_sample
				new_traces[idx,:] = trace[max_rms_sample-(slen_left+excedent):max_rms_sample+remanent]
				new_otraces[idx,:] = otrace[max_rms_sample-(slen_left+excedent):max_rms_sample+remanent]
				low_point.append(max_rms_sample-(slen_left+excedent))
				up_point.append(max_rms_sample+remanent)


	return new_traces, new_otraces, np.array(low_point), np.array(up_point)


##########################################################################################################################################################################

def plotting_histograms(path,name,p_picks,s_picks,fs,delta = None):

	if(delta is not None):

		tstp = delta

	else:

		tstp = abs(s_picks-p_picks)/fs

	fig, ax = plt.subplots(1,3,figsize=(20,10))
	ax[0].hist(p_picks/fs,bins=5,color="b")
	ax[0].set_xlabel("$t_{s} - t_{p}$ (s)",fontsize=16.0)
	ax[0].set_ylabel("Counts",fontsize=16.0)
	ax[1].hist(s_picks/fs,bins=5,color="b")
	ax[1].set_xlabel("$t_{s} - t_{p}$ (s)",fontsize=16.0)
	ax[1].set_ylabel("Counts",fontsize=16.0)
	ax[2].hist(tstp,bins=5,color="b")
	ax[2].set_xlabel("$t_{s} - t_{p}$ (s)",fontsize=16.0)
	ax[2].set_ylabel("Counts",fontsize=16.0)
	plt.savefig(f"{path}/{name}_img.png")
	plt.tight_layout()
	plt.close()

############################################################################################################################################################################

def correct_picks(ch_idx,p_picks,s_picks,name,snr,dx):

	#Convert to array
	parray = np.array(p_picks)
	#ppos = np.arange(0,parray.shape[0])
	ppos = np.array(ch_idx)
	sarray = np.array(s_picks)
	spos = np.array(ch_idx)
	#spos = np.arange(0,sarray.shape[0])

	#Obtain slopes 
	chpdiff = np.diff(ppos)
	chsdiff = np.diff(spos)
	pdiff = np.diff(p_picks)
	sdiff = np.diff(s_picks)
	mpnan = (pdiff/chpdiff); msnan = (sdiff/chsdiff)
	mp = np.nan_to_num(mpnan); ms = np.nan_to_num(msnan)
	distp = np.sqrt(pdiff**2 + chpdiff**2)
	dists = np.sqrt(sdiff**2 + chsdiff**2)

	pmean = np.mean(mp); smean = np.mean(ms)

	new_channelsp = []; new_channelss = [];
	newchp = []; newchs = []
	newp = []; news = []

	fig, ax = plt.subplots()

	for idx in range(0,mp.shape[0]):

		if(ms[idx] <= smean):
			new_channelss.append(ch_idx[idx]); newchs.append(ppos[idx]);news.append(sarray[idx])

		if(mp[idx] <= pmean):
			new_channelsp.append(ch_idx[idx]); newchp.append(spos[idx]);newp.append(parray[idx])

	chpdiff = np.diff(np.array(newchp))
	chsdiff = np.diff(np.array(newchs))
	pdiff = np.diff(np.array(newp))
	sdiff = np.diff(np.array(news))
	slopep = (pdiff/chpdiff); slopes = (sdiff/chsdiff)
	sp = np.nan_to_num(slopep); ss = np.nan_to_num(slopes)

	#Obtaining values for linear regression
	n_mp, bp = np.polyfit(np.array(newchp),np.array(newp),1)
	n_ms, bs = np.polyfit(np.array(newchs),np.array(news),1)

	dp = np.gradient(np.array(newp))
	ds = np.gradient(np.array(news))

	#Applying linear regression for correcting 
	#pred_p = n_mp*np.array(newchp) + bp
	#pred_s = n_ms*np.array(newchs) + bs

	#print(f"mp: {n_mp}, ms: {n_ms}")

	#pred_p, pred_s = fit_obs(newchp,newchs,newp,news)

	parray, sarray = np.array(newp), np.array(news)
	chp_array = np.array(new_channelsp); chs_array = np.array(new_channelss);
	chp = np.reshape(chp_array,(len(new_channelsp),1))
	chs = np.reshape(chs_array,(len(new_channelss),1))

	#Statistics of the pickings
	sortp, sorts = np.sort(parray), np.sort(sarray)
	medianp, medians = np.median(sortp), np.median(sorts)
	iqrp, iqrs = stats.iqr(sortp), stats.iqr(sorts)
	q1p, q3p = np.percentile(sortp,25),np.percentile(sortp,75)
	q1s, q3s = np.percentile(sorts,25),np.percentile(sorts,75)
	sminusp = abs(np.mean(news) - np.mean(newp))

	lenp = chp*dx; lens = chs*dx
	densep_left = np.where(lenp < 50000)[0].shape[0]; densep_right = np.where(lenp > 50000)[0].shape[0]
	denses_left = np.where(lens < 50000)[0].shape[0]; denses_right = np.where(lens > 50000)[0].shape[0]

	print(f"Median p: {medianp}, iqr p: {iqrp}, mean P: {np.mean(newp)}, Ch p:{chp[-1]*dx}")
	print(f"Median s: {medians}, iqr s: {iqrs}, mean S: {np.mean(news)}, Ch s:{chs[-1]*dx}")
	print(f"Dense p left: {densep_left}, Dense p right: {densep_right}")
	print(f"Dense s left: {denses_left}, Dense s right: {denses_right}")
	print(f"S - P: {sminusp}")


	normp = (parray - np.mean(parray))/np.std(parray)
	norms = (sarray - np.mean(sarray))/np.std(sarray)

	svr = GridSearchCV(SVR(kernel="rbf",epsilon=0.1),param_grid={"C":[1e0,1e1,1e2,1e3],"epsilon":[0.001,0.01,0.1,1.0]})

	if(sminusp < 5.0 and iqrp < 10.0):

		#svrp = SVR(kernel="rbf",C=1e2,epsilon=1.0)
		svr = GridSearchCV(SVR(kernel="rbf",epsilon=0.1),param_grid={"C":[1e0,1e1,1e2,1e3],"epsilon":[0.001,0.01,0.1,1.0]})
		#svrp = GridSearchCV(SVR(kernel="rbf",epsilon=0.1),param_grid={"C":[1e0,1e1,1e2,1e3],"epsilon":[0.001,0.01,0.1,1.0]})

	elif(denses_left < 5 or denses_right < 5):

		#svr = SVR(kernel="rbf",C=1.0,epsilon=0.0001)
		#svr = GridSearchCV(LinearSVR(epsilon=0.1),param_grid={"C":[1e0,1e1,1e2,1e3],"epsilon":[0.001,0.01,0.1,1.0]})
		#svr = LinearSVR(C=1.0,epsilon=0.0001)
		svr = SVR(kernel="rbf",C=1.0,epsilon=1.0)

	else:

		svr = SVR(kernel="rbf",C=1.0,epsilon=0.0001)

	# 	

	# if(sminusp < 5.0 and iqrs < 10.0):

	# 	svrs = SVR(kernel="rbf")
	# 	#svrs = GridSearchCV(SVR(kernel="rbf",epsilon=0.1),param_grid={"C":[1e0,1e1,1e2,1e3],"epsilon":[0.001,0.01,0.1,1.0]})

	# else:

	# 	#svrs = GridSearchCV(LinearSVR(epsilon=0.1),param_grid={"C":[1e0,1e1,1e2,1e3],"epsilon":[0.001,0.01,0.1,1.0]})
	# 	#svrs = LinearSVR(C=1.0,epsilon=0.001)
	# 	#svrs = SVR(kernel="linear",C=1.0,epsilon=0.001)
	# 	svrs = SVR(kernel="rbf",C=1.0,epsilon=0.0001)
		
	#Support Vector Regression using RBF kernel

	#Estimate P and S estimated
	
	pmodel = svr.fit(chp,normp)
	smodel = svr.fit(chs,norms)

	#if(len(normp) < 300 and len(norms) < 300):
	#	pmodel = svrp.fit(chp,normp)
	#	smodel = svrs.fit(chs,norms)
	#else:
	#	pmodel = svrp.fit(chp[:200],normp[:200])
	#	smodel = svrs.fit(chs[:200],norms[:200])

	pest = pmodel.predict(chp)
	sest = smodel.predict(chs)

	print(f"Chs: {chp.shape}, pest: {pest.shape}, sest: {sest.shape}")



	# # fig, ax = plt.subplots(3,1)
	# # ax[0].scatter(ppos,parray,s=20,c="b")
	# # ax[0].scatter(newchp,newp,s=20,c="g")
	# # ax[0].scatter(newchp,pred_p,s=20,c="r")
	# # ax[1].scatter(spos,sarray,s=20,c="b")
	# # ax[1].scatter(newchs,news,s=20,c="g")
	# # ax[1].scatter(newchs,pred_s,s=20,c="r")

	# # print(f"pred_p shape:{pred_p.shape}, pred_s:{pred_s.shape}")
	# # ax[2].scatter(newchp,pred_p,s=20,c="r")
	# # ax[2].scatter(newchs,pred_s,s=20,c="b")

	# #Obtaining values for linear regression
	# 
	# n_ms, bs = np.polyfit(np.array(chsfin),np.array(sfin),1)

	# #Applying linear regression for correcting 
	# pred_p = n_mp*np.array(chpfin) + bp
	# pred_s = n_ms*np.array(chsfin) + bs

	# print(f"Chpfinsize")

	# #pred_p, pred_s = fit_obs(chpfin,chsfin,pfin,sfin)

	# a = mp
	# b = ms

	# #Using only the final size
	# if(snr < 0):

	# 	pred_p = np.array(pfin) - 3.0
	# 	pred_s = np.array(sfin) - 3.0

	# else:

	# 	chpfin = ch_idx
	# 	chsfin = ch_idx



	#fig, ax = plt.subplots(2,1,figsize=(10,5))
	#ax[0].plot(a,linewidth=0.7,c="k")
	#ax[0].scatter(np.arange(0,a.shape[0]),a,s=20,c="r")
	#ax[0].set_yscale("log")
	#ax[1].plot(b,linewidth=0.7,c="k")
	#ax[1].scatter(np.arange(0,b.shape[0]),b,s=20,c="r")
	#ax[1].set_yscale("log")
	#plt.savefig(f"/home/gerardomendo/Documents/DAS/Data_labeling/Slopes_correct_{name}")
	#plt.close()
	pest = (pest*np.std(parray)) + np.mean(parray)
	sest = (sest*np.std(sarray)) + np.mean(sarray)

	#return new_channelsp, pred_p, new_channelss, pred_s
	#return chpfin, pred_p, chsfin, pred_s
	#return new_channelsp, np.array(newp), new_channelss, np.array(news)
	#return chpfin, np.array(pfin), chsfin, np.array(sfin)
	return chp,chs,newp,news,pest,sest,pmodel,smodel

##########################################################################################################################################

def plotting_traces(path,name_file,data,channel_list,p,s,fs,datetime,max_snr,n):

	c2, c1 = signal.butter(2,(5.0,15.0),"band",fs=100.0)
	fdata = signal.filtfilt(c2,c1,data,axis=1)
	fdata = signal.filtfilt(c2,c1,fdata,axis=1)

	ch = []
	p_picks = []
	s_picks = []
	tstp = []

	channel_p = channel_list[0]; n_chp = len(channel_p)
	channel_s = channel_list[1]; n_chs = len(channel_s)

	if(n_chp > n_chs):
		 ch_a = channel_s
		 ch_b = channel_p
	else:
		 ch_a = channel_p
		 ch_b = channel_s

	for idx in range(len(ch_a)):

		#Picks average
		name_image = f"{path}/CorrPhases_{name_file}_{datetime}_Channel_{ch_a[idx]}.png"

		loc = np.where(np.array(ch_a) == np.array(ch_b[idx]))[0]
		#print(f"Loc: {loc}")
		trace = np.ravel(fdata[ch_a[idx],:])
		#print(f"time shape: {trace.shape}")
		time = np.arange(0,trace.shape[0])/fs
		#print(f"time shape: {time.shape}")

		p_pick = int(p[idx]*fs); s_pick = int(s[idx]*fs)
		#print(f"P: {p_pick}, S: {s_pick}")
		tstp.append(abs(s_pick - p_pick))

		gp = signal.windows.gaussian(2*n,int(n*0.1))
		gs = signal.windows.gaussian(2*n,int(n*0.1))
		#print(f"data shape: {data.shape}, time shape: {time.shape[0]}, left: {int(p_pick-n)}, right: {int(p_pick+n)} ")
		if(np.mean(p) == 0):

			label_p = np.zeros(time.shape[0])
			try:
				label_s = np.zeros(time.shape[0]); label_s[s_pick-n:s_pick+n] = gs
			except ValueError:
				if(s_pick >= 6000):
					label_s = np.zeros(time.shape[0]); label_s[s_pick-n:s_pick] = gs
				else:
					label_s = np.zeros(time.shape[0]); label_s[s_pick:s_pick+n] = gs

		else:
			try:
				label_p = np.zeros(time.shape[0]); label_p[p_pick-n:p_pick+n] = gp
			except ValueError:
				if(p_pick >= 6000):
					label_p = np.zeros(time.shape[0]); label_p[p_pick-n:p_pick] = gp
				elif(p_pick <= 0):
					pass
				else:
					label_p = np.zeros(time.shape[0]); label_p[p_pick:p_pick+n] = gp
			try:
				label_s = np.zeros(time.shape[0]); label_s[s_pick-n:s_pick+n] = gs
			except ValueError:
				if(s_pick >= 6000):
					print(f"data shape: {data.shape}, time shape: {time.shape[0]}, left: {int(s_pick-n)}, right: {int(s_pick)} ")
					label_s = np.zeros(time.shape[0]); label_s[s_pick-n:s_pick] = gs
				elif(s_pick <= 0):
					pass
				else:
					label_s = np.zeros(time.shape[0]); label_s[s_pick:s_pick+n] = gs

		#Plotting
		fig, ax = plt.subplots(2,1,figsize=(20,10))
		ax[0].plot(time,trace*(1e6),linewidth=0.7,c="k")
		ax[0].plot([time[p_pick],time[p_pick]],[-1,1],linewidth=1.0,c="r")
		ax[0].plot([time[s_pick],time[s_pick]],[-1,1],linewidth=1.0,c="b")
		ax[0].set_xlim(time[0],time[-1])
		ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
		ax[0].tick_params(axis="x",color="k",labelsize=14.0,length=10.0,width=2.0,which="major")
		ax[0].tick_params(axis="x",color="k",labelsize=14.0,length=5.0,width=2.0,which="minor")
		ax[0].tick_params(axis="y",color="k",labelsize=14.0,length=10.0,width=2.0,which="major")
		ax[0].tick_params(axis="y",color="k",labelsize=14.0,length=5.0,width=2.0,which="minor")
		ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10.0))
		ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
		
		ax[1].plot(time,label_p,linewidth=0.7,c="r",label="P")
		ax[1].plot(time,label_s,linewidth=0.7,c="b",label="S")
		ax[1].set_xlim(time[0],time[-1])

		ax[1].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
		ax[1].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
		ax[1].tick_params(axis="x",color="k",labelsize=14.0,length=10.0,width=2.0,which="major")
		ax[1].tick_params(axis="x",color="k",labelsize=14.0,length=5.0,width=2.0,which="minor")
		ax[1].tick_params(axis="y",color="k",labelsize=14.0,length=10.0,width=2.0,which="major")
		ax[1].tick_params(axis="y",color="k",labelsize=14.0,length=5.0,width=2.0,which="minor")

		ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10.0))
		ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1.0))

		ax[1].legend(loc="upper right")
		plt.savefig(f"{name_image}")
		plt.close()

###################################################################################

def fit_obs(chp,chs,pobs,sobs):

	#fitting to 1-order polynomium

	mp, bp = np.polyfit(np.array(chp),np.array(pobs),1)
	ms, bs = np.polyfit(np.array(chs),np.array(sobs),1)

	#fitting to 2-order polynomium

	cp1, cp2, cp3 = np.polyfit(np.array(chp),np.array(pobs),2)
	cs1, cs2, cs3 = np.polyfit(np.array(chs),np.array(sobs),2)

	#Obtaining predictions

	pred_1p = mp*np.array(chp) + bp
	pred_1s = ms*np.array(chs) + bs

	pred_2p = cp1*np.array(chp)**2 + cp2*np.array(chp) + cp3
	pred_2s = cs1*np.array(chs)**2 + cs2*np.array(chs) + cs3

	R1p = stats.pearsonr(pobs,pred_1p)
	R1s = stats.pearsonr(sobs,pred_1s)

	R2p = stats.pearsonr(pobs,pred_2p)
	R2s = stats.pearsonr(sobs,pred_2s)

	print(f"Correlation 1-order: P:{R1p[0]}, S:{R1s[0]}")
	print(f"Correlation 2-order: P:{R2p[0]}, S:{R2s[0]}")

	diffp = abs(R2p[0] - R1p[0])
	diffs = abs(R2s[0] - R1s[0])

	print(f"Difference P:{diffp}, S:{diffs}")

	if(diffp > 0.3 and diffs > 0.1):

		pred_p = pred_2p
		pred_s = pred_2s

	else:

		pred_p = pred_1p
		pred_s = pred_1s

	#if(diffs > 0.25):
	#else:

	return pred_p, pred_s

##############################################################################################################

def plotting_boxplot(path,name,p_picks,s_picks,extra_name=None):

	fig, ax = plt.subplots(1,1,figsize=(10,10))	
	datos_sl = [p_picks,s_picks]
	#try:
	#	datos_sp = [np.array(s_picks) - np.array(p_picks)]
	#except ValueError:
	#	datos_sp = [s_picks[:-1] - p_picks]
	ax.set_title("Phase picking",fontsize=16.0)
	ax.boxplot(datos_sl,showmeans=True,meanline=True)
	ax.set_xticks([x + 1 for x in range(len(datos_sl))],
		labels=["P picks","S picks"])
	ax.set_ylabel("Arrival time (s)",fontsize=14.0)
	#ax[1].set_title("Phase picking - P-S",fontsize=16.0)
	#ax[1].boxplot(datos_sp,showmeans=True,meanline=True)
	#ax[1].set_xticks([x + 1 for x in range(len(datos_sp))],
	#	labels=["S - P"])
	#ax[1].set_ylabel("Difference (s)",fontsize=14.0)
	if(extra_name is None):
		plt.savefig(f"{path}/Accuracy_picks_{name}.pdf")
	else:
		plt.savefig(f"{path}/Accuracy_picks_{name}_{extra_name}.pdf")
	plt.close()

#################################################################################################################

def take_maximum(data,ch_idx,parameters,low_point):

	#Parameters
	fs = parameters["fs"]
	win_size = 0.5
	overlap = 0.1
	th = parameters["threshold"]

	ntraces, nsamples = data.shape[0],data.shape[1]
	maximums = []; channels = []

	for idx in range(0,ntraces):

		trace = data[idx,:]
		rms = rms_detector(trace,fs,win_size,overlap)
		nrms = rms/np.amax(rms)
		maxidx = np.where(nrms > 0.9)[0]
		if(maxidx.shape[0] > 1):
			maximums.append((maxidx[-1]*win_size*overlap)+low_point[idx]/fs)
			channels.append(ch_idx[idx])
		elif(maxidx.shape[0] == 0):
			maximums.append(0)
			channels.append(ch_idx[idx])
		else:
			maximums.append((maxidx[0]*win_size*overlap)+low_point[idx]/fs)
			channels.append(ch_idx[idx])

	return maximums,channels

#################################################################################################################

def concat_hdf5(path_original,original,name_det):

	utc = UTCDateTime(name_det)

	atime = utc + 60

	ahour = atime.hour
	if(ahour < 10):
		ahour = str(f"0{ahour}")
	else:
		ahour = str(ahour)

	amin = atime.minute
	if(amin < 10):
		amin = str(f"0{amin}")
	else:
		amin = str(amin)

	asec = atime.second
	if(asec < 10):
		asec = str(f"0{asec}")
	else:
		asec = str(asec)

	minafter = f"{name_det[:11]}{ahour}{amin}{asec}Z"
	name_after = f"{original[:10]}{minafter}{original[-6:]}"

	atime = utc - 60

	ahour = atime.hour
	if(ahour < 10):
		ahour = str(f"0{ahour}")
	else:
		ahour = str(ahour)

	amin = atime.minute
	if(amin < 10):
		amin = str(f"0{amin}")
	else:
		amin = str(amin)

	asec = atime.second
	if(asec < 10):
		asec = str(f"0{asec}")
	else:
		asec = str(asec)

	minbefore = f"{name_det[:11]}{ahour}{amin}{asec}Z"
	name_before = f"{original[:10]}{minbefore}{original[-6:]}"

	f1 = h5py.File(f"{path_original}/{name_before}"); d1 = np.array(f1["/Acquisition/Raw[0]/RawData"])
	f2 = h5py.File(f"{path_original}/{original}"); d2 = np.array(f2["/Acquisition/Raw[0]/RawData"])
	f3 = h5py.File(f"{path_original}/{name_after}"); d3 = np.array(f3["/Acquisition/Raw[0]/RawData"])

	#Gauge length per sample
	L = f1["Acquisition"].attrs["GaugeLength"]

	#Spatial sampling rate
	dx = f1["Acquisition"].attrs["SpatialSamplingInterval"]

	#Starttime
	starttime = f1["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']

	#Sampling rate
	fs = f1["Acquisition/Raw[0]"].attrs["OutputDataRate"]

	parameters = {"GL":L,"dx":dx,"starttime":starttime,"fs":fs}

	res = np.vstack((d1.T,d2.T,d3.T))

	return res,parameters

###########################################################################################################################################

def check_directories(main_path,list_paths):

	for directory in list_paths:
		
		try:
			os.chdir(f"{directory}")
		except FileNotFoundError:
			os.mkdir(f"{directory}")

	os.chdir(f"{main_path}")


###########################################################################################################################################

def writing_csv(path_csv,name_csv,data,channel_list,p,s,fs,datetime,iqr_snr):

	#Open csv writing
	f = open(f"{path_csv}/{name_csv}","a")

	channel_p = channel_list[0]; n_chp = len(channel_p)
	channel_s = channel_list[1]; n_chs = len(channel_s)

	if(n_chp > n_chs):
		 ch_a = channel_s
		 ch_b = channel_p
	else:
		 ch_a = channel_p
		 ch_b = channel_s

	for idx in range(len(ch_a)):

		#DateTime Channel p_pick(s) s_pick(s) p_pick(sample) s_pick(sample) p_std s_std
		if(iqr_snr > 3):
			f.write(f"{datetime},{ch_a[idx]},{int(ch_a[idx]*(5*8))},{round(p[idx],3)},{int(p[idx]*fs)},{round(s[idx],3)},{int(s[idx]*fs)},{3.0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+1},{int(ch_a[idx]*(5*8))+1},{round(p[idx],3)},{int(p[idx]*fs)},{round(s[idx],3)},{int(s[idx]*fs)},{3.0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+2},{int(ch_a[idx]*(5*8))+2},{round(p[idx],3)},{int(p[idx]*fs)},{round(s[idx],3)},{int(s[idx]*fs)},{3.0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+3},{int(ch_a[idx]*(5*8))+3},{round(p[idx],3)},{int(p[idx]*fs)},{round(s[idx],3)},{int(s[idx]*fs)},{3.0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+4},{int(ch_a[idx]*(5*8))+4},{round(p[idx],3)},{int(p[idx]*fs)},{round(s[idx],3)},{int(s[idx]*fs)},{3.0},{2.0}\n")
		else:
			f.write(f"{datetime},{ch_a[idx]},{int(ch_a[idx]*(5*8))},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+1},{int(ch_a[idx]*(5*8))+1},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+2},{int(ch_a[idx]*(5*8))+2},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+3},{int(ch_a[idx]*(5*8))+3},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")
			f.write(f"{datetime},{ch_a[idx]+4},{int(ch_a[idx]*(5*8))+4},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")

	f.close()

#########################################################################################################################

def fk_filter2(data,fs,dx,vmin,vmax):

	#Retrieving samples and number of traces
	ntraces, npts = data.shape

	#Obtaining frequency and wavenumber samples

	f = np.fft.fftshift(np.fft.fftfreq(npts,(1/fs)))
	k = 4*np.pi*np.fft.fftshift(np.fft.fftfreq(ntraces,dx))

	#Obtain angular wavenumber
	kmax = 4*np.pi*f/vmin
	kmin = 4*np.pi*f/vmax

	#Applying 2D fourier transform
	H = np.fft.fftshift(scipy.fft.fft2(data,s=(ntraces,npts)))

	#estimate apparent velocities
	v = np.abs(f/k[:,np.newaxis])
	v = np.nan_to_num(v,nan=sys.float_info.epsilon)

	#Difference appvel min and max
	#diffv = vmax - vmin

	#v25 = vmin + 0.25*diffv 
	#v75 = vmin + 0.75*diffv

	fac = 0.5*(1 + np.cos(np.pi*(1 - k/(0.1*kmax)[:,np.newaxis])))
	fac = fac.T

	#Kaiser window
	beta = 8
	c1 = 2*k/kmin[:,np.newaxis]
	c2 = np.sqrt(1 - c1**2)
	c3 = special.i0(beta*c2)
	kaiser = c3/special.i0(beta)
	kaiser = kaiser.T
	print(f"Kaiser shape: {kaiser.shape}")
	print(f"fac shape: {fac.shape}")


	fac = np.where((v > -0.4*vmin) & (v < 0.4*vmin),1.0,fac)
	fac = np.where((v > 0.5*vmin) | (v < -0.5*vmin),0.0,fac)
	fac = np.where((v > vmax) & (v < -vmax),1 - kaiser,fac)
	#fkfilt = 1.0 - fac

	#fkfilt = fac*(kaiser)

	IH = H*fac

	filtdata = np.fft.ifft2(np.fft.ifftshift(IH),s=(ntraces,npts)).real[0:ntraces,0:npts]

	return filtdata

#########################################################################################################################################

def labeling_spectrogram(data,channels,parameters):

	fs = parameters["fs"]
	nperseg = parameters["nperseg"]
	noverlap = parameters["noverlap"]

	labels_p = {}
	labels_s = {}
	new_ch = []

	window_length = 300

	for i,trace in enumerate(data):

		available = False

		f,t,s = signal.spectrogram(trace,fs,nperseg=nperseg,noverlap=noverlap)

		ienvelope = (abs(np.gradient(np.sum(s,axis=0))))*50
		envelope = ienvelope/np.amax(ienvelope)
		times_env = np.arange(0,envelope.shape[0])*(t[1]-t[0])
		mean_envelope = np.mean(envelope)
		mad_envelope = stats.median_abs_deviation(envelope)

		if(channels[i] < 1000):
			peaks = detect_peaks(envelope,0.2*mad_envelope,10)
		else:
			peaks = detect_peaks(envelope,0.05*mad_envelope,10)

		try:
			maxindx = np.where(envelope >= 0.8*np.amax(envelope))[0][0]
		except IndexError:
			maxindx = 0
		#print(f"peaks: {peaks}, maxindx: {maxindx}")
		
		#Appending into list of labels

		try:
			assert peaks.shape[0] != 0
			available = True
		except AssertionError:
			print(f"No peaks available")
			labels_p[str(channels[i])] = 0
			labels_s[str(channels[i])] = 0

		if(available):

			try:
				labels_p[str(channels[i])] = int(times_env[peaks[0]]*fs)
			except Exception:
				labels_p[str(channels[i])] = 0

			try:
				labels_s[str(channels[i])] = int(times_env[maxindx]*fs)
			except Exception:
				labels_s[str(channels[i])] = 0

		new_ch.append(channels[i])

	return labels_p,labels_s,new_ch

###########################################################################################################################################

def obtain_pickModel(picks_p,picks_s,channels,lowpoints,nelements,snr,addlowpoints=True,kernelp="rbf",kernels="rbf"):

	if(len(channels) == 1):
		chnsp = channels[0]; chnss = channels[0]
	else:
		chnsp = channels[0]; chnss = channels[1]
		#print(f"Ch S shape: {len(chnss)}")

	#svr = SVR(kernel=kernel,C=0.1,epsilon=0.001)
	if(kernelp == "linear"):
		svrp = make_pipeline(StandardScaler(),LinearSVR(C=0.1,epsilon=0,dual=True,max_iter=10000))
		#svrp = LinearSVR(C=1.0,epsilon=1e-6)
	else:
		if(snr < -0.1):
			svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=0.8,epsilon=0.1))
		else:
			#svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.0,epsilon=0.1))
			#svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.0,epsilon=1e-6))
			svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=0.01,epsilon=1e-6))

	if(kernels == "linear"):
		#svrs = LinearSVR(C=1.0,epsilon=1e-6)
		svrs = make_pipeline(StandardScaler(),LinearSVR(C=0.015,epsilon=0.1,dual=True,max_iter=10000))
	else:
		if(snr < -0.1):
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=1.0,epsilon=0.1))
			svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=0.8,epsilon=0.1))
		else:
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=1.0,epsilon=0.01))
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.5,epsilon=0.01))
			svrs = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=0.05,epsilon=0.01))

	keys = [item for item in picks_p.keys()]
	lp = np.array(lowpoints)

	#For P
	pp = np.array([picks_p[key] for key in keys])
	
	if(addlowpoints):
		pp = pp + lp
	
	normp = (pp - np.mean(pp))/np.std(pp)
	chp = np.reshape(np.array(chnsp),(len(chnsp),1))
	pmodel = svrp.fit(chp,normp)

	newp = {}
	newch = np.arange(0,nelements)
	newch2 = newch.reshape(-1,1)
	pest = pmodel.predict(newch2)
	for i,ch in enumerate(newch):

		newp[str(ch)] = int((pest[i]*np.std(pp))+np.mean(pp))

	#For S
	keyss = [item for item in picks_s.keys()]
	ss = np.array([picks_s[key] for key in keyss])

	if(addlowpoints):
		ss = ss + lp
	
	norms = (ss - np.mean(ss))/np.std(ss)
	chs = np.reshape(np.array(chnss),(len(chnss),1))
	smodel = svrs.fit(chs,norms)
	news = {}
	sest = smodel.predict(newch2)
	for i,ch in enumerate(newch):

		news[str(ch)] = int((sest[i]*np.std(ss))+np.mean(ss))

	return newch, newp, news

######################################################################################################################################################

def create_correlation_matrix(path_detections,file):

	os.chdir(path_detections)
	name_det = file[-33:-15]

	#retrieve files that match name_det (both filtered and non filter)
	h5file = glob.glob(f"*{name_det}*NH8*")

	a = h5py.File(h5file[0])
	traces = list(a['Detections/Data'])

	ccmatrix = np.zeros((len(traces),len(traces)))

	for j,eva in enumerate(traces):

		for k,evb in enumerate(traces):

			ccmatrix[j,k] = np.amax(correlation_coefficient(signal.correlate(eva,evb),eva,evb))

	return ccmatrix

######################################################################################################################################################
######################################################################################################################################

def comparison_picks(ppicks,spicks,channels,low_points,n_methods = 2, c_thres = 300):

	ch_p = []
	ch_s = []
	new_p = {}
	new_s = {}

	cp = 0; pp = 0
	cs = 0; ns = 0

	for i,ch in enumerate(channels):

		exist = True

		#Retrieving methods: 0 - stalta, 1 - ar, 2 - sp
		#p0, p1, p2, p3 = ppicks[0][f"{ch}"], ppicks[1][f"{ch}"], ppicks[2][f"{ch}"], ppicks[3][f"{ch}"],
		#s0, s1, s2, s3 = spicks[0][f"{ch}"], spicks[1][f"{ch}"], spicks[2][f"{ch}"], spicks[3][f"{ch}"],

		#Retrieving picks from
		try:
			ps = [ppicks[p][f"{ch}"] for p in range(n_methods)]
		except KeyError:
			print("Non-existent P picking")
			exist = False

		try:
			ss = [spicks[s][f"{ch}"] for s in range(n_methods)]
		except KeyError:
			print("Non-existent S picking")
			exist = False

		if(exist):

			#Adding the first element to the list as the last element
			ps.append(ppicks[0][f"{ch}"]); ss.append(spicks[0][f"{ch}"])

			#Converting to arrays
			ps = np.array(ps)
			ss = np.array(ss)

			#Compare - p0 is repeated twice in order to compare it with p2
			#ps = np.array([p0,p1,p2,p3,p0])
			#ss = np.array([s0,s1,s2,s3,s0])

			diffp = abs(np.diff(ps)); diffs = abs(np.diff(ss))

			#Evaluate using threshold
			if(np.mean(diffp) < c_thres):

				#print(f"Mean p: {np.mean(abs(diffp))}, c_thres: {c_thres}")
				cp += 1
				ch_p.append(ch); new_p[str(ch)] = int(np.mean(ps)) + low_points[i]

			else:
				pp += 1

			if(np.mean(diffs) < c_thres):

				#print(f"Mean s: {np.mean(abs(diffs))}, c_thres: {c_thres}")
				cs += 1
				ch_s.append(ch); new_s[str(ch)] = int(np.mean(ss))  + low_points[i]
			
			else:
				ns += 1

	#return ch, p_picks, s_picks
	#print(f'Coherent p picks: {cp}, non-coherent p picks: {pp}')
	#print(f'Coherent s picks: {cs}, non-coherent s picks: {ns}')

	return ch_p, ch_s, new_p, new_s

######################################################################################################################################
############################################################################################################################################################################

def correct_picks(channels,p_picks,s_picks):

	#Convert to array
	if(len(channels) == 1):
		ppos = channels[0]; spos = channels[0]
	else:
		ppos = channels[0]; spos = channels[1]

	# Retrieving initial p and s picks
	ini_p = np.array([p_picks[str(ch)] for ch in ppos])
	ini_s = np.array([s_picks[str(ch)] for ch in spos])

	#Obtain slopes 
	chpdiff = np.diff(ppos)
	chsdiff = np.diff(spos)
	pdiff = np.diff(ini_p)
	sdiff = np.diff(ini_s)

	mpnan = (pdiff/chpdiff); msnan = (sdiff/chsdiff)
	slopep = np.nan_to_num(mpnan); slopes = np.nan_to_num(msnan)

	pmean = np.mean(slopep); smean = np.mean(slopes)
	pstd = np.std(slopep); sstd = np.std(slopes)

	new_chp = []; new_chs = []
	newp = {}; news = {}

	low_limp = pmean - 1*pstd
	up_limp = pmean + 1*pstd

	low_lims = smean - 1*sstd
	up_lims = smean + 1*sstd

	for idx in range(0,slopep.shape[0]):

		if(slopep[idx] < up_limp and slopep[idx] > low_limp):
			new_chp.append(ppos[idx]); newp[str(ppos[idx])] = ini_p[idx]

	for idx in range(0,slopes.shape[0]):

		if(slopes[idx] < up_lims and slopes[idx] > low_lims):
			new_chs.append(spos[idx]); news[str(spos[idx])] = ini_s[idx]

	return new_chp, new_chs, newp, news

##########################################################################################################################################

def obtain_pickModelOne(picks_p,channels,lowpoints,nelements,addlowpoints=True,kernel="linear"):

	chnsp = channels

	#svr = SVR(kernel=kernel,C=1.0,epsilon=0.0001)
	svr = make_pipeline(StandardScaler(),SVR(kernel=kernel,C=0.01,epsilon=0))

	keys = [item for item in picks_p.keys()]
	lp = np.array(lowpoints)

	#For P
	pp = np.array([picks_p[key] for key in keys])
	
	if(addlowpoints):
		pp = pp + lp
	
	normp = (pp - np.mean(pp))/np.std(pp)
	chp = np.reshape(np.array(chnsp),(len(chnsp),1))

	if(kernel == "linear"):

		chp = chp[:int(nelements)]
		normp = normp[:int(nelements)]
		svr = make_pipeline(StandardScaler(),LinearSVR(C=1.0,epsilon=0,dual=True,max_iter=10000))

	pmodel = svr.fit(chp,normp)
	newp = {}
	newch = np.arange(0,nelements)
	newch2 = newch.reshape(-1,1)
	pest = pmodel.predict(newch2)
	for i,ch in enumerate(newch):

		newp[str(ch)] = int((pest[i]*np.std(pp))+np.mean(pp))

	return newch, newp

##########################################################################################################################################

def obtain_kernel(ch_p,ch_s,dx,iqr,dthres=50000,pthres=10):

	lengthp = np.array(ch_p)*dx
	lengths = np.array(ch_s)*dx

	np_before = np.where(lengthp > dthres)[0].shape[0]
	ns_before = np.where(lengths > dthres)[0].shape[0]

	np_after = np.where(lengthp < dthres)[0].shape[0]
	ns_after = np.where(lengths < dthres)[0].shape[0]

	#if(np_before < 10 or np_after < 10):
	if(iqr > 2.01 and iqr is not None):

		kernelp = "linear"

	else:

		kernelp = "rbf"

	#if(ns_before < 10 or ns_after < 10):
	if(iqr > 2.01 and iqr is not None):

		kernels = "linear"
		
	else:

		kernels = "rbf"

	print(f"Kernel for P: {kernelp}, Kernel for S: {kernels}")

	return kernelp, kernels

###########################################################################################################################################################################

def refine_models(filtered,nelements,anewch,anewp,anews,params_spec):

	# Refinando modelo de picados para P

	redst_data = np.zeros((nelements,1500))

	for i,trace in enumerate(filtered):

		if(i >= nelements):

			break;

		center = anewp[str(i)]
		redst_data[i] = trace[center-750:center+750]

	sp_np,sp_ns,sp_nch = labeling_spectrogram(redst_data,anewch,params_spec)

	newps = {}

	for ch in anewch:

		center = anewp[str(ch)]
		newps[str(ch)] = (center-750) + sp_np[str(ch)]

	# Refinando modelo de picados para S

	redst_data = np.zeros((nelements,300))

	for i,trace in enumerate(filtered):

		if(i >= nelements):

			break;

		center = anews[str(i)]
		redst_data[i] = trace[center-150:center+150]

	sp_np2,sp_ns2,sp_nch2 = labeling_spectrogram(redst_data,anewch,params_spec)

	newss = {}

	for ch in anewch:

		center = anews[str(ch)]
		newss[str(ch)] = (center-150) + sp_np2[str(ch)]

	return anewch, newss, newps

################################################################################################################################################

def obtain_iqr(nchp,nchs,newp,news):

	if(len(nchp) > 10 and len(nchs) > 10):

		a = np.array(nchp); b = np.array(nchs)
		if(len(a) > len(b)):
			up, low = a, b
		else:
			up, low = b, a

		nchans = []

		for elem in low:
			c = np.where(up == elem)[0]
			if(len(c) != 0):
				nchans.append(elem)

		p = np.array([newp[str(ch)] for ch in nchans])
		s = np.array([news[str(ch)] for ch in nchans])
		sminp = (s-p)*0.01
		sminp.sort()

		#if(stats.iqr(sminp) < 2.1):
		#	print(f"min s-p: {np.amin(sminp)}, max s-p: {np.amax(sminp)}, mean s-p: {np.mean(sminp)}, median s-p: {np.median(sminp)}, std s-p: {np.std(sminp)}, iqr s-p: {stats.iqr(sminp)}")

		iqr_sminp = stats.iqr(sminp)

	elif(len(nchp) < 10 and len(nchp) > 10):

		#print(f"No statistics for {name_det} file.")
		iqr_sminp = 1e6

	elif(len(nchp) > 10 and len(nchp) < 10):

		iqr_sminp = 1e6

	else:

		iqr_sminp = 1e6

	return iqr_sminp

##########################################################################################################

def new_retrieve_centered_traces(channel_index,original_data,fs):

	osize, oshape = original_data.shape[0],original_data.shape[1]
	#print(f"odata.shape: {oshape}")

	#Lengths to retrieve (right and left)
	len_right = 5; slen_right = int(len_right*fs)
	len_left = 10; slen_left = int(len_left*fs) 
	hlen = len_right + len_left
	hlen_samples = int(hlen*fs)

	#ch_rms = np.zeros((channel_index.shape[0],2))

	c2, c1 = signal.butter(4,(5.0,15.0),"band",fs=100.0)
	data = signal.filtfilt(c2,c1,original_data,axis=1)

	low_point = []
	up_point = []

	new_traces = np.zeros((original_data.shape[0],hlen_samples),dtype=float)
	new_otraces = np.zeros((original_data.shape[0],hlen_samples),dtype=float)

	maxs = []
	for idx,trace in enumerate(data):

		maxs.append(np.where(trace==np.amax(trace))[0][0])

	#retrieving the mode of the maximum point in DAS array
	mode_maxs = stats.mode(np.array(maxs))[0]

	#retrieving windows
	for idx,ch in enumerate(data):

		trace = data[idx,:]
		otrace = original_data[idx,:]

		max_rms = mode_maxs/fs
		#print(f"Max at sample {max_rms}")

		#Storing maximum RMS
		#ch_rms[idx,0] = ch; ch_rms[idx,1] = max_rms

		#Maximum RMS sample position  
		max_rms_sample = int(max_rms*fs)

		try:
			new_traces[idx,:] = trace[max_rms_sample-slen_left:max_rms_sample+slen_right]
			new_otraces[idx,:] = otrace[max_rms_sample-slen_left:max_rms_sample+slen_right]
			low_point.append(max_rms_sample-slen_left)
			up_point.append(max_rms_sample+slen_right)
		except Exception:
			if(max_rms_sample-slen_left < 0):

				#print(f"Resetting half length to the right: {max_rms_sample-slen_left}, max_rms_sample: {max_rms_sample}, new traces shape: {slen_left+slen_right}")
				#print(f"New shape: {trace[max_rms_sample:max_rms_sample+(slen_left+slen_right)].shape}")

				new_traces[idx,:] = trace[max_rms_sample:abs(max_rms_sample+slen_left+slen_right)]
				new_otraces[idx,:] = otrace[max_rms_sample:abs(max_rms_sample+slen_left+slen_right)]
				low_point.append(max_rms_sample)
				up_point.append(max_rms_sample+slen_left+slen_right)

			elif(max_rms_sample+slen_right > oshape):
				#print(f"Resetting half length to the left: {max_rms_sample+slen_right},slenright: {slen_right} max_rms_sample: {max_rms_sample}")
				excedent = (max_rms_sample+slen_right) - oshape
				remanent = oshape - max_rms_sample
				new_traces[idx,:] = trace[max_rms_sample-(slen_left+excedent):max_rms_sample+remanent]
				new_otraces[idx,:] = otrace[max_rms_sample-(slen_left+excedent):max_rms_sample+remanent]
				low_point.append(max_rms_sample-(slen_left+excedent))
				up_point.append(max_rms_sample+remanent)


	return new_traces, new_otraces, np.array(low_point), np.array(up_point)


##########################################################################################################################################################################	