import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.signal
from scipy import fftpack
import scipy.stats as stats
import obspy.signal.cross_correlation as ost
import obspy
import os
from detect_peaks import detect_peaks
from scipy.signal import hilbert, savgol_filter
from scipy.signal import decimate as decimate
import obspy.signal.filter as filt
from timeit import default_timer as timer
from obspy.signal.detrend import spline
import scipy.signal as signal
import math
import h5py
from numba import jit
import glob
import os
import timeit

###########################################################################

def divide_trace_DAS(target_matrix,n_t,overlap_factor,envelope_flag=False):

	#Input trace
	matrix = target_matrix

	#Number of samples of the trace
	N = matrix.shape[0]

	#Number of channels
	n_channels = matrix.shape[1]

	#Inverse of overlap
	overlap = 1 - overlap_factor

	#Number of samples of the template is N_t
	#Overlap factor is between 0.1 and 1

	#Estimate number of windows
	n_win = int(N/(n_t*overlap))

	#Initialize matrix of slices
	trace_matrix = np.zeros((n_win,n_t,n_channels),dtype='float32')

	#Starting point of the next window
	inc = n_t*overlap

	for nc in range(n_channels):

		trace = matrix[:,nc]

		for window in range(0,n_win):

			#Updating bounds
			li = int(window*inc); ls = li + n_t 

			#Trimming the trace
			win = trace[li:ls].astype('float32')

			#storing into the 2D array
			try:
				if(envelope_flag):
					trace_matrix[window,:,nc] = envelope(win)
				else:
					trace_matrix[window,:,nc] = win
			except ValueError:
				#print(f"Adding zeros")
				new_win = np.zeros(trace_matrix.shape[1])
				new_win[:win.shape[0]] = win
				if(envelope_flag):
					trace_matrix[window,:,nc] = envelope(new_win)
				else:
					trace_matrix[window,:,nc] = new_win

	#print(f"Number of retrieved windows: {window}")
	return trace_matrix

##############################################################################################################

def template_matching(template, waveforms, overlap, threshold, template_name, path_waveforms, path_save, root=None):

	#Initial array of cc
	cc_total = np.zeros(0)

	#Detection matrix
	detect_traces = []

	#Start times
	start_times = []

	#Sum for cumulative freq
	cumfreq = 0
	cumfreq_list = []
	freqocc = []
	stds = [] 

	for waveform in waveforms:

		#Length of the template
		n_template = template.shape[0]

		#Dividing the matrix - envelopes
		new_matrix = divide_trace_DAS(fwtrace,n_template,overlap,envelope_flag)
		#print(f"new_matrix shape: {new_matrix.shape}")

		#Also with waveforms to retrieve data
		new_waveforms = divide_trace_DAS(fwtrace,n_template,overlap)
		#print(f"Waveforms: {new_waveforms.shape}")

		#Also with waveforms to retrieve data
		new_orig_waveforms = divide_trace_DAS(orig_trace,n_template,overlap)
		#print(f"Orig waveforms: {new_orig_waveforms.shape}")

		#Retrieving number of traces and number of channels
		n_traces = new_matrix.shape[0]
		n_channels = new_matrix.shape[2]

		#Start times of each time window
		inc_window = template.shape[0]*(1-overlap)
		st_array = np.arange(n_traces)*inc_window

		#Arrays of detections, maximum, minimum, and average amplitudes
		mccsub = np.zeros((n_traces,n_channels))
		mmax_amp = np.zeros((n_traces,n_channels))
		mmin_amp = np.zeros((n_traces,n_channels))
		mavg_amp = np.zeros((n_traces,n_channels))

		#Correlation coefficient peaks
		ccpeaks = {}

		for ch in range(n_channels):

			submatrix = new_matrix[:,:,ch]

			ccsub = np.zeros(n_traces)
			max_amp = np.zeros(n_traces)
			min_amp = np.zeros(n_traces)
			avg_amp = np.zeros(n_traces)

			for k in range(0,n_traces):

				subtrace = submatrix[k]

				if(root=="/Users"): #macOS system
					ccf = scipy.signal.correlate(subtrace,template)
				else:
					ccf = scipy.signal.correlate(subtrace,template)
				
				cc = np.amax(correlation_coefficient(ccf,subtrace,template))

				#Adding values
				ccsub[k] = cc; max_amp[k] = np.amax(subtrace); min_amp[k] = np.amin(subtrace); avg_amp[k] = np.mean(subtrace);

			#Adding values
			mccsub[:,ch] = ccsub; mmax_amp[:,ch] = max_amp; mmin_amp[:,ch] = min_amp; mavg_amp[:,ch] = avg_amp;

			#Creating time array per trace
			subtimes = np.arange(0,n_traces)*((template.shape[0]*(1-overlap))/fs)
			
			#Obtaining peaks
			peaks = detect_peaks(ccsub,mph = threshold,mpd=1.0)

			if(peaks.shape[0] != 0):

				ccpeaks[f"{ch}"] = peaks

		traces = []
		orig_traces = []
		
		for index in ccpeaks:

			#print(f"Channel no.: {index}, No. of peaks: {ccpeaks[index].shape[0]}")	

			if(ccpeaks[index].shape[0] >= 1):

				trace = new_waveforms[ccpeaks[index][0],:,int(index)]
				traces.append(trace)

				#Saving original files
				#print(f"Orig. waveforms: {ccpeaks[index].shape}")
				orig_traces.append(new_orig_waveforms[ccpeaks[index][0],:,int(index)])

		params = {"fwtrace":fwtrace.shape[0],"n_channels":n_channels,"fs":fs,"dx":dx}

		#Adding to the list of cumfrequency
		cumfreq += len(traces)
		cumfreq_list.append(cumfreq)
		freqocc.append(len(traces))

		#Plotting
		if(detections):

			plotting_cc(path_save,mccsub,starttime,f"{template_name}_W{waveform}",params)

			# Estimating CC matrix and plotting results
			ccmatrix = create_correlation_matrix(f"{path_detections}/Detections_DAS_{waveform[:-3]}_NH8.hdf5")
			stds.append(np.std(ccmatrix.flatten()))

			#Saving detections info
			writeDetectionsAscii(path_coeff,file,ccpeaks,waveform,fs,conv_fac,n_template)
			
			#Showing to user
			print(f"Detection for {waveform}, stds: {np.std(ccmatrix.flatten())}, traces: {len(traces)}")

		else:

			ccm = create_CC(traces)
			std = np.std(ccm.flatten())
			stds.append(std)
			print(f"No detections for {waveform}, stds: {np.std(ccm.flatten())}, traces: {len(traces)}")
		
		if(download):

			#Erase the file to avoid full storage on local disk
			print(f"Erasing {waveform}")
			os.system(f"rm -r {path_waveforms}/{waveform}")

		params = {"fwtrace":fwtrace.shape[0],"n_channels":n_channels,"fs":fs,"dx":dx}

	return traces, orig_traces, start_times, mccsub, ccpeaks, params, cumfreq_list, freqocc, stds