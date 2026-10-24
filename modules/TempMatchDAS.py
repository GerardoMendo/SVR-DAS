import numpy as np
import obspy
import h5py
import os
import time
import sys

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from matplotlib import ticker

from obspy import read
from obspy import UTCDateTime
from obspy.signal.detrend import spline
import obspy.signal.filter as filt
import obspy.signal.cross_correlation as correlation

import scipy.signal
import scipy.signal as signal
import scipy.stats as stats
from scipy import fftpack

#Adding the modules directory
sys.path.append('/path/to/svrdas/Modules')

from TempMatchDAS_func import *
from detect_peaks import *
from numba import jit
from check_os import *

import multiprocessing

#Retrieving root and media
root, media = check_operating_system()

############################################################################################################################################################################################################################################

def TemplateMatchingDAS(parameters,download=False):

	#Retrieving the parameters from params dictionary

	path_waveforms = parameters["path_wav"]
	path_templates = parameters["path_temp"]
	path_detections = parameters["path_det"]
	path_coeff = parameters["path_coeff"]
	path_traces = parameters["path_tr"]
	envelope_flag, saving_flag = parameters["env_flag"], parameters["save_flag"]
	filter_type = parameters["filter_type"]
	overlap = parameters["overlap"]
	threshold = parameters["threshold"]
	system = parameters["system"]
	
	#Main list that contains all the detections
	dict_events = {}

	# Loading library of templates

	os.chdir(path_templates)
	templates = os.listdir()
	templates_file_list = remove_hidden_files(path_templates,templates)

	#Loading waveform database

	if(download == False):
		os.chdir(path_waveforms)
		waveforms = os.listdir()
		waveforms = remove_hidden_files(path_waveforms,waveforms)
		waveforms.sort()

		#Batch of waveforms
		waveforms = waveforms[5000:20000]

	else:
		waveforms = np.loadtxt("/path/to/svrdas/ListFiles.txt",dtype=str)
		waveforms.sort()

	#Initial array of cc
	cc_total = np.zeros(0)

	#Detection matrix
	detect_traces = []

	#Retrieving the templates array
	templates = retrieve_templates(path_templates,templates_file_list)

	for i,template in enumerate(templates):

		start = time.time()

		print(f"shape: {template.shape[0]}")

		try:
			template_name = templates_file_list[i][:-4]
		except IndexError:
			template_name = str(f"{templates_file_list[0][:-4]}_{i}")

		print(f'Using template: {template_name}')

		if(filter_type=="band"):
				
			coeff_b, coeff_a = scipy.signal.butter(4,(5.0,15.0),"band",fs=100.0)
			ttrace = scipy.signal.filtfilt(coeff_b,coeff_a,template)

		if(filter_type=="high"):

			coeff_b, coeff_a = scipy.signal.butter(2,8.0,"high",fs=100.0)
			ttrace = scipy.signal.filtfilt(coeff_b,coeff_a,template)

		else:
			ttrace = template

		if(envelope_flag):

			ttrace = envelope(ttrace).astype('float32')

		#Applying template matching
		traces, otraces, start_times, mccsub, ccpeaks, params, cumfreq_list,freqocc, stds = template_matching(ttrace,waveforms,overlap,threshold,template_name,path_waveforms,path_coeff,path_traces,path_detections,system=system,envelope_flag=envelope_flag,filter_type=filter_type,root=root,download=download)
		
		#Storing 
		fileName = f"TempMatchDAS_T{template_name}_info.txt"
		writeInfoAscii(path_traces,fileName,{"waveforms":waveforms,"traces":freqocc,"stds":stds,"cumfreq":cumfreq_list})

		end = time.time()

		print(f"Elapsed time: {end - start} seconds")


	
	




