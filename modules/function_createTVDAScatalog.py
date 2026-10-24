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
#from detect_peaks import *
#from numba import jit
#from map_kurt_skew import *

import h5py
import os

###################################################################################################################################

def concatenate_all_dec(path_file,list_dates,max_range,dec=1):

	if(max_range == 9):

		f1 = h5py.File(f"{path_file}/{list_dates[0]}"); d1 = signal.decimate(np.array(f1["/Acquisition/Raw[0]/RawData"]),dec,axis=1)
		f2 = h5py.File(f"{path_file}/{list_dates[1]}"); d2 = signal.decimate(np.array(f2["/Acquisition/Raw[0]/RawData"]),dec,axis=1)
		f3 = h5py.File(f"{path_file}/{list_dates[2]}"); d3 = signal.decimate(np.array(f3["/Acquisition/Raw[0]/RawData"]),dec,axis=1)

		f4 = h5py.File(f"{path_file}/{list_dates[3]}"); d4 = signal.decimate(np.array(f4["/Acquisition/Raw[0]/RawData"]),dec,axis=1)
		f5 = h5py.File(f"{path_file}/{list_dates[4]}"); d5 = signal.decimate(np.array(f5["/Acquisition/Raw[0]/RawData"]),dec,axis=1)
		f6 = h5py.File(f"{path_file}/{list_dates[5]}"); d6 = signal.decimate(np.array(f6["/Acquisition/Raw[0]/RawData"]),dec,axis=1)

		f7 = h5py.File(f"{path_file}/{list_dates[6]}"); d7 = signal.decimate(np.array(f7["/Acquisition/Raw[0]/RawData"]),dec,axis=1)
		f8 = h5py.File(f"{path_file}/{list_dates[7]}"); d8 = signal.decimate(np.array(f8["/Acquisition/Raw[0]/RawData"]),dec,axis=1)
		f9 = h5py.File(f"{path_file}/{list_dates[8]}"); d9 = signal.decimate(np.array(f9["/Acquisition/Raw[0]/RawData"]),dec,axis=1)

		#Gauge length per sample
		L = f1["Acquisition"].attrs["GaugeLength"]

		#Spatial sampling rate
		dx = f1["Acquisition"].attrs["SpatialSamplingInterval"]

		#Starttime
		starttime = f1["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']

		#Sampling rate
		fs = f1["Acquisition/Raw[0]"].attrs["OutputDataRate"]

		parameters = {"GL":L,"dx":dx,"starttime":starttime,"fs":fs}

		res = np.vstack((d1,d2,d3,d4,d5,d6,d7,d8,d9))

	if(max_range == 3):

		f1 = h5py.File(f"{path_file}/{list_dates[0]}"); d1 = np.array(f1["/Acquisition/Raw[0]/RawData"])
		f2 = h5py.File(f"{path_file}/{list_dates[1]}"); d2 = np.array(f2["/Acquisition/Raw[0]/RawData"])
		f3 = h5py.File(f"{path_file}/{list_dates[2]}"); d3 = np.array(f3["/Acquisition/Raw[0]/RawData"])

		#Gauge length per sample
		L = f1["Acquisition"].attrs["GaugeLength"]

		#Spatial sampling rate
		dx = f1["Acquisition"].attrs["SpatialSamplingInterval"]

		#Starttime
		starttime = f1["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']

		#Sampling rate
		fs = f1["Acquisition/Raw[0]"].attrs["OutputDataRate"]

		parameters = {"GL":L,"dx":dx,"starttime":starttime,"fs":fs}

		res = np.hstack((d1,d2,d3))

	return res,parameters

####################################################################################################################################

def concatenate_all(path_file,list_dates,max_range):

	if(max_range == 9):

		f1 = h5py.File(f"{path_file}/{list_dates[0]}"); d1 = np.array(f1["/Acquisition/Raw[0]/RawData"])
		f2 = h5py.File(f"{path_file}/{list_dates[1]}"); d2 = np.array(f2["/Acquisition/Raw[0]/RawData"])
		f3 = h5py.File(f"{path_file}/{list_dates[2]}"); d3 = np.array(f3["/Acquisition/Raw[0]/RawData"])

		f4 = h5py.File(f"{path_file}/{list_dates[3]}"); d4 = np.array(f4["/Acquisition/Raw[0]/RawData"])
		f5 = h5py.File(f"{path_file}/{list_dates[4]}"); d5 = np.array(f5["/Acquisition/Raw[0]/RawData"])
		f6 = h5py.File(f"{path_file}/{list_dates[5]}"); d6 = np.array(f6["/Acquisition/Raw[0]/RawData"])

		f7 = h5py.File(f"{path_file}/{list_dates[6]}"); d7 = np.array(f7["/Acquisition/Raw[0]/RawData"])
		f8 = h5py.File(f"{path_file}/{list_dates[7]}"); d8 = np.array(f8["/Acquisition/Raw[0]/RawData"])
		f9 = h5py.File(f"{path_file}/{list_dates[8]}"); d9 = np.array(f9["/Acquisition/Raw[0]/RawData"])

		#Gauge length per sample
		L = f1["Acquisition"].attrs["GaugeLength"]

		#Spatial sampling rate
		dx = f1["Acquisition"].attrs["SpatialSamplingInterval"]

		#Starttime
		starttime = f1["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']

		#Sampling rate
		fs = f1["Acquisition/Raw[0]"].attrs["OutputDataRate"]

		parameters = {"GL":L,"dx":dx,"starttime":starttime,"fs":fs}

		res = np.vstack((d1,d2,d3,d4,d5,d6,d7,d8,d9))

	if(max_range == 3):

		f1 = h5py.File(f"{path_file}/{list_dates[0]}"); d1 = np.array(f1["/Acquisition/Raw[0]/RawData"])
		f2 = h5py.File(f"{path_file}/{list_dates[1]}"); d2 = np.array(f2["/Acquisition/Raw[0]/RawData"])
		f3 = h5py.File(f"{path_file}/{list_dates[2]}"); d3 = np.array(f3["/Acquisition/Raw[0]/RawData"])

		#Gauge length per sample
		L = f1["Acquisition"].attrs["GaugeLength"]

		#Spatial sampling rate
		dx = f1["Acquisition"].attrs["SpatialSamplingInterval"]

		#Starttime
		starttime = f1["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']

		#Sampling rate
		fs = f1["Acquisition/Raw[0]"].attrs["OutputDataRate"]

		parameters = {"GL":L,"dx":dx,"starttime":starttime,"fs":fs}

		res = np.hstack((d1,d2,d3))

	return res,parameters

####################################################################################################################################

def change_UTCformat(datetime_utc):

	atime = datetime_utc

	amonth = atime.month
	if(amonth < 10):
		amonth = str(f"0{amonth}")
	else:
		amonth = str(amonth)

	aday = atime.day
	if(aday < 10):
		aday = str(f"0{aday}")
	else:
		aday = str(aday)

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

	name_after = f"{atime.year}-{amonth}-{aday}T{ahour}{amin}{asec}Z"

	return name_after

##################################################################################################################################

def create_attrs_lists(channels,parameters):

	attrs = []
	p_sec = parameters[0]
	p_picks = parameters[1]
	s_sec = parameters[2]
	s_picks = parameters[3]
	p_std = parameters[4]
	s_sec = parameters[5]

	data_keys = [

	'back_azimuth_deg', 
	'coda_end_sample', 
	'network_code', 
	'p_arrival_sample', 
	'p_status', 
	'p_travel_sec', 
	'p_weight', 
	'receiver_code', 
	'receiver_elevation_m', 
	'receiver_latitude', 
	'receiver_longitude', 
	'receiver_type', 
	's_arrival_sample', 
	's_status', 
	's_weight', 
	'snr_db',
	'source_depth_km',
	'source_depth_uncertainty_km',
	'source_distance_deg',
	'source_distance_km',
	'source_error_sec',
	'source_gap_deg',
	'source_horizontal_uncertainty_km',
	'source_id',
	'source_latitude',
	'source_longitude',
	'source_magnutude',
	'source_magnitude_author',
	'source_magnitude_type',
	'source_mechanism_strike_dip_rake',
	'source_origin_time',
	'source_origin_uncertainty_sec',
	'trace_category',
	'trace_name',
	'trace_start_time'
	
	]

	for idx,ch in enumerate(channels):

		#print(f"idx: {idx}, ch: {ch}, p_picks: {len(p_picks)}")

		attrs.append([0,0,"DAS",p_picks[idx],"auto",p_sec,1.0,"ERI_GL100",0,0,0,"DAS",s_picks[idx],"auto",1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

	return attrs, data_keys	

##################################################################################################################################

def create_attrs_lists_noise(max_idx):

	attrs = []

	data_keys = [

	'back_azimuth_deg', 
	'coda_end_sample', 
	'network_code', 
	'p_arrival_sample', 
	'p_status', 
	'p_travel_sec', 
	'p_weight', 
	'receiver_code', 
	'receiver_elevation_m', 
	'receiver_latitude', 
	'receiver_longitude', 
	'receiver_type', 
	's_arrival_sample', 
	's_status', 
	's_weight', 
	'snr_db',
	'source_depth_km',
	'source_depth_uncertainty_km',
	'source_distance_deg',
	'source_distance_km',
	'source_error_sec',
	'source_gap_deg',
	'source_horizontal_uncertainty_km',
	'source_id',
	'source_latitude',
	'source_longitude',
	'source_magnutude',
	'source_magnitude_author',
	'source_magnitude_type',
	'source_mechanism_strike_dip_rake',
	'source_origin_time',
	'source_origin_uncertainty_sec',
	'trace_category',
	'trace_name',
	'trace_start_time'
	
	]

	for idx in range(max_idx):

		attrs.append([0,0,"DAS",0,"",0,0.0,"ERI_GL100",0,0,0,"DAS",0,"",0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

	return attrs, data_keys	

####################################################################################################################################################################################

def shift_data(trace,psample,ssample,starttime,fs,npositions=100,winlength=3000):

	length = trace.shape[0]
	random = np.random.rand(npositions)
	rpoints = np.array([int(elem*length) for elem in random])

	sav_rpoints = []
	traces = []
	newsamples = []
	newST = [] 

	for i in range(npositions):
		ini = rpoints[i] - winlength
		end = rpoints[i] + winlength
		#print(f"ini: {ini}, end: {end}")
		condp = bool(np.where(np.arange(ini,end) == psample)[0].shape[0])
		conds = bool(np.where(np.arange(ini,end) == ssample)[0].shape[0])
		if(condp and conds):
			sav_rpoints.append(rpoints[i])
		else:
			#print(f"Range outside picks")
			pass

		#newtrace =trace[rpoints[i]-winlength:rpoints[i]+winlength]
	for point in sav_rpoints:
		traces.append(trace[point-winlength:point+winlength])
		s_sample = int(ssample) - (point-winlength)
		p_sample = int(psample) - (point-winlength)
		newST.append(UTCDateTime(starttime) + ((point-winlength)/fs))
		newsamples.append(np.array([p_sample,s_sample]))

	return np.array(traces), np.array(newsamples), np.array(sav_rpoints), np.array(newST) 

####################################################################################################################################################################################



