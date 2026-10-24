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
sys.path.append('/home/gerardomendo/Documents/DAS/SVRDAS/modules')

from timeit import default_timer as timer
#from ATMC_functions_DAS import *
#from ATMC_functions_2 import *
from detect_peaks import *
from utils import *
from check_os import *
from plotting_functions_DAS import *
from fileNames import *

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

################################################################################################################

def writeReport(filename,stalta_p,stalta_s,ch_stalta,low_point,up_point):

	p = np.array([stalta_p[str(p)] for p in ch_stalta])
	s = np.array([stalta_s[str(s)] for s in ch_stalta])

	print(f"Storing into {filename}")

	with open(filename,"a") as f:

		for i,ch in enumerate(ch_stalta):

			f.write(f"{ch},{p[i]+low_point},{s[i]+low_point},{low_point},{up_point}\n")

	print(f"Stored")

################################################################################################################

def process_Sanriku1_data(path_original,name_original,name_det,system,dfactor=1,concat=False,download=False):

	if(concat):
		if(download):
			traces, params = download_concat_hdf5(path_original,name_original,name_det,system=system,dfactor=dfactor)
		else:
			traces, params = concat_hdf5(path_original,name_original,name_det,download,system=system,dfactor=dfactor)

		if(dfactor > 1):
			params['fs'] /= dfactor

		L = params['GL']
		dx = params['dx']
		starttime = params['starttime']
		fs = params['fs']

	else:

		if(download):
			if(system is not None):
				download_DAS_data(name_original,path_original,system=system)
			else:
				#Default - Sanriku
				download_DAS_data(name_original,path_original,system="Sanriku")

		data = h5py.File(f"{path_original}/{name_original}","r")
		traces = np.array(data["Acquisition/Raw[0]/RawData"])
		print(f"traces.shape: {traces.shape}")

		#Retrieving data
		fs = data['Acquisition/Raw[0]'].attrs['OutputDataRate']

		if(dfactor > 1):
			traces = signal.decimate(traces,dfactor,axis=1)
			fs /= dfactor
	
		#Spatial sampling interval
		dx = data['Acquisition'].attrs['SpatialSamplingInterval']
	
		#Gauge Length
		L = data['Acquisition'].attrs['GaugeLength']

		#Starttime
		starttime = str(data["Acquisition/Raw[0]/RawData"].attrs["PartStartTime"])[2:-1]

		#Setting the params dict
		params = {'GL':L,'dx':dx,'starttime':starttime,'fs':fs}

	#Conversion to strain units
	lamb = 4e-9;nc = 1.457; eps = 0.78;
	strain = lamb/(4*np.pi*eps*nc*L)

	#Estimating strain rate
	srate = np.dot(traces,strain)
	strainrate = np.gradient(srate,axis=0)*fs

	if(concat == False):
		data.close()
	
	return fdata.T, params

################################################################################################################

def process_Sanriku2_data(path_original,name_original,name_det,system,dfactor=1,concat=False,download=False):

	if(concat):
		if(download):
			traces, params = download_concat_hdf5(path_original,name_original,name_det,system=system,dfactor=dfactor)
		else:
			traces, params = concat_hdf5(path_original,name_original,name_det,download,system=system,dfactor=dfactor)

		if(dfactor > 1):
			params['fs'] /= dfactor

		L = params['GL']
		dx = params['dx']
		starttime = params['starttime']
		fs = params['fs']

	else:

		if(download):
			if(system is not None):
				download_DAS_data(name_original,path_original,system=system)
			else:
				#Default - Sanriku
				download_DAS_data(name_original,path_original,system="Sanriku")

		data = h5py.File(f"{path_original}/{name_original}","r")
		traces = np.array(data["Acquisition/Raw[0]/RawData"])
		print(f"traces.shape: {traces.shape}")

		#Retrieving data
		fs = data['Acquisition/Raw[0]'].attrs['OutputDataRate']

		if(dfactor > 1):
			traces = signal.decimate(traces,dfactor,axis=1)
			fs /= dfactor
	
		#Spatial sampling interval
		dx = data['Acquisition'].attrs['SpatialSamplingInterval']
	
		#Gauge Length
		L = data['Acquisition'].attrs['GaugeLength']

		#Starttime
		starttime = str(data["Acquisition/Raw[0]/RawData"].attrs["PartStartTime"])[2:-1]

		#Setting the params dict
		params = {'GL':L,'dx':dx,'starttime':starttime,'fs':fs}

	#Conversion to strain units
	lamb = 4e-9;nc = 1.457; eps = 0.78;
	strain = lamb/(4*np.pi*eps*nc*L)

	#Estimating strain rate
	srate = np.dot(traces,strain)
	strainrate = np.gradient(srate,axis=0)*fs

	if(concat == False):
		data.close()
	
	return fdata.T, params

################################################################################################################

def process_Nnet_data(path_original,name_original,name_det,system,dfactor=1,concat=False,download=False):

	if(concat):
		if(download):
			traces, params = download_concat_hdf5(path_original,name_original,name_det,system=system,dfactor=dfactor)
		else:
			traces, params = concat_hdf5(path_original,name_original,name_det,download,system=system,dfactor=dfactor)

		if(dfactor > 1):
			print(params)
			params['fs'] /= dfactor

		L = params['GL']
		dx = params['dx']
		starttime = params['starttime']
		fs = params['fs']

	else:

		if(download):
			if(system is not None):
				download_DAS_data(name_original,path_original,system=system)
			else:
				#Default - Sanriku
				download_DAS_data(name_original,path_original,system="Sanriku")

		data = h5py.File(f"{path_original}/{name_original}","r")
		traces = np.array(data["Acquisition/Raw[0]/RawData"])
		print(f"traces.shape: {traces.shape}")

		#Retrieving data
		fs = data['Acquisition/Raw[0]'].attrs['OutputDataRate']

		if(dfactor > 1):
			traces = signal.decimate(traces,dfactor,axis=0)
			fs /= dfactor
	
		#Spatial sampling interval
		dx = data['Acquisition'].attrs['SpatialSamplingInterval']
	
		#Gauge Length
		L = data['Acquisition'].attrs['GaugeLength']

		#Starttime
		starttime = str(data["Acquisition/Raw[0]/RawData"].attrs["PartStartTime"])[2:-1]

		#Setting the params dict
		params = {'GL':L,'dx':dx,'starttime':starttime,'fs':fs}

	#Conversion to strain units
	lamb = 4e-9;nc = 1.457; eps = 0.78;
	strain = lamb/(4*np.pi*eps*nc*L)

	#Estimating strain rate
	srate = np.dot(traces,strain)
	strainrate = np.gradient(srate,axis=0)*fs

	if(concat == False):
		data.close()
		
	print(f"strainrate.T shape: {strainrate.T.shape}")
	
	return strainrate.T, params

################################################################################################################

def process_data(path_original,name_original,name_det,system,dfactor=1,concat=False,download=False,length_data=60):

	if(concat):
		traces, params = newConcathdf5(path_original,name_original,name_det,length_data,system=system,dfactor=dfactor,download=download)

		if(dfactor > 1):
			print(params)
			params['fs'] /= dfactor

		L = params['GL']
		dx = params['dx']
		starttime = params['starttime']
		fs = params['fs']

	else:

		if(download):
			if(system is not None):
				download_DAS_data(name_original,path_original,system=system)
			else:
				#Default - Sanriku
				download_DAS_data(name_original,path_original,system="Sanriku1")

		data = h5py.File(f"{path_original}/{name_original}","r")
		traces = np.array(data["Acquisition/Raw[0]/RawData"])
		print(f"traces.shape: {traces.shape}")

		#Retrieving data
		fs = data['Acquisition/Raw[0]'].attrs['OutputDataRate']

		if(dfactor > 1):
			traces = signal.decimate(traces,dfactor,axis=0)
			fs /= dfactor
	
		#Spatial sampling interval
		dx = data['Acquisition'].attrs['SpatialSamplingInterval']
	
		#Gauge Length
		L = data['Acquisition'].attrs['GaugeLength']

		#Starttime
		starttime = str(data["Acquisition/Raw[0]/RawData"].attrs["PartStartTime"])[2:-1]

		#Setting the params dict
		params = {'GL':L,'dx':dx,'starttime':starttime,'fs':fs}

	#Conversion to strain units
	lamb = 4e-9;nc = 1.457; eps = 0.78;
	strain = lamb/(4*np.pi*eps*nc*L)

	#Estimating strain rate
	srate = np.dot(traces,strain)
	strainrate = np.gradient(srate,axis=0)*fs

	if(concat == False):
		data.close()
		
	print(f"strainrate.T shape: {strainrate.T.shape}")
	
	return strainrate, params

################################################################################################################

def load_data(file,Ldata,dict_paths,concat=False,system=None,download=False):

	#Loading list of paths
	path_original = dict_paths['original']
	path_data = dict_paths['data']
	path_ascii = dict_paths['ascii']

	#Retrieve file names of original data and detection data
	#File names for each system
	#N-net    -> Nnet_QX2_2024-08-14T011711Z.h5
	#Sanriku1 -> ERI_GL100_2022-03-01T061320Z.h5
	#Sanriku2 -> Kama_dtest_100Hz_2025-11-17T235708Z.h5

	#Retrieve names from file variable
	name_original, name_det = return_fileNames(file,system=system)

	#Open path where data is stored
	os.chdir(path_data)

	#retrieve files that match name_det (both filtered and non filter)
	match = glob.glob(f"*{name_det}*NH8*")
	match_or = glob.glob(f"*{name_det}*nofilter*")
	#print(f"File: {file}, Match: {match}, Match (no filter): {match_or}")
	print("######################################################################################")
	#print(f"File: {match}")
	print(f"File: {name_original}")

	w = h5py.File(f"{path_data}/{match[0]}","r")
	w_nofil = h5py.File(f"{path_data}/{match_or[0]}","r")
	meta = np.loadtxt(f"{path_ascii}/{file}",dtype=str)

	#Call function to retrieve data
	res, params = process_data(path_original,name_original,name_det,system,2,concat,download,Ldata)

	#Gauge length per sample
	L = params["GL"]

	#Spatial sampling rate
	dx = params["dx"]

	#Starttime
	starttime = params["starttime"]

	#Sampling rate
	fs = params["fs"]

	#Only for 20-s data of Sanriku - DO NOT USE WITH NNET DATA

	# if(system == 'Sanriku1'):

	# 	res, params = process_Sanriku1_data(path_original,name_original,name_det,system,8,concat,download)
	# 	#res, params = concat_hdf5(path_original,name_original,name_det,system=system)
		
	# 	#Gauge length per sample
	# 	L = params["GL"]

	# 	#Spatial sampling rate
	# 	dx = params["dx"]

	# 	#Starttime
	# 	starttime = params["starttime"]

	# 	#Sampling rate
	# 	fs = params["fs"]

	# #ONLY FOR NNET DATA
	# elif(system == "Nnet"):

	# 	print('Aqui 1')

	# 	res, params = process_Nnet_data(path_original,name_original,name_det,system,2,concat,download)

	# 	#Gauge length per sample
	# 	L = params["GL"]

	# 	#Spatial sampling rate
	# 	dx = params["dx"]

	# 	#Starttime
	# 	starttime = params["starttime"]

	# 	#Sampling rate
	# 	fs = params["fs"]

	print(f'st_data.shape: {res.shape}, fs: {fs}')
	
	#Recovering starttimes
	#ch_idx = meta[:,2].astype(int)
	ch_idx = meta[:,1].astype(int)
	st_idx = meta[:,5].astype(int)
	
	#Retrieve data
	if(concat):
		st_data = np.gradient(res.T, axis=1) * fs
	else:
		st_data = res.T

	olddata = np.array(w["Detections/Data"])
	data_nofil = np.array(w_nofil["Detections/Data"])

	flow_point = meta[:,5].astype(int) + 6000
	fup_point = meta[:,6].astype(int) + 6000

	return st_data, olddata, [flow_point,fup_point], ch_idx, st_idx, [L,dx,starttime,fs]

################################################################################################################

def retrieve_data_percentiles(st_data,percentile,hf_dur=2500):

	c2, c1 = signal.butter(2,(5.0,15.0),"band",fs=100.0)
	filtered = signal.filtfilt(c2,c1,st_data,axis=1)
	nsnr = estimate_snr(filtered)
	sort_nsnr = np.sort(nsnr)
	d = np.where(nsnr > np.percentile(sort_nsnr,97),1,0)

	print(f"Min. SNR: {np.amin(nsnr)}, Max. SNR: {np.amax(nsnr)} - Percentiles") 

	#Retrieving events
	events = [filtered[i] for i in range(0,d.shape[0]) if d[i]==1]
	newchsnr = [i for i in range(0,d.shape[0]) if d[i]==1]
	maxindx = [np.where(elem == np.amax(elem))[0][0] for elem in events]
	ev_snr = [nsnr[i] for i in range(0,d.shape[0]) if d[i]==1]

	half_duration = 2500
	data = np.zeros((len(maxindx),int(2*half_duration)))

	for i,event in enumerate(events):

		try:
			data[i] = event[maxindx[i]-half_duration:maxindx[i]+half_duration] 
		except Exception:
			sliced = event[maxindx[i]-half_duration:maxindx[i]+half_duration]
			data[i,:sliced.shape[0]] = sliced
			data[i,sliced.shape[0]:] = data[i,sliced.shape[0]:] + 0.0001

	#data = np.array([events[i][maxindx[i]-half_duration:maxindx[i]+half_duration] for i in range(0,len(events))])
	
	low_point = [maxindx[i]-half_duration for i in range(0,len(events))]
	up_point = [maxindx[i]-half_duration for i in range(0,len(events))]

	return data, filtered, ev_snr, newchsnr, low_point, up_point

###########################################################################################################################

def check_directory(path):

	try:
		os.chdir(path)
	except FileNotFoundError:
		os.mkdir(path)


################################################################################################################################################################################################################################################################

def save_csv(path_csv,date,channels,picks_sample_p,picks_sample_s,picks_time_p,picks_time_s,method='STALTA',type_data='centered'):

	name_csv = f'Picks_{date}_{method}_{type_data}.csv'

	#Storing statistics
	try:
		f = open(f"{main_path}/{name_csv}","x")
	except FileExistsError:
		print(f'The file {name_csv} already exists')
		os.system(f'rm -r {main_path}/{name_csv}')
		f = open(f"{main_path}/{name_csv}","x")

	for i,ch in enumerate(channels):

		f.write(f"{ch} {picks_sample_p} {picks_time_p} {picks_sample_p} {picks_sample_s}\n")

	f.close()

####################################################################################################################################################################################################################################################################

def onclick(event):

	f = open('/home/gerardomendo/Documents/DAS/Picking/Test.csv','a')
	print(f"Pos x: {event.xdata}, pos y: {event.ydata}")
	f.write(f'{float(event.xdata)}\n')
	f.close()

####################################################################################################################################################################################################################################################################

def pick_manual(trace,fs,zoom=None):

	def close_figure(event):
		if event.key == "x":
			plt.close(event.canvas.figure)

	#plt.ion()
	fig,ax = plt.subplots(1,1,figsize=(30,20))
	t_trace = np.arange(0,trace.shape[0])/fs
	ax.plot(t_trace,trace,lw=0.7,c='k')
	#ax.set_xlim(t_trace[0],t_trace[-1])

	if zoom is not None:

		try:
			assert type(zoom)==list
			ax.set_xlim(zoom[0],zoom[1])
		except AssertionError:
			print(f'Setting default')
			ax.set_xlim(max_point-zoom,max_point+zoom)
	else:
		
		max_point = (np.where(trace == np.amax(trace))[0][0])/fs
		zoom = 20
		ax.set_xlim(max_point-zoom,max_point+zoom)

	cid = fig.canvas.mpl_connect('button_press_event',onclick)
	fig.canvas.mpl_connect('key_press_event', close_figure)
	plt.show()
	

####################################################################################################################################################################################################################################################################

def create_csv_picks(file_csv,new_name_csv,channels):

	try:
		f = open(new_name_csv,"x")
	except Exception:
		os.system(f"rm -r {new_name_csv}")
		f = open(new_name_csv,"x")

	#Loading default test data
	data = np.loadtxt(file_csv,dtype=float)
	picks = [(data[i], data[i+1]) for i in range(0,data.shape[0],2)]

	print(f"Num. picks: {len(picks)}")

	for i,elem in enumerate(picks):

		print(f"{i},{channels[i]},{elem}\n")
		f.write(f"{i},{channels[i]},{elem}\n")

	f.close()
	os.system(f'rm -r {file_csv}')


####################################################################################################################################################################################################################################################################

def writing2csvAll(name_file,ppicks,spicks,chns,date,low_point,methods=2,add_lowpoints=True):

	try:
		f = open(name_file,"x")
	except Exception:
		os.system(f"rm -r {name_file}")
		f = open(name_file,"x")

	f.write(f"UTCDate,Channel,p_stalta,s_stalta,p_ar,s_ar,p_rms,s_rms,p_sp,s_sp\n")

	for i,ch in enumerate(chns):

		exist = True

		#Retrieving methods: 0 - stalta, 1 - ar, 2 - rms, 3 - sp
		#Retrieving picks from dictionaries

		try:
			ps = np.array([ppicks[p][f"{ch}"] for p in range(methods)])
			if(add_lowpoints):
				ps = ps + low_point[i]
		except KeyError:
			print("Non-existent P picking")
			exist = False

		try:
			ss = [spicks[s][f"{ch}"] for s in range(methods)]
			if(add_lowpoints):
				ss = ss + low_point[i]
		except KeyError:
			print("Non-existent S picking")
			exist = False

		if(exist):

			for j in range(methods):

				if(j == 0):

					f.write(f"{date},{ch},{ps[j]},{ss[j]}")

				elif(j == methods - 1):

					f.write(f",{ps[j]},{ss[j]}\n")

				else:

					f.write(f",{ps[j]},{ss[j]}")

	f.close()


####################################################################################################################################################################################################################################################################

def writing2csvPick(name_file,ch_p,ppicks,date):

	try:
		f = open(name_file,"x")
	except Exception:
		os.system(f"rm -r {name_file}")
		f = open(name_file,"x")

	f.write(f"UTCDate,channel,ch_p,p_obtained\n")

	for i,ch in enumerate(ch_p):

		p = ppicks[str(ch)]
		f.write(f"{date},{ch},{p}\n")

	f.close()

####################################################################################################################################################################################################################################################################

def writing2csvModel(name_file,chns,ppicks,spicks,nevsnr,date):

	try:
		f = open(name_file,"x")
	except Exception:
		os.system(f"rm -r {name_file}")
		f = open(name_file,"x")

	f.write(f"UTCDate,channel,ch,p_model,s_model,snr\n")

	print(f'Channels: {chns.shape}')

	for i,ch in enumerate(chns):

		p = ppicks[str(ch)]; s = spicks[str(ch)]; snr = nevsnr[i]
		f.write(f"{date},{ch},{p},{s},{snr}\n")

	f.close()

####################################################################################################################################################################################################################################################################

def writing2csv(path_csv,name_csv,data,channel_list,p,s,fs,datetime,iqr_snr,decChannels):

	#Open csv writing
	try:
		f = open(f"{path_csv}/{name_csv}","x")
	except FileExistsError:
		os.system(f"rm -r {path_csv}/{name_csv}")
		f = open(f"{path_csv}/{name_csv}","x")

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
	
		try:
			for number in range(0,decChannels):
				f.write(f"{datetime},{ch_a[idx][0]},{int(ch_a[idx]*(decChannels))+number},{round(p[str(ch_a[idx])]/fs,3)},{int(p[str(ch_a[idx])])},{round(s[str(ch_b[idx])]/fs,3)},{int(s[str(ch_b[idx])])},{3.0},{2.0}\n")
		except IndexError:
			for number in range(0,decChannels):
				f.write(f"{datetime},{ch_a[idx]},{int(ch_a[idx]*(decChannels))+number},{round(p[str(ch_a[idx])]/fs,3)},{int(p[str(ch_a[idx])])},{round(s[str(ch_b[idx])]/fs,3)},{int(s[str(ch_b[idx])])},{3.0},{2.0}\n")

		#else:
		#	try:
		#		for number in range(0,decChannels):
		#			f.write(f"{datetime},{ch_a[idx][0]},{int(ch_a[idx]*(decChannels))+number},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")

		#	except IndexError:
		#		f.write(f"{datetime},{ch_a[idx]},{int(ch_a[idx]*(decChannels))+number},{0},{round(s[idx],3)},{0},{int(s[idx]*fs)},{0},{2.0}\n")

	f.close()