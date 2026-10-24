import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from obspy import read
import scipy.signal
from scipy import fftpack
import scipy.stats as stats
import obspy.signal.cross_correlation as ost
import obspy
from obspy import UTCDateTime
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
from numba import cuda
from download_data import *

#################################################################################################################

def estimateDASConcat(name_det,original,utc,nsecs,system,status="plus"):

	if(status == "plus"):

		atime = utc + nsecs

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

		if(system == 'Nnet'):
			name = f"{original[:9]}{minafter}.h5"
		else:
			name = f"{original[:10]}{minafter}.h5"

	elif(status == "minus"):

		atime = utc - nsecs

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
		if(system == 'Nnet'):
			name = f"{original[:9]}{minbefore}.h5"
		else:
			name = f"{original[:10]}{minbefore}.h5"

	return name

#################################################################################################################

def decimateSanriku1(data,N0,N1):
	t = np.array(data["/Acquisition/Raw[0]/RawData"],dtype=np.float32)[:,N0:N1];
	d = signal.decimate(t,8,axis=1);
	d = signal.decimate(d,8,axis=0);
	return d

###################################################################################################################

def decimateNnet(data):
	t = np.array(data["/Acquisition/Raw[0]/RawData"],dtype=np.float32);
	d = signal.decimate(t,2,axis=0);
	return d

###################################################################################################################

def checkLengthData(size):

	assert size != 0, "Must enter a length data value"

####################################################################################################################

def newConcathdf5(path_original,original,name_det,Ldata,system=None,dfactor=1,download=False):

	#Use assert to evaluate if data is not zero
	checkLengthData(Ldata)

	#Convert date into UTC Date Format
	utc = UTCDateTime(name_det)

	#Number of data based on Ldata to create 60-s DAS sections for template matching
	N = int(60/Ldata)

	#Estimate index where central data is
	idx = int(20/Ldata)

	#Limits to obtain concatenations
	NB = idx - 1
	NA = N - 1

	#Parameter for names_after
	nLimit = N - idx - 1

	#Upper and lower limits for names before and after
	lowLimit = int(Ldata*idx)
	upLimit = int(nLimit*Ldata)+Ldata

	#print(f'Params: N = {N}, idx = {idx}, NB = {NB}, NA = {NA}, nLimit = {nLimit}, lowLimit = {lowLimit}, upLimit = {upLimit}')

	#Estimating names before and after
	names_before = [estimateDASConcat(name_det,original,utc,n,system,status="minus") for n in range(lowLimit,0,-Ldata)]
	names_after = [estimateDASConcat(name_det,original,utc,n,system,status="plus") for n in range(Ldata,upLimit,Ldata)]

	#print(f'Names before: {names_before}')
	#print(f'Names after: {names_after}')
	
	#List to append resultant traces
	ds = []

	#Loop that iterates over names before list
	for i,names in enumerate(names_before):

		#print(f"Reading {path_original}/{names}")

		if(download):
			#print(f"Downloading {names}")
			download_DAS_data(names,path_original,system=system)

		f = h5py.File(f"{path_original}/{names}")
		if(system == 'Sanriku1'):

			d1 = decimateSanriku1(f,0,25000);
			d2 = decimateSanriku1(f,25000,50000);
			ds.append(np.hstack((d1,d2)))

		elif(system == 'Nnet'):
			d = decimateNnet(data)
			ds.append(d)

		if(i == 0):

			#print(f"Aqui")

			#Gauge length per sample
			L = f["Acquisition"].attrs["GaugeLength"]
			
			#Spatial sampling rate
			dx = f["Acquisition"].attrs["SpatialSamplingInterval"]
			
			#Starttime
			starttime = f["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']
			
			#Sampling rate
			fs = f["Acquisition/Raw[0]"].attrs["OutputDataRate"]

		f.close()

		if(download):
			os.system(f"rm -r {path_original}/{names}")

	if(download):
		#print(f"Downloading {original}")
		download_DAS_data(original,path_original,system=system)

	#print(f"Reading {path_original}/{original}")

	f = h5py.File(f"{path_original}/{original}")
	traces = np.array(f["/Acquisition/Raw[0]/RawData"]).astype(np.float32)
	if(system == "Sanriku1"):
		d1 = decimateSanriku1(f,0,25000);
		d2 = decimateSanriku1(f,25000,50000);
		ds.append(np.hstack((d1,d2)))

	f.close()

	if(download):
		os.system(f"rm -r {path_original}/{original}")

	elif(system == 'Nnet'):

		d = decimateNnet(data)
		ds.append(d)

	#Loop that iterates over names after list
	for i,names in enumerate(names_after):

		#print(f"Reading {path_original}/{names}")

		if(download):
			#print(f"Downloading {names}")
			download_DAS_data(names,path_original,system=system)

		f = h5py.File(f"{path_original}/{names}")
				
		if(system == "Sanriku1"):
			d1 = decimateSanriku1(f,0,25000);
			d2 = decimateSanriku1(f,25000,50000);
			ds.append(np.hstack((d1,d2)))

		elif(system == 'Nnet'):
			d = decimateNnet(data)
			ds.append(d)
		
		f.close()

		if(download):
			os.system(f"rm -r {path_original}/{names}")

	res = np.vstack(tuple(ds))

	#print(f'Output shape: {res.shape}')

	parameters = {"GL":L,"dx":dx,"starttime":starttime,"fs":fs}

	return res,parameters

##################################################################################################################################################

def process_data(path_original,name_original,name_det,system,dfactor=1,concat=False,download=False,filter_type=None,stack=False,length_data=60):

	assert system != None, "You must specify a system name"

	if(concat):
		traces, params = newConcathdf5(path_original,name_original,name_det,length_data,system=system,dfactor=dfactor,download=download)

		if(dfactor > 1):
			#print(params)
			params['fs'] /= dfactor

		L = params['GL']
		dx = params['dx']
		starttime = params['starttime']
		fs = params['fs']

	else:

		if(download):
			download_DAS_data(name_original,path_original,system=system)

		data = h5py.File(f"{path_original}/{name_original}","r")
		traces = np.array(data["Acquisition/Raw[0]/RawData"])
		#print(f"traces.shape: {traces.shape}")

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

	if(filter_type == "band"):
		c2, c1 = signal.butter(4,(5.0,15.0),"band",fs=fs)
	else:
		c2, c1 = signal.butter(4,8.0,filter_type,fs=fs)

	#Applying filter
	if(filter_type is not None):
		fwtrace = signal.filtfilt(c2,c1,strainrate,axis=0)
	else:
		fwtrace = strainrate

	if(concat == False):
		data.close()

	#Stacking channels
	if(stack):
		strainrate = stacking(strainrate,factor=5)
		fwtrace = stacking(fwtrace,factor=5)
		
	#print(f"strainrate.T shape: {strainrate.T.shape}")
	if(download and path_original.split('/')[-1] == 'temp'):
		print(f'Erasing {name_original} from temp dir')
		os.chdir(path_original)
		os.system('rm -r *')
	
	return strainrate, fwtrace, params

#####################################################################################################

#writeInfoAscii(path_output,fileName,{"traces":traces,"stds":stds,"cumfreq":{cumfreq_list}})
def writeInfoAscii(path_output,file_name,fields):

	#Retrieving fields
	waveforms = fields["waveforms"]
	traces = fields["traces"]
	std = fields["stds"]
	cumfreq = fields["cumfreq"]

	#Open file to save
	with open(f"{path_output}/{file_name}","a") as f:
		
		#Waveform file name, number of traces, CC std, cumfreq of detections

		for i in range(0,len(waveforms)):
			f.write(f"{waveforms[i]} {traces[i]} {std[i]} {cumfreq[i]}\n")

	f.close()

#####################################################################################################

def writeDetectionsAscii(path_output,file_name,peaks,waveform,fs,factor,template_size,stack=5.0):

	#Form of peaks: peaks[channel] = peak
	#Retrieve channels for dictionary

	peak_keys = [k for k in peaks.keys()]
	#print(f"number of peak keys: {len(peak_keys)}")
	
	try:
		#starttime = UTCDateTime(waveform[10:-6])
		starttime = UTCDateTime(waveform[9:-4])
		#print(f"Waveform: {UTCDateTime(starttime)}, fs:{fs} Hz, path output: {path_output}")
	except TypeError:
		starttime = UTCDateTime(waveform[15:-6])
		#print(f"Waveform: {UTCDateTime(starttime)}, fs:{fs} Hz, path output: {path_output}")

	f = open(f"{path_output}/{file_name}","a")

	#starttime = UTCDateTime(waveform[10:-6])
	c = 0 

	for key in peak_keys:

		ch_detect = peaks[key]
		if(ch_detect.shape[0] > 1):
			ch_detect = [peaks[key][0]]

		for peak in ch_detect:

			time_sec = (factor*int(peak))/fs
			time_det = starttime + time_sec

			startwin = int(peak*factor)
			endwin = int(startwin + template_size)

			if(stack is not None and stack > 0):

				#Starttime(UTC) No.Channel(after stacking and decimating) No.Channel(after decimating) OriginalChannel TimeSeconds
				f.write(f"{time_det} {key} {int(int(key)*(stack))} {int(int(key)*stack*8)} {time_sec} {startwin} {endwin}\n")
				c += 1

			else:

				#Starttime(UTC) No.Channel(after stacking and decimating) No.Channel(after decimating) OriginalChannel TimeSeconds
				f.write(f"{time_det} {key} {key} {int(int(key)*stack)} {time_sec} {startwin} {endwin}\n")

	f.close()

#########################################################################################

#@jit(nopython=True)
def correlation_coefficient(sxy,a,b):
        
    #prod_sx = np.sum((a**2-np.mean(a)**2))
    #prod_sy = np.sum((b**2-np.mean(b)**2))
    #norm = (prod_sx*prod_sy)**0.5


    prod_sx = np.inner(a,a)
    prod_sy = np.inner(b,b)
    norm = np.sqrt(prod_sx*prod_sy)
    #print(f"Norm: {norm}")

    #aa = scipy.signal.correlate(a,a)
    #bb = scipy.signal.correlate(b,b)
    #norm = np.sqrt(np.amax(np.abs(bb))* np.amax(np.abs(aa)))       

    #aa = stats.tstd(a)
    #bb = stats.tstd(b)
    #norm = aa*bb       

    c_coef = abs(sxy)/norm
    
    return c_coef
    
##########################################################################################

def remove_hidden_files(path,list_elements):

	os.chdir(path)
	remove = glob.glob(".*")

	for item in remove:

		list_elements.remove(item)

	return list_elements

###########################################################################################
#Stacking function - for DAS channels

def stacking(array,factor=10):

    #Number of channels
    n_channels = array.shape[1]
    n_samples = array.shape[0]

    new_nchannels = int(n_channels/factor)
    new_data = np.zeros((n_samples,new_nchannels))
    li = 0; ls = factor

    for i in range(new_nchannels):
      
      extract = array[:,li:ls]
      new_data[:,i] = np.sum(extract,axis=1)
      li = ls; ls += factor

    return new_data

##########################################################################################
#Function that creates non-existing files

def create_file(channel,month,day,hour):

	prev_data = 'PO.POCB.'+ channel + '_' + '2021'+ str(month) + day +'_210000.sac'
	a = read(prev_data,debug_headers=True)
	index = a[0].data.shape[0]
	
	#Changes values inside SAC file by zeros
	for i in range(0,index,1):
		a[0].data[i]=1
	
	#new_name=channel+'.'+month+'.'+str(day)+'.'+str(hour)+'.sac'
	new_name = 'PO.POCB.'+ channel + '_' + '2021'+ str(month) + day +'_'+ hour +'0000.sac'
	new_file=a.write(new_name)
	print('Accomplished')
	return new_file

###### Function that extract time windows and do cross correlation #########

def cut_and_correlate(template,time_series,windows,nt,nts):
	l_l = 0
	u_l = nt
	coeff = []
	for l in range(0,windows,1):
		array = time_series[l_l:u_l]
		correl = ost.correlate(array,template,2)
		correl = abs(correl); shift,value = ost.xcorr_max(correl)
		coeff.append(value)
		l_l = l_l + 1
		u_l = u_l + 1
	return coeff

####################### RMS calculation #####################################

def RMS(array):
	suma = 0
	N = array.shape[0]
	for i in range(0,N,1):

		square_comp = array[i]**2
		suma = suma + square_comp
	
	value = suma/N
	rms = np.sqrt(value)
	return rms

################## Estimation of dates ######################################

def estimation_of_date(dwindow,fs,window_length):
	
	#Return in seconds the position of the first sample of the detected window
	dsec_window = dwindow/fs

	#Estimate minutes and hours
	dhour = math.floor(dwindow/(3600*fs))  
	dmin = math.floor(dwindow/(60*fs))
	res_min = abs(dmin-(dwindow/(60*fs)))
	dsec = math.floor(res_min*60)
	dsam_start = dwindow
	dsam_end = dwindow + window_length

	return dsec,dmin,dhour,dsec_window,dsam_start,dsam_end

################# Envelope #################################################

def envelope(input_signal):

	#Calculate envelope
	analytic_signal = hilbert(input_signal)
	envelope = np.abs(analytic_signal)
	smooth = savgol_filter(envelope,30,3)

	return smooth

################# Coefficient ##############################################

#@jit(nopython=True)
def correlation_coefficient_DAS(sxy,a,b):
	
	#Calculate the inner product of 2D arrays
	dima = a.shape[0]
	dimb = b.shape[0]

	prod_sx = np.mean(np.inner(a,a).reshape(dima*dima,1))
	prod_sy = np.mean(np.inner(b,b).reshape(dimb*dimb,1))
	#print(f"prod_sx shape: {prod_sx.shape}, prod_sy shape: {prod_sy.shape}")
	norm = np.sqrt(prod_sx*prod_sy)
	ccoef = np.divide(sxy,a,b)
	abs_ccoef = np.amax(abs(ccoef),axis=0)

	return abs_ccoef

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

###########################################################################

def retrieve_DASdata(input_file,dfactor,len_data=60,filtering=True,stack=True,filter_type=None,dec_axis=None,system=None):

	################################################################################################

	#DAS data - Conversion from optical phase to strain units
	#Parameters 

	lamb = 1550e-9  #Wavelength of user laser light in a vacuum [m]
	nc = 1.457   #Refraction index of a fiber                                                                                  
	eps = 0.78   #Optical-elastic coefficient for the fiber direction in isotropic media
	vp = 3500    #Plane wave velocity [m/s]

	################################################################################################

	#Retrieve data from DAS files - default to OptaSense DAS
	data = input_file.get('Acquisition/Raw[0]/RawData')

	#PENDING - change to support other DAS data hierarhical groups
	if(data == None):

		data = input_file.get('Merge/Data')

	##################################################################################

	#DATA STRUCTURE - MUST BE IN [NSAMPLES,NCHANNELS]
	data = np.array(datax)

	#DECIMATION - APPLICABLE ONLY IF dfactor > 1 AND dec_axis IS EITHER "time" or "length"
	if(dfactor > 1):

		if(dec_axis == "time"):
			data = signal.decimate(data,dfactor,axis=1)
		elif(dec_axis == "length"):
			data = signal.decimate(data,dfactor,axis=0)
		else:
			data = signal.decimate(data,dfactor,axis=1)

	#Recovering Gauge Length (L), sampling rate (fs), and spatial sampling (dx)
	try:

		#Gauge length per sample
		L = input_file["Acquisition"].attrs["GaugeLength"]

		#Spatial sampling rate
		dx = input_file["Acquisition"].attrs["SpatialSamplingInterval"]

		#Sampling rate
		fs = input_file["Acquisition/Raw[0]"].attrs["OutputDataRate"]/dfactor

	except Exception:

		#Gauge length per sample
		L = input_file["Merge/Data"].attrs["GaugeLength"]

		#Spatial sampling rate
		dx = input_file["Merge/Data"].attrs["SpatialSamplingInterval"]

		#Sampling rate
		fs = input_file["Merge/Data"].attrs["OutputDataRate"]/dfactor

	#Strain conversion coefficient
	strain = lamb/(4*np.pi*eps*nc*L)

	#Conversion to strain rate Sanriku - multiply by strain is NOT required for Sanriku one minute files STORED ON DISKS
	
	if(system == "Sanriku1"):

		#UNCOMMENT IF DATA IS DOWNLOADED FROM SERVER OR WAS NOT CONVERTED USING RESIZING PROGRAM
		#st_data = np.dot(data,strain)

		st_data = data
		grad = np.gradient(st_data, axis=0) * fs

	#Conversion to strain rate - Nnet

	elif(system == "Nnet"):

		st_data = np.dot(data,strain)
		grad = np.gradient(st_data, axis=0) * fs

	else:

		grad = data


	#Filtering before dividing
	if(filter_type=="band"):

		#Filter parameters
		w1 = 5.0; w2 = 15.0
		coeff_b, coeff_a = scipy.signal.butter(4,(w1,w2),filter_type,fs=fs)
		fwtrace = scipy.signal.filtfilt(coeff_b,coeff_a,grad,axis=0)

	elif(filter_type=="high"):

		w = 8.0
		coeff_b, coeff_a = scipy.signal.butter(2,w,filter_type,fs=fs)
		fwtrace = scipy.signal.filtfilt(coeff_b,coeff_a,grad,axis=0)

	elif(filter_type==None):
		
		fwtrace = grad

	#Stacking channels
	if(stack):
		traces = stacking(fwtrace,factor=5)
	else:
		traces = fwtrace

	return traces, dx, fs

###########################################################################

def create_hdf5(name_file,path,matrix,fields):

	#Creating new file
	nw = h5py.File(f"{path}/{name_file}","w")

	#Creating subgroups

	grp1 = nw.create_group("Detections")
	grp2 = nw.create_dataset("Detections/Data",data=matrix,dtype=float)

	for i in fields:

		grp2.attrs[i] = fields[i]

###########################################################################

def retrieve_templates(path_templates,templates_file_list):

	#Setting number of templates
	n_temp = len(templates_file_list)
	templates = []

	for i,filename in enumerate(templates_file_list):

		if(filename[-3:] == "txt"):

			trace = np.loadtxt(f"{path_templates}/{filename}",dtype=float)
			templates.append(trace)

		elif(filename[-4:] == "hdf5"):

			trace = h5py.File(f"{path_templates}/{filename}","r")
			templates = trace.get("/Detections/Data")

		else:
			tr = read(f"{path_templates}/{filename}")
			templates.append(tr[0].data)

	if((str(type(templates))=="<class 'numpy.ndarray'>")):

		return templates

	else:

		return np.array(templates)

###########################################################################

def template_matching(template, waveforms, overlap, threshold, template_name, path_waveforms, path_coeff, path_traces, path_detections,system=None,envelope_flag=True,filter_type=None,root=None,download=False):

	assert system != None, "You must specify a system (Sanriku1,Sanriku2,Nnet)"

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

	#Length of waveforms list
	N = len(waveforms)

	#Loop for using each waveform
	#print(f"Len waveforms: {len(waveforms)}")

	#Searching between 1 and N-1 waveform data
	#waveforms = waveforms[1:N-1]

	for i,waveform in enumerate(waveforms):

		#if(i > 0):
		#	break

		#i = 256; waveform = waveforms[256]

		#print(f'Searching in index {i}, waveform: {waveform}')

		start = timer()

		#Retrieve time from waveform file name
		name_det = waveform.split("_")[2][:-3]

		#Flag for detections
		detections = False

		#######################################################################################################################
		# if(download):
		# 	download_DAS_data(waveform,path_waveforms,system=system)

		# #Loading DAS data
		# w = h5py.File(f"{path_waveforms}/{waveform}","r")

		# #Stacking channels (Original)
		# orig_trace, dx, fs = retrieve_DASdata(w,2,stack=True,filter_type=None,dec_axis="length",system=system)

		# #Stacking channels (Filtered)
		# fwtrace, dx, fs = retrieve_DASdata(w,2,stack=True,filter_type=filter_type,dec_axis="length",system=system)

		# # plt.ion()
		# # fig, ax = plt.subplots(1,1,figsize=(10,10))
		# # c = ax.imshow(fwtrace,aspect="auto",interpolation=None,cmap="seismic",origin="upper",vmin=-np.percentile(fwtrace,90),vmax=np.percentile(fwtrace,90))

		# #print(f"fwtrace shape: {fwtrace.shape}")
		# #input("Press Enter to continue")

		# #Parameters
		# try:
		# 	starttime = w["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']
		# 	start_times.append(starttime)
		# except Exception:
		# 	starttime = str(w["Merge/Data"].attrs['PartStartTime'],'utf-8')
		# 	start_times.append(starttime)

		#################################################################################################################################################################

		#Obtaining traces
		orig_trace, fwtrace, paramsData = process_data(path_waveforms,waveform,name_det,system,dfactor=2,concat=False,download=download,filter_type=filter_type,stack=True,length_data=60)
		#print(f'Orig trace shape: {orig_trace.shape}, fwtrace shape:')

		#Retrieving parameters
		dx = paramsData["dx"]
		fs = paramsData["fs"]
		starttime = paramsData["starttime"]

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

		if(len(traces) > 30):

			detections = True
			file = f"TM_Detect_Nnet{template_name}_{waveform}_info_5000_20000.txt"
			conv_fac = n_template*(1 - overlap)

		#Plotting
		if(detections):

			plotting_cc(path_coeff,mccsub,starttime,f"{template_name}_W{waveform}",params)

			# Dictionary for saving parameters
			fields = {"Start_time":starttime,"fs":params["fs"], "dx":params["dx"],"number_traces":len(traces),"number_channels":params["n_channels"]}

			# Creating detection file - filtered data
			create_hdf5(f"Detections_DAS_{waveform[:-3]}_NH8.hdf5",path_detections,np.array(traces),fields)

			# Creating detection file - no filter
			create_hdf5(f"Detections_DAS_{waveform[:-3]}_nofilter.hdf5",path_detections,np.array(orig_traces),fields)

			# Creating detection file - complete trace
			#create_hdf5(f"Detections_DAS_{str(starttime)}_nofilter.hdf5",path_detections,np.array(orig_traces),fields)

			# Estimating CC matrix and plotting results
			ccmatrix = create_correlation_matrix(f"{path_detections}/Detections_DAS_{waveform[:-3]}_NH8.hdf5")
			stds.append(np.std(ccmatrix.flatten()))

			#Saving detections info
			writeDetectionsAscii(path_coeff,file,ccpeaks,waveform,fs,conv_fac,n_template)
			
			#Showing to user
			print(f"Detection for {waveform}, stds: {np.std(ccmatrix.flatten())}, traces: {len(traces)}. Elapsed time: {timer() - start} s.")

		else:

			ccm = create_CC(traces)
			std = np.std(ccm.flatten())
			stds.append(std)
			print(f"No detections for {waveform}, stds: {np.std(ccm.flatten())}, traces: {len(traces)}. Elapsed time: {timer() - start} s.")
		
		params = {"fwtrace":fwtrace.shape[0],"n_channels":n_channels,"fs":fs,"dx":dx}

	return traces, orig_traces, start_times, mccsub, ccpeaks, params, cumfreq_list, freqocc, stds

######################################################################################################################################################
######################################################################################################################################################

def plotting_traces(main_path,file_name,traces,fs):

		fig, ax = plt.subplots(1,1,figsize=(30,20))
		offset = 0; inc = 10; scale = 20

		for i,trace in enumerate(traces):

			new_trace = trace/np.amax(trace)
			arr_trace = np.arange(0,trace.shape[0])/fs
			ax.plot(arr_trace,offset + (scale*new_trace) ,linewidth=0.7,c="k")
			offset += inc

		ax.set_xlim(arr_trace[0],arr_trace[-1])
		ax.set_ylabel("Number of channels",fontsize=16.0,labelpad=10.0)
		ax.set_xlabel("Time [s]",fontsize=16.0,labelpad=10.0)
		plt.savefig(f"{main_path}{file_name}_all_traces.png")
		plt.close()

#Plotting correlation coefficients

######################################################################################################################################################
######################################################################################################################################################

def plotting_cc(path_coeff,mccsub,start_time,file_name,params):

	fig, ax = plt.subplots(1,1,figsize=(20,10))
	t = np.arange(0,params["fwtrace"])/params["fs"]
	#x = np.arange(0,params["n_channels"])*(16.33*5)
	x = np.arange(0,params["n_channels"])*(40.83)
	c = ax.imshow(mccsub, cmap="jet", vmin=0.8, vmax=1.0, aspect="auto", interpolation="lanczos",extent=(x[0], x[-1], t[-1], t[0]))
	ax.set_xlabel("Length (m)",labelpad=10.0,fontsize=20)
	ax.set_ylabel("Time (s)",labelpad=10.0,fontsize=20)
	ax.set_title(f"DAS data - {UTCDateTime(start_time)}")
	cbar = plt.colorbar(c)
	cbar.set_label('|Correlation Coefficient|')
	fig.savefig(f"{path_coeff}{file_name}_absCC.pdf")
	plt.close()

######################################################################################################################################################

def create_correlation_matrix(namefile):

	a = h5py.File(namefile)
	traces = list(a['Detections/Data'])

	ccmatrix = np.zeros((len(traces),len(traces)))

	for j,eva in enumerate(traces):

		for k,evb in enumerate(traces):

			ccmatrix[j,k] = np.amax(correlation_coefficient(signal.correlate(eva,evb),eva,evb))

	return ccmatrix

######################################################################################################################################################

def create_CC(traces):

	ccmatrix = np.zeros((len(traces),len(traces)))

	for j,eva in enumerate(traces):

		for k,evb in enumerate(traces):

			cc = np.amax(correlation_coefficient(signal.correlate(eva,evb),eva,evb))
			#print(cc)

			# plt.ion()
			# fig, ax = plt.subplots(3,1,figsize=(10,10))
			# ax[0].plot(eva,lw=0.7,c="k")
			# ax[1].plot(evb,lw=0.7,c="k")
			# ax[2].plot(signal.correlate(eva,evb),lw=0.7,c="k")
			# print(f"cc: {cc}, ")
			# input("Press Enter to continue...")

			if(cc == np.nan):
				print(f"Found a NaN number")
				ccmatrix[j,k] = 0
			else:
				ccmatrix[j,k] = cc

	return ccmatrix

######################################################################################################################################################

def plot_correlation_matrix(ccmatrix,filename):

	fig, ax = plt.subplots(1,1,figsize=(10,10))
	c = ax.imshow(ccmatrix,aspect="auto",vmin=0.0,vmax=0.5)
	ax.set_xlabel('Event number',fontsize=14.0, labelpad=10.0)
	ax.set_ylabel('Event number',fontsize=14.0, labelpad=10.0)
	cbar = plt.colorbar(c)
	cbar.set_label('|CC|')
	plt.savefig(filename)
	plt.close()

######################################################################################################################################################
####################################################   PENDIENTE        ##############################################################################

# def template_matching2d(template, waveforms, overlap, threshold, template_name, path_waveforms, path_coeff, path_traces, envelope_flag=True,filter_type=None):

# 	#Initial array of cc
# 	cc_total = np.zeros(0)

# 	#Detection matrix
# 	detect_traces = []

# 	#Start times
# 	start_times = []

# 	#Loop for using each waveform

# 	for i,waveform in enumerate(waveforms):

# 		#Flag for deetections
# 		detections = False

# 		print(f'Searching in: {waveform}')

# 		#DAS data - different preprocessing stage
# 		w = h5py.File(f"{path_waveforms}/{waveform}")

# 		#Stacking channels (Original)
# 		orig_trace, dx, fs = retrieve_DASdata(w,stack=True,filter_type=None)

# 		#Stacking channels (Filtered)
# 		fwtrace, dx, fs = retrieve_DASdata(w,stack=True,filter_type=filter_type)

# 		#Parameters
# 		try:
# 			starttime = w["Acquisition/Raw[0]/RawData"].attrs['PartStartTime']
# 		except Exception:
# 			starttime = w["Merge/Data"].attrs['PartStartTime']
# 			start_times.append(starttime)

# 		#Length of the template
# 		n_template = template.shape[0]

# 		#Dividing the matrix - envelopes
# 		new_matrix = divide_trace_DAS(fwtrace,n_template,overlap,envelope_flag)

# 		#Also with waveforms to retrieve data
# 		new_waveforms = divide_trace_DAS(fwtrace,n_template,overlap)

# 		#Also with waveforms to retrieve data
# 		new_orig_waveforms = divide_trace_DAS(orig_trace,n_template,overlap)

# 		#Retrieving number of traces and number of channels
# 		n_traces = new_matrix.shape[0]
# 		n_channels = new_matrix.shape[2]

# 		#Template matrix
# 		t_matrix = np.ones((n_traces,template.shape[0]))

# 		#Arrays of detections, maximum, minimum, and average amplitudes
# 		mccsub = np.zeros((n_traces,n_channels))
# 		#mccsub2 = np.zeros((59970,n_channels))
# 		mmax_amp = np.zeros((n_traces,n_channels))
# 		mmin_amp = np.zeros((n_traces,n_channels))
# 		mavg_amp = np.zeros((n_traces,n_channels))

# 		#Correlation coefficient peaks
# 		ccpeaks = {}

# 		matrix_template = np.zeros((template.shape[0],template.shape[0]))
# 		matrix_template += template

# 		for ch in range(n_channels):

# 			submatrix = new_matrix[:,:,ch]

# 			#ccsub = np.zeros(n_traces)
# 			#max_amp = np.zeros(n_traces)
# 			#min_amp = np.zeros(n_traces)
# 			#avg_amp = np.zeros(n_traces)

# 			################################# Change this section ###################################################################################################

# 			#ccf = scipy.signal.correlate(submatrix,matrix_templates)
# 			#print(f"submatrix shape: {submatrix.shape}, matrix template shape: {matrix_templates.shape}, ccf shape: {ccf.shape}")
# 			ccf = correlate_gpu(submatrix,matrix_template);# test = np.hstack((test,ccf))
# 			#ccsub = correlation_coefficient_DAS(ccf,submatrix,matrix_templates)

# 			#Adding values
# 			#ccsub[k] = cc; max_amp[k] = np.amax(subtrace); min_amp[k] = np.amin(subtrace); avg_amp[k] = np.mean(subtrace);

# 			################################# Change this section ###################################################################################################

# 			#Adding values
# 			#mccsub[:,ch] = ccsub; mmax_amp[:,ch] = max_amp; mmin_amp[:,ch] = min_amp; mavg_amp[:,ch] = avg_amp;
# 			#mccsub2[:,ch] = test;

# 		# 	#Creating time array per trace
# 		# 	subtimes = np.arange(0,n_traces)*((template.shape[0]*(1-overlap))/fs)
			
# 		# 	#Obtaining peaks
# 		# 	peaks = detect_peaks(ccsub,mph = threshold,mpd=1.0)

# 		# 	if(peaks.shape[0] != 0):

# 		# 		ccpeaks[f"{ch}"] = peaks

# 		# traces = []
# 		# orig_traces = []
		
# 	# 	for index in ccpeaks:

# 	# 		#print(f"Channel no.: {index}, No. of peaks: {ccpeaks[index].shape[0]}")	

# 	# 		if(ccpeaks[index].shape[0] == 1):

# 	# 			trace = new_waveforms[ccpeaks[index][0],:,int(index)]
# 	# 			traces.append(trace)

# 	# 			#Saving original files
# 	# 			orig_traces.append(new_orig_waveforms[ccpeaks[index][0],:,int(index)])

# 	# 	params = {"fwtrace":fwtrace.shape[0],"n_channels":n_channels,"fs":fs,"dx":dx}

# 	# 	if(len(traces) > 20):
# 	# 		detections = True

# 	# 	#Plotting
# 	# 	if(detections):
# 	# 		plotting_cc(path_coeff,mccsub,starttime,f"{template_name}_W{waveform}",params)
# 	# 		plotting_traces(path_traces,f"{template_name}_W{waveform}",traces,fs)
# 	# 	else:
# 	# 		print(f"No detections for {waveform}")

# 	# params = {"fwtrace":fwtrace.shape[0],"n_channels":n_channels,"fs":fs,"dx":dx}

# 	#return traces, orig_traces, start_times, mccsub, ccpeaks, params

# 	return ccf

######################################################################################################################################################

