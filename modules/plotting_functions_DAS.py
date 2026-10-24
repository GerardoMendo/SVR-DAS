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
from utils import *
from detect_peaks import *
from numba import jit
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV
#from map_kurt_skew import *

import h5py
import os
import gc

################################################################################################################################################################################

def plotting_test(fdata,params):

	L,dx,starttime,fs = params[0],params[1],params[2],params[3]
	nptsx, nptsy = fdata.shape
	x = np.arange(0,nptsx)*dx
	t = np.arange(0,nptsy)/fs
	print(f"fs: {fs}, nptsx:{nptsx}, nptsy:{nptsy}")

	plt.ion()
	fig, ax = plt.subplots(1,1,figsize=(10,10))
	c = ax.imshow(fdata, cmap="seismic", vmin=-np.percentile(fdata,97), vmax=np.percentile(fdata,97),aspect="auto", interpolation="none",origin="upper")
	cbar = plt.colorbar(c)
	cbar.set_label("Strain rate (μstrain/s)", fontsize=16.0)
	ax.set_title(f"Miyazaki - {UTCDateTime(starttime)}")
	ax.set_xlabel("Length (m)",fontsize=16.0)
	ax.set_ylabel("Time (s)",fontsize=16.0)
	c = input("Presione una tecla para continuar...")

################################################################################################################################################################################

def plotting_label_traces(path,data,channels,picks_p,picks_s,fs,date,method="STALTA",zoom=None):

	for i,trace in enumerate(data):

		time = np.arange(0,trace.shape[0])/fs
		p, s = picks_p[str(channels[i])],picks_s[str(channels[i])] 
		labelp, labels, labeln = creating_psn_labels([p,s],trace.shape[0],100)
		maxp = np.where(labelp==np.amax(labelp))[0][0]
		maxs = np.where(labels==np.amax(labels))[0][0]

		#Plotting
		fig, ax = plt.subplots(2,1,figsize=(20,10))
		ax[0].plot(time,trace,linewidth=0.7,c="k")
		ax[0].plot([maxp/fs,maxp/fs],[-0.7e-5,0.7e-5],lw=4.0,c="r")
		ax[0].plot([maxs/fs,maxs/fs],[-0.7e-5,0.7e-5],lw=4.0,c="b")
		ax[0].set_xlim(time[0],time[-1])
		ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
		ax[0].tick_params(axis="x",color="k",labelsize=14.0)
		ax[0].tick_params(axis="y",color="k",labelsize=14.0)

		ax[1].plot(time,labelp,linewidth=0.7,c="r",label=f"P_{method}",linestyle="solid")
		ax[1].plot(time,labels,linewidth=0.7,c="b",label=f"S_{method}",linestyle="solid")
		ax[1].set_xlim(time[0],time[-1])

		if zoom is not None:

			try:
				assert type(zoom)==list
				ax[0].set_xlim(zoom[0],zoom[1])
				ax[1].set_xlim(zoom[0],zoom[1])
			except AssertionError:
				print(f'Insert a list of two values in seconds for zooming')
				ax[0].set_xlim(time[0],time[-1])
				ax[1].set_xlim(time[0],time[-1])


		ax[1].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
		ax[1].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
		ax[1].tick_params(axis="x",color="k",labelsize=14.0)
		ax[1].tick_params(axis="y",color="k",labelsize=14.0)
		ax[1].legend(loc="upper right")

		name_image = f"DAS_{date}_Ch_{channels[i]}_{method}_picks.png"

		try:
			plt.savefig(f"{path}/{date}/{name_image}")
		except FileNotFoundError:
			os.mkdir(f"{path}/{date}")
			plt.savefig(f"{path}/{date}/{name_image}")
		
		plt.close()

################################################################################################################################################################################

def plotting_label_STALTA_AR(path,data,channels1,picks1_p,picks1_s,channels2,picks2_p,picks2_s,channels3,picks3_p,picks3_s,channels4,picks4_p,picks4_s,fs,date):

	for i,trace in enumerate(data):

		time = np.arange(0,trace.shape[0])/fs
		p1, s1 = picks1_p[str(channels1[i])],picks1_s[str(channels1[i])] 
		labelp1, labels1, labeln1 = creating_psn_labels([p1,s1],trace.shape[0],100)
		maxp1 = np.where(labelp1==np.amax(labelp1))[0][0]
		maxs1 = np.where(labels1==np.amax(labels1))[0][0]

		p2, s2 = picks2_p[str(channels2[i])],picks2_s[str(channels2[i])] 
		labelp2, labels2, labeln2 = creating_psn_labels([p2,s2],trace.shape[0],100)
		maxp2 = np.where(labelp2==np.amax(labelp2))[0][0]
		maxs2 = np.where(labels2==np.amax(labels2))[0][0]

		p3, s3 = picks3_p[str(channels3[i])],picks3_s[str(channels3[i])] 
		labelp3, labels3, labeln3 = creating_psn_labels([p3,s3],trace.shape[0],100)
		maxp3 = np.where(labelp3==np.amax(labelp3))[0][0]
		maxs3 = np.where(labels3==np.amax(labels3))[0][0]

		p4, s4 = picks4_p[str(channels4[i])],picks4_s[str(channels4[i])] 
		labelp4, labels4, labeln4 = creating_psn_labels([p4,s4],trace.shape[0],100)
		maxp4 = np.where(labelp4==np.amax(labelp4))[0][0]
		maxs4 = np.where(labels4==np.amax(labels4))[0][0]

		f,t,s = signal.spectrogram(trace,fs,nperseg=64,noverlap=60)
		
		#fig.colorbar(c,ax=ax[1],location="right")

		ienvelope = (abs(np.gradient(np.sum(s,axis=0)))*1e7)*50
		envelope = ienvelope/np.amax(ienvelope)
		times_env = np.arange(0,envelope.shape[0])*(t[1]-t[0])
		mean_envelope = np.mean(envelope)
		mad_envelope = stats.median_abs_deviation(envelope)
		#peaks = detect_peaks(envelope,mad_envelope,10)
		#try:
		#	maxindx = np.where(envelope >= 0.8*(envelope))[0][0]
		#except Exception:
		#	maxindx = 0
		#print(f"i: {i}, mad envelope: {mad_envelope}, peaks shape: {peaks.shape[0]}")

		#Plotting
		fig, ax = plt.subplots(3,1,figsize=(20,10))
		ax[0].plot(time,trace,linewidth=0.7,c="k")
		ax[0].plot([maxp2/fs,maxp2/fs],[-2e-6,2e-6],lw=2.0,c="r")
		ax[0].plot([maxs2/fs,maxs2/fs],[-2e-6,2e-6],lw=2.0,c="b")
		#ax[0].plot([times_env[peaks[0]],times_env[peaks[0]]],[-5e-6,5e-6],lw=2.0,c="g")
		#ax[0].plot([times_env[maxindx],times_env[maxindx]],[-5e-6,5e-6],lw=2.0,c="g")
		ax[0].plot([maxp3/fs,maxp3/fs],[-5e-6,5e-6],lw=2.0,c="g")
		ax[0].plot([maxs3/fs,maxs3/fs],[-5e-6,5e-6],lw=2.0,c="g")
		ax[0].set_xlim(time[0],time[-1])
		ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
		ax[0].tick_params(axis="x",color="k",labelsize=14.0)
		ax[0].tick_params(axis="y",color="k",labelsize=14.0)
		#ax[0].set_xlim(10,40)

		#for i,peak in enumerate(peaks):

		#	try:
		#		timepeak = times_env[peak]
		#		#print(f'Time peak: {timepeak}')
		#		ax[0].plot([timepeak,timepeak],[-5e-6,5e-6],lw=2.0,c="g")
		#	except Exception:
		#		print(f"No MAD peak value")

		#c=ax[1].imshow(10*np.log10(s),origin="lower",aspect="auto",interpolation="lanczos",cmap="jet",extent=(t[0],t[-1],f[0],f[-1]),vmin=-225,vmax=-125)
		ax[1].plot(times_env,envelope,linewidth=0.7,c="k")
		if(channels3[i] < 1500):
			ax[1].plot(times_env,np.ones(envelope.shape[0])*(0.1*mad_envelope),linewidth=1.0,c="g")
		else:
			ax[1].plot(times_env,np.ones(envelope.shape[0])*(0.25*mad_envelope),linewidth=1.0,c="g")
		ax[1].set_xlim(t[0],t[-1])
		ax[1].set_ylim(0,0.2)
		
		#ax[1].set_xlim(10,40)

		ax[2].plot(time,labelp1,linewidth=0.7,c="r",label="P_STALTA",linestyle="dotted")
		ax[2].plot(time,labels1,linewidth=0.7,c="b",label="S_STALTA",linestyle="dotted")
		ax[2].plot(time,labelp2,linewidth=0.7,c="r",label="P_AR",linestyle="solid")
		ax[2].plot(time,labels2,linewidth=0.7,c="b",label="S_AR",linestyle="solid")	
		ax[2].plot(time,labelp3,linewidth=0.7,c="r",label="P_SP",linestyle="dashdot")
		ax[2].plot(time,labels3,linewidth=0.7,c="b",label="S_SP",linestyle="dashdot")
		ax[2].plot(time,labelp4,linewidth=0.7,c="r",label="P_RMS",linestyle="dashed")
		ax[2].plot(time,labels4,linewidth=0.7,c="b",label="S_RMS",linestyle="dashed")						
		ax[2].set_xlim(time[0],time[-1])
		ax[2].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
		ax[2].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
		ax[2].tick_params(axis="x",color="k",labelsize=14.0)
		ax[2].tick_params(axis="y",color="k",labelsize=14.0)
		ax[2].legend(loc="upper right")
		#ax[2].set_xlim(10,40)

		name_image = f"DAS_{date}_Ch_{channels1[i]}_picks.png"

		try:
			plt.savefig(f"{path}/{date}/{name_image}")
		except FileNotFoundError:
			os.mkdir(f"{path}/{date}")
			plt.savefig(f"{path}/{date}/{name_image}")
		
		plt.close()

####################################################################################################################################################################################################################

def plottingDASpicks(path,list_parameters,data,name,channel_list,p_picks,s_picks,snr,low_point,evsnr,extra_name=None,type="pdf",chidx=False,add_lowpoints=False):

	if(len(channel_list) == 1):
		channelp = channel_list[0]
		channels = channel_list[0]
	else:
		channelp = channel_list[0]
		channels = channel_list[1]

	low_points = np.array(low_point)

	new_ppicks = [p_picks[str(channel)] for channel in channelp]
	new_spicks = [s_picks[str(channel)] for channel in channels]

	#FOR SANRIKU DATA
	st_data = data

	c2, c1 = signal.butter(4,(5.0,10.0),"band",fs=100.0)
	fdata = signal.filtfilt(c2,c1,st_data,axis=0)
	fdata = signal.filtfilt(c2,c1,fdata,axis=0)

	dx,fs = list_parameters[0],list_parameters[1]
	nx, nt = st_data.shape[1], st_data.shape[0]

	#print(f"nx: {nx}, nt: {nt}")
	if(chidx):
		x = np.arange(nx)
		channelp = np.array(channelp)
		channels = np.array(channels)

	else:
		x = np.arange(nx) * dx	
		channelp = np.array(channelp)*dx
		channels = np.array(channels)*dx

	t = np.arange(nt) / fs

	if(add_lowpoints):
		
		new_ppicks = (np.array(new_ppicks)+low_points)/fs
		new_spicks = (np.array(new_spicks)+low_points)/fs

	else:

		new_ppicks = (np.array(new_ppicks))/fs
		new_spicks = (np.array(new_spicks))/fs

	print(f"x: {x[-1]}, t: {t[-1]}")	
	fdata = fdata*1e6

	fig, ax = plt.subplots(1,1, figsize=(20,10))
	c = ax.imshow(fdata, cmap="seismic", vmin=-np.percentile(fdata,97), vmax=np.percentile(fdata,97),aspect="auto", interpolation="none",extent=[x[0],x[-1],t[-1],t[0]],origin="upper")
	#c = ax.imshow(new_das, cmap="seismic", vmin=-np.percentile(new_das,99), vmax=np.percentile(new_das,99),aspect="auto", interpolation="none",origin="upper")
	#a = ax.scatter(channelp,new_ppicks,s=5,c=evsnr,cmap="inferno",vmin=np.amin(evsnr),vmax=np.amax(evsnr))
	#ax.scatter(channels,new_spicks,s=5,c=evsnr,cmap="inferno",vmin=np.amin(evsnr),vmax=np.amax(evsnr))
	ax.scatter(channelp,new_ppicks,s=10,c="r")
	ax.scatter(channels,new_spicks,s=10,c="b")
	cbar = plt.colorbar(c)
	#cbar2 = plt.colorbar(a)
	cbar.set_label("Strain rate (μstrain/s)", fontsize=16.0)
	#cbar2.set_label("SNR", fontsize=16.0)
	ax.set_title(f"Kamaishi - {UTCDateTime(name)} SNR: {snr}")
	ax.set_xlabel("Length (m)",fontsize=16.0)
	ax.set_ylabel("Time (s)",fontsize=16.0)
	#ax.set_ylim(100,0)
	#ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
	#ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
	if(extra_name is not None):
		plt.savefig(f"{path}/{name}_{extra_name}_imgNoSNR.{type}")
	else:
		plt.savefig(f"{path}/{name}_imgNoSNR.{type}")
	plt.tight_layout()
	plt.close('all')
	gc.collect()

################################################################################################################################################################################
################################################################################################################################################################################

#def plotting_label_traces_All(path,data,channels,picks1_p,picks1_s,picks2_p,picks2_s,picks3_p,picks3_s,picks4_p,picks4_s,fs,date,zoom=None):
def plotting_label_traces_All(path,data,channels,picks1_p,picks1_s,picks2_p,picks2_s,fs,date,zoom=None):

	for i,ch in enumerate(channels):

		trace = data[ch]

		time = np.arange(0,trace.shape[0])/fs
		p1, s1 = picks1_p[str(ch)],picks1_s[str(ch)] 
		labelp1, labels1, labeln1 = creating_psn_labels([p1,s1],trace.shape[0],100)
		maxp1 = np.where(labelp1==np.amax(labelp1))[0][0]
		maxs1 = np.where(labels1==np.amax(labels1))[0][0]

		p2, s2 = picks2_p[str(ch)],picks2_s[str(ch)] 
		labelp2, labels2, labeln2 = creating_psn_labels([p2,s2],trace.shape[0],100)
		maxp2 = np.where(labelp2==np.amax(labelp2))[0][0]
		maxs2 = np.where(labels2==np.amax(labels2))[0][0]

		#p3, s3 = picks3_p[str(ch)],picks3_s[str(ch)] 
		#labelp3, labels3, labeln3 = creating_psn_labels([p3,s3],trace.shape[0],100)
		#maxp3 = np.where(labelp3==np.amax(labelp3))[0][0]
		#maxs3 = np.where(labels3==np.amax(labels3))[0][0]

		#p4, s4 = picks4_p[str(ch)],picks4_s[str(ch)] 
		#labelp4, labels4, labeln4 = creating_psn_labels([p4,s4],trace.shape[0],100)
		#maxp4 = np.where(labelp4==np.amax(labelp4))[0][0]
		#maxs4 = np.where(labels4==np.amax(labels4))[0][0]

		# time = np.arange(0,trace.shape[0])/fs
		# p, s = picks_p[str(channels[i])],picks_s[str(channels[i])] 
		# labelp, labels, labeln = creating_psn_labels([p,s],trace.shape[0],100)
		# maxp = np.where(labelp==np.amax(labelp))[0][0]
		# maxs = np.where(labels==np.amax(labels))[0][0]

		#Plotting
		fig, ax = plt.subplots(2,1,figsize=(20,10))
		ax[0].plot(time,trace,linewidth=0.7,c="k")
		#ax[0].plot([maxp/fs,maxp/fs],[-1e-6,1e-6],lw=2.0,c="r")
		#ax[0].plot([maxs/fs,maxs/fs],[-1e-6,1e-6],lw=2.0,c="b")
		ax[0].set_xlim(time[0],time[-1])
		ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
		ax[0].tick_params(axis="x",color="k",labelsize=14.0)
		ax[0].tick_params(axis="y",color="k",labelsize=14.0)

		ax[1].plot(time,labelp1,linewidth=0.7,c="r",label="P_STALTA",linestyle="dotted")
		ax[1].plot(time,labels1,linewidth=0.7,c="b",label="S_STALTA",linestyle="dotted")
		ax[1].plot(time,labelp2,linewidth=0.7,c="r",label="P_AR",linestyle="solid")
		ax[1].plot(time,labels2,linewidth=0.7,c="b",label="S_AR",linestyle="solid")	
		#ax[1].plot(time,labelp3,linewidth=0.7,c="r",label="P_SP",linestyle="dashdot")
		#ax[1].plot(time,labels3,linewidth=0.7,c="b",label="S_SP",linestyle="dashdot")
		#ax[1].plot(time,labelp4,linewidth=0.7,c="r",label="P_RMS",linestyle="dashed")
		#ax[1].plot(time,labels4,linewidth=0.7,c="b",label="S_RMS",linestyle="dashed")	

		#ax[1].plot(time,labelp,linewidth=0.7,c="r",label=f"P_{method}",linestyle="dotted")
		#ax[1].plot(time,labels,linewidth=0.7,c="b",label=f"S_{method}",linestyle="dotted")
		ax[1].set_xlim(time[0],time[-1])

		if zoom is not None:

			try:
				assert type(zoom)==list
				ax[0].set_xlim(zoom[0],zoom[1])
				ax[1].set_xlim(zoom[0],zoom[1])
			except AssertionError:
				print(f'Insert a list of two values in seconds for zooming')
				ax[0].set_xlim(time[0],time[-1])
				ax[1].set_xlim(time[0],time[-1])

		ax[1].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
		ax[1].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
		ax[1].tick_params(axis="x",color="k",labelsize=14.0)
		ax[1].tick_params(axis="y",color="k",labelsize=14.0)
		ax[1].legend(loc="upper right")

		name_image = f"DAS_{date}_Ch_{channels[i]}_All_picks.png"

		try:
			plt.savefig(f"{path}/{date}/{name_image}")
		except FileNotFoundError:
			os.mkdir(f"{path}/{date}")
			plt.savefig(f"{path}/{date}/{name_image}")
		
		plt.close()

################################################################################################################################################################################

def plotting_label_iniTraces(path,data,channels,picks_p,picks_s,fs,date,zoom=None):

	for i in range(len(channels)):

		trace = data[channels[i],:]

		time = np.arange(0,trace.shape[0])/fs
		p, s = picks_p[str(channels[i])],picks_s[str(channels[i])] 
		labelp, labels, labeln = creating_psn_labels([p,s],trace.shape[0],100)
		maxp = np.where(labelp==np.amax(labelp))[0][0]
		maxs = np.where(labels==np.amax(labels))[0][0]

		#Plotting
		fig, ax = plt.subplots(2,1,figsize=(20,10))
		ax[0].plot(time,trace,linewidth=0.7,c="k")
		ax[0].plot([maxp/fs,maxp/fs],[-0.7e-5,0.7e-5],lw=4.0,c="r")
		ax[0].plot([maxs/fs,maxs/fs],[-0.7e-5,0.7e-5],lw=4.0,c="b")
		ax[0].set_xlim(time[0],time[-1])
		ax[0].set_ylabel(f" Amplitude\n(μstrain/s)",fontsize=16.0,labelpad=10.0)
		ax[0].tick_params(axis="x",color="k",labelsize=14.0)
		ax[0].tick_params(axis="y",color="k",labelsize=14.0)

		ax[1].plot(time,labelp,linewidth=0.7,c="r",label=f"P_initial",linestyle="solid")
		ax[1].plot(time,labels,linewidth=0.7,c="b",label=f"S_initial",linestyle="solid")
		ax[1].set_xlim(time[0],time[-1])

		if zoom is not None:

			try:
				assert type(zoom)==list
				ax[0].set_xlim(zoom[0],zoom[1])
				ax[1].set_xlim(zoom[0],zoom[1])
			except AssertionError:
				print(f'Insert a list of two values in seconds for zooming')
				ax[0].set_xlim(time[0],time[-1])
				ax[1].set_xlim(time[0],time[-1])


		ax[1].set_xlabel("Time (s)",fontsize=16.0,labelpad=10.0)
		ax[1].set_ylabel(f"Prob.",fontsize=16.0,labelpad=10.0)
		ax[1].tick_params(axis="x",color="k",labelsize=14.0)
		ax[1].tick_params(axis="y",color="k",labelsize=14.0)
		ax[1].legend(loc="upper right")

		name_image = f"DAS_{date}_Ch_{channels[i]}_Initial_picks.png"

		try:
			plt.savefig(f"{path}/{date}/{name_image}")
		except FileNotFoundError:
			os.mkdir(f"{path}/{date}")
			plt.savefig(f"{path}/{date}/{name_image}")
		
		plt.close()

################################################################################################################################################################################


