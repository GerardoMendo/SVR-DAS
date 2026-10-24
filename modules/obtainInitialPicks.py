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
sys.path.append('/home/gerardomendo/Documents/toServer/DAS/SVRDAS/modules')

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
from pick_params import *
from obtainTraces import *
from runMethods import *

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

#Retrieving root and media
root, media = check_operating_system()

######################################################################################################################################################################

def obtainIniPicks(name_det,pch_idx,st_data,filtered,nelements,dx,fs,len_window,path_stalta,path_ar,path_initial,concat,type_data,plotPicks=True,plotIni=True,lp=None):

	#Obtaining data
	start = timer()
	dRet = DataRetriever(pch_idx,fs,concat,len_window,method="all",system="Nnet",lp=lp)
	paramsData = dRet.get_traces(filtered)
	end = timer()

	print(f"Time elapsed DataRetriever: {end - start} seconds")

	#Retrieving data from paramsData

	data = paramsData["data"] 
	ch_idx = paramsData["channels"] 
	new_original = paramsData["nofiltdata"] 
	low_point = paramsData["lowpoints"]
	up_point = paramsData["uppoints"] 
	maxs = paramsData["max"]

	#Estimate RMS - original

	start = timer()
	snr_total = estimation_snr(filtered,ch_idx,low_point,up_point,decibel=True)
	max_snr = np.amax(snr_total)
	mean_snr = np.mean(snr_total)
	sort_snr = np.sort(snr_total)
	iqr_snr = stats.iqr(sort_snr)
	end = timer()

	print(f"Time elapsed estimation SNR: {end - start} seconds")

	#Params for plotting pickers
	paramsPlotting = {

	"path_sta_lta":path_stalta,
	"path_ar":path_ar,
	"dxfs":[dx,fs],
	"name_det":name_det,
	"maxSNR":max_snr,
	"low_point":low_point,
	"snr_total":snr_total,
	"nameSTALTA":f"STALTA_picks_{type_data}",
	"nameAR":f"AR_picks_{type_data}",
	"type":"png",
	"chidx":False,
	"add_lowpoints":True

	}

	########################################################################################################################################################################################################

	# #If using retrieve centered traces

	newchsnr = list(ch_idx)

	if(max_snr < 5.0):

		#For Sanriku
		params_sta_lta, params_ar = parametersHighSNR(fs)
		print(f"Max_SNR: {max_snr}")
			
	else:

		params_sta_lta, params_ar = parametersLowSNR(fs)

	########################################################################################################################################################################################################

	#Run STA/LTA and AR
	pickers = RunPickers(newchsnr,params_sta_lta,params_ar,paramsPlotting,plot_STALTA=plotPicks,plot_AR=plotPicks,runSTALTA=True,runAR=True)
	stalta, ar = pickers.get_picks(st_data,data)

	#Retrieving data
	stalta_p, stalta_s, ch_stalta = stalta[0], stalta[1], stalta[2]
	ar_p, ar_s, ar_ch = ar[0], ar[1], ar[2]

	#Plotting all
	ppicks = (stalta_p,ar_p)
	spicks = (stalta_s,ar_s)
	chns = (ch_stalta,ar_ch)

	ch_p, ch_s, p_picks, s_picks = comparison_picks(ppicks,spicks,chns[1],low_point,c_thres = 100)
	nchp, nchs, newp, news = correct_picks([ch_p,ch_s],p_picks,s_picks)

	#print(f"newp shape: {len(newp)}, news shape: {len(news)}")

	if(plotIni):
		plottingDASpicks(path_initial,[dx,fs],st_data.T,name_det,[nchp,nchs],newp,news,mean_snr,low_point,snr_total,extra_name="Initial_new",type="png",chidx=False,add_lowpoints=False)

	#Cleaning picks
	if(max_snr < 5.0):
		nchp, newp = remove_outliers(nchp,newp,news)

	#Obtaining interquartile range of S minus P
	iqr_sminp = obtain_iqr(nchp,nchs,newp,news)

	print(f"Min SNR: {np.amin(snr_total)}, Max SNR: {np.amax(snr_total)}, mean SNR: {np.mean(snr_total)}, median SNR: {np.median(snr_total)}, std SNR: {np.std(snr_total)}, iqr SNR: {stats.iqr(snr_total)}")

	return nchp, nchs, newp, news, snr_total, max_snr, iqr_snr, iqr_sminp, low_point, (ppicks,spicks,chns)
