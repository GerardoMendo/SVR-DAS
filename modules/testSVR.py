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
from numba import jit
from utils import *
from check_os import *
from plotting_functions_DAS import *
from TempMatchDAS_func import *
from data_utils import *
from parameters import *
from obtainTraces import *

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

#Retrieving root and media
root, media = check_operating_system()


###########################################################################################################################################

def testpickModel(picks_p,picks_s,channels,lowpoints,nelements,snr,addlowpoints=True,kernelp="rbf",kernels="rbf"):

	if(len(channels) == 1):
		chnsp = channels[0]; chnss = channels[0]
	else:
		chnsp = channels[0]; chnss = channels[1]
		#print(f"Ch S shape: {len(chnss)}")

	#svr = SVR(kernel=kernel,C=0.1,epsilon=0.001)
	if(kernelp == "linear"):
		svrp = make_pipeline(StandardScaler(),LinearSVR(C=100,epsilon=1.0,dual=True,max_iter=10000))
		#svrp = LinearSVR(C=1.0,epsilon=1e-6)
	else:
		if(snr < 5.0):
			print("Aqui")
			svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.0,epsilon=0.01))
			#svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.0,epsilon=0.1))
		else:
			svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.0,epsilon=0.01))
			#svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.0,epsilon=1e-6))
			
			#FOR HIGH SNR - NNET
			#svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=0.01,epsilon=1e-6))

			#FOR LOW SNR - NNET
			#svrp = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=100.0,epsilon=1e-10))

	if(kernels == "linear"):
		#svrs = LinearSVR(C=1.0,epsilon=1e-6)
		svrs = make_pipeline(StandardScaler(),LinearSVR(C=1.0,epsilon=1e-10,dual=True,max_iter=10000))
	else:
		if(snr < 5.0):
			svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=1.0,epsilon=0.01))
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=0.01,epsilon=1e-6))
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=0.8,epsilon=0.1))
		else:
			svrs = make_pipeline(StandardScaler(),SVR(kernel=kernels,C=1.0,epsilon=0.01))
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=1.5,epsilon=0.01))

			#HIGH SNR - NNET
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=0.05,epsilon=0.01))

			#LOW SNR - NNET
			#svrs = make_pipeline(StandardScaler(),SVR(kernel=kernelp,C=10,epsilon=0.01))

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