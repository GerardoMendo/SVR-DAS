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
sys.path.append('/home/gerardomendo/Documents/DAS/toServer/SVRDAS/modules')

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

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

class DataRetriever:

	def __init__(self,channels,fs,concat,len_window,method,system,lp):

		self.channels = channels
		self.method = method
		self.fs = fs
		self.concat=concat
		self.system=system
		self.lenWindow = len_window
		self.lp = lp

	def get_traces(self,st_data):

		if(self.method=="centered"):

			#Retrieving centers
			start = timer()
			data, new_original, low_point, up_point, maxs, ch_idx = retrieve_centered_traces(self.channels,st_data,self.fs,refine=True)
			
			paramsData = {

			"data":data,
			"channels":ch_idx,
			"nofiltdata":new_original,
			"lowpoints":low_point,
			"uppoints":up_point,
			"max":maxs

			}

			end = timer()

		if(self.method == "all"):

			#Retrieving centers
			start = timer()
			if(self.concat):
				endpoint = 180
			else:
				#Uncomment if using 20-m seconds DAS data
				#endpoint = 60
				#endpoint is equal to 180 for 3m data without concat
				endpoint = 180

			data, new_original, low_point, up_point, maxs, ch_idx = retrieve_limits(self.channels,st_data,self.fs,self.lenWindow,endpoint,system=self.system)
			end = timer()

			paramsData = {

			"data":data,
			"channels":ch_idx,
			"nofiltdata":new_original,
			"lowpoints":low_point,
			"uppoints":up_point,
			"max":maxs

			}

		if(self.method == "HSNR"):

			#Retrieving centers
			start = timer()
			data, new_original, low_point, up_point, maxs, ch_idx  = obtainHighSNR(st_data,self.fs,4000,concat=self.concat,lp=self.lp)
			end = timer()

			paramsData = {

			"data":data,
			"channels":ch_idx,
			"nofiltdata":new_original,
			"lowpoints":low_point,
			"uppoints":up_point,
			"max":maxs

			}				

		return paramsData


