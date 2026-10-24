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

import h5py
import os
import logging
import glob

from multiprocessing import Process, Pool

class RunPickers():

	def __init__(self,channels,paramsSTALTA,paramsAR,paramsPlot,plot_STALTA=False,plot_AR=False,runSTALTA=True,runAR=True):

		self.channels = channels
		self.params_sta_lta = paramsSTALTA
		self.params_ar = paramsAR
		self.plot_sta = plot_STALTA
		self.plot_ar = plot_AR
		self.run_sta_lta = runSTALTA
		self.run_ar = runAR
		self.paramsPlot = paramsPlot

	def get_picks(self,st_data,data):

		if(self.run_sta_lta):

			stalta_p, stalta_s, ch_stalta = labeling_stalta(data,self.channels,self.params_sta_lta)

		if(self.run_ar):

			ar_p, ar_s, ar_ch = labeling_ar(data,self.channels,self.params_ar)

		if(self.plot_sta):

			plottingDASpicks(self.paramsPlot["path_sta_lta"],self.paramsPlot["dxfs"],st_data.T,self.paramsPlot["name_det"],[ch_stalta],stalta_p,stalta_s,self.paramsPlot["maxSNR"],self.paramsPlot["low_point"],self.paramsPlot["snr_total"],self.paramsPlot["nameSTALTA"],self.paramsPlot["type"],self.paramsPlot["chidx"],self.paramsPlot["add_lowpoints"])

		if(self.plot_ar):

			plottingDASpicks(self.paramsPlot["path_ar"],self.paramsPlot["dxfs"],st_data.T,self.paramsPlot["name_det"],[ar_ch],ar_p,ar_s,self.paramsPlot["maxSNR"],self.paramsPlot["low_point"],self.paramsPlot["snr_total"],self.paramsPlot["nameAR"],self.paramsPlot["type"],self.paramsPlot["chidx"],self.paramsPlot["add_lowpoints"])

		return (stalta_p,stalta_s,ch_stalta),(ar_p,ar_s,ar_ch)