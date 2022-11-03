import numpy as np 
import pylab as plt
from fbm import FBM 
import os
import sys
import pandas as pd 

def get_fbm_sample(l = 1, h = 0.5, d = 1,n =1):
	#l = end time (from 0)
	#h = hurst parameter
	#d = dimensions (x,y,z .... ) for one realization
    #n = even intervals from 0,l

	f = FBM(length = l, hurst = h,n = n)
	samples = []
	sample_t = []
	for i in range(d):
		fbm_sample = f.fbm()
		fgn_sample = f.fgn()
		t_values = f.times()  
		if n == 1 and n != l:

			samples.append(fbm_sample[1:])
			sample_t.append(t_values[1:])

		else:
			samples.append(fbm_sample)

	return [sample_t, samples]

	
def compute_msd_np(xy, t, t_step):
    shifts = np.floor(t / t_step).astype(np.int)
    msds = np.zeros(shifts.size)
    msds_std = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = xy[:-shift if shift else None] - xy[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std(ddof=1)

    msds = pd.DataFrame({'msds': msds, 'tau': t, 'msds_std': msds_std})
    return msds

def track_decomp(x,y,f,max_track_decomp):
    #takes tracks and finds MSD for various timestep conditions.
    
    #return array-like: 
    #msd = msd values at all tau values considered
    #popt = fitted parameters on MSD equation
    #pcov = covariance matrix of fit
    
    max_decomp = np.floor(len(x)/max_track_decomp)
    tau = list(range(1,int(max_decomp+1.0)))
    msd = []
    for i in tau:
        if i < len(x):
            n_x = np.array(x)[::i]
            n_y = np.array(y)[::i]
            n_f = np.array(f)[::i]
            msd.append(MSD_tavg(n_x,n_y,n_f))
        
    #popt , pcov = curve_fit(fit_MSD,tau,np.array(msd),p0=[1,1],maxfev=10000)
    
    
    return np.array([msd,tau])
