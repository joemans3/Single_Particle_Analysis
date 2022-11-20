import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci

from Analysis_functions import *
from src.databases.trajectory_analysis_script import *

#cols in datafile
#2 =Track_ID
#4 =X (micron)
#5 =Y (micron)
#7 =Time (sec)
#8 =Frame 


t_len_l = 4
t_len_u = 100
path = '/Volumes/BP/rp_ez/analysis/'

files_to_read = glob.glob(path+'*.csv')

list_exposure_time = np.zeros(len(files_to_read))
list_interval_time = np.zeros(len(files_to_read))

x_all = []
y_all = []
f_all = []
for i in files_to_read:

	data = np.loadtxt(i,skiprows = 4, usecols = (2,4,5,7,8),delimiter = ',')
	track_ID = data[:,0]
	tp_x = data[:,1]
	tp_y = data[:,2]
	tp = data[:,3]
	frame_ID = data[:,4]

	u_track, utrack_ind, utrack_count = np.unique(track_ID,return_index=True,return_counts=True)
	cut = u_track[(utrack_count>=t_len_l)*(utrack_count<=t_len_u)]

	for j in range(len(cut)):
		tind=(track_ID==cut[j])

		#sorting by frame per track
		temp = sorted(zip(frame_ID[tind], tp[tind], tp_x[tind], tp_y[tind]))
		nx = [x for f, t, x, y in temp]
		nt = [t for f, t, x, y in temp]
		ny = [y for f, t, x, y in temp]
		nf = [f for f, t, x, y in temp]
		x_all.append(nx)
		y_all.append(ny)
		f_all.append(nf)


msd_all = []
fit_alpha = []
fit_d = []
for i in range(len(x_all)):
	msd_set = track_decomp(x_all[i],y_all[i],f_all[i],1.)
	tt=  msd_set[~np.isnan(msd_set)]
	try:
		#popt , pcov = curve_fit(fit_MSD,np.arange(len(tt))[:6],np.array(tt)[:6],p0=[0.02,0.4],maxfev=1000000)
		popt , pcov = curve_fit(fit_MSD_Linear,np.log(np.arange(len(tt))[:3]),np.log(np.array(tt)[:3]),p0=[np.log(0.02),0.4],maxfev=1000000)
		fit_alpha.append(popt[1])
		fit_d.append(popt[0])
	except:
		print("Too Short")

	msd_all.append(msd_set)
for i in msd_all:
	plt.plot(i,'b')
# plt.yscale("log")
# plt.xscale("log")

msd_all = boolean_indexing(msd_all)
mean_msd = np.nanmean(msd_all, axis = 0)
std_msd = np.nanstd(msd_all, axis = 0)
mean_msd = mean_msd[mean_msd>0]
std_msd = std_msd[std_msd>0]
popt , pcov = curve_fit(fit_MSD,np.arange(len(mean_msd))[:6],np.array(mean_msd)[:6],p0=[0.02,0.4],maxfev=1000000)
#popt , pcov = curve_fit(fit_MSD_Linear,np.log(np.arange(1,len(mean_msd))[:6]),np.log(np.array(mean_msd)[:6]),p0=[np.log(0.02),0.4],maxfev=1000000)
popt_end , pcov_end = curve_fit(fit_MSD,np.arange(len(mean_msd))[6:14],np.array(mean_msd)[6:14],p0=[0.02,0.4],maxfev=1000000)
plt.plot(np.arange(len(mean_msd)),mean_msd)
plt.show()

plt.plot(np.arange(1,len(mean_msd)+1)*0.1,mean_msd*10)
plt.plot(np.arange(1,len(mean_msd)+1)*0.1,10*fit_MSD(np.arange(len(mean_msd)),popt[0],popt[1]))
#plt.plot(np.arange(len(mean_msd))[6:14],fit_MSD(np.arange(len(mean_msd))[6:14],popt_end[0],popt_end[1]))
plt.fill_between(np.arange(1,len(mean_msd)+1)*0.1,(mean_msd-std_msd)*10,(mean_msd+std_msd)*10,alpha = 0.2)
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel("Tau")
plt.ylabel("MSD")
plt.show()