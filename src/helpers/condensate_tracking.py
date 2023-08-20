import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
from scipy.stats import norm
if __name__=="__main__":
	import sys
	sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
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
path = '/Users/baljyot/Documents/test/'

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

len_track = []
for i in range(len(x_all)):
	len_track.append(len(x_all[i]))
plt.hist(len_track,bins=20)
plt.show()

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

# for i in msd_all:
# 	plt.plot(i,'b--')
# plt.yscale("log")
# plt.xscale("log")
def radius_of_confinement_xy(t,r_sqr,D,loc_msd_x,loc_msd_y):
	return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 2*(loc_msd_x**2) + 2*(loc_msd_y**2)
# def msd_calc(track_dic,h=None,tau_lim=None,tick_space=2,save=False,cd=None,data_type=None,plot=True,msd_fit_lim=3):
# 	'''Docstring for msd_calc, this is just a fancy wrapper for the MSD_Tracks function in the Analysis_functions module that also does some plotting
# 	Not very useful for anything other than plotting the MSD curves for a set of tracks.
# 	MSD calculations can be done using this but it is obtuse and not recommended. See MSD_Tracks for a better way to do this.

# 	Parameters:
# 	-----------
# 	track_dic: dictionary
# 		dictionary of tracks with the keys being the track number and the values being the track
# 	h: float
# 		True husrt value for the simulation, if None this does not get plotted
# 	tau_lim: int
# 		The maximum tau value to plot, if None then this is set to the maximum tau value. Only used if plot is True
# 	tick_space: int
# 		Total ticks for colorbar in Van hove Correlation Plot, only used if plot is True
# 	save: bool
# 		If True then the plot is saved to the specified directory
# 	cd: str
# 		The directory to save the plot to, only used if save is True
# 	data_type: str
# 		The type of data that is being plotted, only used if save is True. This is the name of the folder that the plot is saved to
# 	plot: bool
# 		If True then the plot is plotted
# 	msd_fit_lim: int, array-like of length 2, or None, optional
# 		The number of points to fit the line to for the alpha value
# 		if array then the first value is the lower limit and the second value is the upper limit to fit for tau
	

# 	Returns:
# 	--------
# 	Dict containing:
# 	fit_ens: array
# 		The fit parameters for the ensemble of tracks
# 	track_alpha: dict
# 		The alpha values for each track
# 	tavg_t1_msd: dict
# 		The time averaged msd for tau = 1
# 	track_msds: dict
# 		The msd curves for each track
# 	track_alpha_linear_fit: dict
# 		The linear fit parameters for each track
# 	track_diffusion: dict
# 		The diffusion coefficient for each track using a polynomial fit
# 	track_diffusion_linear_fit: dict
# 		The diffusion coefficient for each track using a linear fit
	
	

# 	'''
# 	#if save is True and cd is None then raise an error
# 	if save:
# 		if cd is None:
# 			raise ValueError("cd must be specified if save is True")
# 		if data_type is None:
# 			raise ValueError("data_type must be specified if save is True")

# 	msd_dict,ens_displacements = MSD_Tracks(track_dic,permutation=True,return_type="both",verbose=True,conversion_factor=0.13)
# 	msd = msd_dict["msd_curves"][0]
# 	msd_error = msd_dict["msd_curves"][1]
# 	disp_per_track = msd_dict["displacements"]
# 	#update the disp_per_track dictionary to have the msd curve per track
# 	track_msds = {}
# 	for i,j in disp_per_track.items():
# 		track_msds[i] = msd_avgerage_utility(j)[0]
# 	#fit a line to the msd curves for the first n of the points and find the r2 value
# 	try:
# 		fit_num = 50
# 		fit_num_lower = 0
# 		if isinstance(msd_fit_lim,int):
# 			fit_ens = np.polyfit(np.log(list(msd.keys())[:msd_fit_lim]),np.log(list(msd.values())[:msd_fit_lim]),1,cov=True)
# 		elif isinstance(msd_fit_lim,list|tuple|np.ndarray):
# 			fit_ens = np.polyfit(np.log(list(msd.keys())[msd_fit_lim[0]:msd_fit_lim[1]]),np.log(list(msd.values())[msd_fit_lim[0]:msd_fit_lim[1]]),1,cov=True)
# 			#fit_ens = np.polyfit(np.log(list(msd.keys())[-15:-6]),np.log(list(msd.values())[-15:-6]),1,cov=True)
# 		#fit_ens = np.polyfit(np.log(list(msd.keys())[fit_num_lower:fit_num]),np.log(list(msd.values())[fit_num_lower:fit_num]),1,cov=True)
# 		slope_error = np.sqrt(fit_ens[1])
# 		#fit the first 12 time points to the radius_of_confinement function
# 		fit_ens_con,pcov_fit_ens_con = curve_fit(radius_of_confinement_xy,0.02*(np.array(list(msd.keys()))[:fit_num]),np.array(list(msd.values()))[:fit_num],p0=[1,0.3,0.01,0.01],method='lm')
# 		print(fit_ens_con,pcov_fit_ens_con)
# 		#plot this fit
# 		if plot:
# 			plt.errorbar(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),np.array(list(msd.values()))[fit_num_lower:fit_num],yerr=np.array(list(msd_error.values())[fit_num_lower:fit_num])*1.96,fmt="o",label="Ensemble MSD")
# 			plt.plot(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),radius_of_confinement_xy(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),*fit_ens_con),label="Radius of Confinement Fit")
# 			plt.legend()
# 			plt.xlabel("Time (s)")
# 			plt.ylabel("MSD (um^2)")
# 			plt.show()
# 	except:
# 		fit_ens = None
# 		slope_error = None

# 	track_alphas = {}
# 	track_alphas_linear_fit = {}
# 	tavg_t1_msds = {}
# 	track_diffusion = {}
# 	track_diffusion_linear_fit = {}
# 	loc_err = {}
# 	d_app_loc_corr = {}
# 	# for each track plot the msd_curve 
# 	for i,j in track_msds.items():
# 		#make sure the length of the track is greater than 3 so that the fit can be done
# 		if len(j.keys())<3:
# 			continue
# 		#set the alpha to be 0.1 so that the lines are transparent
# 		#if plot:
# 		#    plt.plot(j.keys(),j.values(),alpha=0.1)
# 		#fit a line to the msd curves for the first 3 of the points and find the r2 value
# 		if isinstance(msd_fit_lim,int):
# 			fit,pcov = curve_fit(fit_MSD_Linear,np.log(list(j.keys())[:msd_fit_lim]),np.log(list(j.values())[:msd_fit_lim]),p0=[1,1])
# 			#repeat this with fitting the msd to a the function fit_MSD from Analysis_functions using curve_fit
# 			fit_curve,pcov = curve_fit(fit_MSD,list(j.keys())[:msd_fit_lim],list(j.values())[:msd_fit_lim],p0=[1,1,0],maxfev=1000000)
# 			#fit using the loc_error function
# 			fit_curve_loc,pcov_loc = curve_fit(fit_MSD_loc_err,list(j.keys())[:msd_fit_lim],list(j.values())[:msd_fit_lim],p0=[1,1,1],maxfev=1000000)
# 		elif isinstance(msd_fit_lim,list|tuple|np.ndarray):
# 			fit,pcov = curve_fit(fit_MSD_Linear,np.log(list(j.keys())[msd_fit_lim[0]:msd_fit_lim[1]]),np.log(list(j.values())[msd_fit_lim[0]:msd_fit_lim[1]]),p0=[1,1])
# 			#repeat this with fitting the msd to a the function fit_MSD from Analysis_functions using curve_fit
# 			fit_curve,pcov = curve_fit(fit_MSD,list(j.keys())[msd_fit_lim[0]:msd_fit_lim[1]],list(j.values())[msd_fit_lim[0]:msd_fit_lim[1]],p0=[1,1,0],maxfev=1000000)
# 			#fit using the loc_error function
# 			fit_curve_loc,pcov_loc = curve_fit(fit_MSD_loc_err,list(j.keys())[msd_fit_lim[0]:msd_fit_lim[1]],list(j.values())[msd_fit_lim[0]:msd_fit_lim[1]],p0=[0.045,0.2,1],maxfev=1000000)
# 		#plot the fitted line
# 		#if plot and fit_curve[1]<0:
# 		#     plt.plot(list(j.keys())[:msd_fit_lim],fit_MSD(list(j.keys())[:msd_fit_lim],fit_curve[0],fit_curve[1]),alpha=0.1)
# 		#     plt.plot(j.keys(),j.values(),alpha=0.1)
# 		# plt.show()
# 		if plot:
# 			plt.plot(list(j.keys()),np.array(list(j.values())),alpha=0.1)
# 		#add the slope of the fitted line to the track_alphas dictionary
# 		track_alphas_linear_fit[i] = fit[1]
# 		track_alphas[i] = fit_curve[1]
# 		#add the msd at tau=1 to the tavg_t1_msds dictionary, divide by 4 to get the correct value t is by default 1 since its tau=1
# 		tavg_t1_msds[i] = j[1]/4.
# 		track_diffusion[i] = fit_curve[0]/4.
# 		track_diffusion_linear_fit[i] = np.exp(fit[0])/4.
# 		loc_err[i] = fit_curve_loc[2]
# 		d_app_loc_corr[i] = fit_curve_loc[0]/4.
		
# 	if plot:
# 		#plot the msd curves and the fitted line
# 		plt.plot(list(msd.keys())[:fit_num],np.array(list(msd.values()))[:fit_num],label="MSD_ensemble",linewidth=3,alpha=1,zorder=1)
# 		if fit_ens != None:
# 			plt.plot(list(msd.keys())[:fit_num],np.exp(fit_ens[0][1])*(np.array(list(msd.keys()))[:fit_num])**fit_ens[0][0],label="fit_ensemble",linewidth=3,alpha=1,zorder=2)

# 		plt.xscale("log")
# 		plt.yscale("log")
# 		#label the plot
# 		plt.xlabel("lag time (au)")
# 		plt.ylabel("MSD (au)")
# 		plt.legend()
# 		#annotate the plot with the slope of the fitted line with 2 decimal places (label the slope as alpha in greek)
# 		#add the error in the slope as well
# 		plt.annotate(r"$\alpha$ = {:.2f} $\pm$ {:.2f}".format(fit_ens[0][0],slope_error[0][0]),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
# 		#annotate the true alpha value (hurst*2)
# 		if h != None:
# 			plt.annotate(r"True $\alpha$ = {:.2f}".format(h*2),xy=(0.05,0.6),xycoords="axes fraction",fontweight="bold",fontsize=16)
# 		if save == True:
# 			plt.savefig(cd+"/{0}_MSD_plot.png".format(data_type))
# 		plt.show()

# 		#repeat on a linear-linear plot
# 		#plot the msd curves and the fitted line
# 		plt.errorbar(list(msd.keys())[:fit_num],np.array(list(msd.values()))[:fit_num],yerr=np.array(list(msd_error.values()))[:fit_num]*1.96,label="MSD_ensemble",linewidth=3,alpha=1,zorder=1)
# 		if fit_ens != None:
# 			plt.plot(list(msd.keys())[:fit_num],np.exp(fit_ens[0][1])*(np.array(list(msd.keys()))[:fit_num])**fit_ens[0][0],label="fit_ensemble",linewidth=3,alpha=1,zorder=2)

# 		# plt.xscale("log")
# 		# plt.yscale("log")
# 		#label the plot
# 		plt.xlabel("lag time (au)")
# 		plt.ylabel("MSD (au)")
# 		plt.legend()
# 		#annotate the plot with the slope of the fitted line with 2 decimal places (label the slope as alpha in greek)
# 		#add the error in the slope as well
# 		plt.annotate(r"$\alpha$ = {:.2f} $\pm$ {:.2f}".format(fit_ens[0][0],slope_error[0][0]),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
# 		#annotate the true alpha value (hurst*2)
# 		if h != None:
# 			plt.annotate(r"True $\alpha$ = {:.2f}".format(h*2),xy=(0.05,0.6),xycoords="axes fraction",fontweight="bold",fontsize=16)
# 		if save == True:
# 			plt.savefig(cd+"/{0}_MSD_plot.png".format(data_type))
# 		plt.show()

# 		#on a new figure plot the histogram of the slopes of the fitted lines
# 		plt.clf()
# 		plt.hist(list(track_alphas.values()),bins=10)
# 		#plot a vertical line at the mean of the track_alphas
# 		plt.axvline(np.mean(list(track_alphas.values())),color="red",label="mean")
# 		#annotate the plot with the mean of the track_alphas
# 		plt.annotate(r"$\alpha$ = {:.2f}".format(np.mean(list(track_alphas.values()))),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
# 		plt.xlabel(r"$\alpha$")
# 		plt.ylabel("count")
# 		if save == True:
# 			plt.savefig(cd+"/{0}_alpha_hist.png".format(data_type))
# 		plt.show()

# 		#repeat the above for the track_alphas_linear_fit
# 		plt.clf()
# 		plt.hist(list(track_alphas_linear_fit.values()),bins=10)
# 		plt.axvline(np.mean(list(track_alphas_linear_fit.values())),color="red",label="mean")
# 		plt.annotate(r"$\alpha$ = {:.2f}".format(np.mean(list(track_alphas_linear_fit.values()))),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
# 		plt.xlabel(r"$\alpha$")
# 		plt.ylabel("count")
# 		if save == True:
# 			plt.savefig(cd+"/{0}_alpha_hist_linear_fit.png".format(data_type))
# 		plt.show()

# 	#the following plots the pdf of the displacements for each tau, right now it sometimes creates infinite loops so it is commented out TODO: fix this
# 	if plot:
# 		#make a figure and axes 2 subplots
# 		fig,ax = plt.subplots(1,2,figsize=(20,10))

# 		#get a collection of N different colours where N is the number of taus
# 		colors = plt.cm.jet(np.linspace(0,1,len(ens_displacements.keys())))

# 	#make a df to store the tau value and the fitted gmm mean for that tau
# 	gmm_tau_df = pd.DataFrame(columns=["tau","mean","sigma"])
# 	#make a histogram of the displacements for each tau from ens_displacements
# 	for i,j in ens_displacements.items():
# 		#convert to r
# 		j_r = np.sqrt(np.sum(np.array(j)**2,axis=1))
# 		#if the tau is greater than the tau_lim then skip it
# 		if (tau_lim != None):
# 			if i > tau_lim: 
# 				continue
# 		if plot:
# 			#make the histogram normalized and transparent for the first subplot
# 			ax[0].hist(np.ndarray.flatten(np.array(j_r)),bins=100,alpha=0.1,color=colors[i-1],density=True)#,stacked=True,weights=np.ones(len(np.ndarray.flatten(np.array(j))))/len(np.ndarray.flatten(np.array(j))))
# 			#make the histogram normalized and transparent for the second subplot for abs displacements
# 			ax[1].hist(np.abs(np.ndarray.flatten(np.array(j_r))),bins=100,alpha=0.1,color=colors[i-1],density=True)#,stacked=True,weights=np.ones(len(np.ndarray.flatten(np.array(j))))/len(np.ndarray.flatten(np.array(j))))
# 			pass
# 		#fit a gaussian to the histogram
# 		mu,sigma = norm.fit(np.ndarray.flatten(np.array(j)))
# 		#fit it for the abs displacements as well
# 		mu_abs,sigma_abs = norm.fit(np.abs(np.ndarray.flatten(np.array(j))))
# 		#store the tau and the mean of the gaussian in the df
# 		gmm_tau_df = gmm_tau_df.append({"tau":i,"mean":mu_abs,"sigma":sigma_abs**2},ignore_index=True)
# 		if plot:
# 			#plot the gaussian
# 			x = np.linspace(np.min(np.ndarray.flatten(np.array(j_r))),np.max(np.ndarray.flatten(np.array(j_r))),100)
# 			x_abs = np.linspace(np.min(np.abs(np.ndarray.flatten(np.array(j_r)))),np.max(np.abs(np.ndarray.flatten(np.array(j_r)))),100)
# 			ax[0].plot(x,norm.pdf(x,mu,sigma),linewidth=1,color=colors[i-1])
# 			ax[1].plot(x_abs,norm.pdf(x_abs,mu_abs,sigma_abs),linewidth=1,color=colors[i-1])

# 	if plot:
# 		#label the plot, in greek the delta_x is P_delta_x
# 		ax[0].set_xlabel(r"$\Delta r$ (au)")
# 		ax[0].set_ylabel(r"$P_{\Delta r}$ ($au^{-1}$)")
# 		ax[1].set_xlabel(r"$|\Delta r|$ (au)")
# 		ax[1].set_ylabel(r"$P_{|\Delta r|}$ ($au^{-1}$)")


# 		v1 = np.linspace(np.min(np.array(list(ens_displacements.keys()),dtype=int)), np.max(np.array(list(ens_displacements.keys()),dtype=int)), tick_space, endpoint=True)
# 		#rather than a legend, make a colorbar with the colors corresponding to the taus
# 		cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet),ticks=v1,ax=ax[0])
# 		cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])

# 		cbar.set_label("tau (au)")


# 		#make the title
# 		ax[0].set_title("PDF of displacements for each tau (van Hove correlation)")
# 		ax[0].set_ylim(0,1.5)
# 		if save == True:
# 			plt.savefig(cd+"/{0}_PDF_plot.png".format(data_type))
# 		plt.show()
# 		#plot the mean of the gaussian for each tau
# 		plt.errorbar(gmm_tau_df["tau"],gmm_tau_df["mean"],yerr=gmm_tau_df["sigma"],fmt="o")
# 		plt.xlabel("tau (au)")
# 		plt.ylabel("mean of gaussian fit (au)")
# 		plt.title("Mean of gaussian fit for each tau")
# 		plt.show()

# 	return {"fit_ens":fit_ens, 
# 			"track_alpha":track_alphas, 
# 			"tavg_t1_msd":tavg_t1_msds, 
# 			"track_msds":track_msds, 
# 			"track_alpha_linear_fit":track_alphas_linear_fit,
# 			"track_diffusion":track_diffusion,
# 			"track_diffusion_linear_fit":track_diffusion_linear_fit,
# 			"D_app_loc_corr":d_app_loc_corr,
# 			"loc_err":loc_err}

# track_dict = {}
# for i in range(len(x_all)):
# 	track = []
# 	for j in range(len(x_all[i])):
# 		track.append([x_all[i][j],y_all[i][j],f_all[i][j]])
# 	track_dict[str(i+1)] = np.array(track)
# msd_calc_t = msd_calc(track_dict,tau_lim=10,plot=True,save=False,msd_fit_lim=[0,5])

# msd_all = boolean_indexing(msd_all)
# mean_msd = np.nanmean(msd_all, axis = 0)
# std_msd = np.nanstd(msd_all, axis = 0)
# mean_msd = mean_msd[mean_msd>0]
# std_msd = std_msd[std_msd>0]
# popt , pcov = curve_fit(fit_MSD,np.arange(len(mean_msd))[:6],np.array(mean_msd)[:6],p0=[0.02,0.4,0.1],maxfev=1000000)
# #popt , pcov = curve_fit(fit_MSD_Linear,np.log(np.arange(1,len(mean_msd))[:6]),np.log(np.array(mean_msd)[:6]),p0=[np.log(0.02),0.4],maxfev=1000000)
# popt_end , pcov_end = curve_fit(fit_MSD,np.arange(len(mean_msd))[6:14],np.array(mean_msd)[6:14],p0=[0.02,0.4,0.1],maxfev=1000000)
# plt.plot(np.arange(len(mean_msd)),mean_msd)
# plt.yscale("log")
# plt.xscale("log")
# plt.show()

# plt.plot(np.arange(1,len(mean_msd)+1),mean_msd)
# plt.plot(np.arange(1,len(mean_msd)+1),fit_MSD(np.arange(len(mean_msd)),*popt))
# #plt.plot(np.arange(len(mean_msd))[6:14],fit_MSD(np.arange(len(mean_msd))[6:14],popt_end[0],popt_end[1]))
# plt.fill_between(np.arange(1,len(mean_msd)+1),(mean_msd-std_msd),(mean_msd+std_msd),alpha = 0.2)
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("Tau")
# plt.ylabel("MSD")
# plt.show()


msd_t1 = []

for i in range(len(x_all)):
	dif_x = np.diff(x_all[i])*0.13
	dif_y = np.diff(y_all[i])*0.13
	dif_r2 = dif_x**2 + dif_y**2
	msd_t1.append(np.mean(dif_r2))

plt.hist(msd_t1,bins=20)
