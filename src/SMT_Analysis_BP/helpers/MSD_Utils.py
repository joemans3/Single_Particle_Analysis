import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Type
from scipy.optimize import curve_fit
from scipy.stats import norm


from src.SMT_Analysis_BP.helpers import Analysis_functions as af
from src.SMT_Analysis_BP.helpers import decorators as dec
from src.SMT_Analysis_BP.helpers import smallestenclosingcircle as sec
from src.SMT_Analysis_BP.Parameter_Store import global_params as gparms
from src.SMT_Analysis_BP.databases import trajectory_analysis_script as tas

#radius of confinment fucntion
def radius_of_confinement(t,r_sqr,D,loc_msd):
    return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 4*(loc_msd**2)

#radius of confinment fucntion
def radius_of_confinement_xy(t,r_sqr,D,loc_msd_x,loc_msd_y):
    return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 4*(loc_msd_x**2) + 4*(loc_msd_y**2)

#power law function with independent x and y
def power_law_xy(t,alpha_x,alpha_y,D,loc_msd_x,loc_msd_y):
    return 4*(loc_msd_x**2) + 4*(loc_msd_y**2) + 4.*D*t**(alpha_x+alpha_y)

#power law function with r_sqr
def power_law(t,alpha,D,loc_msd):
    return 4*(loc_msd**2) + 4.*D*t**(alpha)

def combine_database_tracks(track_dict_bulk:list):
    pass



#lets create a generic MSD analysis class which will store all the MSD analysis for a single dataset
#using encapsulation we will utilize smaller functions to do the analysis
class MsdDatabaseUtil:
    def __init__(self,data_set_RA:Type(tas.run_analysis)) -> None:
        self.data_set = data_set_RA
        self.track_dict_bulk = self.data_set._convert_to_track_dict_bulk() #see tas.run_analysis._convert_to_track_dict_bulk
        
        pass

    


    @property 
    def data_set(self):
        return self._data_set
    @data_set.setter
    def data_set(self,data_set):
        if hasattr(self,'_data_set'):
            raise Exception('data_set already set for {0} data at {1}'.format(self.parameters_dict["t_string"],self.parameters_dict["wd"]))
        self._data_set = data_set
    @property
    def parameters_dict(self):
        if not hasattr(self,'_parameters_dict'):
            self._parameters_dict = self._data_set.parameter_storage
        return self._parameters_dict
    @property
    def track_dict_bulk(self):
        return self._track_dict_bulk
    @track_dict_bulk.setter
    def track_dict_bulk(self,track_dict_bulk):
        #do not allow track_dict_bulk to be set twice
        if hasattr(self,'_track_dict_bulk'):
            raise Exception('track_dict_bulk already set for {0} data at {1}'.format(self.parameters_dict["t_string"],self.parameters_dict["wd"]))
        self._track_dict_bulk = track_dict_bulk







##############################################################################################################
#utilities which are only for internal use, will get removed later
def msd_calc(track_dic,h=None,tau_lim=None,tick_space=2,save=False,cd=None,data_type=None,plot=True,msd_fit_lim=3,convert=None,**kwargs):
    '''Docstring for msd_calc, this is just a fancy wrapper for the MSD_Tracks function in the Analysis_functions module that also does some plotting
    Not very useful for anything other than plotting the MSD curves for a set of tracks.
    MSD calculations can be done using this but it is obtuse and not recommended. See MSD_Tracks for a better way to do this.

    Parameters:
    -----------
    track_dic: dictionary
        dictionary of tracks with the keys being the track number and the values being the track
    h: float
        True husrt value for the simulation, if None this does not get plotted
    tau_lim: int
        The maximum tau value to plot, if None then this is set to the maximum tau value. Only used if plot is True
    tick_space: int
        Total ticks for colorbar in Van hove Correlation Plot, only used if plot is True
    save: bool
        If True then the plot is saved to the specified directory
    cd: str
        The directory to save the plot to, only used if save is True
    data_type: str
        The type of data that is being plotted, only used if save is True. This is the name of the folder that the plot is saved to
    plot: bool
        If True then the plot is plotted
    msd_fit_lim: int, array-like of length 2, or None, optional
        The number of points to fit the line to for the alpha value
        if array then the first value is the lower limit and the second value is the upper limit to fit for tau
    convert: Default, None (takes values in um for the pixel->um conversion)
        covert pixel to um
    

    Returns:
    --------
    Dict containing:
    fit_ens: array
        The fit parameters for the ensemble of tracks
    track_alpha: dict
        The alpha values for each track
    tavg_t1_msd: dict
        The time averaged msd for tau = 1
    track_msds: dict
        The msd curves for each track
    track_alpha_linear_fit: dict
        The linear fit parameters for each track
    track_diffusion: dict
        The diffusion coefficient for each track using a polynomial fit
    track_diffusion_linear_fit: dict
        The diffusion coefficient for each track using a linear fit
    
    

    '''
    #if save is True and cd is None then raise an error
    if save:
        if cd is None:
            raise ValueError("cd must be specified if save is True")
        if data_type is None:
            raise ValueError("data_type must be specified if save is True")

    msd_dict,ens_displacements = af.MSD_Tracks(track_dic,permutation=True,return_type="both",verbose=True,conversion_factor=convert)
    msd = msd_dict["msd_curves"][0]
    msd_error = msd_dict["msd_curves"][1]
    disp_per_track = msd_dict["displacements"]
    #update the disp_per_track dictionary to have the msd curve per track
    track_msds = {}
    for i,j in disp_per_track.items():
        track_msds[i] = af.msd_avgerage_utility(j)[0]
    #fit a line to the msd curves for the first n of the points and find the r2 value
    try:
        fit_num = kwargs.get("fit_num",10)
        fit_num_lower = kwargs.get("fit_num_lower",0)
        if isinstance(msd_fit_lim,int):
            fit_ens = np.polyfit(np.log(list(msd.keys())[:msd_fit_lim]),np.log(list(msd.values())[:msd_fit_lim]),1,cov=True)
            
        elif isinstance(msd_fit_lim,list|tuple|np.ndarray):
            fit_ens = np.polyfit(np.log(list(msd.keys())[msd_fit_lim[0]:msd_fit_lim[1]]),np.log(list(msd.values())[msd_fit_lim[0]:msd_fit_lim[1]]),1,cov=True)
            #fit_ens = np.polyfit(np.log(list(msd.keys())[-15:-6]),np.log(list(msd.values())[-15:-6]),1,cov=True)
        #fit_ens = np.polyfit(np.log(list(msd.keys())[fit_num_lower:fit_num]),np.log(list(msd.values())[fit_num_lower:fit_num]),1,cov=True)
        slope_error = np.sqrt(fit_ens[1])
        #fit the first 12 time points to the radius_of_confinement function
        fit_ens_con,pcov_fit_ens_con = curve_fit(radius_of_confinement_xy,0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),np.array(list(msd.values()))[fit_num_lower:fit_num],p0=[1,0.3,0.01,0.01],method='lm')
        fit_ens_power_law,fit_ens_power_law_con = curve_fit(power_law,0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),np.array(list(msd.values()))[fit_num_lower:fit_num],p0=[1,1,0.03,0.03],method='lm')
        print(fit_ens)
        print(fit_ens_con,pcov_fit_ens_con)
        print(fit_ens_power_law,fit_ens_power_law_con)
        #plot this fit
        if plot:
            plt.errorbar(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),np.array(list(msd.values()))[fit_num_lower:fit_num],yerr=np.array(list(msd_error.values())[fit_num_lower:fit_num])*1.96,fmt="o",label="Ensemble MSD")
            plt.plot(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),radius_of_confinement_xy(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),*fit_ens_con),label="Radius of Confinement Fit")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("MSD (um^2)")
            plt.yscale("log")
            plt.xscale("log")
            plt.show()
            plt.errorbar(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),np.array(list(msd.values()))[fit_num_lower:fit_num],yerr=np.array(list(msd_error.values())[fit_num_lower:fit_num])*1.96,fmt="o",label="Ensemble MSD")
            plt.plot(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),power_law(0.02*(np.array(list(msd.keys()))[fit_num_lower:fit_num]),*fit_ens_power_law),label="Power Law Fit")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("MSD (um^2)")
            plt.yscale("log")
            plt.xscale("log")
            plt.title("Power_law")
            plt.show()

    except:
        fit_ens = None
        slope_error = None

    track_alphas = {}
    track_alphas_linear_fit = {}
    tavg_t1_msds = {}
    track_diffusion = {}
    track_diffusion_linear_fit = {}
    loc_err = {}
    d_app_loc_corr = {}
    # for each track plot the msd_curve 
    for i,j in track_msds.items():
        #make sure the length of the track is greater than 3 so that the fit can be done
        if len(j.keys())<3:
            continue
        #set the alpha to be 0.1 so that the lines are transparent
        #if plot:
        #    plt.plot(j.keys(),j.values(),alpha=0.1)
        #fit a line to the msd curves for the first 3 of the points and find the r2 value
        if isinstance(msd_fit_lim,int):
            #fit,pcov = curve_fit(fit_MSD_Linear,np.log(list(j.keys())[:msd_fit_lim]),np.log(list(j.values())[:msd_fit_lim]),p0=[1,1])
            #repeat this with fitting the msd to a the function fit_MSD from Analysis_functions using curve_fit
            fit_curve,pcov = curve_fit(af.fit_MSD,list(j.keys())[:msd_fit_lim],list(j.values())[:msd_fit_lim],p0=[1,1,0],maxfev=1000000)
            #fit using the loc_error function
            fit_curve_loc,pcov_loc = curve_fit(af.fit_MSD_loc_err,list(j.keys())[:msd_fit_lim],list(j.values())[:msd_fit_lim],p0=[1,1,1],maxfev=1000000)
        elif isinstance(msd_fit_lim,list|tuple|np.ndarray):
            #fit,pcov = curve_fit(fit_MSD_Linear,np.log(list(j.keys())[msd_fit_lim[0]:msd_fit_lim[1]]),np.log(list(j.values())[msd_fit_lim[0]:msd_fit_lim[1]]),p0=[1,1])
            #repeat this with fitting the msd to a the function fit_MSD from Analysis_functions using curve_fit
            fit_curve,pcov = curve_fit(af.fit_MSD,list(j.keys())[msd_fit_lim[0]:msd_fit_lim[1]],list(j.values())[msd_fit_lim[0]:msd_fit_lim[1]],p0=[1,1,0],maxfev=1000000)
            #fit using the loc_error function
            fit_curve_loc,pcov_loc = curve_fit(af.fit_MSD_loc_err,list(j.keys())[msd_fit_lim[0]:msd_fit_lim[1]],list(j.values())[msd_fit_lim[0]:msd_fit_lim[1]],p0=[0.045,0.2,1],maxfev=1000000)
        #plot the fitted line
        #if plot and fit_curve[1]<0:
        #     plt.plot(list(j.keys())[:msd_fit_lim],fit_MSD(list(j.keys())[:msd_fit_lim],fit_curve[0],fit_curve[1]),alpha=0.1)
        #     plt.plot(j.keys(),j.values(),alpha=0.1)
        # plt.show()
        if plot:
            plt.plot(list(j.keys()),np.array(list(j.values())),alpha=0.1)
        #add the slope of the fitted line to the track_alphas dictionary
        #track_alphas_linear_fit[i] = fit[1]
        track_alphas[i] = fit_curve[1]
        #add the msd at tau=1 to the tavg_t1_msds dictionary, divide by 4 to get the correct value t is by default 1 since its tau=1
        tavg_t1_msds[i] = j[1]/4.
        track_diffusion[i] = fit_curve[0]/4.
        #track_diffusion_linear_fit[i] = np.exp(fit[0])/4.
        loc_err[i] = fit_curve_loc[2]
        d_app_loc_corr[i] = fit_curve_loc[0]/4.
        
    if plot:
        #plot the msd curves and the fitted line
        plt.plot(list(msd.keys())[:fit_num],np.array(list(msd.values()))[:fit_num],label="MSD_ensemble",linewidth=3,alpha=1,zorder=1)
        if fit_ens != None:
            plt.plot(list(msd.keys())[:fit_num],np.exp(fit_ens[0][1])*(np.array(list(msd.keys()))[:fit_num])**fit_ens[0][0],label="fit_ensemble",linewidth=3,alpha=1,zorder=2)

        plt.xscale("log")
        plt.yscale("log")
        #label the plot
        plt.xlabel("lag time (au)")
        plt.ylabel("MSD (au)")
        plt.legend()
        #annotate the plot with the slope of the fitted line with 2 decimal places (label the slope as alpha in greek)
        #add the error in the slope as well
        #plt.annotate(r"$\alpha$ = {:.2f} $\pm$ {:.2f}".format(fit_ens[0][0],slope_error[0][0]),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
        #annotate the true alpha value (hurst*2)
        if h != None:
            plt.annotate(r"True $\alpha$ = {:.2f}".format(h*2),xy=(0.05,0.6),xycoords="axes fraction",fontweight="bold",fontsize=16)
        if save == True:
            plt.savefig(cd+"/{0}_MSD_plot.png".format(data_type))
        plt.show()

        #repeat on a linear-linear plot
        #plot the msd curves and the fitted line
        plt.errorbar(list(msd.keys())[:fit_num],np.array(list(msd.values()))[:fit_num],yerr=np.array(list(msd_error.values()))[:fit_num]*1.96,label="MSD_ensemble",linewidth=3,alpha=1,zorder=1)
        if fit_ens != None:
            plt.plot(list(msd.keys())[:fit_num],np.exp(fit_ens[0][1])*(np.array(list(msd.keys()))[:fit_num])**fit_ens[0][0],label="fit_ensemble",linewidth=3,alpha=1,zorder=2)

        # plt.xscale("log")
        # plt.yscale("log")
        #label the plot
        plt.xlabel("lag time (au)")
        plt.ylabel("MSD (au)")
        plt.legend()
        #annotate the plot with the slope of the fitted line with 2 decimal places (label the slope as alpha in greek)
        #add the error in the slope as well
        #plt.annotate(r"$\alpha$ = {:.2f} $\pm$ {:.2f}".format(fit_ens[0][0],slope_error[0][0]),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
        #annotate the true alpha value (hurst*2)
        if h != None:
            plt.annotate(r"True $\alpha$ = {:.2f}".format(h*2),xy=(0.05,0.6),xycoords="axes fraction",fontweight="bold",fontsize=16)
        if save == True:
            plt.savefig(cd+"/{0}_MSD_plot.png".format(data_type))
        plt.show()

        #on a new figure plot the histogram of the slopes of the fitted lines
        plt.clf()
        plt.hist(list(track_alphas.values()),bins=10)
        #plot a vertical line at the mean of the track_alphas
        plt.axvline(np.mean(list(track_alphas.values())),color="red",label="mean")
        #annotate the plot with the mean of the track_alphas
        #plt.annotate(r"$\alpha$ = {:.2f}".format(np.mean(list(track_alphas.values()))),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("count")
        if save == True:
            plt.savefig(cd+"/{0}_alpha_hist.png".format(data_type))
        plt.show()

        #repeat the above for the track_alphas_linear_fit
        plt.clf()
        plt.hist(list(track_alphas_linear_fit.values()),bins=10)
        plt.axvline(np.mean(list(track_alphas_linear_fit.values())),color="red",label="mean")
        #plt.annotate(r"$\alpha$ = {:.2f}".format(np.mean(list(track_alphas_linear_fit.values()))),xy=(0.05,0.7),xycoords="axes fraction",fontweight="bold",fontsize=16)
        plt.xlabel(r"$\alpha$")
        plt.ylabel("count")
        if save == True:
            plt.savefig(cd+"/{0}_alpha_hist_linear_fit.png".format(data_type))
        plt.show()

    #the following plots the pdf of the displacements for each tau, right now it sometimes creates infinite loops so it is commented out TODO: fix this
    if plot:
        #make a figure and axes 2 subplots
        fig,ax = plt.subplots(1,2,figsize=(20,10))

        #get a collection of N different colours where N is the number of taus
        colors = plt.cm.jet(np.linspace(0,1,len(ens_displacements.keys())))

    #make a df to store the tau value and the fitted gmm mean for that tau
    gmm_tau_df = pd.DataFrame(columns=["tau","mean","sigma"])
    #make a histogram of the displacements for each tau from ens_displacements
    for i,j in ens_displacements.items():
        #convert to r
        j_r = np.sqrt(np.sum(np.array(j)**2,axis=1))
        #if the tau is greater than the tau_lim then skip it
        if (tau_lim != None):
            if i > tau_lim: 
                continue
        if plot:
            #make the histogram normalized and transparent for the first subplot
            ax[0].hist(np.ndarray.flatten(np.array(j)),bins=100,alpha=0.1,color=colors[i-1],density=True)#,stacked=True,weights=np.ones(len(np.ndarray.flatten(np.array(j))))/len(np.ndarray.flatten(np.array(j))))
            #make the histogram normalized and transparent for the second subplot for abs displacements
            ax[1].hist(np.abs(np.ndarray.flatten(np.array(j_r))),bins=100,alpha=0.1,color=colors[i-1],density=True)#,stacked=True,weights=np.ones(len(np.ndarray.flatten(np.array(j))))/len(np.ndarray.flatten(np.array(j))))
            pass
        #fit a gaussian to the histogram
        mu,sigma = norm.fit(np.ndarray.flatten(np.array(j)))
        #fit it for the abs displacements as well
        mu_abs,sigma_abs = norm.fit(np.abs(np.ndarray.flatten(np.array(j))))
        #store the tau and the mean of the gaussian in the gmm_tau_df using concat
        gmm_tau_df = pd.concat([gmm_tau_df,pd.DataFrame(columns={"tau":i,"mean":mu,"sigma":sigma**2})],ignore_index=True)
        #gmm_tau_df = gmm_tau_df.append({"tau":i,"mean":mu_abs,"sigma":sigma_abs**2},ignore_index=True)
        if plot:
            #plot the gaussian
            x = np.linspace(np.min(np.ndarray.flatten(np.array(j))),np.max(np.ndarray.flatten(np.array(j))),100)
            x_abs = np.linspace(np.min(np.abs(np.ndarray.flatten(np.array(j_r)))),np.max(np.abs(np.ndarray.flatten(np.array(j_r)))),100)
            ax[0].plot(x,norm.pdf(x,mu,sigma),linewidth=1,color=colors[i-1])
            ax[1].plot(x_abs,norm.pdf(x_abs,mu_abs,sigma_abs),linewidth=1,color=colors[i-1])

    if plot:
        #label the plot, in greek the delta_x is P_delta_x
        ax[0].set_xlabel(r"$\Delta r$ (au)")
        ax[0].set_ylabel(r"$P_{\Delta r}$ ($au^{-1}$)")
        ax[1].set_xlabel(r"$|\Delta r|$ (au)")
        ax[1].set_ylabel(r"$P_{|\Delta r|}$ ($au^{-1}$)")
        ax[0].set_yscale("log")

        v1 = np.linspace(np.min(np.array(list(ens_displacements.keys()),dtype=int)), np.max(np.array(list(ens_displacements.keys()),dtype=int)), tick_space, endpoint=True)
        #rather than a legend, make a colorbar with the colors corresponding to the taus
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet),ticks=v1,ax=ax[0])
        cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])

        cbar.set_label("tau (au)")


        #make the title
        ax[0].set_title("PDF of displacements for each tau (van Hove correlation)")
        #ax[0].set_ylim(0,1.5)
        if save == True:
            plt.savefig(cd+"/{0}_PDF_plot.png".format(data_type))
        plt.show()
        #plot the mean of the gaussian for each tau
        plt.errorbar(gmm_tau_df["tau"],gmm_tau_df["mean"],yerr=gmm_tau_df["sigma"],fmt="o")
        plt.xlabel("tau (au)")
        plt.ylabel("mean of gaussian fit (au)")
        plt.title("Mean of gaussian fit for each tau")
        plt.show()

    return {"fit_ens":fit_ens, 
            "track_alpha":track_alphas, 
            "tavg_t1_msd":tavg_t1_msds, 
            "track_msds":track_msds, 
            "track_alpha_linear_fit":track_alphas_linear_fit,
            "track_diffusion":track_diffusion,
            "track_diffusion_linear_fit":track_diffusion_linear_fit,
            "D_app_loc_corr":d_app_loc_corr,
            "loc_err":loc_err,
            "msd_curve_ens":msd,
            "msd_curve_ens_err":msd_error,
            "Displacements_per_track": disp_per_track,
            "Track_msds":track_msds,
            "ens_displacements":ens_displacements}