from trajectory_analysis_script import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.stats import gaussian_kde
#import tensorflow as tf
import os
from plotting_functions import *
from import_functions import *
from diff_mw import *
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import OPTICS


from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl 
from scalebars import *

from Convert_csv_mat import *

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats  

import csv  


os.chdir("..")



# n = 1000 #n is the number of steps(increase in the value of n increses the compelxity of graph) 
# x = np.zeros(n) # x and y are arrays which store the coordinates of the position 
# y = np.zeros(n) 
# direction=["NORTH","SOUTH","EAST","WEST"] # Assuming the four directions of movement.
# for i in range(1, n): 
#     step = np.random.choice(direction) #Randomly choosing the direction of movement. 
#     if step == "EAST": #updating the direction with respect to the direction of motion choosen.
#         x[i] = x[i - 1] + 1
#         y[i] = y[i - 1] 
#     elif step == "WEST": 
#         x[i] = x[i - 1] - 1
#         y[i] = y[i - 1] 
#     elif step == "NORTH": 
#         x[i] = x[i - 1] 
#         y[i] = y[i - 1] + 1
#     else: 
#         x[i] = x[i - 1] 
#         y[i] = y[i - 1] - 1



def data_which(which):
  which.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 5000,t_len_l = 1)
  which.run_flow()
  total_which = np.array(list(which.i_d_tavg) + list(which.io_d_tavg) + list(which.o_d_tavg))
  total_msd = which.in_msd_track + which.io_msd_track + which.out_msd_track
  total_length = which.in_length + which.out_length + which.inout_length

  return [total_which,total_msd,total_length]







# rp_3_5= run_analysis("DATA/20210926/3_5_c_a_ez","rp_a_ez")
# rp_3_5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l=1)
# rp_3_5.run_flow()
# total_rp_3_5 = np.array(list(rp_3_5.i_d_tavg) + list(rp_3_5.io_d_tavg) + list(rp_3_5.o_d_tavg))


# rp_3_5_2= run_analysis("DATA/20210927/3_5_c_a_ez","rp_a_ez")
# rp_3_5_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l=1)
# rp_3_5_2.run_flow()
# total_rp_3_5_2 = np.array(list(rp_3_5_2.i_d_tavg) + list(rp_3_5_2.io_d_tavg) + list(rp_3_5_2.o_d_tavg))




# test_1 = run_analysis("Scripts/0.001-0.01-0.1_100-100-100_r-r_10-10-10_1000_SD_0.5_test","0.001-0.01-0.1_100-100-100_r-r_10-10-10_1000_SD_0.5_test")
# test_1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 1000,t_len_l = 1)
# test_1.run_flow()	
# total_test_1 = np.array(list(test_1.i_d_tavg) + list(test_1.io_d_tavg) + list(test_1.o_d_tavg))

# fis_ez= run_analysis("/Volumes/WEBERLAB/20200728/fis_mmaple/Folder_20200728","fis_mmaple")
# fis_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l=10)
# fis_ez.run_flow()
# total_fis_ez = np.array(list(fis_ez.i_d_tavg) + list(fis_ez.io_d_tavg) + list(fis_ez.o_d_tavg))


#######################################################     RPOC EZ

# rp_ez_test = run_analysis("DATA/20190620","rpoc_ez")
# rp_ez_test.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_test.run_flow()
# total_rp_ez_test = np.array(list(rp_ez_test.i_d_tavg) + list(rp_ez_test.io_d_tavg) + list(rp_ez_test.o_d_tavg))


rp_ez= run_analysis("DATA/new_days/20190527/rpoc_ez","rpoc_ez")
rp_ez.read_parameters(minimum_percent_per_drop_in = 0.5, t_len_u = 100, t_len_l=10, minimum_tracks_per_drop = 3)#, lower_bp = 0.01)
#rp_ez.read_parameters(minimum_percent_per_drop_in = 0.1, t_len_u = 100, t_len_l=5, minimum_tracks_per_drop = 2, lower_bp = 0.01)
#rp_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_l=10)
rp_ez.run_flow()



# in_o = []
# io_o = []
# o_o = []
# for i,y in rp_ez.Movie.items():
#     for j,x in y.IN_Trajectory_Collection.items():
#         in_o.append(x.MSD_total_um)
#     for j,x in y.IO_Trajectory_Collection.items():
#         io_o.append(x.MSD_total_um)
#     for j,x in y.OUT_Trajectory_Collection.items():
#         o_o.append(x.MSD_total_um)













# rp2= run_analysis("DATA/RPOC_new","RPOC")
# rp2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp2.run_flow()
# total_rp2 = np.array(list(rp2.i_d_tavg) + list(rp2.io_d_tavg) + list(rp2.o_d_tavg))


# rp3= run_analysis("DATA/Files_RPOC","RPOC")
# rp3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp3.run_flow()
# total_rp3 = np.array(list(rp3.i_d_tavg) + list(rp3.io_d_tavg) + list(rp3.o_d_tavg))

# rp1= run_analysis("DATA/Other_RPOC","rpoc")
# rp1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp1.run_flow()
# total_rp1 = np.array(list(rp1.i_d_tavg) + list(rp1.io_d_tavg) + list(rp1.o_d_tavg))



# rp= run_analysis("DATA","RPOC")
# rp.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp.run_flow()
# total_rp = np.array(list(rp.i_d_tavg) + list(rp.io_d_tavg) + list(rp.o_d_tavg))



# ############### RPOC M9


# rp_m9_2= run_analysis("DATA/new_days/20190524/rpoc_m9","rpoc_M9")
# rp_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_m9_2.run_flow()
# total_rp_m9_2 = np.array(list(rp_m9_2.i_d_tavg) + list(rp_m9_2.io_d_tavg) + list(rp_m9_2.o_d_tavg))


# rp_m9= run_analysis("DATA/rpoc_M9/20190515","rpoc_M9")
# rp_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_m9.run_flow()
# total_rp_m9 = np.array(list(rp_m9.i_d_tavg) + list(rp_m9.io_d_tavg) + list(rp_m9.o_d_tavg))

# rp_m9_0212= run_analysis("DATA/12/rpoc_m9","rpoc_ez")
# rp_m9_0212.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_m9_0212.run_flow()
# total_rp_m9_0212 = np.array(list(rp_m9_0212.i_d_tavg) + list(rp_m9_0212.io_d_tavg) + list(rp_m9_0212.o_d_tavg))

# rp_m9_0212_2= run_analysis("DATA/12/rpoc_m9_2","rpoc_ez")
# rp_m9_0212_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_m9_0212_2.run_flow()
# total_rp_m9_0212_2 = np.array(list(rp_m9_0212_2.i_d_tavg) + list(rp_m9_0212_2.io_d_tavg) + list(rp_m9_0212_2.o_d_tavg))


# rpoc_m9 = run_analysis("DATA/20200212/rpoc_m9_2","rpoc_ez")
# rpoc_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_m9.run_flow()
# total_rpoc_m9 = np.array(list(rpoc_m9.i_d_tavg) + list(rpoc_m9.io_d_tavg) + list(rpoc_m9.o_d_tavg))



# ############### RPOC Hex 5%


# rp_ez_h5 = run_analysis("DATA/rpoc_ez_hex_5","rpoc_ez_hex_5")
# rp_ez_h5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5.run_flow()
# total_rp_ez_h5 = np.array(list(rp_ez_h5.i_d_tavg) + list(rp_ez_h5.io_d_tavg) + list(rp_ez_h5.o_d_tavg))



# rp_ez_h5_2 = run_analysis("DATA/rpoc_ez_hex_5_2","rpoc_ez_h_5")
# rp_ez_h5_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5_2.run_flow()
# total_rp_ez_h5_2 = np.array(list(rp_ez_h5_2.i_d_tavg) + list(rp_ez_h5_2.io_d_tavg) + list(rp_ez_h5_2.o_d_tavg))



# rpoc_ez_hex5_binned = run_analysis("DATA/20200210/rpoc_ez_hex5_binned","rpoc_ez_hex5")
# rpoc_ez_hex5_binned.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_ez_hex5_binned.run_flow()
# total_rpoc_ez_hex5_binned = np.array(list(rpoc_ez_hex5_binned.i_d_tavg) + list(rpoc_ez_hex5_binned.io_d_tavg) + list(rpoc_ez_hex5_binned.o_d_tavg))


# rpoc_ez_hex5= run_analysis("DATA/20200210/rpoc_ez_hex5","rpoc_ez_hex5")
# rpoc_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_ez_hex5.run_flow()
# total_rpoc_ez_hex5 = np.array(list(rpoc_ez_hex5.i_d_tavg) + list(rpoc_ez_hex5.io_d_tavg) + list(rpoc_ez_hex5.o_d_tavg))


# rp_ez_h5_15 = run_analysis("DATA/15/rp_ez_hex5","rp_ez_hex5")
# rp_ez_h5_15.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5_15.run_flow()
# total_rp_ez_h5_15 = np.array(list(rp_ez_h5_15.i_d_tavg) + list(rp_ez_h5_15.io_d_tavg) + list(rp_ez_h5_15.o_d_tavg))

# rp_ez_h5_2_15 = run_analysis("DATA/15/rp_ez_hex5_2","rp_ez_hex5")
# rp_ez_h5_2_15.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5_2_15.run_flow()
# total_rp_ez_h5_2_15 = np.array(list(rp_ez_h5_2_15.i_d_tavg) + list(rp_ez_h5_2_15.io_d_tavg) + list(rp_ez_h5_2_15.o_d_tavg))

# rp_ez_h5_16 = run_analysis("DATA/16/rp_ez_hex5","nusa_ez_hex5")
# rp_ez_h5_16.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5_16.run_flow()
# total_rp_ez_h5_16 = np.array(list(rp_ez_h5_16.i_d_tavg) + list(rp_ez_h5_16.io_d_tavg) + list(rp_ez_h5_16.o_d_tavg))

# rp_ez_h5_2_16 = run_analysis("DATA/16/rp_ez_hex5_2","rp_ez_hex5")
# rp_ez_h5_2_16.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5_2_16.run_flow()
# total_rp_ez_h5_2_16 = np.array(list(rp_ez_h5_2_16.i_d_tavg) + list(rp_ez_h5_2_16.io_d_tavg) + list(rp_ez_h5_2_16.o_d_tavg))


# ############### RPOC Hex 3%


# rp_ez_h3 = run_analysis("DATA/new_days/20190527/rpoc_ez_hex_3","rpoc_ez_hex_3")
# rp_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h3.run_flow()
# total_rp_ez_h3 = np.array(list(rp_ez_h3.i_d_tavg) + list(rp_ez_h3.io_d_tavg) + list(rp_ez_h3.o_d_tavg))




# #######################################################     LacI,LacO EZ

# ll= run_analysis("DATA/LACO_LACI","TB54_FAST")
# ll.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll.run_flow()
# total_ll = np.array(list(ll.i_d_tavg) + list(ll.io_d_tavg) + list(ll.o_d_tavg))

# ll_ez= run_analysis("DATA/new_days/20190527/ll_ez","laco_laci_ez")
# ll_ez.read_parameters(minimum_percent_per_drop_in = 0.1, t_len_u = 50, t_len_l=10, minimum_tracks_per_drop = 2)
# ll_ez.run_flow()
# total_ll_ez = np.array(list(ll_ez.i_d_tavg) + list(ll_ez.io_d_tavg) + list(ll_ez.o_d_tavg))



# lI_ez_4= run_analysis("DATA/laci_only","lI")
# lI_ez_4.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l =4)
# lI_ez_4.run_flow()
# total_lI_ez_4 = np.array(list(lI_ez_4.i_d_tavg) + list(lI_ez_4.io_d_tavg) + list(lI_ez_4.o_d_tavg))


# lI_ez= run_analysis("DATA/laci_only","lI")
# lI_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# lI_ez.run_flow()
# total_lI_ez = np.array(list(lI_ez.i_d_tavg) + list(lI_ez.io_d_tavg) + list(lI_ez.o_d_tavg))

# ll_ez_17= run_analysis("DATA/17/ll_ez","ll_ez")
# ll_ez_17.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 4)
# ll_ez_17.run_flow()
# total_ll_ez_17 = np.array(list(ll_ez_17.i_d_tavg) + list(ll_ez_17.io_d_tavg) + list(ll_ez_17.o_d_tavg))






# msd_col = []
# alpha_col = []
# dif_col = []
# for key,value in rp_ez.Movie.items():
#     for k,v in rp_ez.Movie[key].All_Tracjectories.items():
#         x = v.X 
#         y = v.Y 
#         f = v.Frames
#         msd_set = track_decomp(x,y,f,1.)
#         tt=  msd_set[~np.isnan(msd_set)]
#         try:
#             popt , pcov = curve_fit(fit_MSD,np.arange(len(tt))[:6],np.array(tt)[:6],p0=[0.02,0.4],maxfev=1000000)
#             alpha_col.append(popt[1])
#             dif_col.append(popt[0])
#         except:
#             print("Too Short")
#         msd_col.append(msd_set)
# for i in msd_col:
#     plt.plot(i,'b')
# plt.yscale("log")
# plt.xscale("log")
# msd_all = boolean_indexing(msd_col)
# mean_msd = np.nanmean(msd_all, axis = 0)
# len_std = []
# for col in msd_all.T:
#     len_std.append(np.count_nonzero(~np.isnan(col)))

# std_msd = np.nanstd(msd_all, axis = 0)/np.sqrt(len_std)
# error = mean_msd-std_msd
# error_p = mean_msd+std_msd
# mean_msd = mean_msd[mean_msd>0]
# std_msd = std_msd[std_msd>0]
# popt , pcov = curve_fit(fit_MSD,np.arange(1,len(mean_msd)+1)[:6],np.array(mean_msd)[:6],p0=[0.02,0.4],maxfev=1000000)
# plt.plot(np.arange(len(mean_msd)),mean_msd)

# plt.show()

# plt.plot(np.arange(1,len(mean_msd)+1),mean_msd)
# plt.plot(np.arange(1,len(mean_msd)+1),fit_MSD(np.arange(len(mean_msd)),popt[0],popt[1]))
# plt.fill_between(np.arange(1,len(error)+1),error,error_p,alpha = 0.2)
# plt.yscale("log")
# plt.xscale("log")
# plt.xlabel("Tau")
# plt.ylabel("MSD")
# plt.show()



#radius of viable drops

# ll_da = ll_ez 
# rp_da = rp_ez
# def drop(viable_rp):
#     total_radius = []
#     total_in_out_drop = []
#     all_drops = []
#     for key,value in viable_rp.Movie.items():
#         for k,v in viable_rp.Movie[key].Drop_Collection.items():
#             drop_x,drop_y,drop_radius = v
#             total_radius.append(drop_radius)
#             total_in_out_drop.append([len(viable_rp.Movie[key].Trajectory_Collection[k].IN_Trajectory_Collection),len(viable_rp.Movie[key].Trajectory_Collection[k].IO_Trajectory_Collection)])
#         for k,v in viable_rp.Movie[key].All_Drop_Collection.items():
#             drop_x,drop_y,drop_radius = v
#             all_drops.append(drop_radius)
#     return [total_radius,total_in_out_drop,all_drops]

# viable_rp_data = drop(rp_ez)
# viable_ll_data = drop(ll_ez)


# plt.hist(np.array(viable_rp_data[2])*0.13,alpha = 0.2,label = 'RNAP')
# plt.hist(np.array(viable_ll_data[2])*0.13,alpha = 0.2,label = 'LacI')

# plt.legend()
# plt.xlabel("Diameter of Condensate (um)")
# plt.ylabel("Counts")
# plt.show()







# ll_ez_17_2= run_analysis("DATA/17/ll_ez_2","ll_ez")
# ll_ez_17_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_17_2.run_flow()
# total_ll_ez_17_2 = np.array(list(ll_ez_17_2.i_d_tavg) + list(ll_ez_17_2.io_d_tavg) + list(ll_ez_17_2.o_d_tavg))



# ############### LACI,LACO M9

# ll_m9 = run_analysis("DATA/20200215/ll_m9","ll_m9")
# ll_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9.run_flow()
# total_ll_m9 = np.array(list(ll_m9.i_d_tavg) + list(ll_m9.io_d_tavg) + list(ll_m9.o_d_tavg))



# ll_m9= run_analysis("DATA/new_days/20190527/ll_m9","laco_laci_m9")
# ll_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9.run_flow()
# total_ll_m9 = np.array(list(ll_m9.i_d_tavg) + list(ll_m9.io_d_tavg) + list(ll_m9.o_d_tavg))


# ll_m9n= run_analysis("DATA/laco_m9","laco_m9")
# ll_m9n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9n.run_flow()
# total_ll_m9n = np.array(list(ll_m9n.i_d_tavg) + list(ll_m9n.io_d_tavg) + list(ll_m9n.o_d_tavg))


# ll_m9_24 = run_analysis("DATA/new_days/20190524/laco_laci_M9","laco_laci_M9")
# ll_m9_24.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_24.run_flow()
# total_ll_m9_24 = np.array(list(ll_m9_24.i_d_tavg) + list(ll_m9_24.io_d_tavg) + list(ll_m9_24.o_d_tavg))



# lI_4 = run_analysis("DATA/ll_ez_no_laco/","ll_m9")
# lI_4.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l =4)
# lI_4.run_flow()
# total_lI_4 = np.array(list(lI_4.i_d_tavg) + list(lI_4.io_d_tavg) + list(lI_4.o_d_tavg))

# lI = run_analysis("DATA/ll_ez_no_laco/","ll_m9")
# lI.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# lI.run_flow()
# total_lI = np.array(list(lI.i_d_tavg) + list(lI.io_d_tavg) + list(lI.o_d_tavg))


# ll_m9_16 = run_analysis("DATA/16/ll_m9","ll_m9")
# ll_m9_16.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_16.run_flow()
# total_ll_m9_16 = np.array(list(ll_m9_16.i_d_tavg) + list(ll_m9_16.io_d_tavg) + list(ll_m9_16.o_d_tavg))

# ll_m9_16_2 = run_analysis("DATA/16/ll_m9_2","ll_m9")
# ll_m9_16_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_16_2.run_flow()
# total_ll_m9_16_2 = np.array(list(ll_m9_16_2.i_d_tavg) + list(ll_m9_16_2.io_d_tavg) + list(ll_m9_16_2.o_d_tavg))

# ll_m9_16_3 = run_analysis("DATA/16/ll_m9_3","ll_m9")
# ll_m9_16_3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_16_3.run_flow()
# total_ll_m9_16_3 = np.array(list(ll_m9_16_3.i_d_tavg) + list(ll_m9_16_3.io_d_tavg) + list(ll_m9_16_3.o_d_tavg))

# ll_m9_15 = run_analysis("DATA/15/ll_m9","ll_m9")
# ll_m9_15.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_15.run_flow()
# total_ll_m9_15 = np.array(list(ll_m9_15.i_d_tavg) + list(ll_m9_15.io_d_tavg) + list(ll_m9_15.o_d_tavg))

# ll_m9_15_2 = run_analysis("DATA/15/ll_m9_2","ll_m9")
# ll_m9_15_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_15_2.run_flow()
# total_ll_m9_15_2 = np.array(list(ll_m9_15_2.i_d_tavg) + list(ll_m9_15_2.io_d_tavg) + list(ll_m9_15_2.o_d_tavg))

# ll_m9_15_3 = run_analysis("DATA/15/ll_m9_3","ll_m9")
# ll_m9_15_3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_15_3.run_flow()
# total_ll_m9_15_3 = np.array(list(ll_m9_15_3.i_d_tavg) + list(ll_m9_15_3.io_d_tavg) + list(ll_m9_15_3.o_d_tavg))



# ############### LACI,LACO Hex 5%

# ll_ez_hex5 = run_analysis("DATA/20200210/ll_ez_hex5","ll_ez_hex5")
# ll_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5.run_flow()
# total_ll_ez_hex5 = np.array(list(ll_ez_hex5.i_d_tavg) + list(ll_ez_hex5.io_d_tavg) + list(ll_ez_hex5.o_d_tavg))


# ll_ez_hex5l = run_analysis("DATA/20200210/ll_ez_hex5","ll_ez_hex5")
# ll_ez_hex5l.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 1)
# ll_ez_hex5l.run_flow()
# total_ll_ez_hex5l = np.array(list(ll_ez_hex5l.i_d_tavg) + list(ll_ez_hex5l.io_d_tavg) + list(ll_ez_hex5l.o_d_tavg))
# total_msdl = ll_ez_hex5l.in_msd_track + ll_ez_hex5l.io_msd_track + ll_ez_hex5l.out_msd_track
# total_lengthl = ll_ez_hex5l.in_length + ll_ez_hex5l.out_length + ll_ez_hex5l.inout_length

# ############### LACI,LACO Hex 3%

# ll_ez_h3= run_analysis("DATA/new_days/20190527/ll_ez_hex_3","laco_laci_ez__hex_3")
# ll_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_h3.run_flow()
# total_ll_ez_h3 = np.array(list(ll_ez_h3.i_d_tavg) + list(ll_ez_h3.io_d_tavg) + list(ll_ez_h3.o_d_tavg))

# ll_ez_hex5_16 = run_analysis("DATA/16/ll_hex5","ll_ez_hex5")
# ll_ez_hex5_16.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5_16.run_flow()
# total_ll_ez_hex5_16 = np.array(list(ll_ez_hex5_16.i_d_tavg) + list(ll_ez_hex5_16.io_d_tavg) + list(ll_ez_hex5_16.o_d_tavg))

# ll_ez_hex5_16_2 = run_analysis("DATA/16/ll_hex5_2","ll_hex5")
# ll_ez_hex5_16_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5_16_2.run_flow()
# total_ll_ez_hex5_16_2 = np.array(list(ll_ez_hex5_16_2.i_d_tavg) + list(ll_ez_hex5_16_2.io_d_tavg) + list(ll_ez_hex5_16_2.o_d_tavg))

# ll_ez_hex5_16_3 = run_analysis("DATA/16/ll_hex5_3","ll_ez_hex5")
# ll_ez_hex5_16_3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5_16_3.run_flow()
# total_ll_ez_hex5_16_3 = np.array(list(ll_ez_hex5_16_3.i_d_tavg) + list(ll_ez_hex5_16_3.io_d_tavg) + list(ll_ez_hex5_16_3.o_d_tavg))

# ll_ez_hex5_17 = run_analysis("DATA/17/ll_hex5","ll_hex5")
# ll_ez_hex5_17.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5_17.run_flow()
# total_ll_ez_hex5_17= np.array(list(ll_ez_hex5_17.i_d_tavg) + list(ll_ez_hex5_17.io_d_tavg) + list(ll_ez_hex5_17.o_d_tavg))





# #######################################################     Nusa EZ

# n= run_analysis("DATA/newer_NUSA","NUSA")
# n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n.run_flow()
# total_n = np.array(list(n.i_d_tavg) + list(n.io_d_tavg) + list(n.o_d_tavg))

# n1= run_analysis("DATA/New_NUSA","NUSA")
# n1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n1.run_flow()
# total_n1 = np.array(list(n1.i_d_tavg) + list(n1.io_d_tavg) + list(n1.o_d_tavg))


# n2= run_analysis("DATA/Nusa_20190304","NUSA")
# n2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n2.run_flow()
# total_n2 = np.array(list(n2.i_d_tavg) + list(n2.io_d_tavg) + list(n2.o_d_tavg))


# a= run_analysis("DATA/Nusa_20190305","NUSA")
# a.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# a.run_flow()
# total_a = np.array(list(a.i_d_tavg) + list(a.io_d_tavg) + list(a.o_d_tavg))



# ######## NUSA Hex 5%

# nh= run_analysis("DATA/nusa_ez_hex_5","nusa_ez_hex_5")
# nh.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nh.run_flow()
# total_nh = np.array(list(nh.i_d_tavg) + list(nh.io_d_tavg) + list(nh.o_d_tavg))


# nusa_ez_hex5_binned = run_analysis("DATA/20200210/nusa_ez_hex5_binned","nusa_ez_hex5")
# nusa_ez_hex5_binned.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_binned.run_flow()
# total_nusa_ez_hex5_binned = np.array(list(nusa_ez_hex5_binned.i_d_tavg) + list(nusa_ez_hex5_binned.io_d_tavg) + list(nusa_ez_hex5_binned.o_d_tavg))


# nusa_ez_hex5_15 = run_analysis("DATA/15/nusa_ez_hex5","nusa_ez_hex5")
# nusa_ez_hex5_15.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_15.run_flow()
# total_nusa_ez_hex5_15 = np.array(list(nusa_ez_hex5_15.i_d_tavg) + list(nusa_ez_hex5_15.io_d_tavg) + list(nusa_ez_hex5_15.o_d_tavg))

# nusa_ez_hex5_15_2 = run_analysis("DATA/15/nusa_ez_hex5_2","nusa_ez_hex5")
# nusa_ez_hex5_15_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_15_2.run_flow()
# total_nusa_ez_hex5_15_2 = np.array(list(nusa_ez_hex5_15_2.i_d_tavg) + list(nusa_ez_hex5_15_2.io_d_tavg) + list(nusa_ez_hex5_15_2.o_d_tavg))

# nusa_ez_hex5_15_3 = run_analysis("DATA/15/nusa_ez_hex5_3","nusa_ez_hex5")
# nusa_ez_hex5_15_3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_15_3.run_flow()
# total_nusa_ez_hex5_15_3 = np.array(list(nusa_ez_hex5_15_3.i_d_tavg) + list(nusa_ez_hex5_15_3.io_d_tavg) + list(nusa_ez_hex5_15_3.o_d_tavg))


# nusa_ez_hex5_16 = run_analysis("DATA/16/nusa_ez_hex5","nusa_ez_hex5")
# nusa_ez_hex5_16.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_16.run_flow()
# total_nusa_ez_hex5_16 = np.array(list(nusa_ez_hex5_16.i_d_tavg) + list(nusa_ez_hex5_16.io_d_tavg) + list(nusa_ez_hex5_16.o_d_tavg))


# nusa_ez_hex5_16_2 = run_analysis("DATA/16/nusa_ez_hex5_2","nusa_ez_hex5")
# nusa_ez_hex5_16_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_16_2.run_flow()
# total_nusa_ez_hex5_16_2 = np.array(list(nusa_ez_hex5_16_2.i_d_tavg) + list(nusa_ez_hex5_16_2.io_d_tavg) + list(nusa_ez_hex5_16_2.o_d_tavg))




######## NUSA Hex 3%




######### NUSA M9 


# na_m9 = run_analysis("DATA/20191216/nusa_m9","nusa_m9")
# na_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# na_m9.run_flow()
# total_na_m9 = np.array(list(na_m9.i_d_tavg) + list(na_m9.io_d_tavg) + list(na_m9.o_d_tavg))


# na_m9_2 = run_analysis("DATA/20191218/nusa_m9","nusa_m9")
# na_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# na_m9_2.run_flow()
# total_na_m9_2 = np.array(list(na_m9_2.i_d_tavg) + list(na_m9_2.io_d_tavg) + list(na_m9_2.o_d_tavg))

# nusa_m9_2 = run_analysis("DATA/20200212/nusa_m9_2","nusa_m9")
# nusa_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_m9_2.run_flow()
# total_nusa_m9_2 = np.array(list(nusa_m9_2.i_d_tavg) + list(nusa_m9_2.io_d_tavg) + list(nusa_m9_2.o_d_tavg))


# na_m9_12 = run_analysis("DATA/12/nusa_m9","nusa_m9")
# na_m9_12.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# na_m9_12.run_flow()
# total_na_m9_12 = np.array(list(na_m9_12.i_d_tavg) + list(na_m9_12.io_d_tavg) + list(na_m9_12.o_d_tavg))

# na_m9_12_2 = run_analysis("DATA/12/nusa_m9_2","nusa_m9")
# na_m9_12_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# na_m9_12_2.run_flow()
# total_na_m9_12_2 = np.array(list(na_m9_12_2.i_d_tavg) + list(na_m9_12_2.io_d_tavg) + list(na_m9_12_2.o_d_tavg))







# np.savetxt("rp_ez_i.txt",X = con_pix_si(np.log10(rp_ez.i_d_tavg),which = "msd"))
# np.savetxt("rp_ez_o.txt",X = con_pix_si(np.log10(rp_ez.o_d_tavg),which = "msd"))
# np.savetxt("rp_ez_io.txt",X = con_pix_si(np.log10(rp_ez.io_d_tavg),which = "msd"))
# np.savetxt("total_ll.txt",X = con_pix_si(np.log10(total_ll),which = "msd"))








# which_TP = ll
# comp_TP = ll

# def hist_cum_sum(which_TP,comp_TP):
#   plt.hist(con_pix_si(np.log10(which_TP.i_d_tavg)),weights = norm_weights(which_TP.i_d_tavg),alpha = 0.2, label = 'IN')
#   plt.hist(con_pix_si(np.log10(which_TP.o_d_tavg)),weights = norm_weights(which_TP.o_d_tavg),alpha = 0.2, label = 'OUT')
#   plt.hist(con_pix_si(np.log10(which_TP.io_d_tavg)),weights = norm_weights(which_TP.io_d_tavg),alpha = 0.2, label = 'IN/OUT')
#   #plt.hist(con_pix_si(np.log10(total_comp_TP)),weights = norm_weights(total_comp_TP),alpha = 0.2, label = '{0}'.format(comp_TP))
#   #plt.hist(con_pix_si(np.log10(total_rp_ez)),weights = norm_weights(total_rp_ez),alpha = 0.2, label = 'Total')
#   plt.legend()
#   plt.ylabel("Counts")
#   plt.xlabel("log10(dapp)")
#   plt.show()




#   which_TP_i = cum_sum(con_pix_si(which_TP.in_dist,which = 'um'))
#   which_TP_io = cum_sum(con_pix_si(which_TP.io_dist,which = 'um'))
#   which_TP_ot = cum_sum(con_pix_si(which_TP.ot_dist,which = 'um'))
#   #comp_TP_all = cum_sum(con_pix_si(total_comp_TP,which = 'um'),binz = 200)
#   plt.plot(which_TP_i[1][1:],which_TP_i[0],label = "IN")
#   plt.plot(which_TP_io[1][1:],which_TP_io[0],label = "IN/OUT")
#   plt.plot(which_TP_ot[1][1:],which_TP_ot[0],label = "OUT")
#   #plt.plot(comp_TP_all[1][1:],comp_TP_all[0],label = '{0}'.format(comp_TP))
#   plt.xlabel("Distances (um)")
#   plt.ylabel("Cumsum")
#   plt.legend()
#   plt.show()

#   return

# #######
# #alpha values using ensemble averaged track data.


# which_TP = rp_ez
# which_ll = ll
# def alpha_ens(which_TP,which_ll):
#   which_TP_i_a = MSD_a_value_all_ens(which_TP.in_tracksf)
#   which_TP_io_a = MSD_a_value_all_ens(which_TP.io_tracksf)
#   which_TP_o_a = MSD_a_value_all_ens(which_TP.out_tracksf)

#   which_ll_all = MSD_a_value_all_ens(which_ll.all_tracksf,threshold = 6)
#   which_ll_in = MSD_a_value_all_ens(which_ll.in_tracksf,threshold = 6)
#   which_ll_io = MSD_a_value_all_ens(which_ll.io_tracksf,threshold = 3)
#   which_ll_ot = MSD_a_value_all_ens(which_ll.out_tracksf,threshold = 3)
#   np.savetxt("rp_ez_i_a.txt",X = con_pix_si(which_TP_i_a[1],which = "msd"))
#   np.savetxt("rp_ez_io_a.txt",X = con_pix_si(which_TP_io_a[1],which = "msd"))
#   np.savetxt("rp_ez_o_a.txt",X = con_pix_si(which_TP_o_a[1],which = "msd"))
#   np.savetxt("ll_ez_all.txt",X = con_pix_si(which_ll_all[1],which = "msd"))

#   plt.show()

#   plt.plot(np.arange(20*1,20*15),fit_MSD(np.arange(20*1,20*15),1.0,1.0),'k--')
#   plt.plot(np.arange(20*1,20*15),fit_MSD(np.arange(20*1,20*15),0.01,0.0),'k--')
#   plt.plot(20*np.array(which_TP_i_a[2]),con_pix_si(which_TP_i_a[1],which = "msd"),label = "Alpha IN = {0:.3g}".format(which_TP_i_a[0][0]),color = 'blue')
#   plt.plot(20*np.array(which_TP_io_a[2]),con_pix_si(which_TP_io_a[1],which = "msd"),label = "Alpha IN/OUT = {0:.3g}".format(which_TP_io_a[0][0]),color = 'orange')
#   plt.plot(20*np.array(which_TP_o_a[2]),con_pix_si(which_TP_o_a[1],which = "msd"),label = "Alpha OUT = {0:.3g}".format(which_TP_o_a[0][0]),color = 'green')
#   plt.plot(20*np.array(which_ll_all[2]),con_pix_si(which_ll_all[1],which = "msd"),label = "Alpha LL_ALL = {0:.3g}".format(which_ll_all[0][0]),color = 'pink')
#   #plt.plot(20*np.array(which_ll_in[2]),con_pix_si(which_ll_in[1],which = "msd"),label = "Alpha LL_in = {0:.3g}".format(which_ll_in[0][0]))
#   #plt.plot(20*np.array(which_ll_io[2]),con_pix_si(which_ll_io[1],which = "msd"),label = "Alpha LL_io = {0:.3g}".format(which_ll_io[0][0]))
#   #plt.plot(20*np.array(which_ll_ot[2]),con_pix_si(which_ll_ot[1],which = "msd"),label = "Alpha LL_out = {0:.3g}".format(which_ll_ot[0][0]))
#   #plt.yscale("log")
#   plt.ylabel("MSD Ensemble (um/s^2)")
#   plt.xlabel("Tau (ms)")
#   plt.yscale("log")
#   plt.xscale("log")
#   plt.legend(loc = 1)
#   plt.show()
#   return 




















#adapted from https://scipy-cookbook.readthedocs.io/items/Matplotlib_MulticoloredLine.html

########
#plot a trjectory [x,y] as a colored line to show passage of time

# x,y = np.array(rp_ez.in_tracksf[0][0])
# x = x[:]
# y = y[:]
# z = np.arange(0,len(x))  # first derivative

# x_a,y_a = np.array(rp_ez.out_tracksf[0][0])
# x_a = x_a[:]
# y_a = y_a[:]
# z_a = np.arange(0,len(x_a))  # first derivative


# norm = mpl.colors.Normalize(vmin=0, vmax=len(x_a))
# cb1 = mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm")

# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)


# lc = LineCollection(segments, cmap="coolwarm", norm = norm)
# lc.set_array(z)
# lc.set_linewidth(3)

# points_a = np.array([x_a, y_a]).T.reshape(-1, 1, 2)
# segments_a = np.concatenate([points_a[:-1], points_a[1:]], axis=1)

# lc_a = LineCollection(segments_a, cmap="coolwarm", norm = norm)
# lc_a.set_array(z_a)
# lc_a.set_linewidth(3)

# plt.imshow(plt.imread(rp_ez.segmented_drop_files[0][0]),cmap = "gray")
# #plt.gca().add_collection(lc)
# #plt.gca().add_collection(lc_a)
# #plt.xlim(x.min(), x.max())
# #plt.ylim(y.min(), y.max())

# #plt.colorbar(cb1, label = "Time (ms)")

# plt.savefig("figure.eps")
# plt.show()
def draw_item():
    fig, ax = plt.subplots()
    img = mpimg.imread(rp_ez.Movie['0'].Movie_location[3])
    ax.imshow(img)
    Drawing_uncolored_circle = plt.Circle( (rp_ez.Movie['0'].Cells['0'].Drop_Collection['4,1'][0],rp_ez.Movie['0'].Cells['0'].Drop_Collection['4,1'][1]),
                                        rp_ez.Movie['0'].Cells['0'].Drop_Collection['4,1'][2] ,
                                        fill = False )
    #ax.add_artist(Drawing_uncolored_circle)
    x = []
    y = []
    for i in range(len(rp_ez.Movie['0'].Cells['0'].sorted_tracks_frame[0][3])):
        x+=rp_ez.Movie['0'].Cells['0'].sorted_tracks_frame[1][3][i]
        y+=rp_ez.Movie['0'].Cells['0'].sorted_tracks_frame[2][3][i]
    for i in range(len(rp_ez.Movie['0'].Cells['1'].sorted_tracks_frame[0][3])):
        x+=rp_ez.Movie['0'].Cells['1'].sorted_tracks_frame[1][3][i]
        y+=rp_ez.Movie['0'].Cells['1'].sorted_tracks_frame[2][3][i]
    for i,j in rp_ez.Movie['0'].Cells['0'].Drop_Collection.items():
        if i[0] == '4':
            Drawing_uncolored_circle = plt.Circle( (j[0],j[1]),
                                        j[2] ,
                                        fill = False )
            ax.add_artist(Drawing_uncolored_circle)
    for i,j in rp_ez.Movie['0'].Cells['1'].Drop_Collection.items():
        if i[0] == '4':
            Drawing_uncolored_circle = plt.Circle( (j[0],j[1]),
                                        j[2] ,
                                        fill = False )
            ax.add_artist(Drawing_uncolored_circle)
    return [x,y]
x,y=draw_item()
x_y = np.array([[a,b] for a,b in zip(x,y)])
clustering = OPTICS(min_samples=20).fit(x_y)
#for i,j in rp_ez.Movie['0'].Cells['0'].Trajectory_Collection['4,1'].IN_Trajectory_Collection.items():
#   plt.plot(j.X,j.Y)
plt.scatter(x,y,s= 2,c = clustering.labels_)
plt.colorbar()
plt.show()
    
sys.exit()







#distribution of viable drop radius
drop_radius = []
which = rp_ez
for k,kk in which.Movie.items():
  for i,ii in kk.Cells.items():
    for l,ll in ii.Drop_Collection.items():
      drop_radius.append(ll[2])



which = rp_ez
y_collection = []
x_collection = []
in_msd = []
io_msd = []
radius_col = []

#cm of track to boundary vs diff
cm_boundary = []
cm_diff = []
track_recidency_in_drop = []
cm_error = []
end_to_end = []
radius_gyration = []

IO_inside_start = []
IO_inside_start_dist = []
IO_outside_start = []
IO_outside_start_dist = []
#number of tracks that start inside and end inside
tracks_in_in = 0
len_ii = []
tracks_out_out = 0
len_oo = []
tracks_in_out = 0
len_io = []
tracks_out_in = 0
len_oi = []
directional_displacement = []
dist_center = []
long_axis_angle = []

#take notice of tracks which have displacements away from the condensate (in/out only) of >0.2 um
track_xy = []
track_drop = []
track_movie = []
track_cell = []
track_cell_e1_e2 = []
displacement_aligned_long = []
track_drop_loc = []
track_id = []
for k,v in which.Movie.items():
   for o,oo in which.Movie[k].Cells.items():
      for kk,vv in which.Movie[k].Cells[o].Trajectory_Collection.items():
          
          for kkk,vvv in which.Movie[k].Cells[o].Trajectory_Collection[kk].IN_Trajectory_Collection.items():
              track = which.Movie[k].Cells[o].Trajectory_Collection[kk].IN_Trajectory_Collection[kkk]
              x_val = track.X
              y_val = track.Y
              drop_data = which.Movie[k].Cells[o].Drop_Collection[track.Drop_Identifier]

              diff_dist_temp = con_pix_si(dif_dis(x_val,y_val),which = 'um')
              drop_center_dist = dist(x_val,y_val,drop_data[0],drop_data[1]) - drop_data[2]


              #direction of the trajectory
              #r2 -r1 > 0 moving out, r2 - r1 < 0 moving in
              directional = con_pix_si(np.diff(dist(x_val,y_val,drop_data[0],drop_data[1])),which = 'um')
              directional_displacement+=list(directional)
              dist_center+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))

              radius_col.append(drop_data[2])
              y_collection+=list(diff_dist_temp)
              x_collection+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))

              in_msd.append(track.MSD_total_um)

              #center of mass of track relative to boundary vs. diffusion of track
              cm = cm_normal(x_val,y_val)
              cm_dist_boundary = dist(cm[0],cm[1],drop_data[0],drop_data[1]) - drop_data[2]
              cm_boundary.append(con_pix_si(cm_dist_boundary,which = 'um'))
              cm_diff.append(track.MSD_total_um)
              cm_error.append(np.sqrt(np.std(x_val)**2 + np.std(y_val)**2)/np.sqrt(len(x_val)))
              track_recidency_in_drop.append(np.sum(drop_center_dist<0.0)/len(x_val))
              #end ot end distance of trajectory:
              end_to_end.append(end_distance(x_val,y_val))

              #radius of gyration
              radius_gyration.append(radius_of_gyration(x_val,y_val))



              #how aligned is the displacement vector for each displacement to each axis of the cell. 
              #differences in x,y
              dif_x = np.diff(x_val)
              dif_y = np.diff(y_val)
              long_axis_vec = which.Movie[k].Cells[o].cell_long_axis
              angle_xy = []
              for i in range(len(dif_x)):
                  termer = np.arccos(np.dot(long_axis_vec.T[0],[dif_x[i],dif_y[i]])/(np.linalg.norm(long_axis_vec.T[0])*np.linalg.norm([dif_x[i],dif_y[i]])))*180/np.pi
                  angle_xy.append(termer)
              long_axis_angle+=angle_xy


          for kkk,vvv in which.Movie[k].Cells[o].Trajectory_Collection[kk].IO_Trajectory_Collection.items():
              track = which.Movie[k].Cells[o].Trajectory_Collection[kk].IO_Trajectory_Collection[kkk]
              x_val = track.X
              y_val = track.Y
              drop_data = which.Movie[k].Cells[o].Drop_Collection[track.Drop_Identifier]
                  
              diff_dist_temp = con_pix_si(dif_dis(x_val,y_val),which = 'um')
              drop_center_dist = dist(x_val,y_val,drop_data[0],drop_data[1]) - drop_data[2]

              #direction of the trajectory
              #r2 -r1 > 0 moving out, r2 - r1 < 0 moving in
              directional = con_pix_si(np.diff(dist(x_val,y_val,drop_data[0],drop_data[1])),which = 'um')
              directional_displacement+=list(directional)
              dist_center+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))

              radius_col.append(drop_data[2])
              y_collection+=list(diff_dist_temp)
              x_collection+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))
              io_msd.append(track.MSD_total_um)

              #center of mass of track relative to boundary vs. diffusion of track
              cm = cm_normal(x_val,y_val)
              cm_dist_boundary = dist(cm[0],cm[1],drop_data[0],drop_data[1]) - drop_data[2]
              cm_boundary.append(con_pix_si(cm_dist_boundary,which = 'um'))
              cm_diff.append(track.MSD_total_um)
              cm_error.append(np.sqrt(np.std(x_val)**2 + np.std(y_val)**2))
              track_recidency_in_drop.append(np.sum(drop_center_dist<0.0)/len(x_val))
              #end ot end distance of trajectory:
              end_to_end.append(end_distance(x_val,y_val))

              #radius of gyration
              radius_gyration.append(radius_of_gyration(x_val,y_val))

              #how aligned is the displacement vector for each displacement to each axis of the cell. 
              #differences in x,y
              dif_x = np.diff(x_val)
              dif_y = np.diff(y_val)
              long_axis_vec = which.Movie[k].Cells[o].cell_long_axis
              angle_xy = []
              for i in range(len(dif_x)):
                  termer = np.arccos(np.dot(long_axis_vec.T[0],[dif_x[i],dif_y[i]])/(np.linalg.norm(long_axis_vec.T[0])*np.linalg.norm([dif_x[i],dif_y[i]])))*180/np.pi
                  angle_xy.append(termer)
              long_axis_angle+=angle_xy
              #check the tracks which have displacements way outside the condensate and ask how are they oriented relative to the cell axis and where the condensate is
              if np.sum(np.array(con_pix_si(drop_center_dist[:-1], which = 'um'))>-0.4) != 0:
                  track_xy.append([x_val,y_val])
                  track_drop.append(which.Movie[k].Cells[o].Drop_Collection[track.Drop_Identifier])
                  track_drop_loc.append(track.Drop_Identifier)
                  track_id.append(kkk)
                  track_movie.append(k)
                  track_cell.append(o)
                  track_cell_e1_e2.append([which.Movie[k].Cells[o].cell_long_axis,which.Movie[k].Cells[o].cell_short_axis])
                  #how aligned is the displacement vector for each displacement to each axis of the cell. 
                  #differences in x,y
                  dif_x = np.diff(x_val)
                  dif_y = np.diff(y_val)
                  long_axis_vec = which.Movie[k].Cells[o].cell_long_axis
                  angle_xy = []
                  for i in range(len(dif_x)):
                      termer = np.arccos(np.dot(long_axis_vec.T[0],[dif_x[i],dif_y[i]])/(np.linalg.norm(long_axis_vec.T[0])*np.linalg.norm([dif_x[i],dif_y[i]])))*180/np.pi
                      angle_xy.append(termer)



              #for IO trajectories that start in the inside of condensates how do they behave?
              distances_center = dist(x_val,y_val,drop_data[0],drop_data[1]) 
              index_radius = distances_center<drop_data[2]
              # index_index = 0
              # for i in range(len(index_radius)):
              #     if i==0:
              #         index_index = index_radius[i]
              #     else:


              if (index_radius[0] == True) and (index_radius[-1] == True):
                  IO_inside_start+=list(diff_dist_temp)
                  IO_inside_start_dist+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))
                  tracks_in_in+=1
                  len_ii.append(len(index_radius))
              elif (index_radius[0] == False) and (index_radius[-1] == False):
                  IO_outside_start+=list(diff_dist_temp)
                  IO_outside_start_dist+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))
                  tracks_out_out+=1
                  len_oo.append(len(index_radius))
              if (index_radius[0] == False) and (index_radius[-1] == True):
                  tracks_out_in+=1
                  len_oi.append(len(index_radius))
              if (index_radius[0] == True) and (index_radius[-1] == False):
                  tracks_in_out+=1
                  len_io.append(len(index_radius))

          # for kkk,vvv in which.Movie[k].Cells[o].Trajectory_Collection[kk].OUT_Trajectory_Collection.items():
          #     track = which.Movie[k].Cells[o].Trajectory_Collection[kk].OUT_Trajectory_Collection[kkk]
          #     x_val = track.X
          #     y_val = track.Y
          #     drop_data = which.Movie[k].Cells[o].Drop_Collection[track.Drop_Identifier]
                  
          #     diff_dist_temp = con_pix_si(dif_dis(x_val,y_val),which = 'um')
          #     drop_center_dist = dist(x_val,y_val,drop_data[0],drop_data[1]) - drop_data[2]

          #     #direction of the trajectory
          #     #r2 -r1 > 0 moving out, r2 - r1 < 0 moving in
          #     directional = con_pix_si(np.diff(dist(x_val,y_val,drop_data[0],drop_data[1])),which = 'um')
          #     directional_displacement+=list(directional)
          #     dist_center+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))

          #     radius_col.append(drop_data[2])
          #     y_collection+=list(diff_dist_temp)
          #     x_collection+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))
          #     io_msd.append(track.MSD_total_um)

          #     #center of mass of track relative to boundary vs. diffusion of track
          #     cm = cm_normal(x_val,y_val)
          #     cm_dist_boundary = dist(cm[0],cm[1],drop_data[0],drop_data[1]) - drop_data[2]
          #     cm_boundary.append(con_pix_si(cm_dist_boundary,which = 'um'))
          #     cm_diff.append(track.MSD_total_um)
          #     cm_error.append(np.sqrt(np.std(x_val)**2 + np.std(y_val)**2))
          #     track_recidency_in_drop.append(np.sum(drop_center_dist<0.0)/len(x_val))
          #     #end ot end distance of trajectory:
          #     end_to_end.append(end_distance(x_val,y_val))

          #     #radius of gyration
          #     radius_gyration.append(radius_of_gyration(x_val,y_val))





#plotting radius of all drops vs viable ones
plt.hist(np.array(drop_radius)*0.13,alpha = 0.2,label = "Viable")
plt.hist(np.array(rp_ez.radius)[:,2]*0.13,alpha = 0.2,label = "All")
plt.ylabel("Counts")
plt.xlabel("Radius (um)")
plt.legend()
plt.show()
#plotting tracks on cells


#get the '2' movie:
movie_selc = '7'

ind_m = np.array(track_movie) == movie_selc
cells_m = np.array(track_cell)[ind_m]
drops_m = np.array(track_drop)[ind_m]
tracks_m = np.array(track_xy)[ind_m]
drop_loc_m = np.array(track_drop_loc)[ind_m]
track_idm = np.array(track_id)[ind_m]



cmap_all=plt.get_cmap('gray')


'''
for i in range(len(tracks_m)):
    img = mpimg.imread(which.Movie[movie_selc].Movie_location[int(drop_loc_m[i][0])])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pimg = ax.imshow(img,cmap=cmap_all)
    ax.plot(tracks_m[i][0],tracks_m[i][1],'-') 
    cir = Circle([drops_m[i][0],drops_m[i][1]], radius =drops_m[i][2], fill = False, color = 'red')
    ax.add_artist(cir)
    for k,l in which.Movie[movie_selc].Cells[cells_m[i]].Drop_Collection.items():

        print(drops_m[i])
        if k[0] == track_idm[i][0]:
            cir = Circle([l[0],l[1]], radius =l[2], fill = False, color = "black")
            ax.add_artist(cir)
    plt.xlim((50,120))
    plt.ylim((180,240))
    plt.show()
'''
for i in range(len(tracks_m)):
    if i == 0:
      img = mpimg.imread(which.Movie[movie_selc].Movie_location[int(drop_loc_m[i][0])])
      fig = plt.figure()
      ax = plt.axes(projection='3d')
      nx,ny = np.shape(img)
      x = range(nx)
      y = range(ny)
      X, Y = np.meshgrid(x, y)
      ax.plot_surface(X[50:120,190:240], Y[50:120,190:240], img[190:240,50:120].T, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
      cir = Circle([drops_m[i][1],drops_m[i][0]], radius =drops_m[i][2], fill = False, color = 'red')
      ax.add_patch(cir)
      art3d.pathpatch_2d_to_3d(cir, z=200)
      plt.show()
















#directional_displacement
x = np.array(dist_center)
y = np.array(directional_displacement)
xy = np.vstack([dist_center,directional_displacement])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

from scipy.stats import binned_statistic_2d
a,b,c,d = binned_statistic_2d(x,y,None,'count',bins = 50, expand_binnumbers = True)

weights2 = np.ones_like(y[(d[0]==5) | (d[0]==6) | (d[0]==7) | (d[0]==8)]) / (len(y[(d[0]==5) | (d[0]==6) | (d[0]==7) | (d[0]==8)]))
weights1 = np.ones_like(y[(d[0]==9) | (d[0]==10) | (d[0]==11) | (d[0]==12)]) / (len(y[(d[0]==9) | (d[0]==10) | (d[0]==11) | (d[0]==12)]))
plt.hist(y[(d[0]==9) | (d[0]==10) | (d[0]==11) | (d[0]==12)],alpha = 0.3,label = "Boundary",weights=weights1)
plt.hist(y[(d[0]==5) | (d[0]==6) | (d[0]==7) | (d[0]==8)],alpha = 0.3,label = "Droplet Phase",weights=weights2)
plt.xlabel("Directional Displacements (um)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
plt.hist(abs(y[(d[0]==9) | (d[0]==10) | (d[0]==11) | (d[0]==12)]),alpha = 0.3,label = "Boundary",weights=weights1)
plt.hist(abs(y[(d[0]==5) | (d[0]==6) | (d[0]==7) | (d[0]==8)]),alpha = 0.3,label = "Droplet Phase",weights=weights2)
plt.show()

weights1 = np.ones_like(y[(d[0]==2) | (d[0]==3) | (d[0]==4)]) / (len(y[(d[0]==2) | (d[0]==3) | (d[0]==4)]))
weights2 = np.ones_like(y[(d[0]==5) | (d[0]==6) | (d[0]==7)]) / (len(y[(d[0]==5) | (d[0]==6) | (d[0]==7)]))
plt.hist(y[(d[0]==2) | (d[0]==3) | (d[0]==4)],alpha = 0.3,label = "Boundary",weights=weights1)
plt.hist(y[(d[0]==5) | (d[0]==6) | (d[0]==7)],alpha = 0.3,label = "Droplet Phase",weights=weights2)
plt.xlabel("Directional Displacements (um)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
# plt.hist(abs(y[(d[0]==9) | (d[0]==10) | (d[0]==11) | (d[0]==12)]),alpha = 0.3,label = "Boundary",weights=weights1)
# plt.hist(abs(y[(d[0]==5) | (d[0]==6) | (d[0]==7) | (d[0]==8)]),alpha = 0.3,label = "Droplet Phase",weights=weights2)
# plt.show()


weights1 = np.ones_like(y[(d[0]==16) | (d[0]==17) | (d[0]==18)]) / (len(y[(d[0]==16) | (d[0]==17) | (d[0]==18)]))
weights2 = np.ones_like(y[(d[0]==19) | (d[0]==20) | (d[0]==21)]) / (len(y[(d[0]==19) | (d[0]==20) | (d[0]==21)]))
plt.hist(y[(d[0]==16) | (d[0]==17) | (d[0]==18)],alpha = 0.3,label = "Outside Boundary",weights=weights1)
plt.hist(y[(d[0]==19) | (d[0]==20) | (d[0]==21)],alpha = 0.3,label = "Non-Droplet/Boundary Phase",weights=weights2)
plt.xlabel("Directional Displacements (um)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
weights3 = np.ones_like(y[(d[0]==1)]) / (len(y[(d[0]==1)]))
plt.hist(y[(d[0]==1)],alpha = 1,label = "Center of Condensate",weights=weights3)
plt.xlabel("Directional Displacements (um)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()







n, _ = np.histogram(x,bins = 20)
sy, _ = np.histogram(x,bins = 20,weights = y)
sy2, _ = np.histogram(x,bins = 20,weights = y*y)
h, x_bins, y_bins = np.histogram2d(x,y,bins = 20)

mean = sy/n
std = np.sqrt(sy2/n - mean*mean)
plt.scatter(x,y,c = z, s = 50)
plt.plot((_[1:] + _[:-1])/2,mean, 'r-')
#plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 1)/(np.sum(np.sum(h.T,axis = 1))))
#plt.axvline(x=2.5*0.130,linestyle = 'dashed')
plt.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Displacements (um)")
#plt.ylabel("Dapp (um^2/s)")
# plt.ylim((-0.2,1.25))
# plt.xlim((-0.35,1))
plt.colorbar()
plt.show()


n, _ = np.histogram(x,bins = 20)
sy, _ = np.histogram(x,bins = 20,weights = y)
sy2, _ = np.histogram(x,bins = 20,weights = y*y)
h, x_bins, y_bins = np.histogram2d(x,y,bins = 20)

mean = sy/n
std = np.sqrt(sy2/n - mean*mean)
plt.scatter(x,y,c = long_axis_angle, s = 50)
plt.plot((_[1:] + _[:-1])/2,mean, 'r-')
#plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 1)/(np.sum(np.sum(h.T,axis = 1))))
#plt.axvline(x=2.5*0.130,linestyle = 'dashed')
plt.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Displacements (um)")
#plt.ylabel("Dapp (um^2/s)")
# plt.ylim((-0.2,1.25))
# plt.xlim((-0.35,1))
plt.colorbar()
plt.show()





#IO_Crossing inside
x = np.array(IO_inside_start_dist)
y = np.array(IO_inside_start)
xy = np.vstack([IO_inside_start_dist,IO_inside_start])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]



n, _ = np.histogram(x,bins = 20)
sy, _ = np.histogram(x,bins = 20,weights = y)
sy2, _ = np.histogram(x,bins = 20,weights = y*y)
h, x_bins, y_bins = np.histogram2d(x,y,bins = 20)

mean = sy/n
std = np.sqrt(sy2/n - mean*mean)
plt.scatter(x,y,c = z, s = 50)
plt.plot((_[1:] + _[:-1])/2,mean, 'r-')
#plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 1)/(np.sum(np.sum(h.T,axis = 1))))
#plt.axvline(x=2.5*0.130,linestyle = 'dashed')
plt.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Displacements (um)")
#plt.ylabel("Dapp (um^2/s)")
# plt.ylim((-0.2,1.25))
# plt.xlim((-0.35,1))
plt.colorbar()
plt.show()


#IO_Crossing outside
x = np.array(IO_outside_start_dist)
y = np.array(IO_outside_start)
xy = np.vstack([IO_outside_start_dist,IO_outside_start])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]



n, _ = np.histogram(x,bins = 20)
sy, _ = np.histogram(x,bins = 20,weights = y)
sy2, _ = np.histogram(x,bins = 20,weights = y*y)
h, x_bins, y_bins = np.histogram2d(x,y,bins = 20)

mean = sy/n
std = np.sqrt(sy2/n - mean*mean)
plt.scatter(x,y,c = z, s = 50)
plt.plot((_[1:] + _[:-1])/2,mean, 'r-')
#plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 1)/(np.sum(np.sum(h.T,axis = 1))))
#plt.axvline(x=2.5*0.130,linestyle = 'dashed')
plt.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Displacements (um)")
#plt.ylabel("Dapp (um^2/s)")
# plt.ylim((-0.2,1.25))
# plt.xlim((-0.35,1))
plt.colorbar()
plt.show()










which= rp_ez
#cm of track to boundary vs diff
cm_boundaryl = []
cm_diffl = []
track_recidency_in_dropl = []
cm_errorl = []
end_to_endl = []
radius_gyrationl = []
for k,v in which.Movie.items():
  for o,oo in which.Movie[k].Cells.items():
    for kk,vv in which.Movie[k].Cells[o].Trajectory_Collection.items():
        
        for kkk,vvv in which.Movie[k].Cells[o].Trajectory_Collection[kk].IN_Trajectory_Collection.items():
            track = which.Movie[k].Cells[o].Trajectory_Collection[kk].IN_Trajectory_Collection[kkk]
            x_val = track.X
            y_val = track.Y
            drop_data = which.Movie[k].Cells[o].Drop_Collection[track.Drop_Identifier]

            diff_dist_temp = con_pix_si(dif_dis(x_val,y_val),which = 'um')
            drop_center_dist = dist(x_val,y_val,drop_data[0],drop_data[1]) - drop_data[2]
            radius_col.append(drop_data[2])
            #y_collection+=list(diff_dist_temp)
            #x_collection+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))
            #in_msd.append(track.MSD_total_um)

            #center of mass of track relative to boundary vs. diffusion of track
            cm = cm_normal(x_val,y_val)
            cm_dist_boundary = dist(cm[0],cm[1],drop_data[0],drop_data[1]) - drop_data[2]
            cm_boundaryl.append(con_pix_si(cm_dist_boundary,which = 'um'))
            cm_diffl.append(track.MSD_total_um)
            cm_errorl.append(np.sqrt(np.std(x_val)**2 + np.std(y_val)**2)/np.sqrt(len(x_val)))
            track_recidency_in_dropl.append(np.sum(drop_center_dist<0.0)/len(x_val))

  
            #end ot end distance of trajectory:
            end_to_endl.append(end_distance(x_val,y_val))

            #radius of gyration
            radius_gyrationl.append(radius_of_gyration(x_val,y_val))
        for kkk,vvv in which.Movie[k].Cells[o].Trajectory_Collection[kk].IO_Trajectory_Collection.items():
            track = which.Movie[k].Cells[o].Trajectory_Collection[kk].IO_Trajectory_Collection[kkk]
            x_val = track.X
            y_val = track.Y
            drop_data = which.Movie[k].Cells[o].Drop_Collection[track.Drop_Identifier]
                
            diff_dist_temp = con_pix_si(dif_dis(x_val,y_val),which = 'um')
            drop_center_dist = dist(x_val,y_val,drop_data[0],drop_data[1]) - drop_data[2]
            radius_col.append(drop_data[2])
            # y_collection+=list(diff_dist_temp)
            # x_collection+=list(con_pix_si(drop_center_dist[:-1], which = 'um'))
            # io_msd.append(track.MSD_total_um)

            #center of mass of track relative to boundary vs. diffusion of track
            cm = cm_normal(x_val,y_val)
            cm_dist_boundary = dist(cm[0],cm[1],drop_data[0],drop_data[1]) - drop_data[2]
            cm_boundaryl.append(con_pix_si(cm_dist_boundary,which = 'um'))
            cm_diffl.append(track.MSD_total_um)
            cm_errorl.append(np.sqrt(np.std(x_val)**2 + np.std(y_val)**2))
            track_recidency_in_dropl.append(np.sum(drop_center_dist<0.0)/len(x_val))


            #end ot end distance of trajectory:
            end_to_endl.append(end_distance(x_val,y_val))
            #radius of gyration
            radius_gyrationl.append(radius_of_gyration(x_val,y_val))
plt.scatter(cm_boundary,np.log10(cm_diff),c= track_recidency_in_drop,cmap = 'Greens')
#plt.scatter(cm_boundaryl,np.log10(cm_diffl),alpha = 0.3)
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel('Dapp log10(um^2/s)')
plt.colorbar()
plt.show()


h, x_bins, y_bins = np.histogram2d(cm_boundary,np.log10(cm_diff),bins = 20)
plt.plot((x_bins[1:] + x_bins[:-1])/2,np.sum(h.T,axis = 0)/(np.sum(np.sum(h.T,axis = 0))))
plt.ylabel("Normalized Density of Trajectories")
plt.xlabel("Distance of Localization to Boundary (um)")
# plt.ylim((0,1))
# plt.xlim((-0.35,1))
plt.show()


plt.scatter(cm_boundary,np.log10(cm_error),c = track_recidency_in_drop,cmap = 'Greens')
plt.colorbar()
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Error in Center of Mass of Trajectory log10(um)")
plt.show()

plt.scatter(cm_boundary,np.log10(cm_error),c = np.log10(cm_diff),cmap = 'Greens')
plt.colorbar()
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Error in Center of Mass of Trajectory log10(um)")
plt.show()





x = np.array(x_collection)
y = np.array(y_collection)
xy = np.vstack([x_collection,y_collection])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]



n, _ = np.histogram(x,bins = 20)
sy, _ = np.histogram(x,bins = 20,weights = y)
sy2, _ = np.histogram(x,bins = 20,weights = y*y)
h, x_bins, y_bins = np.histogram2d(x,y,bins = 20)

mean = sy/n
std = np.sqrt(sy2/n - mean*mean)
plt.scatter(x,y,c = z, s = 50)
plt.plot((_[1:] + _[:-1])/2,mean, 'r-')
#plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 1)/(np.sum(np.sum(h.T,axis = 1))))
#plt.axvline(x=2.5*0.130,linestyle = 'dashed')
plt.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylabel("Displacements (um)")
#plt.ylabel("Dapp (um^2/s)")
# plt.ylim((-0.2,1.25))
# plt.xlim((-0.35,1))
plt.colorbar()
plt.show()


plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 0)/(np.sum(np.sum(h.T,axis = 0))))
plt.ylabel("Normalized Density of Trajectories")
plt.xlabel("Distance of Localization to Boundary (um)")
plt.ylim((0,1))
plt.xlim((-0.35,1))
plt.show()








# which = rp_ez

# estimated_rad = []


# inst_dist = []
# drop_dist = []
# inst_dist_d_a = []
# drop_dist_d_a = []
# inst_dist_s_a = []
# drop_dist_s_a = []
# for i in range(len(which.segs_frame_xy_data)):
#   inst_dist_s = []
#   drop_dist_s = []
#   for j in range(len(which.segs_frame_xy_data[i])): #this index is the one controlling [in, in/out, out]
#     for k in range(len(which.segs_frame_xy_data[i][j])): 
#       for tt in range(len(which.segs_frame_xy_data[i][j][k])):

#         diff_dist_temp = con_pix_si(dif_dis(which.segs_frame_xy_data[i][j][k][tt][0],which.segs_frame_xy_data[i][j][k][tt][1]),which = 'um')
#         #diff_dist_temp = con_pix_si(MSD_tavg(which.segs_frame_xy_data[i][j][k][tt][0],which.segs_frame_xy_data[i][j][k][tt][1],np.ones(len(which.segs_frame_xy_data[i][j][k][tt][1]))),which = 'msd') * np.ones(len(which.segs_frame_xy_data[i][j][k][tt][1])-1)
        
#         inst_dist_d = []
#         drop_dist_d = []
#         for kk in range(len(which.viable_drop_total[i][j])):
#           check_len = len(which.viable_drop_total[i][j][kk])
#           if check_len == 4:
#             radius = which.viable_drop_total[i][j][kk][3]/2.
#             estimated_rad.append(radius)
#           else:
#             radius = which.viable_drop_total[i][j][kk][2]

#           drop_center_dist = dist(which.segs_frame_xy_data[i][j][k][tt][0],which.segs_frame_xy_data[i][j][k][tt][1],which.viable_drop_total[i][j][kk][0],which.viable_drop_total[i][j][kk][1]) - radius
#           #print(drop_center_dist)
#           drop_center_dist_calc = dist(which.segs_frame_xy_data[i][j][k][tt][0],which.segs_frame_xy_data[i][j][k][tt][1],which.viable_drop_total[i][j][kk][0],which.viable_drop_total[i][j][kk][1])
          
#           #create an upper bound for tracks too far from the drop. 
#           #And create a condition that tracks are only considered if they atleast spend one localization inside the drop. (makes it less messy)
#           #and create a condition that the track also needs to be atleast one step outside so just in tracks arnt used.
#           if np.max(drop_center_dist) < 30 and np.min(drop_center_dist_calc) < radius:# and np.max(drop_center_dist_calc) > 2.5*0.130: 

#             inst_dist_s+=list(diff_dist_temp)
#             drop_dist_s+=list(con_pix_si(drop_center_dist[:-1],which = 'um'))
#             inst_dist_d+=list(diff_dist_temp)
#             drop_dist_d+=list(con_pix_si(drop_center_dist[:-1],which = 'um'))
#             inst_dist+=list(diff_dist_temp)
#             drop_dist+=list(con_pix_si(drop_center_dist[:-1],which = 'um'))
#         inst_dist_d_a.append(inst_dist_d)
#         drop_dist_d_a.append(drop_dist_d)

#   inst_dist_s_a.append(inst_dist_s)
#   drop_dist_s_a.append(drop_dist_s)


# x = np.array(drop_dist)
# y = np.array(inst_dist)
# xy = np.vstack([drop_dist,inst_dist])
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]





# n, _ = np.histogram(x,bins = 50)
# sy, _ = np.histogram(x,bins = 50,weights = y)
# sy2, _ = np.histogram(x,bins = 50,weights = y*y)
# h, x_bins, y_bins = np.histogram2d(x,y,bins = 50)

# mean = sy/n
# std = np.sqrt(sy2/n - mean*mean)
# plt.scatter(x,y,c = z, s = 50)
# plt.plot((_[1:] + _[:-1])/2,mean, 'r-')
# #plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 1)/(np.sum(np.sum(h.T,axis = 1))))
# #plt.axvline(x=2.5*0.130,linestyle = 'dashed')
# plt.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
# plt.xlabel("Distance of Localization to Boundary (um)")
# plt.ylabel("Displacements (um)")
# #plt.ylabel("Dapp (um^2/s)")
# # plt.ylim((-0.2,1.25))
# # plt.xlim((-0.35,1))
# plt.colorbar()
# plt.show()


# plt.plot((_[1:] + _[:-1])/2,np.sum(h.T,axis = 0)/(np.sum(np.sum(h.T,axis = 0))))
# plt.ylabel("Normalized Density of Trajectories")
# plt.xlabel("Distance of Localization to Boundary (um)")
# plt.ylim((0,1))
# plt.xlim((-0.35,1))
# plt.show()

# from scipy.stats import binned_statistic_2d
# a,b,c,d = binned_statistic_2d(x,y,None,'count',bins = 50, expand_binnumbers = True)




# ###########################################################################
# #  Test if there is a difference between the two peaks of the in fraction of RPOC in fraction for rp_ez in terms on the alpha scaling
# #can be changed to look at any fraction 

# # temp_in_tracks = ll_ez.out_tracksf

# def alpha_dif(temp_in_tracks,which_TP):
#   new_in_tracks_below = []
#   new_in_tracks_above = []
#   msd_thresh = -1.3
#   for i in range(len(temp_in_tracks)):

#     temp_inner_tracks_below = []
#     temp_inner_tracks_above = []

#     for j in range(len(temp_in_tracks[i])):
#       temp_msd = con_pix_si(np.log10(MSD_tavg(*temp_in_tracks[i][j],f = 0)))
#       if temp_msd > msd_thresh:
#         temp_inner_tracks_above.append(temp_in_tracks[i][j])
#       else:
#         temp_inner_tracks_below.append(temp_in_tracks[i][j])

#     new_in_tracks_above.append(temp_inner_tracks_above)
#     new_in_tracks_below.append(temp_inner_tracks_below)



#   which_TP_i_a = MSD_a_value_all_ens(which_TP.in_tracksf,threshold = 6)
#   which_TP_io_a = MSD_a_value_all_ens(which_TP.io_tracksf)
#   which_TP_o_a = MSD_a_value_all_ens(which_TP.out_tracksf)
#   which_TP_all = MSD_a_value_all_ens(which_TP.all_tracksf)

#   which_TP_i_above = MSD_a_value_all_ens(new_in_tracks_above,threshold = 6)
#   which_TP_i_below = MSD_a_value_all_ens(new_in_tracks_below,threshold = 6)
#   plt.show()

#   plt.plot(np.arange(20*1,20*15),fit_MSD(np.arange(20*1,20*15),1.0,1.0),'k--')
#   plt.plot(np.arange(20*1,20*15),fit_MSD(np.arange(20*1,20*15),0.01,0.0),'k--')
#   plt.plot(20*np.array(which_TP_i_a[2]),con_pix_si(which_TP_i_a[1],which = "msd"),label = "Alpha IN = {0:.3g}".format(which_TP_i_a[0][0]))
#   plt.plot(20*np.array(which_TP_io_a[2]),con_pix_si(which_TP_io_a[1],which = "msd"),label = "Alpha IN/OUT = {0:.3g}".format(which_TP_io_a[0][0]))
#   plt.plot(20*np.array(which_TP_o_a[2]),con_pix_si(which_TP_o_a[1],which = "msd"),label = "Alpha OUT = {0:.3g}".format(which_TP_o_a[0][0]))
#   plt.plot(20*np.array(which_TP_all[2]),con_pix_si(which_TP_all[1],which = "msd"),label = "Alpha All = {0:.3g}".format(which_TP_all[0][0]))
#   plt.plot(20*np.array(which_TP_i_above[2]),con_pix_si(which_TP_i_above[1],which = "msd"),label = "Alpha rp_ez_in_above -1.3 = {0:.3g}".format(which_TP_i_above[0][0]))
#   plt.plot(20*np.array(which_TP_i_below[2]),con_pix_si(which_TP_i_below[1],which = "msd"),label = "Alpha rp_ez_in_below -1.3 = {0:.3g}".format(which_TP_i_below[0][0]))

#   #plt.yscale("log")
#   plt.ylabel("MSD Ensemble (um/s^2)")
#   plt.xlabel("Tau (ms)")
#   plt.yscale("log")
#   plt.xscale("log")
#   plt.legend(loc = 1)
#   plt.show()
#   return






####################################
#Try to visualize the distribution of tracks at the poles vs the interior.
#May need to segment per cell for each track

















# def sector_mask(shape,centre,radius,angle_range):
#     """
#     Return a boolean mask for a circular sector. The start/stop angles in
#     `angle_range` should be given in clockwise order.
#     """
#     x,y = np.ogrid[:shape[0],:shape[1]]
#     cx,cy = centre
#     tmin,tmax = np.deg2rad(angle_range)
#     # ensure stop angle > start angle
#     if tmax < tmin:
#             tmax += 2*np.pi
#     # convert cartesian --> polar coordinates
#     r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
#     theta = np.arctan2(x-cx,y-cy) - tmin
#     # wrap angles between 0 and 2*pi
#     theta %= (2*np.pi)
#     # circular mask
#     circmask = r2 <= radius*radius
#     # angular mask
#     anglemask = theta <= (tmax-tmin)
#     return circmask*anglemask



# #############################
# #distanced vs drop boundary

# path = "DATA/new_days/20190527/rpoc_ez"

# analysis_path = path + '/Analysis'
# seg_path = path + '/segmented'
# seg_an_path = seg_path + '/Analysis'
# os.chdir(path)
# movies = sorted(glob.glob("*_seg.tif"))


# def load_files(f_path):
#     return io.imread(f_path)


# drop_seg = 1000.
# which = 1
# rp_ez_mov_1 = load_files(movies[which])


# masked_mov = np.zeros(np.shape(rp_ez_mov_1),'uint16')
# shape_mov = np.shape(rp_ez_mov_1)
# fig,ax = plt.subplots()
# ax.imshow(rp_ez_mov_1[0])
# global_dist = []
# global_int = []
# drop_counter_seg = 0
# for i in range(len(rp_ez_mov_1)):
#     temp_mask = np.zeros((len(rp_ez.viable_drop_total[which][drop_counter_seg]),np.shape(rp_ez_mov_1)[1],np.shape(rp_ez_mov_1)[2]))
#     drop_dist = []
#     drop_int = []
#     for j in range(len(rp_ez.viable_drop_total[which][drop_counter_seg])):
#         if len(rp_ez.viable_drop_total[which][drop_counter_seg]) != 0:
#             what_mask = sector_mask(s1111110[1:],(np.floor(rp_ez.viable_drop_total[which][drop_counter_seg][j][1]),np.floor(rp_ez.viable_drop_total[which][drop_counter_seg][j][0])),rp_ez.viable_drop_total[which][drop_counter_seg][j][2] + 2,(0,360))
#             masked_mov[i] += what_mask
#             temp_mask[j] += what_mask
#             cir = Circle((rp_ez.viable_drop_total[which][drop_counter_seg][j][0],rp_ez.viable_drop_total[which][drop_counter_seg][j][1]),rp_ez.viable_drop_total[which][drop_counter_seg][j][2]+2,fill = False)
#             ax.add_artist(cir)
#             temp_mask[j] *= rp_ez_mov_1[i]
#             non_zero = np.where(what_mask!=0)
#             distances = []
#             intensity = []
#             for lll in range(len(non_zero[0])):
#                 dister = dist(non_zero[0][lll],non_zero[1][lll],np.floor(rp_ez.viable_drop_total[which][drop_counter_seg][j][1]),np.floor(rp_ez.viable_drop_total[which][drop_counter_seg][j][0]))
#                 distances.append(dister)
#                 intensity.append(rp_ez_mov_1[i][non_zero[0][lll]][non_zero[1][lll]])
#             if len(distances)!=0:
#                 drop_dist.append(distances)
#                 drop_int.append(intensity)
#     if len(drop_dist) != 0:
#         global_dist.append(drop_dist)
#         global_int.append(drop_int)
#     if int((i+1) % drop_seg) == 0:
#         drop_counter_seg +=1
# #masked_movie = masked_mov*rp_ez_mov_1
# plt.savefig('drop_img')
# plt.show()
# #io.imsave('temp_tiff.tif',masked_movie)
# mov_len = len(global_dist)/1000.
# for i in range(int(mov_len)):
#     new_global_dist = np.array(global_dist[i*1000:(int(i)+1)*1000])
#     new_global_int = np.array(global_int[i*1000:(int(i)+1)*1000])
#     mean_dist = np.mean(new_global_dist,axis = 0)
#     mean_int = np.mean(new_global_int,axis = 0)
#     plt.violinplot(new_global_int[:,0], mean_dist[0],showmeans=False, showextrema=False, showmedians=False)
#     plt.legend()
#     plt.savefig('violin_plot_{0}'.format(i))
#     plt.show()
#     for j in range(len(mean_int)):
#         plt.scatter(mean_dist[j],mean_int[j],label = 'Seg#{0}'.format(i+1))
#     plt.legend()
#     plt.savefig('dot_plot_{0}'.format(i))
#     plt.show()

































# n= run_analysis("DATA/newer_NUSA","NUSA")
# n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n.run_flow()
# total_n = np.array(list(n.i_d_tavg) + list(n.io_d_tavg) + list(n.o_d_tavg))


# new_ax = create_box_plot([np.log10(con_pix_si(total_rp_ez, which = 'msd')),np.log10(con_pix_si(total_n, which = 'msd')),np.log10(con_pix_si(total_ll_ez, which = 'msd')),[0,0]],['rpoc','nusa','laci','rnap'])
# new_ax.plot([1,2,3,4],np.log10(np.array([dif_RPOC,dif_NUSA,dif_LACI,dif_RNAP])),'b.')
# new_ax.set_ylabel("Diffusion Coefficient log10(um^2/s)")
# plt.show()




# ll_ez_hex5 = run_analysis("DATA/20200210/ll_ez_hex5","ll_ez_hex5")
# ll_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5.run_flow()
# total_ll_ez_hex5 = np.array(list(ll_ez_hex5.i_d_tavg) + list(ll_ez_hex5.io_d_tavg) + list(ll_ez_hex5.o_d_tavg))
# total_msd = ll_ez_hex5.in_msd_track + ll_ez_hex5.io_msd_track + ll_ez_hex5.out_msd_track
# total_length = ll_ez_hex5.in_length + ll_ez_hex5.out_length + ll_ez_hex5.inout_length

# ll_ez_hex5l = run_analysis("DATA/20200210/ll_ez_hex5","ll_ez_hex5")
# ll_ez_hex5l.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 1)
# ll_ez_hex5l.run_flow()
# total_ll_ez_hex5l = np.array(list(ll_ez_hex5l.i_d_tavg) + list(ll_ez_hex5l.io_d_tavg) + list(ll_ez_hex5l.o_d_tavg))
# total_msdl = ll_ez_hex5l.in_msd_track + ll_ez_hex5l.io_msd_track + ll_ez_hex5l.out_msd_track
# total_lengthl = ll_ez_hex5l.in_length + ll_ez_hex5l.out_length + ll_ez_hex5l.inout_length

# nusa_ez_hex5 = run_analysis("DATA/20200210/nusa_ez_hex5_binned","nusa_ez_hex5")
# nusa_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5.run_flow()
# total_nusa_ez_hex5 = np.array(list(nusa_ez_hex5.i_d_tavg) + list(nusa_ez_hex5.io_d_tavg) + list(nusa_ez_hex5.o_d_tavg))


# rpoc_ez_hex5 = run_analysis("DATA/20200210/rpoc_ez_hex5","rpoc_ez_hex5")
# rpoc_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_ez_hex5.run_flow()
# total_rpoc_ez_hex5 = np.array(list(rpoc_ez_hex5.i_d_tavg) + list(rpoc_ez_hex5.io_d_tavg) + list(rpoc_ez_hex5.o_d_tavg))


# rpoc_ez_hex5 = run_analysis("DATA/20200210/rpoc_ez_hex5","rpoc_ez_hex5")
# rpoc_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_ez_hex5.run_flow()
# total_rpoc_ez_hex5 = np.array(list(rpoc_ez_hex5.i_d_tavg) + list(rpoc_ez_hex5.io_d_tavg) + list(rpoc_ez_hex5.o_d_tavg))


# nusa_m9 = run_analysis("DATA/20200212/nusa_m9_2","nusa_m9")
# nusa_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_m9.run_flow()
# total_nusa_m9 = np.array(list(nusa_m9.i_d_tavg) + list(nusa_m9.io_d_tavg) + list(nusa_m9.o_d_tavg))


# rpoc_m9 = run_analysis("DATA/20200212/rpoc_m9_2","rpoc_ez")
# rpoc_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_m9.run_flow()
# total_rpoc_m9 = np.array(list(rpoc_m9.i_d_tavg) + list(rpoc_m9.io_d_tavg) + list(rpoc_m9.o_d_tavg))


# rpoc_m9 = run_analysis("DATA/20200212/rpoc_m9_2","rpoc_ez")
# rpoc_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_m9.run_flow()
# total_rpoc_m9 = np.array(list(rpoc_m9.i_d_tavg) + list(rpoc_m9.io_d_tavg) + list(rpoc_m9.o_d_tavg))


# rp2= run_analysis("DATA/RPOC_new","RPOC")
# rp2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp2.run_flow()
# total_rp2 = np.array(list(rp2.i_d_tavg) + list(rp2.io_d_tavg) + list(rp2.o_d_tavg))

# ll_m9 = run_analysis("DATA/20200215/ll_m9","ll_m9")
# ll_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9.run_flow()
# total_ll_m9 = np.array(list(ll_m9.i_d_tavg) + list(ll_m9.io_d_tavg) + list(ll_m9.o_d_tavg))




# rp_ez_test = run_analysis("DATA/20190620","rpoc_ez")
# rp_ez_test.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_test.run_flow()
# total_rp_ez_test = np.array(list(rp_ez_test.i_d_tavg) + list(rp_ez_test.io_d_tavg) + list(rp_ez_test.o_d_tavg))



# ll_m9 = run_analysis("DATA/20200215/ll_m9","ll_m9")
# ll_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9.run_flow()
# total_ll_m9 = np.array(list(ll_m9.i_d_tavg) + list(ll_m9.io_d_tavg) + list(ll_m9.o_d_tavg))


# ll= run_analysis("DATA/LACO_LACI","TB54_FAST")
# ll.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll.run_flow()
# total_ll = np.array(list(ll.i_d_tavg) + list(ll.io_d_tavg) + list(ll.o_d_tavg))





# rp_m9_2= run_analysis("DATA/new_days/20190524/rpoc_m9","rpoc_M9")
# rp_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_m9_2.run_flow()
# total_rp_m9_2 = np.array(list(rp_m9_2.i_d_tavg) + list(rp_m9_2.io_d_tavg) + list(rp_m9_2.o_d_tavg))

# rp_ez_h5 = run_analysis("DATA/rpoc_ez_hex_5","rpoc_ez_hex_5")
# rp_ez_h5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_ez_h5.run_flow()
# total_rp_ez_h5 = np.array(list(rp_ez_h5.i_d_tavg) + list(rp_ez_h5.io_d_tavg) + list(rp_ez_h5.o_d_tavg))
# ll_ez_h3= run_analysis("DATA/new_days/20190527/ll_ez_hex_3","laco_laci_ez__hex_3")
# ll_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_h3.run_flow()
# total_ll_ez_h3 = np.array(list(ll_ez_h3.i_d_tavg) + list(ll_ez_h3.io_d_tavg) + list(ll_ez_h3.o_d_tavg))

# na_m9 = run_analysis("DATA/20191216/nusa_m9","nusa_m9")
# na_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# na_m9.run_flow()
# total_na_m9 = np.array(list(na_m9.i_d_tavg) + list(na_m9.io_d_tavg) + list(na_m9.o_d_tavg))


# nh= run_analysis("DATA/nusa_ez_hex_5","nusa_ez_hex_5")
# nh.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nh.run_flow()
# total_nh = np.array(list(nh.i_d_tavg) + list(nh.io_d_tavg) + list(nh.o_d_tavg))


# a= run_analysis("DATA/Nusa_20190305","NUSA")
# a.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# a.run_flow()
# total_a = np.array(list(a.i_d_tavg) + list(a.io_d_tavg) + list(a.o_d_tavg))
# ll_ez_hex5 = run_analysis("DATA/20200210/ll_ez_hex5","ll_ez_hex5")
# ll_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_hex5.run_flow()
# total_ll_ez_hex5 = np.array(list(ll_ez_hex5.i_d_tavg) + list(ll_ez_hex5.io_d_tavg) + list(ll_ez_hex5.o_d_tavg))





# nusa_ez_hex5_binned = run_analysis("DATA/20200210/nusa_ez_hex5_binned","nusa_ez_hex5")
# nusa_ez_hex5_binned.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nusa_ez_hex5_binned.run_flow()
# total_nusa_ez_hex5_binned = np.array(list(nusa_ez_hex5_binned.i_d_tavg) + list(nusa_ez_hex5_binned.io_d_tavg) + list(nusa_ez_hex5_binned.o_d_tavg))


# rpoc_ez_hex5_binned = run_analysis("DATA/20200210/rpoc_ez_hex5_binned","rpoc_ez_hex5")
# rpoc_ez_hex5_binned.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_ez_hex5_binned.run_flow()
# total_rpoc_ez_hex5_binned = np.array(list(rpoc_ez_hex5_binned.i_d_tavg) + list(rpoc_ez_hex5_binned.io_d_tavg) + list(rpoc_ez_hex5_binned.o_d_tavg))


# rpoc_ez_hex5= run_analysis("DATA/20200210/rpoc_ez_hex5","rpoc_ez_hex5")
# rpoc_ez_hex5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rpoc_ez_hex5.run_flow()
# total_rpoc_ez_hex5 = np.array(list(rpoc_ez_hex5.i_d_tavg) + list(rpoc_ez_hex5.io_d_tavg) + list(rpoc_ez_hex5.o_d_tavg))





# lI_ez_4= run_analysis("DATA/laci_only","lI")
# lI_ez_4.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l =4)
# lI_ez_4.run_flow()
# total_lI_ez_4 = np.array(list(lI_ez_4.i_d_tavg) + list(lI_ez_4.io_d_tavg) + list(lI_ez_4.o_d_tavg))
# #lI_ez_4 = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/laci_only","lI",10,50)
# #lI_ez_4.run()


# lI_ez= run_analysis("DATA/laci_only","lI")
# lI_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# lI_ez.run_flow()
# total_lI_ez = np.array(list(lI_ez.i_d_tavg) + list(lI_ez.io_d_tavg) + list(lI_ez.o_d_tavg))
# #lI_eza = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/laci_only","lI",10,50)
# #lI_eza.run()



# lI_4 = run_analysis("DATA/ll_ez_no_laco/","ll_m9")
# lI_4.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l =4)
# lI_4.run_flow()
# total_lI_4 = np.array(list(lI_4.i_d_tavg) + list(lI_4.io_d_tavg) + list(lI_4.o_d_tavg))

# lI = run_analysis("DATA/ll_ez_no_laco/","ll_m9")
# lI.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# lI.run_flow()
# total_lI = np.array(list(lI.i_d_tavg) + list(lI.io_d_tavg) + list(lI.o_d_tavg))




# fig = plt.figure()
# ax1 = fig.add_subplot(421)
# ax2 = fig.add_subplot(422)
# ax3 = fig.add_subplot(423)
# ax4 = fig.add_subplot(424)


# ax1.hist(np.log10(total_lI),weights=norm_weights(total_lI),label  = "T1: L>=10",alpha = 0.2)
# plt.legend()
# ax2.hist(np.log10(total_lI_4),weights=norm_weights(total_lI_4),label  = "T1: L>=4",alpha = 0.2)
# plt.legend()
# ax3.hist(np.log10(total_lI_ez),weights=norm_weights(total_lI_ez),label  = "T2: L>=10",alpha = 0.2)
# plt.legend()
# ax4.hist(np.log10(total_lI_ez_4),weights=norm_weights(total_lI_ez_4),label  = "T2: L>=4",alpha = 0.2)
# plt.legend()

# plt.show()




# plt.hist(np.log10(con_pix_si(ll_ez.i_d_tavg,'msd')),weights = norm_weights(ll_ez.i_d_tavg),label = "In",alpha = 0.2)
# plt.hist(np.log10(con_pix_si(ll_ez.io_d_tavg,'msd')),weights = norm_weights(ll_ez.io_d_tavg),label = "In/Out",alpha = 0.2)
# plt.hist(np.log10(con_pix_si(ll_ez.o_d_tavg,'msd')),weights = norm_weights(ll_ez.o_d_tavg),label = "Out",alpha = 0.2)

'''

na_m9 = run_analysis("DATA/20191216/nusa_m9","nusa_m9")
na_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
na_m9.run_flow()
total_na_m9 = np.array(list(na_m9.i_d_tavg) + list(na_m9.io_d_tavg) + list(na_m9.o_d_tavg))
na_m9a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/20191216/nusa_m9","nusa_m9",10,50)
na_m9a.run()


na_m9_2 = run_analysis("DATA/20191218/nusa_m9","nusa_m9")
na_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
na_m9_2.run_flow()
total_na_m9_2 = np.array(list(na_m9_2.i_d_tavg) + list(na_m9_2.io_d_tavg) + list(na_m9_2.o_d_tavg))
na_m92a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/20191218/nusa_m9","nusa_m9",10,50)
na_m92a.run()


rp= run_analysis("DATA","RPOC")
rp.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp.run_flow()
total_rp = np.array(list(rp.i_d_tavg) + list(rp.io_d_tavg) + list(rp.o_d_tavg))

rp_m9= run_analysis("DATA/rpoc_M9/20190515","rpoc_M9")
rp_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_m9.run_flow()
total_rp_m9 = np.array(list(rp_m9.i_d_tavg) + list(rp_m9.io_d_tavg) + list(rp_m9.o_d_tavg))
rp_m9a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/rpoc_M9/20190515","rpoc_M9",10,50)
rp_m9a.run()




# sim= run_analysis("DATA/sim_test","5")
# sim.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim.run_flow()
# total_sim = np.array(list(sim.i_d_tavg) + list(sim.io_d_tavg) + list(sim.o_d_tavg))

# sim1= run_analysis("DATA/001","0.01")
# sim1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim1.run_flow()
# total_sim1 = np.array(list(sim1.i_d_tavg) + list(sim1.io_d_tavg) + list(sim1.o_d_tavg))

# sim2= run_analysis("DATA/sim_test1","5")
# sim2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim2.run_flow()
# total_sim2 = np.array(list(sim2.i_d_tavg) + list(sim2.io_d_tavg) + list(sim2.o_d_tavg))

# sim01= run_analysis("DATA/sim_test01","5")
# sim01.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim01.run_flow()
# total_sim01 = np.array(list(sim01.i_d_tavg) + list(sim01.io_d_tavg) + list(sim01.o_d_tavg))




# sim4= run_analysis("DATA/test_noback","5")
# sim4.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim4.run_flow()
# total_sim4 = np.array(list(sim4.i_d_tavg) + list(sim4.io_d_tavg) + list(sim4.o_d_tavg))


# sim5= run_analysis("DATA/01_back","0.1")
# sim5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim5.run_flow()
# total_sim5 = np.array(list(sim5.i_d_tavg) + list(sim5.io_d_tavg) + list(sim5.o_d_tavg))




# sim_all= run_analysis("DATA/pres_test","test_all")
# sim_all.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim_all.run_flow()
# total_sim_all = np.array(list(sim_all.i_d_tavg) + list(sim_all.io_d_tavg) + list(sim_all.o_d_tavg))


# sim_01= run_analysis("DATA/01_what","01_what")
# sim_01.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim_01.run_flow()
# total_sim_01 = np.array(list(sim_01.i_d_tavg) + list(sim_01.io_d_tavg) + list(sim_01.o_d_tavg))


# sim_001= run_analysis("DATA/001_what","001_what")
# sim_001.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l = 2)
# sim_001.run_flow()
# total_sim_001 = np.array(list(sim_001.i_d_tavg) + list(sim_001.io_d_tavg) + list(sim_001.o_d_tavg))


rp_ez= run_analysis("DATA/new_days/20190527/rpoc_ez","rpoc_ez")
rp_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50,t_len_l=1)
rp_ez.run_flow()
total_rp_ez = np.array(list(rp_ez.i_d_tavg) + list(rp_ez.io_d_tavg) + list(rp_ez.o_d_tavg))
rp_eza = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez","rpoc_ez",10,50)
rp_eza.run()


# # rp_ez_M = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez","rpoc_ez",10,50)
# # rp_ez_M.run()


rp_m9_2= run_analysis("DATA/new_days/20190524/rpoc_m9","rpoc_M9")
rp_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_m9_2.run_flow()
total_rp_m9_2 = np.array(list(rp_m9_2.i_d_tavg) + list(rp_m9_2.io_d_tavg) + list(rp_m9_2.o_d_tavg))
rp_m9_2a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190524/rpoc_m9","rpoc_M9",10,50)
rp_m9_2a.run()


ll_m9_24 = run_analysis("DATA/new_days/20190524/laco_laci_M9","laco_laci_M9")
ll_m9_24.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll_m9_24.run_flow()
total_ll_m9_24 = np.array(list(ll_m9_24.i_d_tavg) + list(ll_m9_24.io_d_tavg) + list(ll_m9_24.o_d_tavg))
ll_m9_24a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190524/laco_laci_M9","laco_laci_M9",4,50)
ll_m9_24a.run()



rp_ez_h3 = run_analysis("DATA/new_days/20190527/rpoc_ez_hex_3","rpoc_ez_hex_3")
rp_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez_h3.run_flow()
total_rp_ez_h3 = np.array(list(rp_ez_h3.i_d_tavg) + list(rp_ez_h3.io_d_tavg) + list(rp_ez_h3.o_d_tavg))

rp_ez_h5 = run_analysis("DATA/rpoc_ez_hex_5","rpoc_ez_hex_5")
rp_ez_h5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez_h5.run_flow()
total_rp_ez_h5 = np.array(list(rp_ez_h5.i_d_tavg) + list(rp_ez_h5.io_d_tavg) + list(rp_ez_h5.o_d_tavg))



rp_ez_h5_2 = run_analysis("DATA/rpoc_ez_hex_5_2","rpoc_ez_h_5")
rp_ez_h5_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez_h5_2.run_flow()
total_rp_ez_h5_2 = np.array(list(rp_ez_h5_2.i_d_tavg) + list(rp_ez_h5_2.io_d_tavg) + list(rp_ez_h5_2.o_d_tavg))


# rp_ez_h3M = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez_hex_3","rpoc_ez_hex_3",10,50)
# rp_ez_h3M.run()

# rp_ez_h5M = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/rpoc_ez_hex_5","rpoc_ez_hex_5",10,50)
# rp_ez_h5M.run()

# rp_ez_h52M = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/rpoc_ez_hex_5_2","rpoc_ez_h_5",10,50)
# rp_ez_h52M.run()

ll_ez= run_analysis("DATA/new_days/20190527/ll_ez","laco_laci_ez")
ll_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll_ez.run_flow()
total_ll_ez = np.array(list(ll_ez.i_d_tavg) + list(ll_ez.io_d_tavg) + list(ll_ez.o_d_tavg))
ll_eza = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez","laco_laci_ez",10,50)
ll_eza.run()

#ll_ez_M = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez","laco_laci_ez",10,50)
#ll_ez_M.run()


ll_m9= run_analysis("DATA/new_days/20190527/ll_m9","laco_laci_m9")
ll_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll_m9.run_flow()
total_ll_m9 = np.array(list(ll_m9.i_d_tavg) + list(ll_m9.io_d_tavg) + list(ll_m9.o_d_tavg))
ll_m9a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_m9","laco_laci_m9",10,50)
ll_m9a.run()

ll_m9n= run_analysis("DATA/laco_m9","laco_m9")
ll_m9n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll_m9n.run_flow()
total_ll_m9n = np.array(list(ll_m9n.i_d_tavg) + list(ll_m9n.io_d_tavg) + list(ll_m9n.o_d_tavg))
ll_m9na = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/laco_m9","laco_m9",10,50)
ll_m9na.run()

nh= run_analysis("DATA/nusa_ez_hex_5","nusa_ez_hex_5")
nh.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
nh.run_flow()
total_nh = np.array(list(nh.i_d_tavg) + list(nh.io_d_tavg) + list(nh.o_d_tavg))
nha = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/nusa_ez_hex_5","nusa_ez_hex_5",10,50)
nha.run()


ll_ez_h3= run_analysis("DATA/new_days/20190527/ll_ez_hex_3","laco_laci_ez__hex_3")
ll_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll_ez_h3.run_flow()
total_ll_ez_h3 = np.array(list(ll_ez_h3.i_d_tavg) + list(ll_ez_h3.io_d_tavg) + list(ll_ez_h3.o_d_tavg))
ll_ez_h3a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez_hex_3","laco_laci_ez__hex_3",10,50)
ll_ez_h3a.run()

rp2= run_analysis("DATA/RPOC_new","RPOC")
rp2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp2.run_flow()
total_rp2 = np.array(list(rp2.i_d_tavg) + list(rp2.io_d_tavg) + list(rp2.o_d_tavg))
rp2a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/RPOC_new","RPOC",10,50)
rp2a.run()

rp3= run_analysis("DATA/Files_RPOC","RPOC")
rp3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp3.run_flow()
total_rp3 = np.array(list(rp3.i_d_tavg) + list(rp3.io_d_tavg) + list(rp3.o_d_tavg))
rp3a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/Files_RPOC","RPOC",10,50)
rp3a.run()

rp1= run_analysis("DATA/Other_RPOC","rpoc")
rp1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp1.run_flow()
total_rp1 = np.array(list(rp1.i_d_tavg) + list(rp1.io_d_tavg) + list(rp1.o_d_tavg))
rp1a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/Other_RPOC","rpoc",10,50)
rp1a.run()

ll= run_analysis("DATA/LACO_LACI","TB54_FAST")
ll.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll.run_flow()
total_ll = np.array(list(ll.i_d_tavg) + list(ll.io_d_tavg) + list(ll.o_d_tavg))
lla = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/LACO_LACI","TB54_FAST",10,50)
lla.run()

n= run_analysis("DATA/newer_NUSA","NUSA")
n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
n.run_flow()

total_n = np.array(list(n.i_d_tavg) + list(n.io_d_tavg) + list(n.o_d_tavg))
na = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/newer_NUSA","NUSA",10,50)
na.run()

n1= run_analysis("DATA/New_NUSA","NUSA")
n1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
n1.run_flow()
total_n1 = np.array(list(n1.i_d_tavg) + list(n1.io_d_tavg) + list(n1.o_d_tavg))
n1a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/New_NUSA","NUSA",10,50)
n1a.run()

n2= run_analysis("DATA/Nusa_20190304","NUSA")
n2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
n2.run_flow()
total_n2 = np.array(list(n2.i_d_tavg) + list(n2.io_d_tavg) + list(n2.o_d_tavg))
n2a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/Nusa_20190304","NUSA",10,50)
n2a.run()










fig = plt.figure()
ax = fig.add_subplot(111)

ll_dat = np.log10(con_pix_si(total_ll,which = "msd"))
rp_dat = np.log10(con_pix_si(total_rp_ez,which = "msd"))
n_dat = np.log10(con_pix_si(total_a,which = "msd"))

ll_dat_w = np.ones_like(np.array(ll_dat))/float(len(np.array(ll_dat)))
rp_dat_w = np.ones_like(np.array(rp_dat))/float(len(np.array(rp_dat)))
n_dat_w = np.ones_like(np.array(n_dat))/float(len(np.array(n_dat)))


plt.hist(ll_dat,weights = ll_dat_w, alpha = 0.5)
plt.hist(rp_dat,weights = rp_dat_w, alpha = 0.5)
plt.hist(n_dat,weights = n_dat_w, alpha = 0.5)

plt.savefig("full_overall.svg",format = "svg")

plt.show()




rp_ez.segmented_drop_files

'''














# # ###chephalexin
# ceph_rpoc = run_analysis("DATA/chephalexin/20190731/rpoc_ez_50_ceph", "rpoc_ez_50ceph")
# ceph_rpoc.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ceph_rpoc.run_flow()
# total_ceph_rpoc = np.array(list(ceph_rpoc.i_d_tavg) + list(ceph_rpoc.io_d_tavg) + list(ceph_rpoc.o_d_tavg))




# b = a.viable_drop_total

# c = a.in_track_total 
# c1 = a.io_track_total 
# c2 = a.ot_track_total

# rg = a.in_radius_g
# rg1 = a.io_radius_g
# rg2 = a.ot_radius_g

# cp = a.in_msd_all
# cp1 = a.io_msd_all
# cp2 = a.ot_msd_all

# d = a.segmented_drop_files










# a= run_analysis("DATA/Nusa_20190305","NUSA")
# a.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# a.run_flow()
# total_a = np.array(list(a.i_d_tavg) + list(a.io_d_tavg) + list(a.o_d_tavg))
# # # aaa = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/Nusa_20190305","NUSA",10,50)
# # # aaa.run()
# # # rp_ez= run_analysis("DATA/new_days/20190527/rpoc_ez","rpoc_ez")
# # # rp_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# # # rp_ez.run_flow()
# # # total_rp_ez = np.array(list(rp_ez.i_d_tavg) + list(rp_ez.io_d_tavg) + list(rp_ez.o_d_tavg))
# total_log_msd = np.log10(np.array(total_a))
# # ll_ez= run_analysis("DATA/new_days/20190527/ll_ez","laco_laci_ez")
# # ll_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# # ll_ez.run_flow()
# # total_ll_ez = np.array(list(ll_ez.i_d_tavg) + list(ll_ez.io_d_tavg) + list(ll_ez.o_d_tavg))



# # nusa = run_analysis("DATA/Nusa_20190305","NUSA")
# # nusa.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# # nusa.run_flow()
# # total_nusa = np.array(list(nusa.i_d_tavg) + list(nusa.io_d_tavg) + list(nusa.o_d_tavg))



# cvals  = [np.min(con_pix_si(total_log_msd,which = 'msd')),np.percentile(con_pix_si(total_log_msd,which = 'msd'),25), np.percentile(con_pix_si(total_log_msd,which = 'msd'),75), np.max(con_pix_si(total_log_msd,which = 'msd'))]
# colors = ["green","red","violet","blue"]

# norm=plt.Normalize(min(cvals),max(cvals))
# tuples = list(zip(list(map(norm,cvals)), colors))
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


# # # e1 = a.in_msd_all
# # e2 = a.io_msd_all
# # e3 = a.ot_msd_all


# def plot_msd_n(temp_a,list_number,label,title = " ",log_scale = False):
# 	for i in range(list_number):
# 		plt.hist(np.log10(temp_a[i]),label = label[i],alpha = 0.3,density = True)
# 	if log_scale:
# 		#plt.xscale("log")
# 		pass
# 	plt.xlabel("Log10 MSD in um^2/s")
# 	plt.title(title)
# 	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# 	plt.tight_layout()
# 	plt.show()

# def plot_msd(temp_a,list_number,label,fig,ax,title = " ",log_scale = False):
# 	for i in range(list_number):
# 		ax.hist(np.log10(temp_a[i]),label = label[i],alpha = 0.3,density = True)
# 	if log_scale:
# 		#plt.xscale("log")
# 		pass
# 	ax.set_xlabel("Log10 MSD in um^2/s")
# 	ax.set_title(title)
# 	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# 	return 


# def crop_img(img,top,bottom,left,right):
#     x_len = len(img)
#     y_len = len(img[0])
#     return img[int(top*x_len):int(bottom*x_len),int(left*y_len):int(right*y_len)]



# # fig = plt.figure()
# # d = a.segmented_drop_files
# # #img_ll_seg ='/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/segmented/4_laco_laci_ez_6_seg.tif'
# # #img_ll_ez = mpimg.imread(img_ll_seg)

# # ax = fig.add_subplot(111)
# # img = mpimg.imread(d[2][3])[75:135,65:140]
# # timg_in = ax.imshow(img)
# # #timg = ax.imshow(img_ll_ez,origin = "lower")
# # scalebar = ScaleBar(0.13, 'um')
# # ax.add_artist(scalebar)
# # plt.show()



# drop_color = ["y","b","r","g","m"]


# def overall_plot2D(op, which = "all"):

#   b = op.viable_drop_total
#   d = op.segmented_drop_files
#   c = op.in_track_total
#   c1 = op.io_track_total
#   c2 = op.ot_track_total
#   cp = op.in_msd_all
#   cp1 = op.io_msd_all
#   cp2 = op.ot_msd_all

#   for i in range(len(b)):

#       if len(d[i]) != 0:
#           img = mpimg.imread(d[i][0])
#           timg = plt.imshow(img,cmap=plt.get_cmap('gray'))

#       for j in range(len(b[i])):
#           if which == "all" or which == "in":
#               for l in range(len(c[i][j])):
#                   if len(c[i][j][l])!=0:
#                       temp = np.array(c[i][j][l])
#                       #plt.plot(temp[0],temp[1],'b-')
#                       plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#           if which == "all" or which == "io":
#               for l in range(len(c1[i][j])):
#                   if len(c1[i][j][l])!=0:
#                       temp = np.array(c1[i][j][l])
#                       #plt.plot(temp[0],temp[1],'g-')
#                       plt.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#           if which == "all" or which == "out":

#               for l in range(len(c2[i][j])):
#                   if len(c2[i][j][l])!=0:
#                       temp = np.array(c2[i][j][l])
#                       #plt.plot(temp[0],temp[1],'r-')
#                       plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

#           if (len(b[i][j])>0):
#             for k in range(len(b[i][j])):
#                 circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
#       plt.colorbar()
#       plt.savefig("Frame_{0}".format(i))
#       plt.show()
#   return

# b = rp_ez.viable_drop_total
# d = rp_ez.segmented_drop_files
# c = rp_ez.in_track_total
# c1 = rp_ez.io_track_total
# c2 = rp_ez.ot_track_total
# cp = rp_ez.in_msd_all
# cp1 = rp_ez.io_msd_all
# cp2 = rp_ez.ot_msd_all


# in_in = True

# io_in = True

# ot_in = True

# i =4
# for j in range(len(b[i])):


#     for l in range(len(c[i][j])):
#         if len(c[i][j][l])!=0:
#             temp = np.array(c[i][j][l])
#             if in_in:
#                 plt.plot(temp[0],temp[1])
#                 in_in = False

#     for l in range(len(c1[i][j])):
#         if len(c1[i][j][l])!=0:
#             temp = np.array(c1[i][j][l])
#             if io_in:
#                 plt.plot(temp[0],temp[1])
#                 io_in = False

#     for l in range(len(c2[i][j])):
#         if len(c2[i][j][l])!=0:
#             temp = np.array(c2[i][j][l])
#             if ot_in:
#                 plt.plot(temp[0],temp[1])
#                 ot_in = False

#     if (len(b[i][j])>0):
#         for k in range(len(b[i][j])):
#             circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# plt.savefig("drop.svg")
# plt.show()








# fig = plt.figure()

# img_rpoc_seg ='/Users/baljyot/Documents/2019-2020/RNAP_PAPER/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez/segmented/1_rpoc_ez_1_seg.tif'
# img_ll_seg = '/Users/baljyot/Documents/2019-2020/RNAP_PAPER/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/segmented/4_laco_laci_ez_6_seg.tif'

# img_rp_ez = mpimg.imread(img_rpoc_seg)
# img_ll_ez = mpimg.imread(img_ll_seg)


# ax1 = fig.add_subplot(111)

# ax1.axis('off')


# b = rp_ez.viable_drop_total
# d = rp_ez.segmented_drop_files
# c = rp_ez.in_track_total
# c1 = rp_ez.io_track_total
# c2 = rp_ez.ot_track_total
# cp = rp_ez.in_msd_all
# cp1 = rp_ez.io_msd_all
# cp2 = rp_ez.ot_msd_all


# which = "all"
# show = False
# line = True
# scatter = True
# good = 0
# i = 0
# img_ori = mpimg.imread(d[0][0])


# cmap_all=plt.get_cmap('gray')


# cmap_all.set_bad(color = 'white')

# img = np.ma.masked_where(img_ori == 126,img_ori)


# def masked(img,do = True):
#     if do:
#         return np.ma.masked_where(img == 126,img)
#     else:
#         return img

# timg = ax1.imshow(masked(img_rp_ez,True),cmap=cmap_all)
# #other_tim = ax2.imshow(masked(img_rp_ez,True),cmap=cmap_all)


# #rp_ez_im = ax.imshow(masked(img_rp_ez,True),cmap = cmap_all)



# copy_array_in = np.zeros(np.shape(img))
# copy_array_io = np.zeros(np.shape(img))
# copy_array_ot = np.zeros(np.shape(img))
# copy_array_all = np.zeros(np.shape(img))


# random_choose_c = [0,3]
# random_choose_c1 = [0,3]
# random_choose_c2 = [0,3]
# choose_b = np.random.randint(0,len(b[i]),2)

# in_track_used = []
# io_track_used = []
# ot_track_used = []

# track_x = [] #x values of the track localizations
# track_y = [] #y values of the track localizations
# track_z = [] #msd values of the track localizations


# for j in range(len(b[i])):


#     for l in range(len(c[i][j])):
#         if len(c[i][j][l])!=0:
#             temp = np.array(c[i][j][l])

#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'b-')
#             if (which == "all" or which == "in") and scatter :
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l])))

#                 #ax2.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "in") and line:
#                 if (good == 0):
#                     if len(temp[0]) != 0:
#                         print(temp[0])
#                         #ax.plot(temp[0],temp[1],c = 'r')
#                         in_track_used = [temp[0],temp[1],cmap(cp[i][j][l])]
#                         good = 1
#                         print("good = {0},{1}".format(j,l))

#     for l in range(len(c1[i][j])):
#         if len(c1[i][j][l])!=0:
#             temp = np.array(c1[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'g-')
#             if (which == "all" or which == "io") and scatter:
#                 #ax2.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l])))
#             if (which == "all" or which == "io") and line and (j == 0) and (l == 0):
#                 #ax.plot(temp[0],temp[1],c = 'b')
#                 io_track_used = [temp[0],temp[1],cmap(cp1[i][j][l])]

#     for l in range(len(c2[i][j])):
#         if len(c2[i][j][l])!=0:
#             temp = np.array(c2[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'r-')
#             if (which == "all" or which == "out") and scatter:
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l])))
#                 #what = ax2.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "out") and line and (j == 0) and (l == 6):
#                 #ax.plot(temp[0],temp[1],c = 'g')
#                 ot_track_used = [temp[0],temp[1],cmap(cp2[i][j][l])]
#     # if (len(b[i][j])>0):
#     #   for k in range(len(b[i][j])):
#     #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# #ax2.colorbar()

#     #fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.15, 0.85, 0.7, 0.03])
# #####sorted scatter points
# idx = np.array(track_z).argsort()[::-1]
# #ax2.scatter(np.array(track_x)[idx],np.array(track_y)[idx],s= 2,c = np.array(track_z)[idx],cmap=cmap, norm = norm)
# #fig.colorbar(what,cax=cbar_ax,orientation="horizontal")
# cont = ax1.contourf(copy_array_all)
# fig.colorbar(cont,cax=cbar_ax,orientation="horizontal")
# # ax2.annotate('$log\\left( \\frac{um^2}{s}\\right) $', xy=(2.4*ax2.bbox.width, 2.2*ax2.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
# # ax.annotate('RPOC', xy=(-1.0*ax.bbox.width, 0.65*ax.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
# # axins_in = zoomed_inset_axes(ax, 3, loc=1)
# # axins_in.plot(in_track_used[0],in_track_used[1],c = 'r',lw = 1)
# # axins_in.get_xaxis().set_visible(False)
# # axins_in.get_yaxis().set_visible(False)
# # axins_io = zoomed_inset_axes(ax, 3, loc=3)
# # axins_io.plot(io_track_used[0],io_track_used[1],c = 'b',lw = 1)
# # axins_io.get_xaxis().set_visible(False)
# # axins_io.get_yaxis().set_visible(False)
# # axins_ot = zoomed_inset_axes(ax, 3, loc=4)
# # axins_ot.plot(ot_track_used[0],ot_track_used[1],c = 'g',lw = 1)
# # axins_ot.get_xaxis().set_visible(False)
# # axins_ot.get_yaxis().set_visible(False)
# # mark_inset(ax, axins_in, loc1=2, loc2=4, fc="none", ec="0.5")
# # mark_inset(ax, axins_io, loc1=2, loc2=4, fc="none", ec="0.5")
# # mark_inset(ax, axins_ot, loc1=2, loc2=1, fc="none", ec="0.5")

# # ax.get_xaxis().set_visible(False)
# # ax.get_yaxis().set_visible(False)
# # ax.set_xlim((80,210))
# # ax.set_ylim((80,210))
# ax1.get_xaxis().set_visible(False)
# ax1.get_yaxis().set_visible(False)
# # ax1.set_xlim((80,210))
# # ax1.set_ylim((80,210))
# # ax2.get_xaxis().set_visible(False)
# # ax2.get_yaxis().set_visible(False)
# # ax2.set_xlim((80,210))
# # ax2.set_ylim((80,210))
# #obj2 = add_scalebar(ax2) 
# scalebar = ScaleBar(0.13, 'um',location = 1)
# ax1.add_artist(scalebar)
# fig.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('new.svg',format="svg")
# plt.show()






##################################################################################
#figure_1

# fig = plt.figure()

# img_rpoc_seg ='/Users/baljyot/Documents/2019-2020/RNAP_PAPER/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez/segmented/4_rpoc_ez_4_seg.tif'
# img_ll_seg = '/Users/baljyot/Documents/2019-2020/RNAP_PAPER/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/segmented/4_laco_laci_ez_6_seg.tif'

# img_rp_ez = mpimg.imread(img_rpoc_seg)
# img_ll_ez = mpimg.imread(img_ll_seg)


# ax = fig.add_subplot(331)
# ax1 = fig.add_subplot(332)
# ax2 = fig.add_subplot(333)

# ax3 = fig.add_subplot(334)
# ax4 = fig.add_subplot(335)
# ax5 = fig.add_subplot(336)


# ax6 = fig.add_subplot(337)
# ax7 = fig.add_subplot(338)
# ax8 = fig.add_subplot(339)



# ax.axis('off')
# ax1.axis('off')
# ax2.axis('off')
# ax3.axis('off')
# ax4.axis('off')
# ax5.axis('off')
# ax6.axis('off')
# ax7.axis('off')
# ax8.axis('off')

# b = rp_ez.viable_drop_total
# d = rp_ez.segmented_drop_files
# c = rp_ez.in_track_total
# c1 = rp_ez.io_track_total
# c2 = rp_ez.ot_track_total
# cp = rp_ez.in_msd_all
# cp1 = rp_ez.io_msd_all
# cp2 = rp_ez.ot_msd_all


# which = "all"
# show = False
# line = True
# scatter = True
# good = 0
# i = 4
# img_ori = mpimg.imread(d[4][3])


# cmap_all=plt.get_cmap('gray')


# cmap_all.set_bad(color = 'white')

# img = np.ma.masked_where(img_ori == 126,img_ori)


# def masked(img,do = True):
#     if do:
#         return np.ma.masked_where(img == 126,img)
#     else:
#         return img

# timg = ax1.imshow(masked(img_rp_ez,True),cmap=cmap_all,origin = "lower")
# other_tim = ax2.imshow(masked(img_rp_ez,True),cmap=cmap_all,origin = "lower")


# rp_ez_im = ax.imshow(masked(img_rp_ez,True),origin = 'lower',cmap = cmap_all)
# ll_ez_im = ax3.imshow(masked(img_ll_ez,True),origin = 'lower',cmap = cmap_all)



# copy_array_in = np.zeros(np.shape(img))
# copy_array_io = np.zeros(np.shape(img))
# copy_array_ot = np.zeros(np.shape(img))
# copy_array_all = np.zeros(np.shape(img))


# random_choose_c = [0,3]
# random_choose_c1 = [0,3]
# random_choose_c2 = [0,3]
# choose_b = np.random.randint(0,len(b[i]),2)

# in_track_used = []
# io_track_used = []
# ot_track_used = []

# track_x = [] #x values of the track localizations
# track_y = [] #y values of the track localizations
# track_z = [] #msd values of the track localizations


# for j in range(len(b[i])):


#     for l in range(len(c[i][j])):
#         if len(c[i][j][l])!=0:
#             temp = np.array(c[i][j][l])

#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'b-')
#             if (which == "all" or which == "in") and scatter :
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l])))

#                 #ax2.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "in") and line:
#                 if (good == 0):
#                     if len(temp[0]) != 0:
#                         print(temp[0])
#                         ax.plot(temp[0],temp[1],c = 'r')
#                         in_track_used = [temp[0],temp[1],cmap(cp[i][j][l])]
#                         good = 1
#                         print("good = {0},{1}".format(j,l))

#     for l in range(len(c1[i][j])):
#         if len(c1[i][j][l])!=0:
#             temp = np.array(c1[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'g-')
#             if (which == "all" or which == "io") and scatter:
#                 #ax2.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l])))
#             if (which == "all" or which == "io") and line and (j == 0) and (l == 0):
#                 ax.plot(temp[0],temp[1],c = 'b')
#                 io_track_used = [temp[0],temp[1],cmap(cp1[i][j][l])]

#     for l in range(len(c2[i][j])):
#         if len(c2[i][j][l])!=0:
#             temp = np.array(c2[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'r-')
#             if (which == "all" or which == "out") and scatter:
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l])))
#                 #what = ax2.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "out") and line and (j == 0) and (l == 6):
#                 ax.plot(temp[0],temp[1],c = 'g')
#                 ot_track_used = [temp[0],temp[1],cmap(cp2[i][j][l])]
#     # if (len(b[i][j])>0):
#     #   for k in range(len(b[i][j])):
#     #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# #ax2.colorbar()

#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.15, 0.85, 0.7, 0.03])
# #####sorted scatter points
# idx = np.array(track_z).argsort()[::-1]
# ax2.scatter(np.array(track_x)[idx],np.array(track_y)[idx],s= 2,c = np.array(track_z)[idx],cmap=cmap, norm = norm)
# #fig.colorbar(what,cax=cbar_ax,orientation="horizontal")
# cont = ax1.contour(copy_array_all)

# ax2.annotate('$log\\left( \\frac{um^2}{s}\\right) $', xy=(2.4*ax2.bbox.width, 2.2*ax2.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
# ax.annotate('RPOC', xy=(-1.0*ax.bbox.width, 0.65*ax.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
# ax3.annotate('LacO \n LacI', xy=(-1.3*ax3.bbox.width, 0.65*ax3.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
# ax6.annotate('NUSA', xy=(-1.0*ax3.bbox.width, 0.65*ax3.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
# axins_in = zoomed_inset_axes(ax, 3, loc=1)
# axins_in.plot(in_track_used[0],in_track_used[1],c = 'r',lw = 1)
# axins_in.get_xaxis().set_visible(False)
# axins_in.get_yaxis().set_visible(False)
# axins_io = zoomed_inset_axes(ax, 3, loc=3)
# axins_io.plot(io_track_used[0],io_track_used[1],c = 'b',lw = 1)
# axins_io.get_xaxis().set_visible(False)
# axins_io.get_yaxis().set_visible(False)
# axins_ot = zoomed_inset_axes(ax, 3, loc=4)
# axins_ot.plot(ot_track_used[0],ot_track_used[1],c = 'g',lw = 1)
# axins_ot.get_xaxis().set_visible(False)
# axins_ot.get_yaxis().set_visible(False)
# mark_inset(ax, axins_in, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins_io, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax, axins_ot, loc1=2, loc2=1, fc="none", ec="0.5")






# b = ll_ez.viable_drop_total
# d = ll_ez.segmented_drop_files
# c = ll_ez.in_track_total
# c1 = ll_ez.io_track_total
# c2 = ll_ez.ot_track_total
# cp = ll_ez.in_msd_all
# cp1 = ll_ez.io_msd_all
# cp2 = ll_ez.ot_msd_all


# which = "all"
# show = False
# line = True
# scatter = True
# good = 0
# i = 5
# img = mpimg.imread(d[5][3])
# timg = ax4.imshow(masked(img,True),cmap=cmap_all,origin = "lower")
# other_tim = ax5.imshow(masked(img,True),cmap=cmap_all,origin = "lower")
# copy_array_in = np.zeros(np.shape(img))
# copy_array_io = np.zeros(np.shape(img))
# copy_array_ot = np.zeros(np.shape(img))
# copy_array_all = np.zeros(np.shape(img))


# random_choose_c = [0,3]
# random_choose_c1 = [0,3]
# random_choose_c2 = [0,3]
# choose_b = np.random.randint(0,len(b[i]),2)

# in_track_used = []
# io_track_used = []
# ot_track_used = []



# track_x = [] #x values of the track localizations
# track_y = [] #y values of the track localizations
# track_z = [] #msd values of the track localizations


# for j in range(len(b[i])):

#     which_in = 0
#     for l in range(len(c[i][j])):
#         if len(c[i][j][l])!=0:
#             temp = np.array(c[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'b-')
#             if (which == "all" or which == "in") and scatter :
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l])))
#                 #ax5.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "in") and line:
#                 if (good == 0):
#                     if len(temp[0]) != 0:
#                         if which_in == 12:
#                             print(temp[0])
#                             ax3.plot(temp[0],temp[1],c = 'r')
#                             in_track_used = [temp[0],temp[1]]
#                             good = 1
#                             print("good = {0},{1}".format(j,l))
#                         which_in+=1

#     for l in range(len(c1[i][j])):
#         if len(c1[i][j][l])!=0:
#             temp = np.array(c1[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'g-')
#             if (which == "all" or which == "io") and scatter:
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l])))
#                 #ax5.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "io") and line and (j == 0) and (l == 0):
#                 ax3.plot(temp[0],temp[1],c = 'b')
#                 io_track_used = [temp[0],temp[1]]
#     which_out = 0
#     for l in range(len(c2[i][j])):
#         if len(c2[i][j][l])!=0:
#             temp = np.array(c2[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'r-')
#             if (which == "all" or which == "out") and scatter:
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l])))
#                 #ax5.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "out") and line and (j == 0) and (l == 6):
#                 ax3.plot(temp[0],temp[1],c = 'g')
#                 ot_track_used = [temp[0],temp[1]]
#             which_out += 1
#     # if (len(b[i][j])>0):
#     #   for k in range(len(b[i][j])):
#     #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)

# #####sorted scatter points
# idx = np.array(track_z).argsort()[::-1]
# ax5.scatter(np.array(track_x)[idx],np.array(track_y)[idx],s= 2,c = np.array(track_z)[idx],cmap=cmap, norm = norm)

# axins_in = zoomed_inset_axes(ax3, 3, loc=4)
# axins_in.plot(in_track_used[0],in_track_used[1],'r')
# axins_in.set_xlim((np.min(in_track_used[0]) - 1, np.max(in_track_used[0]) + 1 ))
# axins_in.set_ylim((np.min(in_track_used[1]) - 1, np.max(in_track_used[1]) + 1 ))
# axins_in.get_xaxis().set_visible(False)
# axins_in.get_yaxis().set_visible(False)
# # axins_io = zoomed_inset_axes(ax3, 6, loc=7)
# # axins_io.plot(io_track_used[0],io_track_used[1],'b')
# # axins_io.get_xaxis().set_visible(False)
# # axins_io.get_yaxis().set_visible(False)
# # axins_ot = zoomed_inset_axes(ax3, 3, loc=1)
# # axins_ot.plot(ot_track_used[0],ot_track_used[1],'g')
# # axins_ot.set_xlim((np.min(ot_track_used[0]) - 1, np.max(ot_track_used[0]) + 1 ))
# # axins_ot.set_ylim((np.min(ot_track_used[1]) - 1, np.max(ot_track_used[1]) + 1 ))
# # axins_ot.get_xaxis().set_visible(False)
# # axins_ot.get_yaxis().set_visible(False)
# mark_inset(ax3, axins_in, loc1=3, loc2=1, fc="none", ec="0.5")
# # mark_inset(ax3, axins_io, loc1=2, loc2=4, fc="none", ec="0.5")
# #mark_inset(ax3, axins_ot, loc1=2, loc2=4, fc="none", ec="0.5")
# cont = ax4.contour(copy_array_all)#[35:100,200:255])







# #####NUSA

# b = nusa.viable_drop_total
# d = nusa.segmented_drop_files
# c = nusa.in_track_total
# c1 = nusa.io_track_total
# c2 = nusa.ot_track_total
# cp = nusa.in_msd_all
# cp1 = nusa.io_msd_all
# cp2 = nusa.ot_msd_all


# which = "all"
# show = False
# line = True
# scatter = True
# good = 0
# i = 14
# img = mpimg.imread(d[14][1])
# timg_in = ax6.imshow(masked(img,True),cmap=cmap_all,origin = "lower")
# timg = ax7.imshow(masked(img,True),cmap=cmap_all,origin = "lower")
# other_tim = ax8.imshow(masked(img,True),cmap=cmap_all,origin = "lower")
# copy_array_in = np.zeros(np.shape(img))
# copy_array_io = np.zeros(np.shape(img))
# copy_array_ot = np.zeros(np.shape(img))
# copy_array_all = np.zeros(np.shape(img))


# random_choose_c = [0,3]
# random_choose_c1 = [0,3]
# random_choose_c2 = [0,3]
# choose_b = np.random.randint(0,len(b[i]),2)

# in_track_used = []
# io_track_used = []
# ot_track_used = []



# track_x = [] #x values of the track localizations
# track_y = [] #y values of the track localizations
# track_z = [] #msd values of the track localizations


# for j in range(len(b[i])):

#     which_in = 0
#     for l in range(len(c[i][j])):
#         if len(c[i][j][l])!=0:
#             temp = np.array(c[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'b-')
#             if (which == "all" or which == "in") and scatter :
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l])))
#                 #ax8.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "in") and line:
#                 if (good == 0):
#                     if len(temp[0]) != 0:
#                         if which_in == 0:
#                             print(temp[0])
#                             ax6.plot(temp[0],temp[1],c = 'r')
#                             in_track_used = [temp[0],temp[1]]
#                             good = 1
#                             print("good = {0},{1}".format(j,l))
#                         which_in+=1

#     for l in range(len(c1[i][j])):
#         if len(c1[i][j][l])!=0:
#             temp = np.array(c1[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'g-')
#             if (which == "all" or which == "io") and scatter:
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l])))
#                 #ax8.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "io") and line and (j == 0) and (l == 0):
#                 ax6.plot(temp[0],temp[1],c = 'b')
#                 io_track_used = [temp[0],temp[1]]
#     which_out = 0
#     for l in range(len(c2[i][j])):
#         if len(c2[i][j][l])!=0:
#             temp = np.array(c2[i][j][l])
#             copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#             #plt.plot(temp[0],temp[1],'r-')
#             if (which == "all" or which == "out") and scatter:
#                 track_x += list(temp[0])
#                 track_y += list(temp[1])
#                 track_z += list(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l])))
#                 #ax8.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
#             if (which == "all" or which == "out") and line and (j == 0) and (l == 6):
#                 ax6.plot(temp[0],temp[1],c = 'g')
#                 ot_track_used = [temp[0],temp[1]]
#             which_out += 1
#     # if (len(b[i][j])>0):
#     #   for k in range(len(b[i][j])):
#     #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)


# #####sorted scatter points
# idx = np.array(track_z).argsort()[::-1]
# ax8.scatter(np.array(track_x)[idx],np.array(track_y)[idx],s= 2,c = np.array(track_z)[idx],cmap=cmap, norm = norm)


# axins_in = zoomed_inset_axes(ax6, 3, loc=1)
# axins_in.plot(in_track_used[0],in_track_used[1],'r')
# axins_in.set_xlim((np.min(in_track_used[0]) - 1, np.max(in_track_used[0]) + 1 ))
# axins_in.set_ylim((np.min(in_track_used[1]) - 1, np.max(in_track_used[1]) + 1 ))
# axins_in.get_xaxis().set_visible(False)
# axins_in.get_yaxis().set_visible(False)
# axins_io = zoomed_inset_axes(ax6, 3, loc=3)
# axins_io.plot(io_track_used[0],io_track_used[1],'b')
# axins_io.get_xaxis().set_visible(False)
# axins_io.get_yaxis().set_visible(False)
# axins_ot = zoomed_inset_axes(ax6, 3, loc=4)
# axins_ot.plot(ot_track_used[0],ot_track_used[1],'g')
# axins_ot.set_xlim((np.min(ot_track_used[0]) - 1, np.max(ot_track_used[0]) + 1 ))
# axins_ot.set_ylim((np.min(ot_track_used[1]) - 1, np.max(ot_track_used[1]) + 1 ))
# axins_ot.get_xaxis().set_visible(False)
# axins_ot.get_yaxis().set_visible(False)
# mark_inset(ax6, axins_in, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax6, axins_io, loc1=2, loc2=4, fc="none", ec="0.5")
# mark_inset(ax6, axins_ot, loc1=3, loc2=1, fc="none", ec="0.5")
# cont = ax7.contour(copy_array_all)#[75:135,65:140])






















# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax.set_xlim((80,210))
# ax.set_ylim((80,210))
# ax1.get_xaxis().set_visible(False)
# ax1.get_yaxis().set_visible(False)
# ax1.set_xlim((80,210))
# ax1.set_ylim((80,210))
# ax2.get_xaxis().set_visible(False)
# ax2.get_yaxis().set_visible(False)
# ax2.set_xlim((80,210))
# ax2.set_ylim((80,210))
# #obj2 = add_scalebar(ax2) 
# scalebar = ScaleBar(0.13, 'um',location = 1)
# ax2.add_artist(scalebar)



# ax3.get_xaxis().set_visible(False)
# ax3.get_yaxis().set_visible(False)
# ax3.set_xlim((200,280))
# ax3.set_ylim((20,150))

# ax4.get_xaxis().set_visible(False)
# ax4.get_yaxis().set_visible(False)
# ax4.set_xlim((200,280))
# ax4.set_ylim((20,150))

# ax5.get_xaxis().set_visible(False)
# ax5.get_yaxis().set_visible(False)
# ax5.set_xlim((200,280))
# ax5.set_ylim((20,150))
# scalebar = ScaleBar(0.13, 'um',location = 1)
# ax5.add_artist(scalebar)




# ax6.get_xaxis().set_visible(False)
# ax6.get_yaxis().set_visible(False)
# ax6.set_xlim((50,180))
# ax6.set_ylim((50,180))
# ax7.get_xaxis().set_visible(False)
# ax7.get_yaxis().set_visible(False)
# ax7.set_xlim((50,180))
# ax7.set_ylim((50,180))
# ax8.get_xaxis().set_visible(False)
# ax8.get_yaxis().set_visible(False)
# ax8.set_xlim((50,180))
# ax8.set_ylim((50,180))
# scalebar = ScaleBar(0.13, 'um',location = 1)
# ax8.add_artist(scalebar)


# fig.subplots_adjust(wspace=0, hspace=0)

# plt.savefig('11.svg',format="svg")
# plt.show()


# '''
# # plot_msd([a.i_d_tavg,ll.i_d_tavg,rp_ez.i_d_tavg,rp_m9_2.i_d_tavg],4,["Nusa","LACO_LACI","RPOC","RPOC_M9"])

# # plot_msd([a.i_d_tavg,a.io_d_tavg,a.o_d_tavg,total_ll],4,["in","in/out","out","LACO_LACI"],"Nusa_201905")
# # plot_msd([rp_m9.i_d_tavg,rp_m9.io_d_tavg,rp_m9.o_d_tavg,total_ll],4,["in","in/out","out","LACO_LACI"],"Rpoc_M9")

# # plot_msd([rp_m9.i_d_tavg,rp_m9.io_d_tavg,rp_m9.o_d_tavg],3,["in","in/out","out"],"Rpoc_M9")

# # plot_msd([rp.i_d_tavg,rp.io_d_tavg,rp.o_d_tavg],3,["in","in/out","out"],"First")	
# # plot_msd([rp1.i_d_tavg,rp1.io_d_tavg,rp1.o_d_tavg],3,["in","in/out","out"],"RPOC_new")
# # plot_msd([rp2.i_d_tavg,rp2.io_d_tavg,rp2.o_d_tavg],3,["in","in/out","out"],"Files_RPOC")
# # plot_msd([rp3.i_d_tavg,rp3.io_d_tavg,rp3.o_d_tavg],3,["in","in/out","out"],"Other_RPOC")
# # plot_msd([ll.i_d_tavg,ll.io_d_tavg,ll.o_d_tavg],3,["in","in/out","out"],"LACO_LACI")
# # plot_msd([a.i_d_tavg,a.io_d_tavg,a.o_d_tavg],3,["in","in/out","out"],"Nusa_201905")
# # plot_msd([n.i_d_tavg,n.io_d_tavg,n.o_d_tavg],3,["in","in/out","out"],"newer_NUSA")
# # plot_msd([n1.i_d_tavg,n1.io_d_tavg,n1.o_d_tavg],3,["in","in/out","out"],"New_NUSA")
# # plot_msd([n2.i_d_tavg,n2.io_d_tavg,n2.o_d_tavg],3,["in","in/out","out"],"Nusa_201904")

# # plot_msd([a.i_d_tavg,n.i_d_tavg,n1.i_d_tavg,n2.i_d_tavg],4,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904"])
# # plot_msd([rp.i_d_tavg,rp1.i_d_tavg,rp2.i_d_tavg,rp3.i_d_tavg],4,["First","RPOC_new","Files_RPOC","Other_RPOC"])
# #plot_msd_n([a.o_d_tavg,n.o_d_tavg,n1.o_d_tavg,n2.o_d_tavg],4,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904"])
# # plot_msd([rp_m9_2.i_d_tavg,rp_m9_2.io_d_tavg,rp_m9_2.o_d_tavg,total_ll],4,["in","in/out","out","control"],"Rpoc_m9_20190524")
# # plot_msd([rp_ez.i_d_tavg,rp_ez.io_d_tavg,rp_ez.o_d_tavg,total_ll],4,["in","in/out","out","control"],"Rpoc_ez_20190527")

# # #laco_laci com
# # plot_msd([total_ll,total_ll_m9,total_ll_ez,total_ll_ez_h3],4,["Nic's","m9","ez","ez + hex 3%"])
# #plot_msd_n([total_ll,ll_m9n.in_sorted_experiment[0] + ll_m9n.io_sorted_experiment[0] + ll_m9n.ot_sorted_experiment[0],total_ll_ez,total_ll_ez_h3],4,["Nic's","m9","ez","ez + hex 3%"])
# # #rpoc_compare total
# # plot_msd([total_rp,total_rp2,total_rp3,total_rp_ez,total_rp_m9,total_rp_m9_2,total_rp_ez_h3,total_ll],8,["First","second","third","new_ez","old_m9","new_m9","ez_hex 3%","control"])
# # #in only
# # plot_msd([rp.i_d_tavg, rp2.i_d_tavg,rp3.i_d_tavg,rp_ez.i_d_tavg,rp_m9.i_d_tavg,rp_m9_2.i_d_tavg,rp_ez_h3.i_d_tavg, total_ll],8,["First","second","third","new_ez","old_m9","new_m9","ez_hex 3%","control"])
# #plot_msd_n([rp.i_d_tavg, rp2.i_d_tavg,rp3.i_d_tavg,rp_ez.i_d_tavg,rp_m9.i_d_tavg,rp_m9_2.i_d_tavg,rp_ez_h3.i_d_tavg, total_ll],8,["First","second","third","new_ez","old_m9","new_m9","ez_hex 3%","control"])
# #plot_msd_n([rp_m9_2.i_d_tavg,rp_m9_2.io_d_tavg,rp_m9_2.o_d_tavg,total_ll],4,["in","in/out","out","control"],"Rpoc_m9_20190524")

# #plot_msd([ll_m9n.i_d_tavg, ll_m9n.io_d_tavg,ll_m9n.o_d_tavg, total_ll],4,["in","io","out","control"],fig,ax)
# #plot_msd([np.array(ll_m9n.in_sorted_experiment), np.array(ll_m9n.io_sorted_experiment),np.array(ll_m9n.ot_sorted_experiment), total_ll],4,["in","io","out","control"],fig,ax2)






# #chephalexin rpoc no flophore
# #plot_msd_n([ceph_rpoc.i_d_tavg,ceph_rpoc.io_d_tavg,ceph_rpoc.o_d_tavg,total_ll],4,["rpoc_in","rpoc_io","rpoc_out","ll_control"])



# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for i in range(len(rp_ez_h5.in_sorted_experiment)):
# # 	ax.hist(np.log10(np.array(rp_ez_h5.in_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# # ax.legend()

# # plt.show()
# # fig.clear()

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for i in range(len(rp_ez_h5.io_sorted_experiment)):
# # 	ax.hist(np.log10(np.array(rp_ez_h5.io_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# # ax.legend()

# # plt.show()
# # fig.clear()

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for i in range(len(rp_ez_h5.ot_sorted_experiment)):
# # 	ax.hist(np.log10(np.array(rp_ez_h5.ot_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# # ax.legend()

# # plt.show()
# # fig.clear()

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for i in range(len(nh.in_sorted_experiment)):
# # 	ax.hist(np.log10(np.array(nh.in_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# # ax.legend()

# # plt.show()
# # fig.clear()

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for i in range(len(nh.io_sorted_experiment)):
# # 	ax.hist(np.log10(np.array(nh.io_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# # ax.legend()

# # plt.show()
# # fig.clear()

# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # for i in range(len(nh.ot_sorted_experiment)):
# # 	ax.hist(np.log10(np.array(nh.ot_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# # ax.legend()

# # plt.show()
# # fig.clear()


# # plot_msd([total_a,total_n,total_n1,total_n2],4,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904"])
# # plot_msd([total_rp,total_rp1,total_rp2,total_rp3,total_rp_m9],4,["First","RPOC_new","Files_RPOC","RPOC_M9"])

# # plot_msd([list(total_a)+list(total_n)+list(total_n1)+list(total_n2),list(total_rp)+list(total_rp1)+list(total_rp2)+list(total_rp3),list(total_ll),list(total_rp_m9)],4,["Nusa","RPOC","LACO_LACI","RPOC_M9"])
# # plot_msd([list(a.i_d_tavg)+list(n.i_d_tavg)+list(n2.i_d_tavg),list(rp.i_d_tavg)+list(rp1.i_d_tavg)+list(rp2.i_d_tavg)+list(rp3.i_d_tavg),total_ll,list(rp_m9.i_d_tavg)],4,["Nusa","RPOC","LACO_LACI","RPOC_M9"])
# # plot_msd([a.i_d_tavg,n.i_d_tavg,n1.i_d_tavg,n2.i_d_tavg,rp.i_d_tavg,rp1.i_d_tavg,rp2.i_d_tavg,rp3.i_d_tavg,total_ll],9,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904","First","RPOC_new","Files_RPOC","Other_RPOC","LACO_LACI"])


# # plt.hist(np.log10(a.i_d_tavg),label = "in",alpha = 0.3,density = True)
# # plt.hist(np.log10(a.o_d_tavg),label = "out",alpha = 0.3,density= True)
# # plt.hist(np.log10(a.io_d_tavg),label = "in/out",alpha = 0.3,density = True)
# # plt.hist(np.log10(a.unrestricted_msd),label = "BASELINE",alpha = 0.3,density = True)
# # #plt.xscale("log")
# # plt.xlabel("Log10 MSD in um^2/s")
# # plt.legend()
# # plt.show()

# # #plot_msd([rp_ez_h5.in_sorted_experiment[7],rp_ez_h5.io_sorted_experiment[7],rp_ez_h5.ot_sorted_experiment[7],total_ll],4,["in","in/out","out","control"],fig,ax3,"Rpoc_hex")
# # #############
# # #try plotting radius of gyration and msd with distinctions
# # plt.scatter(np.log10(a.i_d_tavg),np.log10(rg),label = "in", alpha = 0.3)
# # plt.scatter(np.log10(a.io_d_tavg),np.log10(rg1),label = "in/out", alpha = 0.3)
# # plt.scatter(np.log10(a.o_d_tavg),np.log10(rg2),label = "out", alpha = 0.03)
# # plt.legend()
# # plt.xlabel("Log10 MSD in um^2/s")
# # plt.ylabel("Log10 Radius of Gyration (um)")
# # plt.title("Radius of Gyration vs MSD with Classification")
# # plt.show()

# # plt.scatter(np.log10(rp_m9.i_d_tavg),np.log10(rp_m9.in_radius_g),label = "in", alpha = 0.3)
# # plt.scatter(np.log10(rp_m9.io_d_tavg),np.log10(rp_m9.io_radius_g),label = "in/out", alpha = 0.3)
# # plt.scatter(np.log10(rp_m9.o_d_tavg),np.log10(rp_m9.ot_radius_g),label = "out", alpha = 0.03)
# # plt.legend()
# # plt.xlabel("Log10 MSD in um^2/s")
# # plt.ylabel("Log10 Radius of Gyration (um)")
# # plt.title("Radius of Gyration vs MSD with Classification")
# # plt.show()

# # plt.scatter(np.log10(con_pix_si(rp_ez.i_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.in_radius_g,which = 'um')),label = "in", alpha = 0.3)
# # plt.scatter(np.log10(con_pix_si(rp_ez.io_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.io_radius_g,which = 'um')),label = "in/out", alpha = 0.3)
# # plt.scatter(np.log10(con_pix_si(rp_ez.o_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.ot_radius_g,which = 'um')),label = "out", alpha = 0.03)
# # plt.legend()
# # plt.xlabel("Log10 MSD in um^2/s")
# # plt.ylabel("Log10 Radius of Gyration (um)")
# # plt.title("Radius of Gyration vs MSD with Classification")
# # plt.show()


# # plt.scatter(np.log10(con_pix_si(lI_ez.i_d_tavg,which = 'msd')),np.log10(con_pix_si(lI_ez.in_radius_g,which = 'um')),label = "in", alpha = 0.3)
# # plt.scatter(np.log10(con_pix_si(lI_ez.io_d_tavg,which = 'msd')),np.log10(con_pix_si(lI_ez.io_radius_g,which = 'um')),label = "in/out", alpha = 0.3)
# # plt.scatter(np.log10(con_pix_si(lI_ez.o_d_tavg,which = 'msd')),np.log10(con_pix_si(lI_ez.ot_radius_g,which = 'um')),label = "out", alpha = 0.03)
# # plt.legend()
# # plt.xlabel("Log10 MSD in um^2/s")
# # plt.ylabel("Log10 Radius of Gyration (um)")
# # plt.title("Radius of Gyration vs MSD with Classification")
# # plt.show()

# # #compare the in fraction with laco-laci and rpoc
# # plt.scatter(np.log10(rp.i_d_tavg),np.log10(rp.in_radius_g),label = "in_RPOC", alpha = 0.1)
# # plt.scatter(np.log10(ll.i_d_tavg),np.log10(ll.in_radius_g),label = "in_LACO_LACI", alpha = 0.3)
# # plt.scatter(np.log10(a.i_d_tavg),np.log10(a.in_radius_g),label = "in_NUSA", alpha = 0.1)
# # plt.legend()
# # plt.xlabel("Log10 MSD in um^2/s")
# # plt.ylabel("Log10 Radius of Gyration (um)")
# # plt.title("Radius of Gyration vs MSD with Classification")
# # plt.show()

# # #compare the in fraction with laco-laci and rpoc
# # plt.scatter(np.log10(rp.i_d_tavg),np.log10(rp.in_radius_g),label = "in_RPOC", alpha = 0.3)
# plt.scatter(np.log10(ll.i_d_tavg),np.log10(ll.in_radius_g),label = "in_LACO_LACI", alpha = 0.3)
# #plt.scatter(np.log10(a.i_d_tavg),np.log10(a.in_radius_g),label = "in_NUSA", alpha = 0.3)
# plt.scatter(np.log10(rp_m9.i_d_tavg),np.log10(rp_m9.in_radius_g),label = "in_RPOC_M9", alpha = 0.3)
# plt.legend()
# plt.xlabel("Log10 MSD in um^2/s")
# plt.ylabel("Log10 Radius of Gyration (um)")
# plt.title("Radius of Gyration vs MSD with Classification")
# plt.show()


# ###radius of gyration + end to end + msd
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(a.i_d_tavg),np.log10(rg),np.log10(a.in_ete),label = "in", alpha = 0.3)
# ax.scatter(np.log10(a.io_d_tavg),np.log10(rg1),np.log10(a.io_ete),label = "in/out", alpha = 0.3)
# ax.scatter(np.log10(a.o_d_tavg),np.log10(rg2),np.log10(a.ot_ete),label = "out", alpha = 0.03)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# def init():
#     ax.scatter(np.log10(con_pix_si(rp_ez.i_d_tavg,which ='msd')),np.log10(con_pix_si(rp_ez.in_radius_g,which = 'um')),np.log10(con_pix_si(rp_ez.in_ete,which = 'um')),label = "in", alpha = 0.3)
#     ax.scatter(np.log10(con_pix_si(rp_ez.io_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.io_radius_g,which = 'um')),np.log10(con_pix_si(rp_ez.io_ete,which = 'um')),label = "in/out", alpha = 0.3)
#     ax.scatter(np.log10(con_pix_si(rp_ez.o_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.ot_radius_g,which = 'um')),np.log10(con_pix_si(rp_ez.ot_ete,which = 'um')),label = "out", alpha = 0.03)
#     ax.set_xlabel("Log10 MSD in um^2/s")
#     ax.set_ylabel("Log10 Radius of Gyration (um)")
#     ax.set_zlabel("Log10 End to End Distance (um)")
#     ax.legend()
#     return fig,
# def animate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig,
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# writer=animation.FFMpegWriter(bitrate=5000,fps = 60)
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=360, interval=20, blit=True)
# # Save
# anim.save('basic_animation.mp4', writer = writer)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(rp.i_d_tavg),np.log10(rp.in_radius_g),np.log10(rp.in_ete),label = "in_RPOC", alpha = 0.1)
# ax.scatter(np.log10(ll.i_d_tavg),np.log10(ll.in_radius_g),np.log10(ll.in_ete),label = "in_LACO_LACI", alpha = 0.3)
# ax.scatter(np.log10(a.i_d_tavg),np.log10(a.in_radius_g),np.log10(a.in_ete),label = "in_NUSA", alpha = 0.1)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(total_rp_ez),np.log10(np.array(rp_ez.in_radius_g + rp_ez.io_radius_g + rp_ez.ot_radius_g)),np.log10(np.array(rp_ez.in_ete + rp_ez.io_ete + rp_ez.ot_ete)),label = "RPOC", alpha = 0.1)
# ax.scatter(np.log10(total_ll),np.log10(np.array(ll.in_radius_g + ll.io_radius_g + ll.ot_radius_g)),np.log10(np.array(ll.in_ete + ll.io_ete + ll.ot_ete)),label = "LACO_LACI", alpha = 0.3)
# ax.scatter(np.log10(total_a),np.log10(np.array(a.in_radius_g + a.io_radius_g + a.ot_radius_g)),np.log10(np.array(a.in_ete + a.io_ete + a.ot_ete)),label = "NUSA", alpha = 0.1)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()




# total_len = len(a.i_d_tavg) + len(a.io_d_tavg) + len(a.o_d_tavg)
# in_len = float(len(a.i_d_tavg))/total_len
# io_len = float(len(a.io_d_tavg))/total_len
# out_len = float(len(a.o_d_tavg))/total_len


# #Dapi controls
# #rpoc_noflorophore + dapi at 0.1 ug/ml
# path_rp_dapi = '/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/dapi_2/20190820/rpoc_nofl_dapi_0.1'
# #hup_mCherry + dapi at 0.1 ug/ml
# path_hup_mCherry_dapi = '/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/dapi_2/20190820/hup_mcherry_dapi_0.1'


# #read dapi+rpoc_nofl
# rp_dapi_files = glob.glob(path_rp_dapi + "/" + "rpoc_fapi_" + "**.tif")
# #read hup_mcherry + dapi
# hup_mcherry_files = glob.glob(path_hup_mCherry_dapi + "/" + "hup_mcherry_" + "**.tif")


# for i in rp_dapi_files:
# 	img_read = read_imag(i,show = 0)
# 	contour_intens(img_read,perc = 99)
# for i in hup_mcherry_files:
# 	img_read = read_imag(i,show = 0)
# 	contour_intens(img_read,perc = 99)
'''


# for i in range(len(e1)):
# 	for j in range(len(e1[i])):
# 		if len(e1[i][j]) != 0:
# 			plt.plot(e1[i][j],np.ones(len(e1[i][j]))*j,'b.')
# 		if len(e2[i][j]) != 0:
# 			plt.plot(e2[i][j],np.ones(len(e2[i][j]))*j,'y.')
# 		if len(e3[i][j]) != 0:
# 			plt.plot(e3[i][j],np.ones(len(e3[i][j]))*j,'r.')

# 	plt.xscale("log")
# 	plt.xlabel("MSD in um^2/s")
# 	plt.ylabel("Frame Subset (index from 0)")
# 	plt.legend(["Blue: in","Yellow: In/Out","Red: Out"])
# 	plt.show()


weights = np.ones_like(np.array(rp_ez.i_d_tavg))/float(len(np.array(rp_ez.i_d_tavg)))
weights1 = np.ones_like(np.array(rp_ez.io_d_tavg))/float(len(np.array(rp_ez.io_d_tavg)))
weights2 = np.ones_like(np.array(rp_ez.o_d_tavg))/float(len(np.array(rp_ez.o_d_tavg)))
weights3 = np.ones_like(total_rp_ez)/float(len(total_rp_ez))
weights4 = np.ones_like(total_ll)/float(len(total_ll))
weights00 = np.ones_like(np.array(rp_3_5.i_d_tavg))/float(len(np.array(rp_3_5.i_d_tavg)))
weights11 = np.ones_like(np.array(rp_3_5.io_d_tavg))/float(len(np.array(rp_3_5.io_d_tavg)))
weights21 = np.ones_like(np.array(rp_3_5.o_d_tavg))/float(len(np.array(rp_3_5.o_d_tavg)))
weights31 = np.ones_like(total_rp_3_5)/float(len(total_rp_3_5))

weights000 = np.ones_like(np.array(rp_3_5_2.i_d_tavg))/float(len(np.array(rp_3_5_2.i_d_tavg)))
weights110 = np.ones_like(np.array(rp_3_5_2.io_d_tavg))/float(len(np.array(rp_3_5_2.io_d_tavg)))
weights210 = np.ones_like(np.array(rp_3_5_2.o_d_tavg))/float(len(np.array(rp_3_5_2.o_d_tavg)))
weights310 = np.ones_like(total_rp_3_5_2)/float(len(total_rp_3_5_2))

# plt.hist(np.log10(con_pix_si(np.array(rp_ez.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights)
# plt.hist(np.log10(con_pix_si(np.array(rp_ez.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights1)
# plt.hist(np.log10(con_pix_si(np.array(rp_ez.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights2)

# plt.hist(np.log10(con_pix_si(np.array(total_ll),which = 'msd')),label = "OUT",alpha = 0.5,weights=weights4, color= (245./255.,181./255.,183./255.))

# plt.hist(np.log10(con_pix_si(np.array(rp_3_5.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.5,weights=weights00,color = (175./255.,207./255.,227./255.))
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.5,weights=weights11,color = (255./255.,210./255.,176./255.))
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.5,weights=weights21,color = (179./255.,222./255.,185./255.))

plt.hist(np.log10(con_pix_si(np.array(rp_3_5_2.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights000)
plt.hist(np.log10(con_pix_si(np.array(rp_3_5_2.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights110)
plt.hist(np.log10(con_pix_si(np.array(rp_3_5_2.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights210)
plt.hist(np.log10(con_pix_si(np.array(total_ll),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights4)

# plt.hist(np.log10(con_pix_si(np.array(rp_3_5.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights00)
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights11)
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights21)
plt.xlabel('$log\\left( \\frac{um^2}{s}\\right) $')
plt.ylabel('Probability')
plt.ylabel('Probability')
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5_2.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights000)
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5_2.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights110)
# plt.hist(np.log10(con_pix_si(np.array(rp_3_5_2.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights210)
plt.show()

# drop_color = ["y","b","r","g","m"]



fig = plt.figure()
ax = plt.subplot(211)
ax1 = plt.subplot(212, sharex = ax)

weights = np.ones_like(np.array(rp_ez.i_d_tavg))/float(len(np.array(rp_ez.i_d_tavg)))
weights1 = np.ones_like(np.array(rp_ez.io_d_tavg))/float(len(np.array(rp_ez.io_d_tavg)))
weights2 = np.ones_like(np.array(rp_ez.o_d_tavg))/float(len(np.array(rp_ez.o_d_tavg)))
weights3 = np.ones_like(total_rp_ez)/float(len(total_rp_ez))
weights4 = np.ones_like(total_ll)/float(len(total_ll))

ax.hist(np.log10(con_pix_si(np.array(rp_ez.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights, bins = 12)
ax.hist(np.log10(con_pix_si(np.array(rp_ez.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights1)
ax.hist(np.log10(con_pix_si(np.array(rp_ez.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights2)
ax.hist(np.log10(con_pix_si(np.array(total_ll),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights4)
ax1.hist(np.log10(con_pix_si(total_rp_ez,which = 'msd')),label = "control",alpha = 0.3,weights=weights3)
ax1.hist(np.log10(con_pix_si(np.array(total_ll),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights4)

#ax.set_xlabel('$log\\left( \\frac{um^2}{s}\\right) $')
ax1.set_xlabel('$log\\left( \\frac{um^2}{s}\\right) $')
ax.set_ylabel('Probability')
ax1.set_ylabel('Probability')

plt.show()

fig = plt.figure()
ax = plt.subplot(321)
ax1 = plt.subplot(323, sharex = ax, sharey = ax)
ax2 = plt.subplot(325, sharex = ax, sharey = ax)


ax3 = plt.subplot(322)
ax4 = plt.subplot(324, sharex = ax, sharey = ax)
ax5 = plt.subplot(326, sharex = ax, sharey = ax)



weights = np.ones_like(np.array(rp_ez.i_d_tavg))/float(len(np.array(rp_ez.i_d_tavg)))
weights1 = np.ones_like(np.array(rp_ez.io_d_tavg))/float(len(np.array(rp_ez.io_d_tavg)))
weights2 = np.ones_like(np.array(rp_ez.o_d_tavg))/float(len(np.array(rp_ez.o_d_tavg)))
weights3 = np.ones_like(total_ll)/float(len(total_ll))

weights4 = np.ones_like(np.array(rp_m9_2.i_d_tavg))/float(len(np.array(rp_m9_2.i_d_tavg)))
weights5 = np.ones_like(np.array(rp_m9_2.io_d_tavg))/float(len(np.array(rp_m9_2.io_d_tavg)))
weights6 = np.ones_like(np.array(rp_m9_2.o_d_tavg))/float(len(np.array(rp_m9_2.o_d_tavg)))
weights7 = np.ones_like(ll_m9.i_d_tavg)/float(len(ll_m9.i_d_tavg))

weights8 = np.ones_like(np.array(rp_ez_h5.i_d_tavg))/float(len(np.array(rp_ez_h5.i_d_tavg)))
weights9 = np.ones_like(np.array(rp_ez_h5.io_d_tavg))/float(len(np.array(rp_ez_h5.io_d_tavg)))
weights01 = np.ones_like(np.array(rp_ez_h5.o_d_tavg))/float(len(np.array(rp_ez_h5.o_d_tavg)))
weights11 = np.ones_like(np.array(ll_ez_h3.i_d_tavg))/float(len(np.array(ll_ez_h3.i_d_tavg)))


weights12 = np.ones_like(np.array(a.i_d_tavg))/float(len(np.array(a.i_d_tavg)))
weights13 = np.ones_like(np.array(a.io_d_tavg))/float(len(np.array(a.io_d_tavg)))
weights14 = np.ones_like(np.array(a.o_d_tavg))/float(len(np.array(a.o_d_tavg)))


weights15 = np.ones_like(np.array(na_m9.i_d_tavg))/float(len(np.array(na_m9.i_d_tavg)))
weights16 = np.ones_like(np.array(na_m9.io_d_tavg))/float(len(np.array(na_m9.io_d_tavg)))
weights17 = np.ones_like(np.array(na_m9.o_d_tavg))/float(len(np.array(na_m9.o_d_tavg)))


weights18 = np.ones_like(np.array(nh.i_d_tavg))/float(len(np.array(nh.i_d_tavg)))
weights19 = np.ones_like(np.array(nh.io_d_tavg))/float(len(np.array(nh.io_d_tavg)))
weights20 = np.ones_like(np.array(nh.o_d_tavg))/float(len(np.array(nh.o_d_tavg)))




ax.hist(np.log10(con_pix_si(np.array(rp_ez.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights)
ax.hist(np.log10(con_pix_si(np.array(rp_ez.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights1)
ax.hist(np.log10(con_pix_si(np.array(rp_ez.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights2)
ax.hist(np.log10(con_pix_si(total_ll,which = 'msd')),label = "control",alpha = 0.3,weights=weights3)

ax1.hist(np.log10(con_pix_si(np.array(rp_m9_2.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights4)
ax1.hist(np.log10(con_pix_si(np.array(rp_m9_2.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights5)
ax1.hist(np.log10(con_pix_si(np.array(rp_m9_2.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights6)
ax1.hist(np.log10(con_pix_si(ll_m9.i_d_tavg,which = 'msd')),label = "control",alpha = 0.3,weights=weights7)

ax2.hist(np.log10(con_pix_si(np.array(rp_ez_h5.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights8)
ax2.hist(np.log10(con_pix_si(np.array(rp_ez_h5.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights9)
ax2.hist(np.log10(con_pix_si(np.array(rp_ez_h5.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights01)
ax2.hist(np.log10(con_pix_si(np.array(ll_ez_h3.i_d_tavg),which = 'msd')),label = "control",alpha = 0.3,weights=weights11)





ax3.hist(np.log10(con_pix_si(np.array(a.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights12)
ax3.hist(np.log10(con_pix_si(np.array(a.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights13)
ax3.hist(np.log10(con_pix_si(np.array(a.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights14)
ax3.hist(np.log10(con_pix_si(total_ll,which = 'msd')),label = "control",alpha = 0.3,weights=weights3)

ax4.hist(np.log10(con_pix_si(np.array(na_m9.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights15)
ax4.hist(np.log10(con_pix_si(np.array(na_m9.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights16)
ax4.hist(np.log10(con_pix_si(np.array(na_m9.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights17)
ax4.hist(np.log10(con_pix_si(ll_m9.i_d_tavg,which = 'msd')),label = "control",alpha = 0.3,weights=weights7)

ax5.hist(np.log10(con_pix_si(np.array(nh.i_d_tavg),which = 'msd')),label = "IN",alpha = 0.3,weights=weights18)
ax5.hist(np.log10(con_pix_si(np.array(nh.io_d_tavg),which = 'msd')),label = "IN/OUT",alpha = 0.3,weights=weights19)
ax5.hist(np.log10(con_pix_si(np.array(nh.o_d_tavg),which = 'msd')),label = "OUT",alpha = 0.3,weights=weights20)
ax5.hist(np.log10(con_pix_si(np.array(ll_ez_h3.i_d_tavg),which = 'msd')),label = "control",alpha = 0.3,weights=weights11)


#plt.legend()

ax5.set_xlabel('$log\\left( \\frac{um^2}{s}\\right) $')
#ax6.set_xlabel('$log\\left( \\frac{um^2}{s}\\right) $')

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
#ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
#ax3.get_xaxis().set_visible(False)
#ax3.get_yaxis().set_visible(False)
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
#ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)


fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig('2.svg',format="svg")
plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(111)


# weights = np.ones_like(np.array(total_rp_ez))/float(len(np.array(total_rp_ez)))
# weights2 = np.ones_like(total_ll)/float(len(total_ll))



# weightsr1 = np.ones_like(np.array(rp_ez.i_d_tavg))/float(len(np.array(total_rp_ez)))



# ax.hist(np.log10(con_pix_si(np.array(total_rp_ez),which = 'msd')),color='red',label = "RPOC_EZ",alpha = 0.3,weights=weights)
# ax.hist(np.log10(con_pix_si(np.array(rp_ez.i_d_tavg),which = 'msd')),color='red',label = "RPOC_EZ",alpha = 0.3,weights=weightsr1)

# ax.hist(np.log10(con_pix_si(total_ll,which = 'msd')),color='blue',label = "control_EZ",alpha = 0.3,weights=weights2)
# ax.set_xlabel('$log\\left( \\frac{um^2}{s}\\right) $')

# ax.set_ylabel('Probability')

# fig.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('3.svg',format="svg")
# plt.show()





#########
#stacked bar plots

#from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots

def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    
    # Add super ylabel or xlabel to the figure
    # Similar to matplotlib.suptitle
    # axis       - string: "x" or "y"
    # label      - string
    # label_prop - keyword dictionary for Text
    # labelpad   - padding from the axis (default: 5)
    # ha         - horizontal alignment (default: "center")
    # va         - vertical alignment (default: "center")

    
    fig = plt.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None: 
        label_prop = dict()
    plt.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)






















# rp_bar = (1./len(total_rp_ez)) * np.array([len(rp_ez.i_d_tavg),len(rp_ez.io_d_tavg),len(rp_ez.o_d_tavg)]) * 100.
# ll_bar = (1./len(total_ll_ez)) * np.array([len(ll_ez.i_d_tavg),len(ll_ez.io_d_tavg),len(ll_ez.o_d_tavg)]) * 100.
# na_bar = (1./len(total_n)) * np.array([len(n.i_d_tavg),len(n.io_d_tavg),len(n.o_d_tavg)]) * 100.


# rp_barm = (1./len(total_rp_m9_2)) * np.array([len(rp_m9_2.i_d_tavg),len(rp_m9_2.io_d_tavg),len(rp_m9_2.o_d_tavg)]) * 100.
# ll_barm = (1./len(total_ll_m9)) * np.array([len(ll_m9.i_d_tavg),len(ll_m9.io_d_tavg),len(ll_m9.o_d_tavg)]) * 100.
# na_barm = (1./len(total_na_m9)) * np.array([len(na_m9.i_d_tavg),len(na_m9.io_d_tavg),len(na_m9.o_d_tavg)]) * 100.


# rp_barh = (1./len(total_rp_ez_h3)) * np.array([len(rp_ez_h3.i_d_tavg),len(rp_ez_h3.io_d_tavg),len(rp_ez_h3.o_d_tavg)]) * 100.
# ll_barh = (1./len(total_ll_ez_h3)) * np.array([len(ll_ez_h3.i_d_tavg),len(ll_ez_h3.io_d_tavg),len(ll_ez_h3.o_d_tavg)]) * 100.
# na_barh = (1./len(total_nh)) * np.array([len(nh.i_d_tavg),len(nh.io_d_tavg),len(nh.o_d_tavg)]) * 100.


# fig = plt.figure()

# ax = fig.add_subplot(311)
# ax1 = fig.add_subplot(312,sharey = ax)
# ax2 = fig.add_subplot(313,sharey = ax)

# r = [0,1,2]

# new_r = [rp_bar[0],rp_barm[0],rp_barh[0]]
# new_r1 = [rp_bar[1],rp_barm[1],rp_barh[1]]
# new_r2 = [rp_bar[2],rp_barm[2],rp_barh[2]]
# names = ('RPOC EZ', 'RPOC M9', 'RPOC 5% HEX')
# ax.bar(r,new_r)
# ax.bar(r,new_r1,bottom=new_r)
# ax.bar(r,new_r2,bottom=[i+j for i,j in zip(new_r, new_r1)])
# ax.set_xticks(r,names)




# new_r = [ll_bar[0],ll_barm[0],ll_barh[0]]
# new_r1 = [ll_bar[1],ll_barm[1],ll_barh[1]]
# new_r2 = [ll_bar[2],ll_barm[2],ll_barh[2]]
# names = ('LL EZ', 'LL M9', 'LL 5% HEX')
# ax1.bar(r,new_r)
# ax1.bar(r,new_r1,bottom=new_r)
# ax1.bar(r,new_r2,bottom=[i+j for i,j in zip(new_r, new_r1)])
# ax1.set_xticks(r,names)



# new_r = [na_bar[0],na_barm[0],na_barh[0]]
# new_r1 = [na_bar[1],na_barm[1],na_barh[1]]
# new_r2 = [na_bar[2],na_barm[2],na_barh[2]]
# names = ('LL EZ', 'LL M9', 'LL 5% HEX')
# ax2.bar(r,new_r)
# ax2.bar(r,new_r1,bottom=new_r)
# ax2.bar(r,new_r2,bottom=[i+j for i,j in zip(new_r, new_r1)])
# ax2.set_xticks(r,names)






# suplabel('y','Percentages')
# plt.savefig('4.svg',format="svg")
# plt.show()




def overall_plot2D_contour(op, which = "all", scatter = 0, line = 0):

	b = op.viable_drop_total
	d = op.segmented_drop_files
	c = op.in_track_total
	c1 = op.io_track_total
	c2 = op.ot_track_total
	cp = op.in_msd_all
	cp1 = op.io_msd_all
	cp2 = op.ot_msd_all

	for i in range(len(b)):

		if len(d[i]) != 0:
			img = mpimg.imread(d[i][0])
			timg = plt.imshow(img,cmap=plt.get_cmap('gray'),origin = "lower")
			copy_array_in = np.zeros(np.shape(img))
			copy_array_io = np.zeros(np.shape(img))
			copy_array_ot = np.zeros(np.shape(img))
			copy_array_all = np.zeros(np.shape(img))


		random_choose_c = [0,3]
		random_choose_c1 = [0,3]
		random_choose_c2 = [0,3]
		choose_b = np.random.randint(0,len(b[i]),2)
		for j in range(len(b[i])):

			for l in range(len(c[i][j])):
				if len(c[i][j][l])!=0:
					temp = np.array(c[i][j][l])
					copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
					copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
					#plt.plot(temp[0],temp[1],'b-')
					if (which == "all" or which == "in") and scatter :
						plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
					if (which == "all" or which == "in") and line and (l in random_choose_c) and (j in choose_b):
						plt.plot(temp[0],temp[1],c = 'r')

			for l in range(len(c1[i][j])):
				if len(c1[i][j][l])!=0:
					temp = np.array(c1[i][j][l])
					copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
					copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
					#plt.plot(temp[0],temp[1],'g-')
					if (which == "all" or which == "io") and scatter:
						plt.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
					if (which == "all" or which == "in") and line and l in random_choose_c1:
						plt.plot(temp[0],temp[1],c = 'b')

			for l in range(len(c2[i][j])):
				if len(c2[i][j][l])!=0:
					temp = np.array(c2[i][j][l])
					copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
					copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
					#plt.plot(temp[0],temp[1],'r-')
					if (which == "all" or which == "out") and scatter:
						plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
					if (which == "all" or which == "in") and line and l in random_choose_c2:
						plt.plot(temp[0],temp[1],c = 'g')
			# if (len(b[i][j])>0):
			# 	for k in range(len(b[i][j])):
			# 		circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
		plt.contour(copy_array_all)
		plt.colorbar()
		#plt.savefig("Frame_{0}".format(i))
		plt.show()
	return


# def animate(i,ax):
#     # azimuth angle : 0 deg to 360 deg
#     ax.view_init(elev=i, azim=i*4)

#     return 



# def overall_plot3D(op,which = "all"):

# 	b = op.viable_drop_total
# 	d = op.segmented_drop_files
# 	c = op.in_track_total
# 	c1 = op.io_track_total
# 	c2 = op.ot_track_total
# 	cp = op.in_msd_all
# 	cp1 = op.io_msd_all
# 	cp2 = op.ot_msd_all


# 	for i in range(len(b)):
# 		fig = plt.figure()
# 		ax = fig.add_subplot(111,projection = '3d')
# 		if len(d[i]) != 0:
# 			img = mpimg.imread(d[i][0])
# 			#timg = ax2.imshow(img, cmap=plt.get_cmap('gray'))
		
# 		for j in range(len(b[i])):
# 			if which == "all" or which == "in":
# 				for l in range(len(c[i][j])):
# 					if len(c[i][j][l])!=0:
# 						temp = np.array(c[i][j][l])
# 						#plt.plot(temp[0],temp[1],'b-')
# 						im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
# 			if which == "all" or which == "io":
# 				for l in range(len(c1[i][j])):
# 					if len(c1[i][j][l])!=0:
# 						temp = np.array(c1[i][j][l])
# 						#plt.plot(temp[0],temp[1],'g-')
# 						im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
# 			if which == "all" or which == "out":
# 				for l in range(len(c2[i][j])):
# 					if len(c2[i][j][l])!=0:
# 						temp = np.array(c2[i][j][l])
# 						#plt.plot(temp[0],temp[1],'r-')
# 						im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

# 			# if (len(b[i][j])>0):
# 			# 	for k in range(len(b[i][j])):
# 			# 		circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# 		#fig.colorbar(im,ax=ax3)
# 		#plt.savefig("Frame_{0}".format(i))
# 		#fig.show()
# 		ani = animation.FuncAnimation(fig, animate,fargs = [ax],frames=180, interval=50)
# 		fn = op.wd + "{0}".format(i)
# 		ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
# 		ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)
# 	return


# def spacialplot_msd(op):
# 	x = op.all_tracks_x
# 	y = op.all_tracks_y
# 	z = op.all_msd
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(x, y, z, marker = '.',alpha = 0.3)
# 	plt.show()

# 	return

#overall_plot(rp2.viable_drop_total,rp2.segmented_drop_files,rp2.in_track_total ,rp2.io_track_total ,rp2.ot_track_total,rp2.in_msd_all, rp2.io_msd_all, rp2.ot_msd_all)

# def other_plot(op):
# 	fraction_tick = [i for i in range(1,int(op.frame_total/op.frame_step)+1)]
# 	create_box_plot(op.tmframe_occ,fraction_tick,y_label = "Fraction inside the drop",x_label = "Frame number",y_lim = (),title = "Percent Occupation of Track in Drop per Frame Over All Experiments")

# 	for i in op.tmframe_occ:

# 		w_i = np.ones_like(i)/float(len(i))
# 		plt.hist(i,histtype = 'step',weights=w_i)
# 		plt.xlabel("Fraction inside the drop")
# 		plt.ylabel("Probability")
# 		plt.title("Percent Occupation of Track in Drop per Frame")
# 	plt.show()
# 	return
#other_plot(rp2)







which = 0
def track_plot(what,which,a_in=0.3,a_io=0.2,a_ot=0.1):
    for i in range(len(what.in_track_total[which])):
        for j in range(len(what.in_track_total[which][i])):
            plt.plot(what.in_track_total[which][i][j][0],what.in_track_total[which][i][j][1],'-r',alpha = a_in,linewidth = 1)
            

    for i in range(len(what.io_track_total[which])):
        for j in range(len(what.io_track_total[which][i])):
            plt.plot(what.io_track_total[which][i][j][0],what.io_track_total[which][i][j][1],'-b',alpha = a_io,linewidth = 1)


    for i in range(len(what.ot_track_total[which])):
        for j in range(len(what.ot_track_total[which][i])):
            plt.plot(what.ot_track_total[which][i][j][0],what.ot_track_total[which][i][j][1],'-g',alpha = a_ot,linewidth = 1)

    return


#fitting displacement curves: 

def fit_dist(type,gaus = "both"):

    collected_dist = []

    collected_dist += type.in_distances_x
    collected_dist += type.io_distances_x
    collected_dist += type.ot_distances_x
    collected_dist += type.in_distances_y
    collected_dist += type.io_distances_y
    collected_dist += type.ot_distances_y

    weights = np.ones_like(np.array(collected_dist))/float(len(np.array(collected_dist)))

    hist, bins = np.histogram(con_pix_si(np.array(collected_dist),which ='um'),weights = weights,bins = 20)
    bin_middle = bins[:-1] + np.diff(bins)/2.
    plt.bar(bin_middle,hist)
    plt.show()
    plt.plot(bin_middle,hist,'.r')
    plt.show()

    if gaus == "both":
        #plt.bar(bin_middle,hist)
        popt1,pcov1 = curve_fit(gaus1D,bin_middle,hist,maxfev = 100000)
        popt2,pcov2 = curve_fit(gaus2D,bin_middle,hist,maxfev = 100000)


        plt.plot(np.linspace(bin_middle[0],bin_middle[-1],100),gaus1D(np.linspace(bin_middle[0],bin_middle[-1],100),popt1[0],popt1[1],popt1[2]))
        plt.show()
        return [popt1,pcov1,popt2,pcov2]


'''





