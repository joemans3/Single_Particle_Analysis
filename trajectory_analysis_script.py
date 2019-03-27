import os
import sys
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit
import glob
from sklearn import mixture

from plotting_functions import *
from Analysis_functions import *
from draw_circle import *

import pdb


class run_analysis:

	def __init__(self,wd,t_string):
		self.wd = wd
		self.t_string = t_string


		self.frame_step = 1 #change manual
		self.frame_total = 1 #change manual
		#cutoff for track length
		self.t_len_l = 0 #change manual #people use 5
		self.t_len_u = 0 #change manual #100 #use 20
		self.MSD_avg_threshold = 0
		#upper and lower "bound proportion" threshold for determining if in_out track or out only.
		self.upper_bp = 0
		self.lower_bp = 0
		self.max_track_decomp = 0
		self.conversion_p_nm = 0
		self.minimum_tracks_per_drop = 0
		self.minimum_percent_per_drop_in = 0.50
		self.frames = int(self.frame_total/self.frame_step)




		self.occupency_per_drop__per_frame_per_experiment = []
		self.tmframe_occ = [[] for i in range(self.frames)] #occupation per frame step for all experiments 

		#averaged MSD (time) over all frame steps
		self.i_d_tavg = []
		self.o_d_tavg = []
		self.io_d_tavg = []

		self.i_d_tavg_x = []
		self.o_d_tavg_x = []
		self.io_d_tavg_x = []

		self.i_d_tavg_y = []
		self.o_d_tavg_y = []
		self.io_d_tavg_y = []

		#Lengths of tracks

		self.in_length = []
		self.out_length =[]
		self.inout_length = []


		self.in_dist = []
		self.ot_dist = [] 
		self.io_dist = []

		self.minimum_center = []
		    
		    
		self.minimum_msd = []

		#averaged MSD (time) per frame step
		self.in_drop = [[] for i in range(self.frames)]
		self.out_drop = [[] for i in range(self.frames)]
		self.in_out_drop = [[] for i in range(self.frames)]

		self.in_drop_x = [[] for i in range(self.frames)]
		self.out_drop_x = [[] for i in range(self.frames)]
		self.in_out_drop_x = [[] for i in range(self.frames)]

		self.in_drop_y = [[] for i in range(self.frames)]
		self.out_drop_y = [[] for i in range(self.frames)]
		self.in_out_drop_y = [[] for i in range(self.frames)]

		#single track decomposition per frame step over all experiments
		self.max_in = np.zeros(self.frames)
		self.max_out = np.zeros(self.frames)
		self.max_io = np.zeros(self.frames)


		self.in_msd_track = [[] for i in range(self.frames)]
		self.out_msd_track = [[] for i in range(self.frames)]
		self.io_msd_track = [[] for i in range(self.frames)]

		self.in_msd_track_x = [[] for i in range(self.frames)]
		self.out_msd_track_x = [[] for i in range(self.frames)]
		self.io_msd_track_x = [[] for i in range(self.frames)]

		self.in_msd_track_y = [[] for i in range(self.frames)]
		self.out_msd_track_y = [[] for i in range(self.frames)]
		self.io_msd_track_y = [[] for i in range(self.frames)]
		self.distances = []


		self.in_tracksf = []
		self.out_tracksf = []
		self.io_tracksf = []

		self.in_track_total = []
		self.io_track_total = []
		self.ot_track_total = []

		self.viable_drop_total = []

		self.segmented_drop_files = []

		self.in_msd_all = []
		self.ot_msd_all = []
		self.io_msd_all = []
	def read_track_data(self,wd,t_string):
		'''
		wd: this is the current woring directory for the dataset you are interested in.
		t_string: Eg. NUSA, nusA, rpoC, RPOC etc.
		'''

		cd = wd

		all_files = sorted(glob.glob(cd + "/Analysis/" + t_string + "_**.tif_spots.csv"))
		print all_files
		max_tag = np.max([len(i) for i in all_files])

		tracks = []
		drops = []
		segf = []
		for pp in all_files:

		    test = np.loadtxt("{0}".format(pp),delimiter=",")
		    tracks.append(test)
		    if len(pp) == max_tag:
		        tag = pp[len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+2]
		    else: 
		        tag = pp[len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+1]

		    drop_files = sorted(glob.glob("{0}/Segmented/Analysis/*_".format(cd)+t_string+"_{0}_seg.tif_spots.csv".format(tag)))
		    seg_files = sorted(glob.glob("{0}/raw_pic/Segmented/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag)))
		    segf.append(seg_files)
		    point_data = []
		    print(drop_files)
		    for i in drop_files:
		        point_data.append(np.loadtxt("{0}".format(i),delimiter=",",usecols=(0,1,2),skiprows=1))
		    drops.append(point_data)
		self.segmented_drop_files = segf
		return [tracks,drops]

	def convert_track_frame(self,track_set):
		'''
		track_set: the set of tracks for one specific frame of reference
		This function preps the data such that the tracks satisfy a length
		and segregates the data in respect to the frame step.
		'''
		track_ID = track_set[:,0]
		frame_ID = track_set[:,1]
		x_ID = track_set[:,2]
		y_ID = track_set[:,3]
		intensity_ID = track_set[:,4]

		tp=[]
		tp_x=[]
		tp_y=[]
		tp_intensity=[]
		fframe_ID = []
		for i in np.arange(0,self.frame_total,self.frame_step):
			a=(i<frame_ID) & (frame_ID<(i + self.frame_step))
			fframe_ID.append(frame_ID[a])
			tp.append(track_ID[a])
			tp_x.append(x_ID[a])
			tp_y.append(y_ID[a])
			tp_intensity.append(intensity_ID[a])


		track_n=[]
		x_n=[]
		y_n=[]
		i_n=[]
		f_n=[]
		for i in range(len(tp)):
			u_track, utrack_ind, utrack_count = np.unique(tp[i],return_index=True,return_counts=True)
			track_t=[]
			x_t=[]
			y_t=[]
			i_t=[]
			f_t=[]
			cut = u_track[(utrack_count>=self.t_len_l)*(utrack_count<=self.t_len_u)]

			for j in range(len(cut)):
				tind=(tp[i]==cut[j])

				#sorting by frame per track
				temp = sorted(zip(fframe_ID[i][tind], tp[i][tind], tp_x[i][tind], tp_y[i][tind], tp_intensity[i][tind]))
				nx = [x for f, t, x, y, it in temp]
				nt = [t for f, t, x, y, it in temp]
				ny = [y for f, t, x, y, it in temp]
				ni = [it for f, t, x, y, it in temp]
				nf = [f for f, t, x, y, it in temp]

	        
	        
				track_t.append(nt)
				x_t.append(nx)
				y_t.append(ny)
				i_t.append(ni)
				f_t.append(nf)
			track_n.append(track_t)
			x_n.append(x_t)
			y_n.append(y_t)
			i_n.append(i_t)
			f_n.append(f_t)

		return [track_n,x_n,y_n,i_n,f_n]

	def analyse_tracks(self,point_data,track_n,x_n,y_n,i_n,f_n):
		x_f = []
		y_f = []
		i_f = []
		f_f = []   




		#Per_frame

		tmframe_occ_f=[]
		pc_h=[]


		diff_in = []
		diff_out = []
		diff_io = []


		'''
		for i in range(len(f_n)):
		    for j in range(len(f_n[i])):
		        print np.max(np.array(f_n[i][j])-np.min(f_n[i][j])), len(f_n[i][j])-1
		'''    
		viable_drop_t = []

		in_track_t = []
		ot_track_t = []
		io_track_t = [] 

		in_msd_f = []
		ot_msd_f = []
		io_msd_f = []

		occupency_per_drop__per_frame = []
		for i in range(len(track_n)):
			in_msd_ff = []
			ot_msd_ff = []
			io_msd_ff = []




			in_tracks = []
			out_tracks = []
			io_tracks = []
			in_drop_f = []
			out_drop_f = []
			in_out_drop_f = []
			pc_tf =[]
			frame_occ = []
			xf = []
			yf = []
			ipf = []
			ff = []
			diff_in_f = []
			diff_out_f = []
			diff_io_f = []

			in_msd_track_f = []
			out_msd_track_f = []
			io_msd_track_f = []

			in_msd_track_f_x = []
			out_msd_track_f_x = []
			io_msd_track_f_x = []

			in_msd_track_f_y = []
			out_msd_track_f_y = []
			io_msd_track_f_y = []


			ki=0
			ko=0
			kio=0
			counter = 0
			occupency_per_drop = np.zeros(len(point_data[i]))

			viable_drop_f = []
			for k in range(len(track_n[i])):
				thresh = MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k])
				if thresh > self.MSD_avg_threshold:
					pc_th = []
			    	if len(np.shape(point_data[i])) == 1:
			    	    point_data[i] = [point_data[i]]
			            
			    	for j in range(len(point_data[i])):
			    	    n_dist=dist(np.array(x_n[i][k]),np.array(y_n[i][k]),point_data[i][j][0],point_data[i][j][1])
			    	    if (np.sum(n_dist < point_data[i][j][2]))/float(len(n_dist)) >= self.minimum_percent_per_drop_in:
			    	        occupency_per_drop[j] += 1

			for j in range(len(point_data[i])):
				if occupency_per_drop[j] > self.minimum_tracks_per_drop:
					viable_drop_f.append(point_data[i][j])

			occupency_per_drop__per_frame.append(occupency_per_drop)       
			            

			for k in range(len(track_n[i])):
				thresh = MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k])
				if thresh > self.MSD_avg_threshold:
					counter += 1
					xf.append(x_n[i][k])
					yf.append(y_n[i][k])
					ipf.append(i_n[i][k])	
					ff.append(f_n[i][k])
					pc_th = []
					if len(np.shape(point_data[i])) == 1:
						print "hi"
						point_data[i] = [point_data[i]]
		                
					dist_center = []
					msd_center = []
					for j in range(len(point_data[i])):
						if occupency_per_drop[j] > self.minimum_tracks_per_drop:
							dist_center.append(np.min(dist(np.array(x_n[i][k]),np.array(y_n[i][k]),point_data[i][j][0],point_data[i][j][1])))
							msd_center.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))


							n_dist=dist(np.array(x_n[i][k]),np.array(y_n[i][k]),point_data[i][j][0],point_data[i][j][1])
							self.distances.append(n_dist)
							pc_th.append((np.sum(n_dist < point_data[i][j][2]))/float(len(n_dist)))
					if len(dist_center) != 0:
						index = np.where(np.array(dist_center) == np.array(dist_center).min())
						self.minimum_center.append(np.array(dist_center).min())

						self.minimum_msd.append(msd_center[index[0][0]])

					if len(pc_th) != 0:
						if np.sum(np.array(pc_th)==0.0) < len(point_data[i]): #asking for how many drops does the track in question not interact with.
		                	#for all tracks which interact with any drop < all drops; find the mean occupation of track taken over all said drops
							frame_occ.append(np.mean(np.array(pc_th)[np.array(pc_th)>=0.0]))
							w_1 = np.where(np.array(pc_th)==1) #where are the occupation 1? i.e when is it alway in the drop?

							if len(w_1[0])>0: #if it is in the drop (atleast in one drop all the time) calculate MSD
		                    
								in_tracks.append([x_n[i][k],y_n[i][k]])
								in_msd_ff.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
								in_drop_f.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k])) #if track is inside atleast one drop all the T.
								self.i_d_tavg.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
								self.i_d_tavg_x.append(MSD_tavg_single(x_n[i][k],f_n[i][k]))
								self.i_d_tavg_y.append(MSD_tavg_single(y_n[i][k],f_n[i][k]))
								self.in_length.append(len(x_n[i][k]))
								self.in_dist+=list(dif_dis(x_n[i][k],y_n[i][k]))
								track_info = track_decomp(x_n[i][k],y_n[i][k],f_n[i][k],self.max_track_decomp)
								track_info_x = track_decomp_single(x_n[i][k],f_n[i][k],self.max_track_decomp)
								track_info_y = track_decomp_single(y_n[i][k],f_n[i][k],self.max_track_decomp)
								in_msd_track_f.append(track_info)
								if len(track_info) > ki:
								    ki=len(track_info)
								diff_in_f.append(cumsum(x_n[i][k],y_n[i][k]))
		                
							else: #if not count it as in and out of the drop; not counting if never in any drop
		                    #####test
								if np.max(pc_th) > self.lower_bp and np.max(pc_th) < self.upper_bp:
									io_tracks.append([x_n[i][k],y_n[i][k]])
									io_msd_ff.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
									in_out_drop_f.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
									self.io_d_tavg.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
									self.io_d_tavg_x.append(MSD_tavg_single(x_n[i][k],f_n[i][k]))
									self.io_d_tavg_y.append(MSD_tavg_single(y_n[i][k],f_n[i][k]))
									self.inout_length.append(len(x_n[i][k]))
									self.io_dist+=list(dif_dis(x_n[i][k],y_n[i][k]))
									track_info = track_decomp(x_n[i][k],y_n[i][k],f_n[i][k],self.max_track_decomp)
									track_info_x = track_decomp_single(x_n[i][k],f_n[i][k],self.max_track_decomp)
									track_info_y = track_decomp_single(y_n[i][k],f_n[i][k],self.max_track_decomp)
									io_msd_track_f.append(track_info)
									if len(track_info) > kio:
										kio=len(track_info)
									diff_io_f.append(cumsum(x_n[i][k],y_n[i][k]))
								else:
									out_tracks.append([x_n[i][k],y_n[i][k]])
									ot_msd_ff.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
									out_drop_f.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
									self.o_d_tavg.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
									self.o_d_tavg_x.append(MSD_tavg_single(x_n[i][k],f_n[i][k]))
									self.o_d_tavg_y.append(MSD_tavg_single(y_n[i][k],f_n[i][k]))
									self.out_length.append(len(x_n[i][k]))
									self.ot_dist+=list(dif_dis(x_n[i][k],y_n[i][k]))
									track_info = track_decomp(x_n[i][k],y_n[i][k],f_n[i][k],self.max_track_decomp)
									track_info_x = track_decomp_single(x_n[i][k],f_n[i][k],self.max_track_decomp)
									track_info_y = track_decomp_single(y_n[i][k],f_n[i][k],self.max_track_decomp)
									out_msd_track_f.append(track_info)
									if len(track_info) > ko:
										ko=len(track_info)
									diff_out_f.append(cumsum(x_n[i][k],y_n[i][k]))
		                    
		                
		                
						else:
							out_tracks.append([x_n[i][k],y_n[i][k]])
							ot_msd_ff.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
							frame_occ.append(0) #if track interacts with none of the drops then input 0 occupancy average. 
							out_drop_f.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
							self.o_d_tavg.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
							self.o_d_tavg_x.append(MSD_tavg_single(x_n[i][k],f_n[i][k]))
							self.o_d_tavg_y.append(MSD_tavg_single(y_n[i][k],f_n[i][k]))
							self.out_length.append(len(x_n[i][k]))
							self.ot_dist+=list(dif_dis(x_n[i][k],y_n[i][k]))
							track_info = track_decomp(x_n[i][k],y_n[i][k],f_n[i][k],self.max_track_decomp)
							track_info_x = track_decomp_single(x_n[i][k],f_n[i][k],self.max_track_decomp)
							track_info_y = track_decomp_single(y_n[i][k],f_n[i][k],self.max_track_decomp)
							out_msd_track_f.append(track_info)
							if len(track_info) > ko:
							    ko=len(track_info)
							diff_out_f.append(cumsum(x_n[i][k],y_n[i][k]))

						pc_tf.append(pc_th)

					else:
						out_tracks.append([x_n[i][k],y_n[i][k]])
						pc_tf.append(0)
						ot_msd_ff.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
						frame_occ.append(0) #if track interacts with none of the drops then input 0 occupancy average. 
						out_drop_f.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
						self.o_d_tavg.append(MSD_tavg(x_n[i][k],y_n[i][k],f_n[i][k]))
						self.o_d_tavg_x.append(MSD_tavg_single(x_n[i][k],f_n[i][k]))
						self.o_d_tavg_y.append(MSD_tavg_single(y_n[i][k],f_n[i][k]))
						self.out_length.append(len(x_n[i][k]))
						self.ot_dist+=list(dif_dis(x_n[i][k],y_n[i][k]))
						track_info = track_decomp(x_n[i][k],y_n[i][k],f_n[i][k],self.max_track_decomp)
						track_info_x = track_decomp_single(x_n[i][k],f_n[i][k],self.max_track_decomp)
						track_info_y = track_decomp_single(y_n[i][k],f_n[i][k],self.max_track_decomp)
						out_msd_track_f.append(track_info)
						if len(track_info) > ko:
						    ko=len(track_info)
						diff_out_f.append(cumsum(x_n[i][k],y_n[i][k]))
			
			in_track_t.append(in_tracks)
			io_track_t.append(io_tracks)
			ot_track_t.append(out_tracks)

			in_msd_f.append(in_msd_ff)
			io_msd_f.append(io_msd_ff)
			ot_msd_f.append(ot_msd_ff)

			if len(frame_occ) != 0:        
				self.in_tracksf.append(in_tracks) 
				self.io_tracksf.append(io_tracks) 
				self.out_tracksf.append(out_tracks) 
				print counter
				self.in_msd_track[i] += in_msd_track_f
				self.out_msd_track[i] += out_msd_track_f
				self.io_msd_track[i] += io_msd_track_f

				self.in_msd_track_x[i] += in_msd_track_f
				self.out_msd_track_x[i] += out_msd_track_f_x
				self.io_msd_track_x[i] += io_msd_track_f_x

				self.in_msd_track_y[i] += in_msd_track_f
				self.out_msd_track_y[i] += out_msd_track_f_y
				self.io_msd_track_y[i] += io_msd_track_f_y

				self.max_in[i] = np.max([self.max_in[i],ki])
				self.max_out[i] = np.max([self.max_out[i],ko])
				self.max_io[i] = np.max([self.max_io[i],kio])

				diff_in.append(diff_in_f)
				diff_out.append(diff_out_f)
				diff_io.append(diff_io_f)

				x_f.append(xf)
				y_f.append(yf)
				i_f.append(ipf)
				self.tmframe_occ[i] = self.tmframe_occ[i]+frame_occ
				tmframe_occ_f.append(frame_occ)
				self.in_drop[i] = self.in_drop[i] + in_drop_f
				self.out_drop[i] = self.out_drop[i] + out_drop_f
				self.in_out_drop[i] = self.in_out_drop[i] + in_out_drop_f
				
				pc_h.append(pc_tf)

				viable_drop_t.append(viable_drop_f)

		self.in_msd_all.append(in_msd_f)
		self.io_msd_all.append(io_msd_f)
		self.ot_msd_all.append(ot_msd_f)


		self.occupency_per_drop__per_frame_per_experiment.append(occupency_per_drop__per_frame)
		self.viable_drop_total.append(viable_drop_t)

		self.in_track_total.append(in_track_t)
		self.io_track_total.append(io_track_t)
		self.ot_track_total.append(ot_track_t)

		return 

	def run_flow(self):

		tracks, drops = self.read_track_data(self.wd,self.t_string)
		for i in range(len(tracks)):
			track_n,x_n,y_n,i_n,f_n = self.convert_track_frame(tracks[i])	
			self.analyse_tracks(drops[i],track_n,x_n,y_n,i_n,f_n)
		return

	def read_parameters(self,frame_step = 1000,frame_total = 5000,t_len_l = 10,t_len_u = 1000,
		MSD_avg_threshold  = 0.0001,upper_bp = 0.99 ,lower_bp = 0.80,max_track_decomp = 1.0,
		conversion_p_nm = 130,minimum_tracks_per_drop = 3, minimum_percent_per_drop_in = 1.0):

		self.frame_step = frame_step #change manual
		self.frame_total = frame_total #change manual
		#cutoff for track length
		self.t_len_l = t_len_l #change manual #people use 5
		self.t_len_u = t_len_u #change manual #100 #use 20
		self.MSD_avg_threshold = MSD_avg_threshold
		#upper and lower "bound proportion" threshold for determining if in_out track or out only.
		self.upper_bp = upper_bp
		self.lower_bp = lower_bp
		self.max_track_decomp = max_track_decomp
		self.conversion_p_nm = conversion_p_nm
		self.minimum_tracks_per_drop = minimum_tracks_per_drop
		self.minimum_percent_per_drop_in = minimum_percent_per_drop_in
		self.frames = int(self.frame_total/self.frame_step)

		self.tmframe_occ = [[] for i in range(self.frames)]
				#averaged MSD (time) per frame step
		self.in_drop = [[] for i in range(self.frames)]
		self.out_drop = [[] for i in range(self.frames)]
		self.in_out_drop = [[] for i in range(self.frames)]

		self.in_drop_x = [[] for i in range(self.frames)]
		self.out_drop_x = [[] for i in range(self.frames)]
		self.in_out_drop_x = [[] for i in range(self.frames)]

		self.in_drop_y = [[] for i in range(self.frames)]
		self.out_drop_y = [[] for i in range(self.frames)]
		self.in_out_drop_y = [[] for i in range(self.frames)]

		#single track decomposition per frame step over all experiments
		self.max_in = np.zeros(self.frames)
		self.max_out = np.zeros(self.frames)
		self.max_io = np.zeros(self.frames)


		self.in_msd_track = [[] for i in range(self.frames)]
		self.out_msd_track = [[] for i in range(self.frames)]
		self.io_msd_track = [[] for i in range(self.frames)]

		self.in_msd_track_x = [[] for i in range(self.frames)]
		self.out_msd_track_x = [[] for i in range(self.frames)]
		self.io_msd_track_x = [[] for i in range(self.frames)]

		self.in_msd_track_y = [[] for i in range(self.frames)]
		self.out_msd_track_y = [[] for i in range(self.frames)]
		self.io_msd_track_y = [[] for i in range(self.frames)]


		return 

	def correct_msd_vectors(self):

		######################################################################################################################
		#pad msd vectors with NaNs
		  
		for i in range(len(self.in_msd_track)):
		    for j in range(len(self.in_msd_track[i])):
		        self.in_msd_track[i][j] = np.pad(self.in_msd_track[i][j],(0,int(self.max_in[i]-len(self.in_msd_track[i][j]))),'constant',constant_values=(np.nan,np.nan))
		for i in range(len(self.out_msd_track)):
		    for j in range(len(self.out_msd_track[i])):
		        self.out_msd_track[i][j] = np.pad(self.out_msd_track[i][j],(0,int(self.max_out[i]-len(self.out_msd_track[i][j]))),'constant',constant_values=(np.nan,np.nan)) 
		for i in range(len(self.io_msd_track)):
		    for j in range(len(self.io_msd_track[i])):
		        self.io_msd_track[i][j] = np.pad(self.io_msd_track[i][j],(0,int(self.max_io[i]-len(self.io_msd_track[i][j]))),'constant',constant_values=(np.nan,np.nan))
		return 
	


