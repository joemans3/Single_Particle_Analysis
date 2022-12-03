from scipy.io import savemat
import numpy as np
import glob
import os


class Prepare_Tracks:
	'''_summary_
	'''
	def __init__(self,wd,t_string,lower_length,upper_length):
		'''_summary_

		Parameters
		----------
		wd : _type_
			_description_
		t_string : _type_
			_description_
		lower_length : _type_
			_description_
		upper_length : _type_
			_description_
		'''
		self.wd = wd
		self.t_string = t_string
		self.t_len_l = lower_length
		self.t_len_u = upper_length
		self.tracks_clist = []

	def update_lengths(self,lower_length,upper_length):
		'''_summary_

		Parameters
		----------
		lower_length : _type_
			_description_
		upper_length : _type_
			_description_
		'''
		self.t_len_l = lower_length
		self.t_len_u = upper_length
		return

	def read_track_data(self,wd,t_string):
		'''wd: this is the current woring directory for the dataset you are interested in.
		t_string: Eg. NUSA, nusA, rpoC, RPOC etc.

		Returns
		-------
		_type_
			_description_
		'''
		cd = wd

		all_files = sorted(glob.glob(cd + "/Analysis/" + t_string + "_**.tif_spots.csv"))

		tracks = []
		max_tag = np.max([len(i) for i in all_files]) 

		for pp in all_files:
			tag = 0
			if len(pp) == max_tag:
				tag = pp[len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+2]
			else: 
				tag = pp[len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+1]

			test = np.loadtxt("{0}".format(pp),delimiter=",")
			tracks.append(test)
		for i in range(len(all_files)):
			self.tracks_clist.append([[],[],[],[],[]])

		return [tracks]

	def convert_track_frame(self,track_set):
		'''track_set: the set of tracks for one specific frame of reference
		This function preps the data such that the tracks satisfy a length
		and segregates the data in respect to the frame step.

		Returns
		-------
		_type_
			_description_
		'''
		track_ID = track_set[:,0]
		frame_ID = track_set[:,1]
		x_ID = track_set[:,2]
		y_ID = track_set[:,3]
		intensity_ID = track_set[:,4]


		u_track, utrack_ind, utrack_count = np.unique(track_ID,return_index=True,return_counts=True)

		track_t=[]
		x_t=[]
		y_t=[]
		i_t=[]
		f_t=[]
		c_f_t = []
		cut = u_track[(utrack_count>=self.t_len_l)*(utrack_count<=self.t_len_u)]

		for j in range(len(cut)):
			tind=(track_ID==cut[j])
			track_ID[tind] = np.zeros(len(track_ID[tind])) + j + 1
			#sorting by frame per track
			temp = sorted(zip(frame_ID[tind].astype(int), track_ID[tind].astype(int), x_ID[tind], y_ID[tind], intensity_ID[tind]))
			nx = [x for f, t, x, y, it in temp]
			nt = [t for f, t, x, y, it in temp]
			ny = [y for f, t, x, y, it in temp]
			ni = [it for f, t, x, y, it in temp]
			nf = [f for f, t, x, y, it in temp]

			cf = nf - (np.min(nf)) + 1
        	 
			track_t.append(nt)
			x_t.append(nx)
			y_t.append(ny)
			i_t.append(ni)
			f_t.append(nf)
			c_f_t.append(cf.astype(int))
		return [track_t,x_t,y_t,i_t,f_t,c_f_t]



	def save_to_mat(self,track,x,y,i,f,cf,k):
		'''_summary_

		Parameters
		----------
		track : _type_
			_description_
		x : _type_
			_description_
		y : _type_
			_description_
		i : _type_
			_description_
		f : _type_
			_description_
		cf : _type_
			_description_
		k : _type_
			_description_
		'''
		for i in range(len(track)):
			self.tracks_clist[k][0] += list(track[i])
			self.tracks_clist[k][1] += list(cf[i])
			self.tracks_clist[k][2] += list(f[i])
			self.tracks_clist[k][3] += list(x[i])
			self.tracks_clist[k][4] += list(y[i])

		return 

	def run(self):
		
		directory = "{0}/MAT_FILES_{1}".format(self.wd,self.t_string)
		if not os.path.exists(directory):
			os.makedirs(directory)
		t = self.read_track_data(self.wd,self.t_string)
		for i in range(len(t[0])):

			self.save_to_mat(*self.convert_track_frame(t[0][i]),k =i)
			temp = self.tracks_clist[i]
			dic = {	
			"trfile" : np.transpose(temp)
			}

			savemat("{0}/MAT_FILES_{1}/{2}_{3}.mat".format(self.wd,self.t_string,self.t_string,i),dic)
		return
'''
a = Prepare_Tracks("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/Nusa_20190305","NUSA",10,20)
a.run()
'''
	