'''
Documentation for trajectory_analysis_script.py

This script is used to analyse the data from the trajectory analysis script
It does this by reading the data from the trajectory analysis script and then making mappings of the SMT data to drops and cells

The core of the script is the run_analysis class, which is used to analyse a single dataset
Each database in a given experiment is analysed by a separate instance of this class
Each instance of this class is initialised with the working directory and the unique string identifier for the dataset
The mapping of the SMT data to drops and cells is done by the analyse_cell_tracks method among others
The core mapping is as follows:
Movies -> Cells -> Drops -> Trajectories
The class also contains methods to plot the data and to save the data to a .mat file

The script also contains a number of helper functions that are used by the class
These are mostly used to read the data from the trajectory analysis script and to plot the data and make classifications

The script also contains a number of functions that are used to analyse the data
These are mostly used to analyse the data and to plot the data and make classifications

Classes:
--------
1. run_analysis: class for each dataset to analyse
2. Movie_frame: class for each frame of view in a movie
3. Cell: class for each cell in a frame of view
4. Drop: class for each drop in a cell
5. Trajectory: class for each trajectory in a drop
6. Tracjectory_Drop_Mapping: class for each mapping of a trajectory to a drop

7. boundary_analysis: class for each boundary analysis of a dataset

Author: Baljyot Singh Parmar
'''
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon

import src.helpers.import_functions as import_functions
import src.helpers.nucleoid_detection as nucleoid_detection
from src.helpers.Analysis_functions import *
from src.helpers.blob_detection import *
from src.helpers.Convert_csv_mat import *
from src.helpers.plotting_functions import *
from src.helpers.decorators import deprecated

class run_analysis:
	'''
	Define a class for each dataset to analyse
	Contains multiple frames of view for movies

	Parameters;
	-----------
	wd : str
		working directory of the stored data
	t_string : str
		unique string identifier for dataset
	sim : bool
		if True, this is a simulated dataset


	Methods:
	--------
	get_fitting_parameters(): get the fitting parameters for the dataset
	get_blob_parameters(): get the blob detection parameters for the dataset
	read_parameters(): read the global parameters for the dataset
	run_flow(): run the analysis for the dataset to build the mappings in self.Cells, and self.movie


	_analyse_cell_tracks_utility(): utility function for analyse_cell_tracks
	_makeTrackCls(): make the track class for a given trajectory
	_make_TrueDrop(): make the true drop class for a given drop
	_map_TrackstoDrops(): map the trajectories to the drops
	_analyse_cell_tracks(): analyse the cell tracks and make the mappings
	_convert_track_frame(): reorder the tracks by subframe rather than the original linking
	_find_nucleoid(): find the nucleoid in a given cell
	_read_track_data(): read the track data 
	_load_segmented_image_data(): load the segmented image data 
	_load_segmented_image_locations(): load the segmented image locations 
	_read_supersegger(): read the supersegger data
	_blob_detection_utility(): utility function for blob detection
	_get_nucleoid_path(): get the path to the nucleoid data
	_get_frame_cell_mask(): get the cell mask for a given frame
	_get_movie_path(): get the path to the movie data
	_reinitalizeVariables(): reinitalize the variables for a new dataset

	'''
	def __init__(self,wd,t_string,sim = False):
		'''
		Parameters:
		-----------
		wd : str
			working directory of the stored data
		t_string : str
			unique string identifier for dataset
		sim : bool
			if True, this is a simulated dataset

		self.Cells : list
			list of cells in the dataset
		self.movie : list
			list of movies in the dataset
		
		'''
		self.sim = sim
		#global system parameters
		self.pixel_to_nm = 0
		self.pixel_to_um = 0

		self.total_experiments = 0

		self.wd = wd
		self.t_string = t_string
		self.mat_path_dir = 0

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
		
		self.minimum_tracks_per_drop = 0
		self.minimum_percent_per_drop_in = 0.50
		self.frames = int(self.frame_total/self.frame_step)

		self.segmented_drop_files = []
		##########################
		#blob detection parameters
		self.threshold_blob = 1e-2
		self.overlap_blob = 0.5
		self.min_sigma_blob = 0.5
		self.max_sigma_blob = 3
		self.num_sigma_blob = 500
		self.blob_median_filter = False
		self.detection_name = 'bp'
		self.log_scale = False
		
		self.type_of_blob = "Scale" #takes values of "Fitted" or "Scale", "Fitted" is not implemented yet and the code will crash if you try to use it

		self.blob_parameters = {"threshold": self.threshold_blob, \
					"overlap": self.overlap_blob, \
					"median": self.blob_median_filter, \
					"min_sigma": self.min_sigma_blob, \
					"max_sigma": self.max_sigma_blob, \
					"num_sigma": self.num_sigma_blob, \
					"detection": self.detection_name, \
					"log_scale": self.log_scale}

		self.fitting_parameters = {}
		
		##########################
		#condensed analysis data
		#Cells in the specific movie being analysed
		self.Movie = {}
	def _reinitalizeVariables(self):
		self.segmented_drop_files = []
		##########################
		#condensed analysis data
		#Cells in the specific movie being analysed
		self.Cells = {}
		self.Movie = {}
	def _get_movie_path(self,movie_ID,frame):
		'''
		Gives the path of the specific time projected frame (0-4) of the movie (reference frame)

		Parameters
		----------
		movie_ID : str
			key identifier for the frame of reference, i.e the movie in this whole dataset
		frame : int, or array-like of ints
			the specific time projected subframe of the movie
		
		Returns
		-------
		string, or array-like of strings
			if frame is a single integer then returns the path to that specific subframe
			if frame is a set of integers defining the subframes then array of paths of length frame

		Note
		----
		This function only works if the run_flow() method has already been applied to an instance of the run_analysis class
		'''
		#check to make sure run_flow() has occured by checking the length of self.Movie
		#check the type of frame
		if len(self.Movie) != 0:
			if isinstance(frame, int):
				return self.Movie[movie_ID].Movie_location[frame]
			else:
				return np.asarray(self.Movie[movie_ID].Movie_location)[frame]
		else:
			raise Exception("There are no Movies in this dataset yet.")
	def _get_frame_cell_mask(self,mask,frame,movie_ID):
		if len(self.Movie) != 0:
			if isinstance(frame, int):
				return mask*self.Movie[movie_ID].Movie_location[frame]
			else:
				return mask*np.asarray(self.Movie[movie_ID].Movie_location)[frame] #what does this do?
		else:
			raise Exception("There are no Movies in this dataset yet.")
	def _get_nucleoid_path(self,movie_ID,cell_ID,full_path = False):
		''' Returns the gfp image location or the image used to nuceloid segmentation'''
		if len(self.Movie) != 0:
			if full_path == True:
				return self.Movie[movie_ID].Movie_nucleoid
			else:
				return self.Movie[movie_ID].Cells[cell_ID].Cell_Nucleoid_Mask
	def _blob_detection_utility(self,seg_files,movie_ID,plot = False,kwargs={}):
		'''
		Utility function for the use of blob_dections to find the candidate spots

		Parameteres
		-----------
		seg_files : array-like of img locations (str)
			location of the images
		plot : bool
			if true plot the images with the circles ontop
			else don't plot and don't print the possible drops
		movie_ID : str
			key identifier for the frame of reference, i.e the movie in this whole dataset
		kwarg : dict
			keyword arguments for the blob_detection function


		KWARGS:
		-------
		threshold : float
			threshold for the blob detection	
		overlap : float
			overlap for the blob detection
		median : bool
			if true then apply a median filter to the image
		min_sigma : float
			minimum sigma for the blob detection
		max_sigma : float
			maximum sigma for the blob detection
		num_sigma : int
			number of sigmas for the blob detection
		detection : str
			which detection method to use
		log_scale : bool
			if true then use a log scale for the blob detection


		Returns
		-------
		array-like 
			for each seg_file find the possible blobs and return the center coordinates and radius
			[len(seg_files),# circles identified, 3 ([x,y,r])]
		'''

		blob_data = []
		for ff in range(len(seg_files)):

			blob_class = blob_detection(
				seg_files[ff],
				threshold = kwargs.get("threshold",1e-4),
				overlap = kwargs.get("overlap",0.5),
				median=kwargs.get("median",False),
				min_sigma=kwargs.get("min_sigma",1),
				max_sigma=kwargs.get("max_sigma",2),
				num_sigma=kwargs.get("num_sigma",500),
				logscale=kwargs.get("log_scale",False),
				verbose=True)

			blob_class._update_fitting_parameters(kwargs=self.fitting_parameters)
			#if blob_detection.verbose:
			#this returns a dictionary of the {Fitted: fitting results, Scale: scale space fit, Fit: Fit object}
			blob = blob_class.detection(type = kwargs.get("detection",'bp'))

			fitted = blob["Fitted"]
			scale = blob["Scale"]
			blob["Fitted"] = reshape_col2d(fitted,[1,0,2,3])
			blob["Scale"] = reshape_col2d(scale,[1,0,2])
			blobs = blob
			
			blob_data.append(blobs)
		return blob_data
	def _read_supersegger(self,sorted_cells):
		'''
		Reads the structured cell data from supersegger and returns a nested array with the structure

		Parameters
		----------
		sorted_cells : array-like of strings of directories paths
			the directories of the different frames of reference that are segemented
			
		Returns
		-------
		array-like of structured data for each parameter is read, see below
		'''

		movies = []
		for i in sorted_cells:
			cells = []
			all_files = sorted(glob.glob(i +"/cell"+ "/cell**.mat"))

			for j in range(len(all_files)):
				
				f = sio.loadmat(all_files[j])
				bounding_box = f['CellA'][0][0]['coord'][0][0]['box'][0][0]
				r_offset = f['CellA'][0][0]['r_offset'][0][0][0]
				cell_area = f['CellA'][0][0]['coord'][0][0]['A'][0][0][0]
				cell_center = f['CellA'][0][0]['coord'][0][0]['r_center'][0][0]
				cell_long_axis = f['CellA'][0][0]['coord'][0][0]['e1'][0][0]
				cell_short_axis = f['CellA'][0][0]['coord'][0][0]['e2'][0][0]
				cell_mask = f['CellA'][0][0]['mask'][0][0]
				cell_axis_lengths = f['CellA'][0][0]['length'][0][0][0]
				cells.append([bounding_box,r_offset,cell_area,cell_center,cell_long_axis,cell_short_axis,cell_axis_lengths,cell_mask])

			movies.append(cells)
		return movies
	def _load_segmented_image_locations(self,pp,cd,t_string,max_tag,min_tag):

		if len(pp) == max_tag:
			tag = pp[len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+2]
		else: 
			tag = pp[len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+1]

		drop_files = 0
		seg_files = 0

		if max_tag != min_tag:
			drop_files = sorted(glob.glob("{0}/Segmented_mean/Analysis/*_".format(cd)+t_string+"_{0}_seg.tif_spots.csv".format(tag[:])))
			if len(sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[:])))) == 0:
				seg_files = sorted(glob.glob("{0}/Segmented_mean/*".format(cd)+"_{0}_seg.tif".format(tag[:])))
			else:
				seg_files = sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[:])))
		else:
			drop_files = sorted(glob.glob("{0}/Segmented_mean/Analysis/*_".format(cd)+t_string+"_{0}_seg.tif_spots.csv".format(tag[0])))
			if len(sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[0])))) == 0:
				seg_files = sorted(glob.glob("{0}/Segmented_mean/*".format(cd)+"_{0}_seg.tif".format(tag[0])))
			else:
				seg_files = sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[0])))
		return drop_files,seg_files
	def _load_segmented_image_data(self,drop_files,use_cols=(0,1,2),skiprows=0):
		point_data = []

		for i in drop_files:

			points = np.loadtxt("{0}".format(i),delimiter=",",usecols=(0,1,2),skiprows=0)

			point_data.append(points)
		return point_data

#	def _load_segmented_image_data(self,all_files,cd,t_string,max_tag):

		blob_total = []
		tracks = []
		drops = []
		segf = []
		for pp in range(len(all_files)):

			test = np.loadtxt("{0}".format(all_files[pp]),delimiter=",")
			IO_run_analysis._save_sptanalysis_data(pp,test)

			tracks.append(test)
			if len(all_files[pp]) == max_tag:
				tag = all_files[pp][len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+2]
			else: 
				tag = all_files[pp][len(cd)+len("/Analysis/"+t_string+"_"):len(cd)+len("/Analysis/"+t_string+"_")+1]


			drop_files = 0
			seg_files = 0
			if max_tag != np.min([len(i) for i in all_files]):
				drop_files = sorted(glob.glob("{0}/Segmented_mean/Analysis/*_".format(cd)+t_string+"_{0}_seg.tif_spots.csv".format(tag[:])))
				if len(sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[:])))) == 0:
					seg_files = sorted(glob.glob("{0}/Segmented_mean/*".format(cd)+"_{0}_seg.tif".format(tag[:])))
				else:
					seg_files = sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[:])))
			else:
				drop_files = sorted(glob.glob("{0}/Segmented_mean/Analysis/*_".format(cd)+t_string+"_{0}_seg.tif_spots.csv".format(tag[0])))
				if len(sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[0])))) == 0:
					seg_files = sorted(glob.glob("{0}/Segmented_mean/*".format(cd)+"_{0}_seg.tif".format(tag[0])))
				else:
					seg_files = sorted(glob.glob("{0}/Segmented_mean/*_".format(cd)+t_string+"_{0}_seg.tif".format(tag[0])))

			blob_total.append(self._blob_detection_utility(seg_files,plot = False))
			point_data = []
			segf.append(seg_files)

			for i in drop_files:

				points = np.loadtxt("{0}".format(i),delimiter=",",usecols=(0,1,2))

				point_data.append(points)

			drops.append(point_data)
		return blob_total,tracks,drops,segf
	def _read_track_data_nocells(self,wd,t_string,**kwargs):
		'''Docstring
		Alternative to _read_track_data for cases when there is no cell segmentation via the supersegger method

		Parameters:
		-----------
		wd : str
			directory location of the dataset
		t_string : str
			unique identifier string for the dataset. Eg. NUSA, nusA, rpoC, RPOC etc.
		
		Returns:
		--------
		array-like : [tracks,drops,blob_total]
			tracks : array-like
				all raw tracks from the dataset of trajectories from TrackMate
			drops : array-like
				drop statistics from TrackMate on time projected images
			blob_total
				drop statistics from blob_detection on time projected images
		'''
		cd = wd
		all_files = sorted(glob.glob(cd + "/Analysis/" + t_string + "_**.tif_spots.csv"))
		max_tag = np.max([len(i) for i in all_files]) 
		min_tag = np.min([len(i) for i in all_files])

		blob_total = []
		tracks = []
		drops = []
		segf = []

		for pp in range(len(all_files)):
			test = np.loadtxt("{0}".format(all_files[pp]),delimiter=",")
			IO_run_analysis._save_sptanalysis_data(all_files[pp],test)
			tracks.append(test)
			drop_files, seg_files = self._load_segmented_image_locations(pp = all_files[pp], \
											cd = cd, \
											t_string = t_string, \
											max_tag = max_tag, \
											min_tag = min_tag)
			#store seg_files
			segf.append(seg_files)
			#blob analysis
			#TODO make sure to use the bounded box image created from Analysis_functions.subarray2D()
			blob_total.append(self._blob_detection_utility(seg_files=seg_files,
														movie_ID=pp,
														plot = False,
														kwargs=self.blob_parameters))
			#blob segmented data
			drops.append(self._load_segmented_image_data(drop_files))
			self.Movie[str(pp)] = Movie_frame(pp,all_files[pp],segf[pp])
			#depending on the type of blob to use "fitted","scale" use that for the blob mapping.
			self.Movie[str(pp)].Movie_nucleoid = None
			drop_s = blob_total[pp]
			#each movie is a cell in this scenario since there is no cell segmentation
			self.Movie[str(pp)].Cells[0] = Cell(Cell_ID = 0, \
										Cell_Movie_Name = all_files[pp], \
										bounding_box = None, \
										r_offset = None, \
										cell_area = None, \
										cell_center = None, \
										cell_long_axis = None, \
										cell_short_axis = None, \
										cell_axis_lengths = None, \
										cell_mask = None, \
										Cell_Nucleoid_Mask = None)
			self.Movie[str(pp)].Cells[str(0)].raw_tracks=tracks[pp]
			
			for j in range(len(drop_s)):
				for k in range(len(drop_s[j][self.type_of_blob])): #only use the blob type that is being used for the analysis

					#name the drop with j = sub-frame number (0-4), and k = unique ID for this drop in the j-th sub-frame
					self.Movie[str(pp)].Cells[str(0)].All_Drop_Collection[str(j)+','+str(k)] = drop_s[j][self.type_of_blob][k]
					self.Movie[str(pp)].Cells[str(0)].All_Drop_Verbose[str(j)+','+str(k)] = {"Fitted":drop_s[j]["Fitted"][k],\
																							"Scale":drop_s[j]["Scale"][k],\
																							"Fit":drop_s[j]["Fit"][k]}
		self.segmented_drop_files = segf
		return [tracks,drops,blob_total]
			
	def _read_track_data(self,wd,t_string,**kwargs):
		'''
		TODO: full explination

		Parameters
		----------
		wd : str
			directory location of the dataset
		t_string : str
			unique identifier string for the dataset. Eg. NUSA, nusA, rpoC, RPOC etc.
		
		Returns
		-------
		array-like : [tracks,drops,blob_total]
			tracks : array-like
				all raw tracks from the dataset of trajectories from TrackMate
			drops : array-like
				drop statistics from TrackMate on time projected images
			blob_total
				drop statistics from blob_detection on time projected images

		Notes
		-----
		This function does more than just the returns.
		It also sets up the class substructure for the Movie.Cell.Drop.Trajectory mapping and updates many of their attributes
		'''

		cd = wd
		#use gfp images for nuceloid segmentation
		nucleoid_path = kwargs.get("nucleoid_path",cd + '/gfp')
		#check if cd+/gfp exists
		if not os.path.exists(nucleoid_path):
			Warning("No gfp folder found in {0}. Assuming no cell segmentation exists and running analysis without use of segmentation. \n Note this mean the whole frame is a considered one cell".format(cd))
			return self._read_track_data_nocells(wd,t_string,**kwargs)
		
		nucleoid_imgs = find_image(nucleoid_path,full_path=True)
		nucleoid_imgs_sorted = sorted_alphanumeric(nucleoid_imgs)

		#load the data of segmented cells from SuperSegger (cell files)
		xy_frame_dir_names = IO_run_analysis._load_superSegger(cd,'/gfp/Inverted_Images')
		
		movies = self._read_supersegger(np.sort(xy_frame_dir_names))

		all_files = sorted(glob.glob(cd + "/Analysis/" + t_string + "_**.tif_spots.csv"))
		
		#make a matlab folder to store data for SMAUG analysis
		self.mat_path_dir = cd + "/Analysis/" + t_string + "MATLAB_dat/"
		
		#tag for segmented images

		max_tag = np.max([len(i) for i in all_files]) 
		min_tag = np.min([len(i) for i in all_files])

		blob_total = []
		tracks = []
		drops = []
		segf = []

		#initialize data structure
		for pp in range(len(movies)):
			#loading the track data  (formated as [track_ID,frame_ID,x,y,intensity])
			test = np.loadtxt("{0}".format(all_files[pp]),delimiter=",")

			IO_run_analysis._save_sptanalysis_data(all_files[pp],test)

			tracks.append(test)

			drop_files, seg_files = self._load_segmented_image_locations(pp = all_files[pp], \
											cd = cd, \
											t_string = t_string, \
											max_tag = max_tag, \
											min_tag = min_tag)
			#store seg_files
			segf.append(seg_files)
			#blob analysis
			#TODO make sure to use the bounded box image created from Analysis_functions.subarray2D()
			blob_total.append(self._blob_detection_utility(seg_files=seg_files,
														movie_ID=pp,
														plot = False,
														kwargs=self.blob_parameters))
			#blob segmented data
			drops.append(self._load_segmented_image_data(drop_files))

			self.Movie[str(pp)] = Movie_frame(pp,all_files[pp],segf[pp])
			#depending on the type of blob to use "fitted","scale" use that for the blob mapping.
			drop_s = blob_total[pp]
			self.Movie[str(pp)].Movie_nucleoid = nucleoid_imgs_sorted[pp]

			for i in range(len(movies[pp])):
				nuc_img = import_functions.read_file(self.Movie[str(pp)].Movie_nucleoid)
				padded_mask = pad_array(movies[pp][i][7],np.shape(nuc_img),movies[pp][i][1])

				self.Movie[str(pp)].Cells[str(i)] = Cell(Cell_ID = i, \
														Cell_Movie_Name = all_files[pp], \
														bounding_box = movies[pp][i][0], \
														r_offset = movies[pp][i][1], \
														cell_area = movies[pp][i][2], \
														cell_center = movies[pp][i][3], \
														cell_long_axis = movies[pp][i][4], \
														cell_short_axis = movies[pp][i][5], \
														cell_axis_lengths = movies[pp][i][6], \
														cell_mask = padded_mask, \
														Cell_Nucleoid_Mask = nuc_img*padded_mask)
				
				#perform nucleoid segmentation
				feature_mask, regions = self._find_nucleoid(str(pp),str(i),nuc_img*padded_mask)
				#store the nucleoid area by using the 1s in the feature mask
				self.Movie[str(pp)].Cells[str(i)].nucleoid_area = float(np.sum(feature_mask[feature_mask == 1]))
				
				#sort points into cells
				poly_cord = []
				for temp in movies[pp][i][0]:
					poly_cord.append((temp[0],temp[1]))
				poly = Polygon(poly_cord)
				x_points = tracks[pp][:,2]
				y_points = tracks[pp][:,3]
				for j in range(len(x_points)):
					point = Point(x_points[j],y_points[j])
					if poly.contains(point) or poly.touches(point):
						self.Movie[str(pp)].Cells[str(i)].raw_tracks.append(tracks[pp][j])
				#once the raw tracks are added, calculate the points_per_frame
				if len(self.Movie[str(pp)].Cells[str(i)].raw_tracks) != 0:	
					self.Movie[str(pp)].Cells[str(i)].points_per_frame = points_per_frame_bulk_sort(x=np.array(self.Movie[str(pp)].Cells[str(i)].raw_tracks)[:,2],
																									y=np.array(self.Movie[str(pp)].Cells[str(i)].raw_tracks)[:,3],
																									t=np.array(self.Movie[str(pp)].Cells[str(i)].raw_tracks)[:,1])
				for j in range(len(drop_s)):
					for k in range(len(drop_s[j][self.type_of_blob])): #only use the blob type that is being used for the analysis

						point = Point(drop_s[j][self.type_of_blob][k][0],drop_s[j][self.type_of_blob][k][1])
						if poly.contains(point) or poly.touches(point):
							#name the drop with j = sub-frame number (0-4), and k = unique ID for this drop in the j-th sub-frame
							if self.type_of_blob == "Fitted":
								self.Movie[str(pp)].Cells[str(i)].All_Drop_Collection[str(j)+','+str(k)] = drop_s[j][self.type_of_blob][k][:2]
							else:
								self.Movie[str(pp)].Cells[str(i)].All_Drop_Collection[str(j)+','+str(k)] = drop_s[j][self.type_of_blob][k]
								self.Movie[str(pp)].Cells[str(i)].All_Drop_Verbose[str(j)+','+str(k)] = {"Fitted":drop_s[j]["Fitted"][k],\
																									"Scale":drop_s[j]["Scale"][k],\
																									"Fit":drop_s[j]["Fit"][k]}

		self.segmented_drop_files = segf

		return [tracks,drops,blob_total]

	def _find_nucleoid(self,movie_ID,cell_ID,img=0):
		'''
		This function finds the nucleoid for a given cell. This is done by using the
		segmented image of the nucleoid and finding the largest blob in the image.

		Parameters:
		-----------
		movie_ID : int
			the movie ID number
		cell_ID : int
			the cell ID number
		img : array-like
			the image to use for nucleoid detection. If not provided, the function will
			load the image from the Movie object.
		
		Returns:
		--------
		feature_mask : array-like
			the mask of the nucleoid with pixel labels = 1
		regions : regionprops object
			the regionprops object for the nucleoid detections, see skimage.measure.regionprops and nucleoid_detection.find_nuc
		'''
		
		
		#find the location of the the nucleoid labeled images
		#load the image
		if isinstance(img,int):

			img = import_functions.read_file(self.Movie[movie_ID].Movie_nucleoid)*self.Movie[movie_ID].Cells[cell_ID].Cell_Nucleoid_Mask

		return nucleoid_detection.find_nuc(img,typee=None,given_type="Threshold_12")
	def _set_nucleoid_area_bulk(self):
		'''
		This function sets the nucleoid area for each cell in the movie.
		'''
		for movie in self.Movie:
			for cell in self.Movie[movie].Cells:
				feature_mask, regions = self._find_nucleoid(movie,cell)
				self.Movie[movie].Cells[cell].nucleoid_area = float(np.sum(feature_mask[feature_mask == 1]))
	def _convert_track_frame(self,track_set,**kwargs):
		'''
		This function preps the data such that the tracks satisfy a length
		and segregates the data in respect to the frame step.

		Parameters
		----------
		track_set : array-like
			the set of tracks for one specific frame of reference from TrackMate (unfiltered)
			This is a 2D array with the following columns: [track_ID,frame_ID,x,y,intensity]
		**kwargs : dict
			frame_total : int
				the total number of frames in the movie
			frame_step : int
				the step size for the frame
			track_len_upper : int
				the upper limit of the track length
			track_len_lower : int
				the lower limit of the track length
			order : tuple
				the order of the data in track_set

		
		Returns
		-------
		array-like : [track_n,x_n,y_n,i_n,f_n]
			track_n : array-like of ints
				track_IDs from TrackMate
			x_n : array-like of floats
				x coordinates of the localization belonging to index of track_ID
			y_n : array-like of floats
				y coordinates of the localization belonging to index of track_ID
			i_n : array-like of floats
				intensity of the localization belonging to index of track_ID
			f_n : array-like of floats
				frame of the movie the localization belonging to index of track_ID
		'''

		frame_total = kwargs.get("frame_total", self.frame_total)
		frame_step = kwargs.get("frame_step", self.frame_step)
		track_len_upper = kwargs.get("t_len_u",self.t_len_u)
		track_len_lower = kwargs.get("t_len_l",self.t_len_l)

		data_order = kwargs.get("order",(0,1,2,3,4))
		
		track_ID = track_set[:,data_order[0]]
		frame_ID = track_set[:,data_order[1]]
		x_ID = track_set[:,data_order[2]]
		y_ID = track_set[:,data_order[3]]
		intensity_ID = track_set[:,data_order[4]]

		tp=[]
		tp_x=[]
		tp_y=[]
		tp_intensity=[]
		fframe_ID = []
		for i in np.arange(0,frame_total,frame_step):
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

		track_all=[]
		x_all=[]
		y_all=[]
		i_all=[]
		f_all=[]

		for i in range(len(tp)):
			u_track, utrack_ind, utrack_count = np.unique(tp[i],return_index=True,return_counts=True)

			track_t=[]
			x_t=[]
			y_t=[]
			i_t=[]
			f_t=[]
			cut = u_track[(utrack_count>=track_len_lower)*(utrack_count<=track_len_upper)]

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

				track_all.append(nt)
				x_all.append(nx)
				y_all.append(ny)
				i_all.append(ni)
				f_all.append(nf)

			track_n.append(track_t)
			x_n.append(x_t)
			y_n.append(y_t)
			i_n.append(i_t)
			f_n.append(f_t)

		return [track_n,x_n,y_n,i_n,f_n]

	def _analyse_cell_tracks(self):
		'''
		Helper function to create mapping of Movie.Cell.Drop.Trajectory
		'''
		#go over each movie in this dataset
		for i,j in self.Movie.items():
			#go over each Cell in this movie
			for k,l in j.Cells.items():
				#sort the tracks based on the frame segmentation and cutoff criteria
				if len(l.raw_tracks)!=0:
					if self.sim:
						sorted_track = self._convert_track_frame(np.array(l.raw_tracks),order=(0,3,1,2,4))
					else:
						sorted_track = self._convert_track_frame(np.array(l.raw_tracks))
					self._analyse_cell_tracks_utility(i,k,sorted_track)
		return
	def _map_TrackstoDrops(self,i,k,sorted_tracks):
		#can i use scipy tree to do this?
		
		#make a list of dics to store the drops (index of list) and tracks (key in drop), with values (value of dic key) that 
		#defines the percent of the track inside the drop
		list_drop_track = [{} for i in range(len(self.Movie[i].Cells[k].All_Drop_Collection))]
		#list of all IDs of the drops
		drop_ID_list = []
		#go over all drops
		for ii,j in self.Movie[i].Cells[k].All_Drop_Collection.items():
			drop_ID_list.append(ii)#add ID 
			for kk in range(len(sorted_tracks[0])):#go over all sorted tracks subframes
				if int(ii[0]) == kk: #make sure tracks are of the same subframe ID as the drop we are looping over
					for l in range(len(sorted_tracks[0][kk])): #running over the tracks in frame k
						#find the distance of track from the drop center
						n_dist=dist(np.array(sorted_tracks[1][kk][l]),np.array(sorted_tracks[2][kk][l]),j[0],j[1]) 
						percent_drop_in = np.float(np.sum(n_dist <= j[2]))/len(n_dist) #find percentage of track loc inside
						if percent_drop_in >= self.minimum_percent_per_drop_in: #if condition holds
							key = str(kk)+","+str(l)
							checks_low = 0
							checks_none = 0 #need len(seg frames) to go forward and use this track
							for list_drop in range(len(list_drop_track)): #loop over all the drops
								if key in list_drop_track[list_drop]: #is this track already in a drop? 
									#is the value of the percent drop in larger that the one we find in the database?
									if list_drop_track[list_drop][key] < percent_drop_in:
										del list_drop_track[list_drop][key] #delete that mapping from the database
										checks_low = 1 #tells us to store this track in a drop later
								else:
									checks_none+=1 #if its not found in that drop, add 1 for all such drops
							#if we deleted the track fromt he database in the check or if it doesnt exist in any drop
							if (checks_low == 1) or (checks_none == len(list_drop_track)):
								list_drop_track[len(drop_ID_list)-1][key] = percent_drop_in
		
		return list_drop_track,drop_ID_list
	def _make_TrueDrop(self,i,k,drop_ID_list,list_drop_track,sorted_tracks):
		true_drop_per_frame = [[] for kk in range(len(sorted_tracks[0]))] #holder to true drops per frame
		#if the updated list of occupancies matches a critetion make it a true drop
		for ii,j in self.Movie[i].Cells[k].All_Drop_Collection.items():
			ind = drop_ID_list.index(ii)
			len_drop_tracks = len(list_drop_track[ind])
			if len_drop_tracks >= self.minimum_tracks_per_drop:
				self.Movie[i].Cells[k].Drop_Collection[ii] = j
				self.Movie[i].Cells[k].Drop_Verbose[ii] = self.Movie[i].Cells[k].All_Drop_Verbose[ii] #store verbose info for this viable drop
				true_drop_per_frame[int(ii[0])].append(j)
		return true_drop_per_frame
	def _makeTrackCls(self,temp,which_type,drop_ID,sorted_tracks,kk,l):
		track = Trajectory(Track_ID = str(kk)+','+str(l), \
					Frame_number = kk, X = sorted_tracks[1][kk][l], \
					Y = sorted_tracks[2][kk][l], \
					Classification = which_type, \
					Drop_Identifier = drop_ID, \
					Frames = sorted_tracks[4][kk][l], \
					MSD_total_um = con_pix_si(MSD_tavg(sorted_tracks[1][kk][l], \
													sorted_tracks[2][kk][l], \
													sorted_tracks[4][kk][l]), \
													which = 'msd'), \
					distance_from_drop = temp[drop_ID],
					Intensity = sorted_tracks[3][kk][l])
		return track
	def _analyse_cell_tracks_utility(self,i,k,sorted_tracks):
		'''
		Main function that: 
			1) Identifies viable drops
			2) Classifies trajecotries based on 1)
		'''
		#store the sorted frames in the database
		self.Movie[i].Cells[k].sorted_tracks_frame = sorted_tracks
		list_drop_track,drop_ID_list = self._map_TrackstoDrops(i=i,\
																k=k,\
																sorted_tracks=sorted_tracks)
		#find the true drops
		true_drop_per_frame = self._make_TrueDrop(i=i,\
													k=k,\
													drop_ID_list=drop_ID_list,\
													list_drop_track=list_drop_track,\
													sorted_tracks=sorted_tracks)

		
		#if true drops don't exist make all the tracks out tracks
		for ii in range(len(sorted_tracks[0])):
			if len(true_drop_per_frame[ii]) == 0:
				for l in range(len(sorted_tracks[0][ii])): #running over the tracks in frame k
					#update the All_Trajectories dictionary with a unique key if (i,k) and value of a Trajectory() object.
					track = Trajectory(Track_ID = str(ii)+','+str(l), 
										Frame_number = ii, 
										X = sorted_tracks[1][ii][l], 
										Y = sorted_tracks[2][ii][l], 
										Classification = None, 
										Drop_Identifier = None, 
										Frames = sorted_tracks[4][ii][l], 
										MSD_total_um = con_pix_si(MSD_tavg(sorted_tracks[1][ii][l],sorted_tracks[2][ii][l],sorted_tracks[4][ii][l]),which = 'msd'), 
										Intensity = sorted_tracks[3][ii][l])
					self.Movie[i].Cells[k].All_Tracjectories[str(ii)+','+str(l)] = track
					self.Movie[i].Cells[k].No_Drops_Trajectory_Collection[str(ii)+','+str(l)] = track



		#create the mapping for each viable drop
		for ii,j in self.Movie[i].Cells[k].Drop_Collection.items():
			self.Movie[i].Cells[k].Trajectory_Collection[ii] = Trajectory_Drop_Mapping(ii)

		
		for kk in range(len(sorted_tracks[0])): #loop over the subframes
			if len(true_drop_per_frame[kk]) != 0: #make sure we only consider drops in this subframe
				for l in range(len(sorted_tracks[0][kk])): #loop over tracks in this subframe
					temp_in = {}
					temp_io = {}
					temp_ot = {}
					inx_in = 0
					inx_io = 0
					inx_ot = 0
					for ii,j in self.Movie[i].Cells[k].Drop_Collection.items(): #loop over viable drops
						if int(ii[0]) == kk: #make sure its in this subframe
							#find distance from drop center
							n_dist=dist(np.array(sorted_tracks[1][kk][l]),np.array(sorted_tracks[2][kk][l]),j[0],j[1])
							#for in/out and out we need to know the percentage of in and out.
							percent_in = np.float(np.sum(n_dist <= j[2]))/len(n_dist) 
							

							if (n_dist <= j[2]).all():#if all localizations are inside 
								if len(temp_in) == 0:
									temp_in[ii] = [n_dist,percent_in]
									inx_in = ii
								else:
									print("IN track occurs twice! Indexs={0}".format(str(i)+","+str(k)+","+str(kk)+","+str(l)))
									index_tt = 0
									for tt,tv in temp_in.items():
										index_tt=tt #find the drop it belonged to
									if np.mean(temp_in[index_tt][0]) > np.mean(n_dist):
										del temp_in[index_tt]
										temp_in[ii] = [n_dist,percent_in]
										inx_in = ii
							#if the distances are all out then class as "OUT" also for <50% occupency
							elif (n_dist >= j[2]).all() or (percent_in <= self.lower_bp):
								if len(temp_ot) == 0:#if this track doesn't belong to any drop add it this drop
									temp_ot[ii] = [n_dist,percent_in]
									inx_ot = ii
								else: #if it does, find the drop it is closest to
									index_tt = 0
									for tt,tv in temp_ot.items():
										index_tt=tt #find the drop it belonged to
									if temp_ot[index_tt][1]<percent_in: #if the percent in smaller delete the entry and add this one
										del temp_ot[index_tt]
										temp_ot[ii] = [n_dist,percent_in]
										inx_ot = ii
									elif temp_ot[index_tt][1]==percent_in: #if percent is same, base it on which is closer in distance
										if np.mean(temp_ot[index_tt][0]) > np.mean(n_dist):
											del temp_ot[index_tt]
											temp_ot[ii] = [n_dist,percent_in]
											inx_ot = ii

							elif (percent_in > self.lower_bp) and (percent_in < self.upper_bp): #repeat for in/out
								if len(temp_io) == 0:
									temp_io[ii] = [n_dist,percent_in]
									inx_io = ii
								else:
									index_tt = 0
									for tt,tv in temp_io.items():
										index_tt=tt
									if temp_io[index_tt][1]<percent_in:
										del temp_io[index_tt]
										temp_io[ii] = [n_dist,percent_in]
										inx_io = ii
									elif temp_io[index_tt][1]==percent_in:
										if np.mean(temp_io[index_tt][0]) > np.mean(n_dist):
											del temp_io[index_tt]
											temp_io[ii] = [n_dist,percent_in]
											inx_io = ii
					

					if len(temp_in) == 0:
						if len(temp_io) == 0:
							#now store this is the Trajectory_Collection array of the specific cell that this trajectory belongs to
							track = self._makeTrackCls(temp = temp_ot, \
														which_type = "OUT", \
														drop_ID = inx_ot, \
														sorted_tracks = sorted_tracks, \
														kk = kk, \
														l = l)
							self.Movie[i].Cells[k].Trajectory_Collection[inx_ot].OUT_Trajectory_Collection[str(kk)+','+str(l)] = track
							#update the All_Trajectories dictionary with a unique key if (i,k) and value of a Trajectory() object.
							self.Movie[i].Cells[k].All_Tracjectories[str(kk)+','+str(l)] = track
						else:
							track = self._makeTrackCls(temp = temp_io, \
														which_type = "IO", \
														drop_ID = inx_io, \
														sorted_tracks = sorted_tracks, \
														kk = kk, \
														l = l)
							self.Movie[i].Cells[k].Trajectory_Collection[inx_io].IO_Trajectory_Collection[str(kk)+','+str(l)] = track
							#update the All_Trajectories dictionary with a unique key if (i,k) and value of a Trajectory() object.
							self.Movie[i].Cells[k].All_Tracjectories[str(kk)+','+str(l)] = track
					else:
						#now store this is the Trajectory_Collection array of the specific cell that this trajectory belongs to
						track = self._makeTrackCls(temp = temp_in, \
														which_type = "IN", \
														drop_ID = inx_in, \
														sorted_tracks = sorted_tracks, \
														kk = kk, \
														l = l)
						self.Movie[i].Cells[k].Trajectory_Collection[inx_in].IN_Trajectory_Collection[str(kk)+','+str(l)] = track
						#update the All_Trajectories dictionary with a unique key if (i,k) and value of a Trajectory() object.
						self.Movie[i].Cells[k].All_Tracjectories[str(kk)+','+str(l)] = track
	def run_flow(self):
		'''
		Controls the flow of this dataset analysis
		'''
		tracks, _, _ = self._read_track_data(self.wd,self.t_string)
		self.total_experiments = len(tracks)
		self._analyse_cell_tracks()

		return
	def run_flow_sim(self,cd,t_string): #very hacky to get this to work for simulation data. Assumes the whole movie is one cell. 
		all_files = sorted(glob.glob(cd + "/Analysis/" + t_string + ".tif_spots.csv"))

		blob_total = []
		tracks = []
		drops = []
		segf = []
		#initialize data structure
		for pp in range(len(all_files)):
			test = np.loadtxt("{0}".format(all_files[pp]),delimiter=",",skiprows=4,usecols=(2,4,5,8,12))
			tracks.append(test)

			seg_files = sorted(glob.glob("{0}/segmented/**.tif".format(cd)))

			#store seg_files
			segf.append(seg_files)
			#blob analysis
			#TODO make sure to use the bounded box image created from Analysis_functions.subarray2D()
			blob_total.append(self._blob_detection_utility(seg_files=seg_files,
														movie_ID=pp,
														plot = False,
														kwargs=self.blob_parameters))

			self.Movie[str(pp)] = Movie_frame(pp,all_files[pp],segf[pp])
			#depending on the type of blob to use "fitted","scale" use that for the blob mapping.
			drop_s = blob_total[pp]

			self.Movie[str(pp)].Cells[str(0)] = Cell(Cell_ID = 0, \
													Cell_Movie_Name = 0, \
													bounding_box = 0, \
													r_offset = 0, \
													cell_area = 0, \
													cell_center = 0, \
													cell_long_axis = 0, \
													cell_short_axis = 0, \
													cell_axis_lengths = 0, \
													cell_mask = 0, \
													Cell_Nucleoid_Mask = 0)


			self.Movie[str(pp)].Cells[str(0)].raw_tracks=tracks[pp]
			
			for j in range(len(drop_s)):
				for k in range(len(drop_s[j][self.type_of_blob])): #only use the blob type that is being used for the analysis

					#name the drop with j = sub-frame number (0-4), and k = unique ID for this drop in the j-th sub-frame
					self.Movie[str(pp)].Cells[str(0)].All_Drop_Collection[str(j)+','+str(k)] = drop_s[j][self.type_of_blob][k]
					self.Movie[str(pp)].Cells[str(0)].All_Drop_Verbose[str(j)+','+str(k)] = {"Fitted":drop_s[j]["Fitted"][k],\
																							"Scale":drop_s[j]["Scale"][k],\
																							"Fit":drop_s[j]["Fit"][k]}

		self.segmented_drop_files = segf
		self._analyse_cell_tracks()
		return [tracks,drops,blob_total]

	def read_parameters(self,frame_step = 1000,frame_total = 5000,t_len_l = 10,t_len_u = 1000,
						MSD_avg_threshold  = 0.0001,upper_bp = 0.99 ,lower_bp = 0.50,max_track_decomp = 1.0,
						conversion_p_nm = 130,minimum_tracks_per_drop = 3, minimum_percent_per_drop_in = 1.0):
		'''
		Reads in the parameters needed for the analysis

		Parameters:
		----------
		frame_step : int
			Number of subframes used in the analysis
		frame_total : int
			Total number of frames in the movies
		t_len_l : int
			Lower bound for the length of the tracks
		t_len_u : int
			Upper bound for the length of the tracks
		MSD_avg_threshold : float
			Lower bound threshold for the MSD average, to determine if the track is a valid track and not auto-fluorescence or dirt
		upper_bp : float
			Upper bound proportion threshold for determining if in_out track or out only.
		lower_bp : float
			Lower bound proportion threshold for determining if in_out track or out only.
		max_track_decomp : float
			Maximum track decomposition value
		conversion_p_nm : float
			Conversion factor from pixels to nanometers
		minimum_tracks_per_drop : int
			Minimum number of tracks per drop to be considered a valid drop
		minimum_percent_per_drop_in : float
			Minimum percentage of tracks per drop that are in tracks to be considered a valid drop

		Notes:
		-----
		This function reads in variables and assigns them to the attributed of this class instance
		'''
		self.pixel_to_um = conversion_p_nm/1000.
		self.pixel_to_nm = conversion_p_nm
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
		
		self.minimum_tracks_per_drop = minimum_tracks_per_drop
		self.minimum_percent_per_drop_in = minimum_percent_per_drop_in
		self.frames = int(self.frame_total/self.frame_step)

		return 
	def get_blob_parameters(self,threshold = 1e-4,median= False,overlap = 0.5,num_sigma = 500,
							min_sigma = 1, max_sigma = 3, log_scale = False, detection_name = 'bp'):
		'''_summary_

		Parameters
		----------
		threshold : _type_, optional
			_description_, by default 1e-4
		median : bool, optional
			_description_, by default False
		overlap : float, optional
			_description_, by default 0.5
		num_sigma : int, optional
			_description_, by default 500
		min_sigma : int, optional
			_description_, by default 1
		max_sigma : int, optional
			_description_, by default 3
		log_scale : bool, optional
			_description_, by default False
		detection_name : str, optional
			_description_, by default 'bp'
		'''
		self.max_sigma_blob = max_sigma
		self.min_sigma_blob = min_sigma
		self.num_sigma_blob = num_sigma
		self.overlap_blob = overlap
		self.threshold_blob = threshold
		self.blob_median_filter = median
		self.detection_name = detection_name
		self.log_scale = log_scale
		self.blob_parameters = {"threshold": self.threshold_blob, \
			"overlap": self.overlap_blob, \
			"median": self.blob_median_filter, \
			"min_sigma": self.min_sigma_blob, \
			"max_sigma": self.max_sigma_blob, \
			"num_sigma": self.num_sigma_blob, \
			"detection": self.detection_name, \
			"log_scale": self.log_scale}
		return

	def get_fitting_parameters(self,kwargs={}):
		'''
		Updates the fitting_parameters to be used in each iteration of this class object

		Kwargs
		------
		mask_size: int
			when fitting the image with a function this is size of square round a reference point to use for fit
		residual_func: functional
			function to use when defining the residuals for the fitting
		fit_method: string, default 'least squares'
			method of the fitting to use 
		radius_func: functional, default numpy.mean
			function to use as a method to take two sigams and convert to one radius 
		plot_fit: bool
			if True, plots each fit with the fit statistics
		centroid_range: int or float-like
			controls the bounds on the fit for the centroid (x,y). Ie: the min fit is x-centroid_range, and max is x+centroid_range
			same for y.
		sigma_range: int or float-like
			controls the bounds on the fit for the sigmas (s_x,s_y). Ie: the min fit is s_x-sigma_range, and max is s_x+sigma_range
			same for y.
		fitting_image: string
			if "Original" use the original image to fit function
			else use the Laplacian image created with the sigma that maximized the laplacian
		
		Notes
		-----
		Some of these expect a certain type to work. This is not fully coded yet and might break if you give inputs which dont make sense
		to it.
		'''
		self.fitting_parameters = kwargs

	def _correct_msd_vectors(self):
		'''
		pad msd vectors with NaNs so they are the same length as you increase tau
		'''
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
	@deprecated("This seems useless right now, and sometimes breaks into an infinite loop. Should not be used, but keeping for testing before removing.")
	def _bulk_msd_plot(self,Movies=None,plot=False):
		''' Docstring for _bulk_msd_plot
		
		'''
		if Movies is None:
			Movies = self.Movie
		no_drop = []
		in_drop = []
		ot_drop = []
		io_drop = []
		for i,j in Movies.items():
			for k,l in j.Cells.items():
				for n,m in l.All_Tracjectories.items():
					if m.Classification == None:
						no_drop.append([m.MSD_total_um,len(m.X)])
					if m.Classification == "IN":
						in_drop.append([m.MSD_total_um,len(m.X)])
					if m.Classification == "OUT":
						ot_drop.append([m.MSD_total_um,len(m.X)])
					if m.Classification == "IO":
						io_drop.append([m.MSD_total_um,len(m.X)])

		return {"no_drop": no_drop, "in_drop": in_drop, "ot_drop": ot_drop, "io_drop": io_drop}
	def _convert_to_track_dict_bulk(self,Movie=None):
		''' Docstring for _convert_to_track_dict_bulk
		Purpose: convert the Movie object into a dictionary of tracks for the bulk analysis of MSD

		Parameters:
		-----------
		Movie: Movie object, default None
			Movie object to convert to a dictionary of tracks

		Returns:
		--------
		track_dict: dictionary
			dictionary of tracks with keys "IN","OUT","IO" and values being a dictionary of tracks with keys being the track number and values being the track [(x,y,T),...,(x,y,T)]
		'''
		if Movie is None:
			Movie = self.Movie
		#check if Movie is a dictionary and not empty
		if type(Movie) is not dict:
			raise TypeError("Movie must be a dictionary")
		if len(Movie) == 0:
			raise ValueError("Movie dictionary is empty")
		
		track_dict_out = {}
		track_dict_in = {}
		track_dict_io = {}
		#make a collection for all the tracks
		track_dict_all = {}
		for i,j in Movie.items():
			for k,l in j.Cells.items():
				for n,m in l.All_Tracjectories.items():
					track_dict_all[n] = np.array([m.X,m.Y,m.Frames]).T
					if m.Classification == "IN":
						track_dict_in[n] = np.array([m.X,m.Y,m.Frames]).T
					if m.Classification == "IO":
						track_dict_io[n] = np.array([m.X,m.Y,m.Frames]).T
					else:
						track_dict_out[n] = np.array([m.X,m.Y,m.Frames]).T
		return {"IN": track_dict_in, "OUT": track_dict_out, "IO": track_dict_io, "ALL": track_dict_all}
					
class Movie_frame:
	'''
	Frame of reference for one movie

	Parameters
	----------
	__init__:
		Movie_ID: int
			unique identifier for each movie
		Movie_name: string
			name of the movie, used to find the movie. This is the name of the movie file!
		Movie_location: string or list of strings, optional (size is the number of frames \n 
			(i.e 5 for 5 subframes from a 5000 frame movie with 1000 per subframe))
			This is the sub segmented frames of the movie. This is just a list of strings that are the location of the frames images
			nucleoid_location: string, optional
			location of the image that shows the nucleoid, by default 0
	Cells: dictionary
		dictionary of Cell class objects belonging to this Movie, identified by a label (0,1,...,n)


	'''
	def __init__(self,Movie_ID,Movie_name,Movie_location=0,nucleoid_location = 0):
		self.Movie_ID = Movie_ID
		self.Movie_name = Movie_name
		self.Movie_location = Movie_location
		self.Movie_nucleoid = nucleoid_location

		self.Cells = {}

class Cell:
	'''
	each cell class is built of two main things:
	1) A dictionary of drops (x,y,r) identified by a label (0,1,...,n)
	2) A collection of Trajectory class objects defining the trajectories in that Cell class object
	
	Parameters
	----------
	__init__:
		Cell_ID: int
			unique identifier for each cell
		Cell_Movie_Name: string
			name of the movie the cell is in, used to find the movie. This is the name of the movie file!
		bounding_box: list of 4 ints
			coordinates of the bounding box of the cell
		r_offset: list of 2 ints
			coordinate of the top left corner of the bounding box
		cell_area: int
			area of the cell
		cell_center: list of 2 ints
			coordinates of the center of the cell
		cell_long_axis: int
			length of the long axis of the cell
		cell_short_axis: int
			length of the short axis of the cell
		cell_axis_lengths: list of 2 ints
			lengths of the long and short axis of the cell
		cell_mask: 2D array of bools
			mask of the cell
		Cell_Nucleoid_Mask: 2D array of bools
			mask of the nucleoid in the cell
	
	Methods:
	-----------
	__init__:
		Initialize the Cell class object
	_convert_viableDrop_list(self):
		converts the viableDrop_list to a dictionary of viable drops
	
	'''
	def __init__(self, Cell_ID, Cell_Movie_Name,bounding_box=0,r_offset=0,cell_area=0,cell_center=0,cell_long_axis=0,cell_short_axis=0,cell_axis_lengths=0,cell_mask=0,Cell_Nucleoid_Mask=0):

		self.Cell_ID = Cell_ID
		self.Cell_Movie_Name = Cell_Movie_Name

		self.Cell_Nucleoid_Mask = Cell_Nucleoid_Mask

		#Cell global variables
		self.cell_mask = cell_mask
		self.cell_area = cell_area
		self.cell_center = cell_center
		#bounding box is first coordinate is the bottom right one. Highest x -> counter clockwise (i.e lowest x -> rest)
		self.bounding_box = bounding_box 
		#r_offset is the top left edge of the cell's bounding box.
		self.r_offset = r_offset
		self.cell_long_axis = cell_long_axis
		self.cell_short_axis = cell_short_axis
		self.cell_axis_lengths = cell_axis_lengths

		#properties below are not initialized in the __init__ method but are defined later

		#keys in Trajectory_Collection are String(i,j) (frame,viable_drop index) while the values are instances of Trajectory_Drop_Mapping
		self.Trajectory_Collection = {}
		#If no drops are identified in this frame then put all "OUT" trajectories in this collection with (i,k) -> (frame,trajectory index)
		self.No_Drops_Trajectory_Collection = {}
		#viable drops only
		self.Drop_Collection = {}
		#dict for viable drops with verbose information
		self.Drop_Verbose = {}
		#all drops before viability test
		self.All_Drop_Collection = {}
		#dic to store verbose information about the drops
		self.All_Drop_Verbose = {}
		#dict for all trajectories without mapping but still classified
		self.All_Tracjectories = {}


		#unsorted raw track data 
		self.raw_tracks = []
		#sorted tracks per sub frame
		self.sorted_tracks_frame = []
	def _convert_viableDrop_list(self,subframes = 5):
		list_sorted = [[] for i in range(subframes)]
		for i,j in self.Drop_Collection.items():
			list_sorted[int(i[0])].append(j)
		return list_sorted

	#points_per_frame allows us to see how many points are in each frame of this cell
	#this is of the form {frame_number: number_of_points,...}
	@property
	def points_per_frame(self)->dict:
		return self._points_per_frame
	@points_per_frame.setter
	def points_per_frame(self,points_per_frame: dict)->None:
		#make sure points_per_frame is a dictionary with keys as frame numbers and values as the number of points in that frame
		if not isinstance(points_per_frame,dict):
			raise TypeError("points_per_frame must be a dictionary")
		self._points_per_frame = points_per_frame
		#everytime we set this also change the area_of_points_per_frame using Analysis_functions.area_points_per_frame
		self.area_of_points_per_frame = area_points_per_frame(self._points_per_frame,radius_of_gyration)

	#in conjunction with points_per_frame, this allows us to see how much area the points in each frame cover
	#this is of the form {frame_number: area_of_points,...}
	@property
	def area_of_points_per_frame(self)->dict:
		return self._area_of_points_per_frame
	@area_of_points_per_frame.setter
	def area_of_points_per_frame(self,area_of_points_per_frame: dict)->None:
		#make sure area_of_points_per_frame is a dictionary with keys as frame numbers and values as the area of points in that frame
		if not isinstance(area_of_points_per_frame,dict):
			raise TypeError("area_of_points_per_frame must be a dictionary")
		self._area_of_points_per_frame = area_of_points_per_frame
	
	@property
	def density_per_frame(self)->dict:
		#check if points_per_frame and area_of_points_per_frame are defined
		if not hasattr(self,"points_per_frame"):
			raise AttributeError("points_per_frame is not defined")
		if not hasattr(self,"area_of_points_per_frame"):
			raise AttributeError("area_of_points_per_frame is not defined")
		#for each frame number, calculate the density of points in that frame by dividing the number of points by the area of points
		self._density_per_frame = {frame: self.points_per_frame[frame]/self.area_of_points_per_frame[frame] for frame in self.points_per_frame.keys()}
		return self._density_per_frame

	@property
	def nucleoid_area(self)->float:
		return self._nucleoid_area
	@nucleoid_area.setter
	def nucleoid_area(self,nucleoid_area: float|int)->None:
		if not isinstance(nucleoid_area,(float,int)):
			raise TypeError("nucleoid_area must be a float or an int")
		self._nucleoid_area = nucleoid_area
	
class Trajectory_Drop_Mapping:
	'''
	create a mapping for each viable drop to store all the Trajectory instances that belong to it in terms of Classification
	
	Parameters
	----------
	__init__ :
		Drop_ID : str
			unique identifier for each viable drop (i.e "0,1" for the first drop in the first sub-frame)
			Note that the first number is the sub-frame number and the second number is the drop index in that sub-frame
			See notes for more information
	
	IN_Trajectory_Collection : dict
		dictionary of all the IN trajectories that belong to the viable drop
	OUT_Trajectory_Collection : dict
		dictionary of all the OUT trajectories that belong to the viable drop
	IO_Trajectory_Collection : dict
		dictionary of all the IN and OUT trajectories that belong to the viable drop

	Notes:
	------
	1. Each dictionary is of the form (i,j) -> (frame,trajectory index) where i is the frame number and j is the trajectory index
	where the above represent the keys, the values are instances of the Trajectory class
	2. The frame number is based on the sub segmented frames (i.e 0,1,2,3,4 if subframes = 5)
	3. The trajectory index is the index of the trajectory in the raw_tracks list of the Cell class object
	4. The viable drop index is the index of the detected drop in the Drop_Collection dictionary of the Cell class object
	5. 4) shows that the viable drop index is the same as the drop index in the raw_tracks list of the Cell class object, \n
	(i.e the drop index is determined by the blob detection. After the viability criterion is applied the non-viable drops are removed, but the order of the drop IDs is perserved)\n
	(For a collection of drops with IDs (0,1),(0,2),(0,3), if (0,2) is not viable then the viable drops are: (0,1),(0,3) and they keep this order and ID name)
	'''
	def __init__(self,Drop_ID):

		self.Drop_ID = Drop_ID
		self.IN_Trajectory_Collection = {}
		self.OUT_Trajectory_Collection = {}
		self.IO_Trajectory_Collection = {}

class Trajectory:
	'''
	Trajectory attribute class

	Parameters
	----------
	__init__ :
		Track_ID : int
			unique identifier for each trajectory
		Frame_number : int
			frame number of the trajectory (based on the sub segmented frames) (i.e 0,1,2,3,4 if subframes = 5)
		X : list or np.array
			x coordinates of the trajectory
		Y : list or np.array
			y coordinates of the trajectory
		Classification : str 
			"IN" or "OUT" or "IO", for In Drop, Out Drop, In and Out Drop
		Drop_Identifier : str
			ID of the drop that the trajectory belongs to
		Frames : int
			number of frames in the trajectory
		Intensity : list or np.array
			intensity of the trajectory (same length as X and Y and Frames)
		MSD_total_um : float, default = None
			total MSD of the trajectory
		
		Kwargs:
		-------
		distance_from_drop : float
			distance from the drop center to the trajectory
		
		
	'''
	def __init__(self, Track_ID, Frame_number, X, Y, Classification, Drop_Identifier, Frames, MSD_total_um = None,**kwargs):

		self.Track_ID = Track_ID
		self.Frame_number = Frame_number
		self.Frames = Frames
		self.X = X 
		self.Y = Y 
		self.Intensity = kwargs.get("Intensity",None)
		self.Classification = Classification
		self.Drop_Identifier = Drop_Identifier
		self.MSD_total_um = MSD_total_um
		self.distance_from_drop = kwargs.get("distance_from_drop",0)

#define a custom class for boundary analysis

class boundary_analysis:
	'''
	TODO - make this better
	class for storing analysis intermediates and boundary analysis
	'''
	def __init__(self,**kwargs) -> None:
		self.dataset = kwargs.get("dataset",None)
	@staticmethod
	def _xy_angle_from_drop(collection_traj,cell_obj):
		angles = []
		for i,k in collection_traj.items():
			track = k
			x_val = track.X
			y_val = track.Y
			drop_data = cell_obj.Drop_Collection[track.Drop_Identifier]
			drop_center_dist = (dist(x_val,y_val,drop_data[0],drop_data[1]))/drop_data[2]
			v1 = list(zip(x_val-drop_data[0],y_val-drop_data[1]))
			angles += list(angle_multi(v1))[:-1]
		return angles
	@staticmethod
	def _track_to_closest_drop(x,y,drops):
		''' for a track defined by x,y returns the closest circle it is from a collenction of circles [[d_x,d_y,d_r],...]'''
		drop_x = np.asarray(drops)[:,0]
		drop_y = np.asarray(drops)[:,1]
		dists = dist(drop_x,drop_y,x,y)
		if len(drops) > 1:
			return drops[np.argmin(dists)]
		else:
			return drops[0]

	@staticmethod
	def	_directional_variableTracks(x,y,drops):
		mapped = []
		for i in range(len(x)):
			min_drop = boundary_analysis._track_to_closest_drop(x[i],y[i],drops)
			mapped.append([x[i],y[i],min_drop])
		angles,dist_center,directional_displacements = boundary_analysis._directional_displacement_utility(mapped)
		return [angles,dist_center,directional_displacements]

	@staticmethod
	def _directional_displacement_utility(obj):
		angles = []
		dist_center = []
		directional_displacement = []
		for i in obj:
			drop_center_dist = (dist(i[0],i[1],i[2][0],i[2][1]))/i[2][2]
			v1 = list(zip(i[0]-i[2][0],i[1]-i[2][1]))
			angles += list(angle_multi(v1))[:-1]
			directional = con_pix_si(np.diff(dist(i[0],i[1],i[2][0],i[2][1])),which = 'um')
			directional_displacement+=list(directional)
			dist_center+=list(drop_center_dist)[:-1]
		return [angles,dist_center,directional_displacement]

	@staticmethod
	def directional_displacement(collection_traj,cell_obj):
		angles = []
		dist_center = []
		directional_displacement = []
		for i,k in collection_traj.items():
			track = k
			x_val = track.X
			y_val = track.Y
			drop_data = cell_obj.Drop_Collection[track.Drop_Identifier]

			drop_center_dist = (dist(x_val,y_val,drop_data[0],drop_data[1]))/drop_data[2]
			v1 = list(zip(x_val-drop_data[0],y_val-drop_data[1]))
			angles += list(angle_multi(v1))[:-1]

			#direction of the trajectory
			#r2 -r1 > 0 moving out, r2 - r1 < 0 moving in
			directional = con_pix_si(np.diff(dist(x_val,y_val,drop_data[0],drop_data[1])),which = 'um')
			directional_displacement+=list(directional)
			dist_center+=list(drop_center_dist)[:-1]
		return [angles,dist_center,directional_displacement]
		
	def directional_displacement_bulk(self,**kwargs):
		'''
		return the directional_displacement vs distance from center with angles
		'''
		displacements = []
		dist_centers = []
		angles = []
		for k,i in self.dataset.items(): #movies
			for n,m in i.Cells.items(): #cells
				for o,p in m.Trajectory_Collection.items(): #drop trajectory mapping
					if kwargs.get("IN",False) == True:
						angle,dist_center,dir_displacement = self.directional_displacement(p.IN_Trajectory_Collection,m)
						angles+=angle
						displacements+=dir_displacement
						dist_centers+=dist_center
					if kwargs.get("IO",False) == True:
						angle,dist_center,dir_displacement = self.directional_displacement(p.IO_Trajectory_Collection,m)
						angles+=angle
						displacements+=dir_displacement
						dist_centers+=dist_center
					if kwargs.get("OT",False) == True:
						angle,dist_center,dir_displacement = self.directional_displacement(p.OUT_Trajectory_Collection,m)
						angles+=angle
						displacements+=dir_displacement
						dist_centers+=dist_center
		return [displacements,dist_centers,angles]
	
	@staticmethod
	def plot_directional_displacements(**kwargs):
		displacements = kwargs.get("dir_displacements")
		dist_center = kwargs.get("dist_center")
		angles = kwargs.get("angles")
		x = np.array(dist_center)
		y = np.array(displacements)
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]
		angles = np.array(angles)
		n, _ = np.histogram(x,bins = 20)
		sy, _ = np.histogram(x,bins = 20,weights = y)
		sy2, _ = np.histogram(x,bins = 20,weights = y*y)
		#h, x_bins, y_bins = np.histogram2d(x,y,bins = 20)

		mean = sy/n
		std = np.sqrt(sy2/n - mean*mean)
		fig = plt.figure()
		ax_1 = fig.add_subplot(211)
		ax_2 = fig.add_subplot(212)
		a = ax_1.scatter(x,y,c = z, s = 50)


		b = ax_2.scatter(*rt_to_xy(np.array(dist_center),angles),s = 0.1)
		cir = plt.Circle( (0,0) ,1,fill = False )
		ax_2.plot(0,0,'bo',markersize = 2)
		plt.colorbar(b,ax = ax_2)
		ax_2.add_artist(cir)

		ax_1.plot((_[1:] + _[:-1])/2,mean, 'r-')
		ax_1.errorbar((_[1:] + _[:-1])/2, mean,yerr = std/np.sqrt(len(mean)),fmt = 'r-')
		ax_1.set_xlabel("Ralative Distance of Localization to Boundary (relative to radius)")
		ax_1.set_ylabel("Displacements (um)")
		plt.colorbar(a,ax = ax_1)
		fig.tight_layout()
		if kwargs.get("plot",False):
			plt.show()
			return
		else:
			return [fig,ax_1,ax_2] 
