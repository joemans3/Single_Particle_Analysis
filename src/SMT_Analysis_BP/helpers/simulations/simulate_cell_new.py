import json
import numpy as np
import os
import random
import SMT_Analysis_BP.helpers.simulations.simulate_foci_new as sf
import skimage as skimage
from PIL import Image
import pickle
import SMT_Analysis_BP.helpers.misc.decorators as decorators
import SMT_Analysis_BP.helpers.simulations.probability_functions as pf
import SMT_Analysis_BP.helpers.misc.errors as errors

def save_tiff(image,path,img_name=None):
	''' Docstring for save_tiff: save the image as a tiff file
	
	Parameters:
	-----------
	image : array-like
		image to be saved
	path : str
		path to save the image
	img_name : str, Default = None
		name of the image
	
	Returns:
	--------
	None
	'''
	if img_name is None:
		skimage.io.imsave(path,image)
	else:
		skimage.io.imsave(path+img_name+".tiff",image)
	return  

#function to perform the subsegmentation
def sub_segment(img,sub_frame_num,img_name=None,subsegment_type="mean"):
	''' Docstring for sub_segment: perform subsegmentation on the image
	
	Parameters:
	-----------
	img : array-like
		image to be subsegmented
	sub_frame_num : int
		number of subsegments to be created
	img_name : str, Default = None
		name of the image
	subsegment_type : str, Default = "mean"
		type of subsegmentation to be performed, currently only "mean" is supported
	
	Returns:
	--------
	hold_img : list
		list of subsegments
	hold_name : list
		list of names of the subsegments
	
	
	'''
	#get the dimensions of the image
	dims = img.shape
	#get the number of frames
	num_frames = dims[0]
	#find the number of frames per subsegment
	frames_per_subsegment = int(num_frames/sub_frame_num)
	hold_img = []
	hold_name = []
	for j in np.arange(sub_frame_num):
		if subsegment_type == "mean":
			hold_img.append(np.mean(img[int(j*frames_per_subsegment):int((j+1)*frames_per_subsegment)],axis=0))
		elif subsegment_type == "max":
			hold_img.append(np.max(img[int(j*frames_per_subsegment):int((j+1)*frames_per_subsegment)],axis=0))
		elif subsegment_type == "std":
			hold_img.append(np.std(img[int(j*frames_per_subsegment):int((j+1)*frames_per_subsegment)],axis=0))
		
	return hold_img

def make_directory_structure(cd,img_name,img,subsegment_type,sub_frame_num,**kwargs):
	''' Docstring for make_directory_structure: make the directory structure for the simulation and save the image + the data and parameters
	Also perform the subsegmentation and save the subsegments in the appropriate directory
	
	Parameters:
	-----------
	cd : str
		directory to save the simulation
	img_name : str
		name of the image
	img : array-like    
		image to be subsegmented
	subsegment_type : str
		type of subsegmentation to be performed, currently only "mean" is supported
	sub_frame_num : int
		number of subsegments to be created
	**kwargs : dict
		dictionary of keyword arguments
	
	KWARGS:
	-------
	data : dict, Default = None
		dictionary of data to be saved, Keys = "map","tracks","points_per_frame" Values = array-like. 
		See the return of the function simulate_cell_tracks for more details
	parameters : dict, Default = None (dict with the keys "cell_parameters","global_parameters")
		cell_parameters : dict, Default = None
			dictionary of cell parameters, Keys = "num_tracks",
												  "diffusion_coefficients",
												  "initials",
												  "simulation_cube",
												  "hursts",
												  "track_type",
												  "mean_track_length",
												  "track_length_distribution",
												  "exposure_time"
		global_parameters : dict, Default = None
			dictionary of global parameters, Keys = "num_cells",
													"num_subsegments",
													"subsegment_type",
													"subsegment_num_frames",
													"subsegment_num_points",
													"subsegment_num_tracks",
													"subsegment_num_diffusion_coefficients",
													"subsegment_num_initials",
													"subsegment_num_hursts",
													"subsegment_num_track_type",
													"subsegment_num_mean_track_length",
													"subsegment_num_track_length_distribution",
													"subsegment_num_exposure_time"
													(names might be different, but should be self explanatory once you load the dictionary)
	
	Returns:
	--------
	array-like
		list of subsegment images
	'''
	#make the directory if it does not exist
	if not os.path.exists(cd):
		os.makedirs(cd)

	#saves the data if it is passed as a keyword argument (map,tracks,points_per_frame)
	with open(cd+'Track_dump.pkl', 'wb+') as f:
		pickle.dump(kwargs.get("data",{}), f)
	#saves the parameters used to generate the simulation
	with open(cd+'params_dump.pkl', 'wb+') as f:
		pickle.dump(kwargs.get("parameters",{}), f)

	#in this directory, dump the parameters into a json file
	if "parameters" in kwargs.keys():
		with open(os.path.join(cd,"parameters.json"), 'w') as fp:
			#check if parameter values are dictionaries
			for i,j in kwargs["parameters"].items():
				if type(j) == dict:
					for k,l in j.items():
						if type(l) == np.ndarray and k !="initials":
							#if true then find the unique values in the array with the number of times they occur and save it as a dictionary "unique_values":count
							unique, counts = np.unique(l, return_counts=True)
							#convert the arrays to lists
							unique = unique.tolist()
							counts = counts.tolist()
							kwargs["parameters"][i][k] = dict(zip(unique, counts))
						elif k=="initials":
							if isinstance(l,dict):
								temp_dict = {}
								for m,n in l.items():
									temp_dict[m] = n.tolist()

								kwargs["parameters"][i][k] = temp_dict
							else:
								#from the collection of [[x,y]...] find the unique values of [x,y] combinations and save it as a dictionary "unique_values":count
								unique, counts = np.unique(l, axis=0,return_counts=True)
								#convert the arrays to lists
								unique = map(str,unique.tolist())
								counts = counts.tolist()
								kwargs["parameters"][i][k] = dict(zip(unique, counts))
						
							
			json.dump(kwargs["parameters"], fp,indent=4)


	#make a diretory inside cd called Analysis if it does not exist
	if not os.path.exists(os.path.join(cd,"Analysis")):
		os.makedirs(os.path.join(cd,"Analysis"))
	#save the img file with its name in the cd directory
	save_tiff(img,cd,img_name=img_name)
	#make a directory inside cd called segmented if it does not exist
	if not os.path.exists(os.path.join(cd,"segmented")):
		os.makedirs(os.path.join(cd,"segmented"))
	#perform subsegmentation on the image
	hold_img = sub_segment(img,sub_frame_num,img_name=img_name,subsegment_type=subsegment_type)
	#create the names for the subsegmented images
	hold_name = []
	for i in np.arange(sub_frame_num):
		hold_name.append(os.path.join(cd,"segmented",str(int(i)+1)+"_"+img_name+".tif"))
	#save the subsegmented images
	for i in np.arange(sub_frame_num):
	
		img = Image.fromarray(hold_img[i])
		img.save(hold_name[i])
	return hold_img


class Simulate_cells():
	def __init__(self,init_dict_json:dict|str):
		''' Docstring for Simulate_cells: Class for simulating cells in space.
		
		Parameters:
		-----------
		init_dict_json : dict|str
			dictionary of parameters or path to the json file containing the parameters
			see sim_config.md for more details
		
		Returns:
		--------
		None
		'''
		if isinstance(init_dict_json,str):
			self.init_dict = self._read_json(init_dict_json)
		elif isinstance(init_dict_json,dict):
			self.init_dict = init_dict_json
		
		#store the times
		self.frame_count = self.init_dict["Global_Parameters"]["frame_count"]
		self.interval_time = self.init_dict["Global_Parameters"]["interval_time"]
		self.oversample_motion_time = self.init_dict["Global_Parameters"]["oversample_motion_time"]
		self.exposure_time = self.init_dict["Global_Parameters"]["exposure_time"]
		self.total_time = self._convert_frame_to_time(self.frame_count,self.exposure_time,self.interval_time,self.oversample_motion_time)
		#convert the track_length_mean from frame to time
		self.track_length_mean = self._convert_frame_to_time(self.init_dict["Track_Parameters"]["track_length_mean"],self.exposure_time,self.interval_time,self.oversample_motion_time) 

		#update the diffusion coefficients from um^2/s to pix^2/ms
		self.track_diffusion_updated = self._update_units(self.init_dict["Track_Parameters"]["diffusion_coefficient"],'um^2/s','pix^2/(oversample_motion_time)ms)') 
		self.condensate_diffusion_updated = self._update_units(self.init_dict["Condensate_Parameters"]["diffusion_coefficient"],'um^2/s','pix^2/(oversample_motion_time)ms)')

		return
	def _convert_frame_to_time(self,frame:int,exposure_time:int,interval_time:int,oversample_motion_time:int)->int:
		''' Docstring for _convert_frame_to_time: convert the frame number to time
		
		Parameters:
		-----------
		frame : int
			frame number
		exposure_time : int
			exposure time
		interval_time : int
			interval time
		oversample_motion_time : int
			oversample motion time
		
		Returns:
		--------
		int
			time in the appropriate units
		'''
		return (frame*(exposure_time+interval_time))/oversample_motion_time
	def _update_units(self,unit,orig_type,update_type):
		''' Docstring for _update_units: update the unit from one type to another
		
		Parameters:
		-----------
		unit : int
			unit to be updated
		orig_type : str
			original type of unit
		update_type : str
			type to update unit to
		'''
		if orig_type == 'nm':
			if update_type == 'pix':
				return unit/self.init_dict["Global_Parameters"]["pixel_size"]['pixel_size']
		elif orig_type == 'pix':
			if update_type == 'nm':
				return unit*self.init_dict["Global_Parameters"]["pixel_size"]['pixel_size']
		elif orig_type == 'ms':
			if update_type == 's':
				return unit/1000.
		elif orig_type == 's':
			if update_type == 'ms':
				return unit*1000.
		elif orig_type == 'um^2/s':
			if update_type == 'pix^2/(oversample_motion_time)ms)':
				return unit*(self.init_dict["Global_Parameters"]["pixel_size"]**2)/(1000./self.init_dict["Global_Parameters"]["oversample_motion_time"])
		return 
	def _check_init_dict(self)->bool:
		''' Docstring for _check_init_dict: check the init_dict for the required keys, and if they are consistent with other keys
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		bool: True if the init_dict is correct

		Raises:
		-------
		InitializationKeys: if the init_dict does not have the required keys
		InitializationValues: if the init_dict values are not consistent with each other
		'''
		#check if the init_dict has the required keys
		#TODO
		return True
	def _read_json(self, json_file: str) -> dict:
		''' Docstring for _read_json: read the json file and return the dictionary
		
		Parameters:
		-----------
		json_file : str
			path to the json file
		
		Returns:
		--------
		dict
			dictionary of parameters
		'''
		# Open the json file
		with open(json_file) as f:
			# Load the json file
			data = json.load(f)
		
		# Function to recursively convert lists to NumPy arrays
		def convert_lists_to_arrays(obj):
			if isinstance(obj, list):
				return np.array(obj)
			elif isinstance(obj, dict):
				return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
			else:
				return obj
		
		# Convert lists to NumPy arrays
		data = convert_lists_to_arrays(data)
		
		return data
	def _define_space(self,dims=(100,100),movie_frames=500):
		''' Docstring for _define_space: make the empty space for simulation
		
		Parameters:
		-----------
		dims : tuple, Default = (100,100)
			dimensions of the space to be simulated
		movie_frames : int, Default = 500
			number of frames to be simulated
		
		Returns:
		--------
		space : array-like, shape = (movie_frames,dims[0],dims[1])
			empty space for simulation
		'''
		space = np.zeros((movie_frames,dims[0],dims[1]))
		return space
	def _convert_track_dict_points_per_frame(self,tracks,movie_frames):
		'''Docstring for _convert_track_dict_points_per_frame: convert the track dictionary to a dictionary of points per frame

		Parameters:
		-----------
		tracks : dict
			dictionary of tracks, keys = track number, values = dictionary with keys = 'xy','frames','diffusion_coefficient','initial','hurst'
		movie_frames : int
			number of frames in the movie
		
		Returns:
		--------
		points_per_frame : dict
			dictionary of points per frame, keys = frame number, values = list of (x,y,z) tuples
		
		'''
		points_per_frame = dict(zip([str(i) for i in range(movie_frames)],[[] for i in range(movie_frames)]))
		for i,j in tracks.items():
			for k in range(len(j["frames"])):
				points_per_frame[str(j["frames"][k])].append(j["xy"][k])

		return points_per_frame
	def _convert_track_dict_msd(self,tracks):
		'''Docstring for _convert_track_dict_msd: convert the track dictionary to a dictionary of tracks with the format
		required for the msd function

		Parameters:
		-----------
		tracks : dict
			dictionary of tracks, keys = track number, values = dictionary with keys = 'xy','frames','diffusion_coefficient','initial','hurst'
		
		Returns:
		--------
		track_msd : dict
			dictionary of tracks with the format required for the msd function, keys = track number, values = list of (x,y,T) tuples
			
		
		'''
		track_msd = {}
		for i,j in tracks.items():
			#make a (x,y,T) tuple for each point
			track_msd[i] = []
			for k in range(len(j["xy"])):
				track_msd[i].append((j["xy"][k][0],j["xy"][k][1],j["frames"][k]))
			#add this to the dictionary
			track_msd[i] = np.array(track_msd[i])
		return track_msd
	def _create_track_pop_dict(self,simulation_cube:np.ndarray,**kwargs):
		''' Docstring for _create_cell_tracks: create the tracks for the cell

		Parameters:
		-----------
		simulation_cube : array-like
			empty space for simulation
		Returns:
		--------
		tracks : list
			list of tracks for each cell
		points_per_track : list
			list of number of points in each frame
		'''
		#get the number of frames to be simulated from simulation_cube
		movie_frames = simulation_cube.shape[0]
		#get the lengths of the tracks given a distribution
		track_lengths = sf.get_lengths(track_distribution=self.init_dict["Track_Parameters"]["track_distribution"],
									   track_length_mean=self.track_length_mean,
									   total_tracks=self.init_dict["Track_Parameters"]["num_tracks"]
									   )
		#if track_lengths is larger than the number of frames then set that to the number of frames -1
		track_lengths = np.array([i if i < movie_frames else movie_frames-1 for i in track_lengths])
		#for each track_lengths find the starting frame
		starting_frames = np.array([random.randint(0,movie_frames-i) for i in track_lengths])
		#initialize the Condensates.
		#find area assuming cell_space is [[min_x,max_x],[min_y,max_y]]
		area_cell = np.abs(np.diff(self.init_dict["Cell_Parameters"]['cell_space'][0]))*np.abs(np.diff(self.init_dict["Cell_Parameters"]['cell_space'][1]))
		self.condensates = sf.create_condensate_dict(initial_centers=self.init_dict["Condensate_Parameters"]["initial_centers"],
									initial_scale=self.init_dict["Condensate_Parameters"]["initial_scale"],
									diffusion_coefficients=self.condensate_diffusion_updated,
									hurst_exponent=self.init_dict["Condensate_Parameters"]["hurst_exponent"],
									units_time=np.array([str(self.init_dict["Global_Parameters"]["oversample_motion_time"])+self.init_dict["time_unit"]]))
		#define the top_hat class that will be used to sample the condensates
		top_hat_func = pf.multiple_top_hat_probability(
			num_subspace = len(self.condensate_diffusion_updated),
			subspace_centers = self.init_dict["Condensate_Parameters"]["initial_centers"],
			subspace_radius = self.init_dict["Condensate_Parameters"]["initial_scale"],
			density_dif =  self.init_dict["Condensate_Parameters"]["density_dif"],
			space_size = np.array(area_cell)
			)
		#make a placeholder for the initial position array with all 0s
		initials = np.zeros((self.init_dict["Track_Parameters"]["num_tracks"],3))
		#lets use the starting frames to find the inital position based on the position of the condensates
		for i in range(self.init_dict["Track_Parameters"]["num_tracks"]):
			#get the starting time from the frame, oversample_motion_time, and interval_time
			starting_frame = starting_frames[i]*self.init_dict["Global_Parameters"]["oversample_motion_time"]*self.init_dict["Global_Parameters"]["interval_time"]
			#condensate positions
			condensate_positions = np.zeros((len(self.condensates),2))
			#loop through the condensates
			for ID,cond in self.condensates.items():
				condensate_positions[int(ID)] = cond(int(starting_frame),str(self.init_dict["Global_Parameters"]["oversample_motion_time"])+self.init_dict["time_unit"])["Position"]
			#update the top_hat_func with the new condensate positions
			top_hat_func.update_parameters(subspace_centers=condensate_positions)
			#sample the top hat to get the initial position
			initials[i][:2] = sf.generate_points_from_cls(top_hat_func,
															total_points=1,
															min_x=self.init_dict["Cell_Parameters"]['cell_space'][0][0],
															max_x=self.init_dict["Cell_Parameters"]['cell_space'][0][1],
															min_y=self.init_dict["Cell_Parameters"]['cell_space'][1][0],
															max_y=self.init_dict["Cell_Parameters"]['cell_space'][1][1],
															density_dif=self.init_dict["Condensate_Parameters"]["density_dif"])[0]
		#check to see if there is 2 or 3 values in the second dimension of initials
		if initials.shape[1] == 2:
			#add a third dimension of zeros so that the final shape is (num_tracks,3) with (:,3) being 0s
			initials = np.hstack((initials,np.zeros((self.init_dict["Track_Parameters"]["num_tracks"],1)))) 
		#create tracks
		tracks = {}
		points_per_frame = dict(zip([str(i) for i in range(movie_frames)],[[] for i in range(movie_frames)]))
		if self.init_dict["Track_Parameters"]["track_type"]=="constant":
			for i in range(self.init_dict["Track_Parameters"]["num_tracks"]):
				#make a constant track
				xyz = np.array([initials[i] for ll in range(int(track_lengths[i]))])
				#make the time array
				t = np.arange(track_lengths[i],dtype=int)
				start_frame = starting_frames[i]
				#shift the frames to start at the start_frame
				frames = start_frame + t
				#add this to the dictionary of tracks
				tracks[i] = {'xy':xyz,'frames':frames,'diffusion_coefficient':0,'initial':initials[i],'hurst':0}
				#add the number of points per frame to the dictionary
				for j in range(len(frames)):
					points_per_frame[str(frames[j])].append(xyz[j])
		return tracks, points_per_frame
	

	@property
	def condensates(self)->dict:
		return self._condensates
	@condensates.setter
	def condensates(self,condensates: dict):
		self._condensates = condensates



	@decorators.deprecated("This function is not useful, but is still here for a while in case I need it later")
	def _format_points_per_frame(self,points_per_frame):
		'''
		Docstring for _format_points_per_frame: format the points per frame dictionary so that for each key the set of tracks in it are 
		converted to a numpy array of N x 2 where N is the total amount of points in that frame. You don't need this function. 
		
		Parameters:
		-----------
		points_per_frame : dict
			keys = str(i) for i in range(movie_frames), values = list of tracks, which are collections of [x,y] coordinates
		
		Returns:
		--------
		points_per_frame : dict
			keys = str(i) for i in range(movie_frames), values = numpy array of N x 2 where N is the total amount of points in that frame
		
		'''
		for i in points_per_frame.keys():
			#each value is a list of K lists that are composed of M x 2 arrays where M can be different for each list
			#we want to convert this to a numpy array of N x 2 where N is the total amount of points in that frame
			point_holder = []
			for j in points_per_frame[i]:
				point_holder.append(j)
			points_per_frame[i] = np.array(point_holder)
		return points_per_frame