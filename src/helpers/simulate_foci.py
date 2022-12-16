'''
Documentation for the simulate_foci.py file.
This file contains the class for simulating foci in space.

It contains the following classes:

Classes:
--------
1. sim_foci: Class for simulating foci in space.
2. Track_generator: Class for generating tracks for the foci in space.
3. sim_focii: Class for simulating multiple space maps with foci in space and detecting the foci in the space maps.

Notes:
------
1. The class sim_foci is used to simulate foci in space.
2. The class Track_generator is used to generate tracks for the foci in space.
3. The class sim_focii is used to simulate multiple space maps with foci in space and detect the foci in the space maps.

Author: Baljyot Singh Parmar
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.patches import Circle

if __name__=="__main__":
	import sys
	sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
import src.helpers.blob_detection as blob_detection
import src.helpers.Analysis_functions as Analysis_functions
import src.helpers.fbm_utility as fbm

class sim_foci():
	''' 
	Class for simulating foci in space.

	Initalization parameters:
	-------------------------
	max_x: maximum x value for the space, default is 200
	min_x: minimum x value for the space, default is 0
	radius: radius of the blob, default is 20
	center: center of the blob, default is [100,100]
	total_points: total number of points to be generated, default is 500
	density_dif: difference in density between the center and the edge of the blob, default is 5
	bias: bias for the space, default is density_dif/(max_x*max_x)
	space: space probability for the points to be generated outside the space, default is b/(max_x*max_x)
	pdf: probability distribution function for the points to be generated, default is tophat_function_2d
	psf_sigma: sigma for the psf, default is 10
	point_intensity: intensity of the point, default is 1
	noise: noise for the point, default is 0
	uniform_blob: if the blob is uniform or not, default is False

	Methods:
	--------
	_define_space: defines the space probability for the points to be generated in the space, returns the space probability as a float.
	_makePoints: generates the points in the space, returns the points as a numpy array of shape (total_points,2).
	simulate_point: simulates the point in space and returns the space map as a numpy array of shape (max_x,max_x) and the points as a numpy array of shape (total_points,2).

	'''
	def __init__(self,**kwargs) -> None:
		self.max_x = kwargs.get("max_x",200)
		self.min_x = kwargs.get("min_x",0)
		self.radius = kwargs.get("radius",20)
		self.center = kwargs.get("center",[100,100])
		self.total_points = kwargs.get("total_points",500)
		self.x = tf.cast(tf.linspace(self.min_x,self.max_x,self.max_x),tf.float64)
		self.y = tf.cast(tf.linspace(self.min_x,self.max_x,self.max_x),tf.float64)
		self.density_dif = kwargs.get("density_dif",5)
		self.bias_subspace = kwargs.get("bias",self.density_dif/(self.max_x*self.max_x))
		self.space_prob = kwargs.get("space",self._define_space())
		self.pdf = kwargs.get("pdf",tophat_function_2d)
		self.psf_sigma = kwargs.get("psf_sigma",10) #~200nm if using 1 bin = 10 nm
		self.point_intensity = 1 # can change for multiple different intensity points
		self.noise = 0 #

		self.uniform_blob = kwargs.get("unifrom",False)
	
	def _define_space(self,**kwargs):
		''' 
		Docstring for _define_space:
		---------------------------
		Defines the space probability for the points to be generated in the space.

		Parameters:
		-----------
		kwargs: keyword arguments

		Returns:
		--------
		space probability as a float
		
		'''
		b = ((self.max_x**2) - (np.pi*(self.radius)**2)*self.density_dif)/((self.max_x**2) - (np.pi*(self.radius)**2))
		return b/(self.max_x*self.max_x)
	
	def _makePoints(self,points=None):
		''' 
		Docstring for _makePoints:
		---------------------------
		Generates the points in the space.

		Returns:
		--------
		points as a numpy array of shape (total_points,2), where each row is a point in the space.
		
		Notes:
		------
		1. If the blob is uniform, then the points are generated using generate_radial_points function.
		2. If the blob is not uniform, then the points are generated using generate_points function.
		'''

		point_num = 0
		if points==None:
			point_num = self.total_points
		else:
			point_num = points
		if self.uniform_blob:
			return generate_radial_points(total_points=point_num,\
										center=self.center,\
										radius=self.radius)
		else:								
			return generate_points(pdf = self.pdf, \
								total_points = point_num, \
								min_x = self.min_x, \
								max_x = self.max_x, \
								center = self.center, \
								radius = self.radius, \
								bias_subspace_x = self.bias_subspace, \
								space_prob = self.space_prob, \
								density_dif = self.density_dif)

	def simulate_point(self,**kwargs):
		''' 
		Docstring for simulate_point:
		---------------------------
		Simulates the point in space and returns the point and the space map.

		Parameters:
		-----------
		kwargs: keyword arguments

		Returns:
		--------
		space map as a numpy array of shape (max_x,max_x)
		points as a numpy array of shape (total_points,2)

		Notes:
		------
		1. The space map is generated using get_gaussian function.
		2. The points are generated using _makePoints function.
		3. The point intensity and the number of points can be changed using the keyword arguments.
		'''
		num_points = kwargs.get("points",self.total_points)
		point_intensity = kwargs.get("intensity",np.ones(num_points)*self.point_intensity)
		if np.isscalar(point_intensity):
			point_intensity *= np.ones(len(num_points))
		x = np.arange(self.min_x,self.max_x,1.)
		points = self._makePoints()
		space_map = np.zeros((len(x),len(x)))
		for i in points:
			space_map += get_gaussian(i,np.ones(2)*self.psf_sigma,domain=[x,x])
		return space_map,points
	
	def generate_map_from_points(self,points):
		''' 
		Docstring for generate_map_from_points:
		---------------------------
		Generates the space map from the points.

		Parameters:
		-----------
		points: numpy array of shape (total_points,2)

		Returns:
		--------
		1. space map as a numpy array of shape (max_x,max_x)
		2. points as a numpy array of shape (total_points,2)
		
		
		Notes:
		------
		1. The space map is generated using get_gaussian function.
		'''
		x = np.arange(self.min_x,self.max_x,1.)
		space_map = np.zeros((len(x),len(x)))
		for i in points:
			space_map += get_gaussian(i,np.ones(2)*self.psf_sigma,domain=[x,x])
		return space_map,points

class Track_generator(sim_foci):
	'''
	Generic class for generating tracks.
	Inherits from sim_foci class.
	MRO: Track_generator -> sim_foci -> object

	Initalization parameters:
	-------------------------
	track_length_mean: mean track length
	track_type: type of track to generate. Options are "fbm" and "ctrw"
	track_hurst: hurst parameter for fbm
	track_distribution: distribution of track lengths. Options are "exponential" and "uniform", default is "exponential"
	
	Method:
	-------
	_get_Track: returns a track of mean length track_length_mean from the distribution track_distribution
	_get_fbm: returns a fractional brownian motion track with hurst parameter track_hurst
	_get_ctrw: returns a continuous time random walk track
	'''	
	def __init__(self,track_parameters={},sim_parameters={}) -> None:
		# Get the track parameters
		self.track_length_mean=track_parameters.get("track_length_mean",10)
		self.track_type=track_parameters.get("track_type","fbm")
		self.track_hurst=track_parameters.get("track_hurst",0.5)
		self.track_distribution=track_parameters.get("track_distribution","exponential")
		self.diffusion_coefficient=track_parameters.get("diffusion_coefficient",1)
		#total tracks is calcualted from the total points defined in sim_parameters and the mean track length
		self.total_tracks = int(sim_parameters.get("total_points")/self.track_length_mean)
		# Initialize the base class
		super(Track_generator,self).__init__(**sim_parameters)
		pass

	def _get_total_tracks(self):
		''' 
		Docstring for _get_total_tracks:
		---------------------------
		Returns the total number of tracks to be generated from the total number of points and the mean track length.
		Sets the total_tracks attribute.

		Returns:
		--------
		total_tracks as an integer
		
		Notes:
		------
		1. The total number of tracks is calculated from the total number of points and the mean track length.
		'''
		self.total_tracks = int(self.total_points/self.track_length_mean)
		return self.total_tracks
	def _get_lengths(self):
		''' 
		Docstring for _get_lengths:
		---------------------------
		Returns the track lengths from the distribution track_distribution.
		The lengths are returned as the closest integer

		Returns:
		--------
		track lengths as a numpy array of shape (total_tracks,1)
		
		Notes:
		------
		1. If the distribution is exponential, then the track lengths are generated using exponential distribution.
		2. If the distribution is uniform, then the track lengths are generated using uniform distribution between 0 and 2*track_length_mean.

		Exceptions:
		-----------
		ValueError: if the distribution is not recognized.
		'''
		if self.track_distribution=="exponential":
			#make sure each of the lengths is an integer and is greater than or equal to 1
			return np.round(np.random.exponential(scale=self.track_length_mean,size=self.total_tracks))
		elif self.track_distribution=="uniform":
			#make sure each of the lengths is an integer
			return np.round(np.random.uniform(low=1,high=2*(self.track_length_mean)-1,size=self.total_tracks))
		else:
			raise ValueError("Distribution not recognized")

	def _get_Track(self):
		'''
		Docstring for _get_Track:
		-------------------------
		For each track length, generates a track of that length, and returns the tracks using the track_type.
		First generates the track lengths using _get_lengths function and then generates the tracks using _get_fbm or _get_ctrw functions.
		Returns the tracks as a list of shape (total_tracks,track_length_mean,2). 

		Returns:
		--------
		tracks as a list of shape (total_tracks,track_length_mean,2)

		Notes:
		------
		1. If the track_type is "fbm", then the tracks are generated using _get_fbm function.
		2. If the track_type is "ctrw", then the tracks are generated using _get_ctrw function.

		Exceptions:
		-----------
		ValueError: if the track_type is not recognized.
		'''
		track_lengths = self._get_lengths()
		tracks = []
		for i in track_lengths:
			if self.track_type=="fbm":
				tracks.append(self._make_fbm_track(i))
			elif self.track_type=="ctrw":
				tracks.append(self._get_ctrw(i))
			else:
				raise ValueError("Track type not recognized")
		return tracks
	
	def _flatten_tracks(self,tracks):
		'''
		Docstring for _flatten_tracks:
		------------------------------
		Flattens the tracks list into a numpy array of shape (total_tracks*track_length_mean,2)

		Parameters:
		-----------
		tracks: list of tracks

		Returns:
		--------
		flattened tracks as a numpy array of shape (total_tracks*track_length_mean,2)
		'''
		if len(tracks) == 0:
			return None
		flattened_tracks = np.array(tracks[0])
		for i in tracks[1:]:
			flattened_tracks = np.concatenate((flattened_tracks,i),axis=0)
		return flattened_tracks
	
	def _tracks_to_points(self,tracks):
		'''
		Docstring for _tracks_to_points:
		-------------------------------
		Converts the tracks list into a numpy array of points of shape (total_tracks*track_length_mean,2)

		Parameters:
		-----------
		tracks: list of tracks

		Returns:
		--------
		points as a numpy array of shape (total_tracks*track_length_mean,2)
		'''
		flattened_tracks = self._flatten_tracks(tracks)
		return flattened_tracks

	def _get_initials(self):
		'''
		Docstring for _get_initials:
		----------------------------
		Generate random points as initials for the tracks. Returns the initials as a numpy array of shape (total_tracks,2)
		Uses the _makePoints function from the base class.

		Returns:
		--------
		initials as a numpy array of shape (total_tracks,2)
		'''
		return self._makePoints(self.total_tracks)

	def create_points(self,diffusion_coefficients):
		'''
		Docstring for create_points:
		-----------------------
		Runs the flow of the class. First generates the tracks using _get_Track function, and then converts the tracks to points using Tracks_to_points function.
		Returns the points as a numpy array of shape (total_tracks*track_length_mean,2)

		Parameters:
		-----------
		diffusion_coefficients: list of diffusion coefficients for each track, if scalar, then all tracks have the same diffusion coefficient

		Returns:
		--------
		points as a numpy array of shape (total_tracks*track_length_mean,2)
		'''
		#if diffusion_coefficients is a scalar, then make it a list of length total_tracks
		#updates the total_tracks attribute
		_ = self._get_total_tracks()
		if np.isscalar(diffusion_coefficients):
			diffusion_coefficients = np.ones(self.total_tracks)*diffusion_coefficients
		tracks = self._get_Track()
		#use the initials to shift the tracks to the right place, also shift each track by the diffusion coefficient
		initials = self._get_initials()
		for i in range(len(tracks)):
			tracks[i] *= np.sqrt(2*diffusion_coefficients[i])
			tracks[i] += initials[i]
			#make sure the track x,y values are bounded by a circle of radius self.radius and center self.center=(x,y)
			#TODO this destroys the diffusion coefficient of the track and should be fixed
			#Currently, the points outside the circle are not removed, but are just shifted to the boundary of the circle
			#this creates a larger density of points near the boundary of the circle and makes the blob detection algorithm less accurate
			tracks[i] = self._bound_points(tracks[i])
		points = self._tracks_to_points(tracks)
		return points

	def _bound_points(self,points):
		'''
		Docstring for _bound_points:
		----------------------------
		Bound the points to a circle of radius self.radius and center self.center=(x,y)
		Theory: if the distance of a point from the center is greater than the radius, then the point is outside the circle.
		We can get the angle of the point from the center and then get the point on the circle that is at that angle and at a distance of self.radius from the center.
		We can then replace the point with the new point.

		Parameters:
		-----------
		points: numpy array of shape (n,2)

		Returns:
		--------
		bounded points as a numpy array of shape (n,2)
		'''
		#make sure the points are within a circle of radius self.radius and center self.center=(x,y)
		#first get the distance of each point from the center
		distances = np.sqrt((points[:,0]-self.center[0])**2 + (points[:,1]-self.center[1])**2)
		#now get the points that are outside the circle
		outside_points = points[distances>self.radius]
		#now get the points that are inside the circle
		inside_points = points[distances<=self.radius]
		#now get the angles of the outside points
		angles = np.arctan2(outside_points[:,1]-self.center[1],outside_points[:,0]-self.center[0])
		#now get the new x,y values of the outside points
		new_x = self.center[0] + self.radius*np.cos(angles)
		new_y = self.center[1] + self.radius*np.sin(angles)
		#now make the new points
		new_points = np.stack((new_x,new_y),axis=-1)
		#now combine the inside and outside points
		return np.concatenate((inside_points,new_points),axis=0)

	def _make_fbm_track(self,length):
		''' 
		Docstring for _make_fbm_track:
		------------------------------
		Returns a fractional brownian motion track of length length and hurst parameter self.track_hurst.

		Parameters:
		-----------
		length: length of the track

		Returns:
		--------
		track as a numpy array of shape (length,2)
		'''
		#make the length an int if it is not
		length = int(length)
		x,y = fbm.get_fbm_sample(l=1,n=length,h=self.track_hurst,d=2)[1]
		return np.stack((x,y),axis =-1)
	def _get_ctrw(self):
		pass

class sim_focii(Track_generator): #is this usefull or not? Turns out to be slower ~x2 than the brute force way.
	''' 
	Class for simulating focii in space, and detecting them.
	Inherits from Track_generator class, which inherits from sim_foci class.
	MRO: sim_focii -> Track_generator -> sim_foci -> object

	Initalization parameters:
	-------------------------
	radii: list of radii to simulate, default is [1,2,3], in pixels
	repeats: number of times to repeat the simulation for each radii, default is 3
	detection_kwargs: dictionary of parameters for the detection algorithm
	sim_kwargs: dictionary of parameters for the simulation algorithm
	fitting_parm: dictionary of parameters for the fitting algorithm
	track_parm: dictionary of parameters for the track generation algorithm

	Methods:
	--------
		_create_sim: creates a simulation object
		_repeat_sim: repeats the simulation for the given number of repeats and radii
		_blob_detection_object: creates a blob detection object, with the given parameters
		_map_detection: maps the detection algorithm to the simulation
		_found_utils: returns the number of focii found, and the number of focii that were simulated
		radius_analysis: returns the number of focii found, and the number of focii that were simulated, for each radius
		total_point_analysis: returns the number of focii found, and the number of focii that were simulated, for each number of points simulated
	
	Notes:
	------
		- The detection algorithm is a blob detection algorithm, with the parameters given in the detection_kwargs dictionary.
		- The simulation algorithm is a simulation of focii in space, with the parameters given in the sim_kwargs dictionary.
		- The fitting algorithm is a fitting algorithm for the focii, with the parameters given in the fitting_parm dictionary.
		- The track generation algorithm is a track generation algorithm, with the parameters given in the track_parm dictionary.

	'''
	def __init__(self,radii=None,repeats=3,detection_kwargs={},sim_kwargs={},fitting_parm={},track_parm={}) -> None:
		'''
		Docstring for __init__:
		-----------------------
		Initializes the class, and creates the blob detection object.

		Parameters:
		-----------
		radii: array-like or list, default is None, in pixels
			list of radii to simulate
		repeats: int, default is 3
			number of times to repeat the simulation for each radii
		detection_kwargs: dictionary
			dictionary of parameters for the detection algorithm
		sim_kwargs: dictionary
			dictionary of parameters for the simulation algorithm
		fitting_parm: dictionary
			dictionary of parameters for the fitting algorithm
		track_parm: dictionary
			dictionary of parameters for the track generation algorithm

		Notes:
		------
		1. The detection algorithm is a blob detection algorithm, with the parameters given in the detection_kwargs dictionary.
		2. The simulation algorithm is a simulation of focii in space, with the parameters given in the sim_kwargs dictionary.
		3. The fitting algorithm is a fitting algorithm for the focii, with the parameters given in the fitting_parm dictionary.
		4. The track generation algorithm is a track generation algorithm, with the parameters given in the track_parm dictionary.
		'''
		self.radii = radii
		self.detection_kwargs = detection_kwargs
		self.sim_kwargs = sim_kwargs
		self.fitting_parm = fitting_parm
		self.repeats = repeats
		super(sim_focii,self).__init__(sim_parameters=sim_kwargs,track_parameters=track_parm)
		self.blob_detector = None
		self._blob_detection_object(detection_kwargs=detection_kwargs,fitting_parm=fitting_parm)
		#if self.use_points is True: then points are simualted independently and then the detection algorithm is applied to each point.
		# if self.use_points is False: then tracks are simulated, and then the detection algorithm is applied to each track.
		self.use_points = True

	def _create_sim(self,radius):
		'''
		Docstring for _create_sim:
		--------------------------
		Creates a simulation object.

		Parameters:
		-----------
		radius: float
			radius of the focii to simulate, in pixels
		
		Returns:
		--------
		sim_obj: simulation object, with the given radius, and the simulation parameters given in the sim_kwargs dictionary.

		'''
		self.radius=radius
		if self.use_points is True:
			sim_obj = self.simulate_point()
		else:
			sim_obj = self.generate_map_from_points(self.create_points(self.diffusion_coefficient))
		return sim_obj

	def _repeat_sim(self,repeats,radius):
		'''
		Docstring for _repeat_sim:
		--------------------------
		Repeats the simulation for the given number of repeats and radii.

		Parameters:
		-----------
		repeats: number of times to repeat the simulation for each radii, default is 3
		radius: radius of the focii to simulate, in pixels

		Returns:
		--------
		repeat_obj: dictionary of simulation objects, with the given radius, and the simulation parameters given in the sim_kwargs dictionary.
		'''
		repeat_obj = {}
		for i in range(repeats):
			repeat_obj[i+1] = self._create_sim(radius=radius)
		return repeat_obj

	def _blob_detection_object(self,detection_kwargs={},fitting_parm={}):
		'''
		Docstring for _blob_detection_object:
		-------------------------------------
		Creates a blob detection object, with the given parameters.

		Parameters:
		-----------
		detection_kwargs: dictionary of parameters for the detection algorithm
		fitting_parm: dictionary of parameters for the fitting algorithm

		Notes:
		------
		1. The detection algorithm is a blob detection algorithm, with the parameters given in the detection_kwargs dictionary.
		2. The fitting algorithm is a fitting algorithm for the focii, with the parameters given in the fitting_parm dictionary.
		3. This function creates a blob detection object, and assigns it to the class variable self.blob_detector.
		'''
		self.blob_detector = blob_detection.blob_detection(path=0,**detection_kwargs)
		self.blob_detector._update_fitting_parameters(kwargs=fitting_parm)

	def _mapdetection(self,radius,repeats):
		'''
		Docstring for _mapdetection:
		----------------------------
		Maps the detection algorithm to the simulation.

		Parameters:
		-----------
		radius: radius of the focii to simulate, in pixels
		repeats: number of times to repeat the simulation for each radii, default is 3

		Returns:
		--------
		found_map: dictionary of the found focii, with the given radius, and the simulation parameters given in the sim_kwargs dictionary.
		repeat_map: dictionary of the simulation objects, with the given radius, and the simulation parameters given in the sim_kwargs dictionary.

		Notes:
		------
		1. The detection algorithm is a blob detection algorithm, with the parameters given in the detection_kwargs dictionary.
		2. All the simulation objects are stored in the repeat_map dictionary.
		'''
		repeat_map = self._repeat_sim(repeats=repeats,radius=radius)
		found_map = {}
		for i,j in repeat_map.items():
			self.blob_detector.img = np.array(j[0])
			found_map[i] = self.blob_detector.detection(type = "bp")
		return found_map,repeat_map

	def _found_utils(self,found_spots,method="single"):
		'''
		Docstring for _found_utils:
		---------------------------
		Extracts the found focii from the found_spots dictionary, and returns the mean and standard deviation of the found focii.

		Parameters:
		-----------
		found_spots: dictionary of the found focii, with the given radius, and the simulation parameters given in the sim_kwargs dictionary.
		method: if detection is done on both the scale space and the fit, then the method is "both", else it is "single"

		Returns:
		--------
		if method=="single":
		mean: mean of the found focii
		std: standard deviation of the found focii
		if method=="both":
		sig_mean: mean of the found focii from the scale space
		sig_std: standard deviation of the found focii from the scale space
		fit_mean: mean of the found focii from the fit
		fit_std: standard deviation of the found focii from the fit

		'''
		
		if method=="single":
			try:
				temp = [k[:,2:] for i,k in found_spots.items()]
			except:
				raise ValueError("More than one spot found")
			return np.mean(temp),np.std(temp)
		if method=="both":
			sig_mean = []
			fit_mean = []
			for i,j in found_spots.items():
				sig_mean.append(j[1][:,2:])
				fit_mean.append(j[0][:,2:])
			
			return np.mean(Analysis_functions.flatten(sig_mean)),np.std(Analysis_functions.flatten(sig_mean)),np.mean(Analysis_functions.flatten(fit_mean)),np.std(Analysis_functions.flatten(fit_mean))

	def radius_analysis(self):
		'''
		Docstring for radius_analysis:
		------------------------------
		Performs the radius analysis, and returns the mean and standard deviation of the found focii.

		Returns:
		--------
		if self.blob_detector.verbose:
			sig_means: mean of the found focii from the scale space
			sig_stds: standard deviation of the found focii from the scale space
			fit_means: mean of the found focii from the fit
			fit_stds: standard deviation of the found focii from the fit
		if not self.blob_detector.verbose:
			fit_means: mean of the found focii
			fit_stds: standard deviation of the found focii

		'''
		sig_means = []
		fit_means = []
		sig_stds = []
		fit_stds = []
		if self.blob_detector.verbose:
			for i in self.radii:
				found,repeat = self._mapdetection(radius=i,repeats=self.repeats)
				sig_mean,sig_std,fit_mean,fit_std = self._found_utils(found_spots=found,method="both")
				sig_means.append(sig_mean)
				sig_stds.append(sig_std)
				fit_means.append(fit_mean)
				fit_stds.append(fit_std)
		else:
			for i in self.radii:
				found,repeat = self._mapdetection(radius=i,repeats=self.repeats)
				fit_mean,fit_std = self._found_utils(found_spots=found)
				fit_means.append(fit_mean)
				fit_stds.append(fit_std)
		if self.blob_detector.verbose:
			return {"sig_mean":sig_means,"sig_std":sig_stds,"fit_mean":fit_means,"fit_stds":fit_stds}
		else:
			return {"fit_mean":fit_means,"fit_stds":fit_stds}
	def total_points_radius_analysis(self,total_points):
		'''
		Docstring for total_points_radius_analysis:
		-------------------------------------------
		Performs the radius analysis, and returns the mean and standard deviation of the found focii, with the total number of points in the simulation set to total_points.
		
		Parameters:
		-----------
		total_points: total number of points in the simulation

		Returns:
		--------
		if self.blob_detector.verbose:
			sig_means: mean of the found focii from the scale space
			sig_stds: standard deviation of the found focii from the scale space
			fit_means: mean of the found focii from the fit
			fit_stds: standard deviation of the found focii from the fit
		if not self.blob_detector.verbose:
			fit_means: mean of the found focii
			fit_stds: standard deviation of the found focii
		
		Notes:
		------
		1. Total points is set to the total number of points in the simulation, and the radius is set to the radius of the top hat distribution.

		'''
		self.total_points = total_points
		return self.radius_analysis()



def tophat_function_2d(var,center,radius,bias_subspace,space_prob):
	'''
	Defines a circular top hat probability distribution with a single biased region defining the hat.
	The rest of the space is uniformly distrubuted in 2D

	Parameters
	----------
	var : array-like, float
		[x,y] defining sampling on the x,y span of this distribution
	center : array-like, float
		[c1,c2] defining the center coordinates of the top hat region
	radius : float
		defines the radius of the circular tophat from the center
	bias_subspace : float
		probability at the top position of the top hat
	space_prob : float 
		probability everywhere not in the bias_subspace
	
	Returns
	-------
	float, can be array-like if var[0],var[1] is array-like
		returns the value of bias_subspace or space_prob depending on where the [x,y] data lies
	
	'''
	x = var[0]
	y = var[1]
	if ((x-center[0])**2+(y-center[1])**2) <= radius**2:
		return bias_subspace
	else:
		return space_prob

def generate_points(pdf,total_points,min_x,max_x,center,radius,bias_subspace_x,space_prob,density_dif):
	''' 
	genereates random array of (x,y) points given a distribution using accept/reject method

	Parameters
	----------
	pdf : function
		function which defines the distribution to sample from
	total_points : int
		total points to sample
	min_x : float
		lower bound to the support of the distribution
	max_x : float
		upper bound to the support of the distribution
	center : array-like of float
		coordinates of the center of the top hat
	redius : float
		raidus of the top hat
	bias_subspace : float
		probability at the top hat
	space_prob : float
		probaility everywhere not at the top hat
	
	Returns
	-------
	array-like
		[x,y] coordinates of the points sampled from the distribution defined in pdf
	'''
	xy_coords = []
	while len(xy_coords) < total_points:
		#generate candidate variable
		var = np.random.uniform([min_x,min_x],[max_x,max_x])
		#generate varibale to condition var1
		var2 = np.random.uniform(0,1)
		#apply condition
		pdf_val = pdf(var,center,radius,bias_subspace_x,space_prob)
		if (var2 < ((1./density_dif)*(max_x-min_x)**2) * pdf_val):
			xy_coords.append(var)
	return np.array(xy_coords)
	
def generate_radial_points(total_points,center,radius):
	'''Genereate uniformly distributed points in a circle of radius.

	Parameters
	----------
	total_points : int
		total points from this distribution
	center : array-like or tuple 
		coordinate of the center of the radius. [x,y,...]
	radius : float-like
		radius of the region on which to 

	Returns
	-------
	(n,2) size array
		array of coordinates of points genereated (N,3) N = # of points, 2 = dimentions
	'''
	theta = 2.*np.pi*np.random.random(size=total_points)
	rad = radius*np.sqrt(np.random.random(size=total_points))
	x = rad*np.cos(theta)+center[0]
	y = rad*np.sin(theta)+center[1]
	return np.stack((x,y),axis =-1)
 
def get_gaussian(mu, sigma,domain = [list(range(10)),list(range(10))]):
	'''
	Parameters
	----------
	mu : array-like or float of floats
		center position of gaussian (x,y) or collection of (x,y)
	sigma : float or array-like of floats of shape mu
		sigma of the gaussian
	domain : array-like, Defaults to 0->9 for x,y
		x,y domain over which this gassuain is over


	Returns
	-------
	array-like 2D 
		values of the gaussian centered at mu with sigma across the (x,y) points defined in domain
	
	'''

	mvn = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
	x = domain[0] 
	y = domain[1]
	# meshgrid as a list of [x,y] coordinates
	coords = tf.reshape(tf.stack(tf.meshgrid(x,y),axis=-1),(-1,2))
	gauss = mvn.prob(coords)
	return tf.reshape(gauss, (len(x),len(y)))


if __name__ == "__main__":
	#define whole space
	#E. coli cell
	max_x = 200 #nm
	min_x = 0

	#define the ranges of the biased subspace
	radius = 20. #nm
	center = [100,100] #nm
	total_points = 500

	#create a mesh 
	x,y = tf.cast(tf.linspace(min_x,max_x,max_x),tf.float64), tf.cast(tf.linspace(min_x,max_x,max_x), tf.float64)



	density_dif = 5.0

	b = ((max_x**2) - (np.pi*(radius)**2)*density_dif)/((max_x**2) - (np.pi*(radius)**2))




	space_prob = b/(max_x**2)
	bias_subspace_x = density_dif/(max_x**2)
	result = generate_points(tophat_function_2d,total_points,min_x,max_x,center,radius,bias_subspace_x,space_prob,density_dif)

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(result[:,0],result[:,1],'r.')
	cir = Circle(center,radius = radius,fill = False)
	ax.add_artist(cir)
	plt.xlim((min_x,max_x))
	plt.ylim((min_x,max_x))
	plt.show()

	sigma = np.array([20.,20.],dtype = type(result[0][0]))


	full_img = []
	for i in range(len(result)):

		full_img.append(np.array(get_gaussian(result[i], sigma,domain = [x,y])))


	# im = Image.fromarray(np.sum(np.array(full_img),axis = 0))
	# im.save("sum.tif")
	# im = Image.fromarray(np.mean(np.array(full_img),axis = 0))
	# im.save("mean.tif")
	# im = Image.fromarray(np.std(np.array(full_img),axis = 0))
	# im.save("std.tif")
	# im = Image.fromarray(np.max(np.array(full_img),axis = 0))
	# im.save("max.tif")


	#test CLT
	ratio_inside = []
	for k in range(1000):
		inside = 0
		result = generate_points(tophat_function_2d,total_points,min_x,max_x,center,radius,bias_subspace_x,space_prob)
		for i in result:
			x = i[0]
			y = i[1]
			if ((x-center[0])**2+(y-center[1])**2) <= radius**2:
				inside +=1
		ratio_inside.append(inside/total_points)
	print("expected: {0}, acutal: {1} +/- {2}".format(density_dif*(np.pi*radius**2)/max_x**2,np.mean(ratio_inside),np.std(ratio_inside)))

	plt.hist(ratio_inside)
	plt.xlabel("Probability to be in subsection")
	plt.axvline(x=density_dif*(np.pi*radius**2)/max_x**2)
	plt.show()






