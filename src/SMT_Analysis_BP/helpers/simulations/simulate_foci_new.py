'''
Documentation for the simulate_foci.py file.
This file contains the class for simulating foci in space.

Author: Baljyot Singh Parmar
'''
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import SMT_Analysis_BP.helpers.simulations.fbm_utility as fbm
import SMT_Analysis_BP.helpers.simulations.condensate_movement as condensate_movement

def get_lengths(track_distribution:str,track_length_mean:int,total_tracks:int):
	''' 
	Returns the track lengths from the distribution track_distribution. The lengths are returned as the closest integer

	Parameters:
	-----------
	track_distribution: distribution of track lengths. Options are "exponential" and "uniform"
	track_length_mean: mean track length
	total_tracks: total number of tracks to be generated

	Returns:
	--------
	track lengths as a numpy array of shape (total_tracks,1)
	
	Notes:
	------
	1. If the distribution is exponential, then the track lengths are generated using exponential distribution.
	2. If the distribution is uniform, then the track lengths are generated using uniform distribution between 0 and 2*track_length_mean.
	3. If the distribution is constant, then all the track lengths are set to the mean track length. (track_length_mean)

	Exceptions:
	-----------
	ValueError: if the distribution is not recognized.
	'''
	if track_distribution=="exponential":
		#make sure each of the lengths is an integer and is greater than or equal to 1
		return np.ceil(np.random.exponential(scale=track_length_mean,size=total_tracks))
	elif track_distribution=="uniform":
		#make sure each of the lengths is an integer
		return np.ceil(np.random.uniform(low=1,high=2*(track_length_mean)-1,size=total_tracks))
	elif track_distribution=="constant":
		return np.ones(total_tracks)*track_length_mean
	else:
		raise ValueError("Distribution not recognized")
def create_condensate_dict(initial_centers: np.ndarray,
							initial_scale: np.ndarray,
							diffusion_coefficient: np.ndarray,
							hurst_exponent: np.ndarray,
							**kwargs) -> dict:
	'''
	Docstring for create_condensate_dict:

	Parameters:
	-----------
	inital_centers: numpy array of shape (num_condensates,2) with the initial centers of the condensates
	initial_scale: numpy array of shape (num_condensates,2) with the initial scales of the condensates
	diffusion_coefficient: numpy array of shape (num_condensates,2) with the diffusion coefficients of the condensates
	hurst_exponent: numpy array of shape (num_condensates,2) with the hurst exponents of the condensates
	**kwargs: additional arguments to be passed to the condensate_movement.Condensate class
	'''
	# check the length of diffusion_coefficient to find the number of condensates
	num_condensates = len(diffusion_coefficient)
	condensates = {}
	units_time = kwargs.get("units_time", ["ms"] * num_condensates)
	for i in range(num_condensates):
		condensates[str(i)] = condensate_movement.Condensate(
			inital_position=initial_centers[i],
			initial_scale=initial_scale[i],
			diffusion_coefficient=diffusion_coefficient[i],
			hurst_exponent=hurst_exponent[i],
			condensate_id=str(i),
			units_time=units_time[i]
		)
	return condensates
def tophat_function_2d(var,center,radius,bias_subspace,space_prob,**kwargs):
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
def generate_points_from_cls(pdf,total_points,min_x,max_x,min_y,max_y,density_dif):
	xy_coords = []
	area = (max_x-min_x)*(max_y-min_y)
	while len(xy_coords) < total_points:
		#generate candidate variable
		var = np.random.uniform([min_x,min_y],[max_x,max_y])
		#generate varibale to condition var1
		var2 = np.random.uniform(0,1)
		#apply condition
		pdf_val = pdf(var)
		if (var2 < ((1./density_dif)*area) * pdf_val):
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
def generate_sphere_points(total_points,center,radius):
	'''Genereate uniformly distributed points in a sphere of radius.

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
	#check to see if the center is an array of size 3
	if len(center) != 3:
		#make it an array of size 3 with the last element being 0
		center = np.array([center[0],center[1],0])

	theta = 2.*np.pi*np.random.random(size=total_points)
	phi = np.arccos(2.*np.random.random(size=total_points)-1.)
	rad = radius*np.cbrt(np.random.random(size=total_points))
	x = rad*np.cos(theta)*np.sin(phi)+center[0]
	y = rad*np.sin(theta)*np.sin(phi)+center[1]
	z = rad*np.cos(phi)+center[2]
	return np.stack((x,y,z),axis =-1)
def radius_spherical_cap(R,center,z_slice):
	''' Find the radius of a spherical cap given the radius of the sphere and the z coordinate of the slice
	Theory: https://en.wikipedia.org/wiki/Spherical_cap, https://mathworld.wolfram.com/SphericalCap.html

	Parameters:
	-----------
	R : float,int
		radius of the sphere
	center : array-like
		[x,y,z] coordinates of the center of the sphere
	z_slice : float,int
		z coordinate of the slice relative to the center of the sphere, z_slice = 0 is the center of the sphere

	Returns:
	--------
	float
		radius of the spherical cap at the z_slice
	
	Notes:
	------
	1. This is a special case of the spherical cap equation where the center of the sphere is at the origin
	'''
	#check if z_slice is within the sphere
	if z_slice > R:
		raise ValueError('z_slice is outside the sphere')
	#check if z_slice is at the edge of the sphere
	if z_slice == R:
		return 0
	#check if z_slice is at the center of the sphere
	if z_slice == 0:
		return R
	#calculate the radius of the spherical cap
	return np.sqrt(R**2 - (z_slice)**2)
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
	
	Notes:
	------
	THIS IS IMPORTANT: MAKE SURE THE TYPES IN EACH PARAMETER ARE THE SAME!!!!
	'''

	mvn = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
	x = domain[0] 
	y = domain[1]
	# meshgrid as a list of [x,y] coordinates
	coords = tf.reshape(tf.stack(tf.meshgrid(x,y),axis=-1),(-1,2))
	gauss = mvn.prob(coords)
	return tf.reshape(gauss, (len(x),len(y)))
def axial_intensity_factor(abs_axial_pos: float|np.ndarray,**kwargs) -> float|np.ndarray:
	'''Docstring
	Calculate the factor for the axial intensity of the PSF given the absolute axial position from the 0 position of 
	the focal plane. This is the factor that is multiplied by the intensity of the PSF

	For now this is a negative exponential decay i.e:
		I = I_0*e^(-|z-z_0|) 
	This function returns the factor e^(-|z-z_0|**2 / (2*2.2**2)) only. 

	Parameters:
	-----------
	abs_axial_pos : float|np.ndarray
		absolute axial position from the 0 position of the focal plane
	kwargs : dict

	Returns:
	--------
	float|np.ndarray
		factor for the axial intensity of the PSF
	'''
	func_type = kwargs.get("func","ones")
	if func_type == "ones":
		return np.ones(len(abs_axial_pos))
	elif func_type == "exponential":
		#for now this uses a negative exponential decay
		return np.exp(-abs_axial_pos**2 / (2*2.2**2))

class Track_generator:
	def __init__(self,
			  cell_space:np.ndarray|list,
			  cell_axial_range:int|float,
			  frame_count:int,
			  exposure_time:int|float,
			  interval_time:int|float,
			  oversample_motion_time:int|float) -> None:
		self.cell_space = cell_space
		self.min_x = self.cell_space[0][0]
		self.max_x = self.cell_space[0][1]
		self.min_y = self.cell_space[1][0]
		self.max_y = self.cell_space[1][1]
		self._min_rel_x = 0
		self._max_rel_x = self.max_x - self.min_x
		self._min_rel_y = 0
		self._max_rel_y = self.max_y - self.min_y
		self.cell_axial_range = cell_axial_range
		self.frame_count = frame_count #count of frames
		self.exposure_time = exposure_time #in ms
		self.interval_time = interval_time #in ms
		self.oversample_motion_time = oversample_motion_time #in ms
		#total time in ms is the exposure time + interval time * (frame_count) / oversample_motion_time
		#in ms
		self.total_time = ((self.exposure_time + self.interval_time)*self.frame_count)/self.oversample_motion_time
	def track_generation_no_transition(self,diffusion_coefficient:np.ndarray|list,
							  hurst_exponent:np.ndarray|list,
							  diffusion_track_amount:np.ndarray|list,
							  hurst_track_amount:np.ndarray|list,
							  track_length_mean:int)->list:
		'''
		'''
		return	[]
	def track_generation_with_transition(self,diffusion_coefficient:np.ndarray|list,
							  hurst_exponent:np.ndarray|list,
							  diffusion_transition_matrix:np.ndarray|list,
							  hurst_transition_matrix:np.ndarray|list,
							  track_length_mean:int)->list:

		'''
		'''
		return []
	def track_generation_constant(self,track_length_mean:int)->list:
		'''
		'''
		return []