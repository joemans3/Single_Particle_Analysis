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

class sim_foci():
	'''_summary_
	'''
	def __init__(self,**kwargs) -> None:
		'''_summary_
		'''
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
		'''_summary_

		Returns
		-------
		_type_
			_description_
		'''
		b = ((self.max_x**2) - (np.pi*(self.radius)**2)*self.density_dif)/((self.max_x**2) - (np.pi*(self.radius)**2))
		return b/(self.max_x*self.max_x)
	
	def _makePoints(self):
		'''_summary_

		Returns
		-------
		_type_
			_description_
		'''
		if self.uniform_blob:
			return generate_radial_points(total_points=self.total_points,\
										center=self.center,\
										radius=self.radius)
		else:								
			return generate_points(pdf = self.pdf, \
								total_points = self.total_points, \
								min_x = self.min_x, \
								max_x = self.max_x, \
								center = self.center, \
								radius = self.radius, \
								bias_subspace_x = self.bias_subspace, \
								space_prob = self.space_prob, \
								density_dif = self.density_dif)

	def simulate_point(self,**kwargs):
		'''_summary_

		Returns
		-------
		_type_
			_description_
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
		
class sim_focii(sim_foci): #is this usefull or not? Turns out to be slower ~x2 than the brute force way.
	def __init__(self,radii=None,repeats=3,detection_kwargs={},sim_kwargs={},fitting_parm={}) -> None:
		'''_summary_

		Parameters
		----------
		radii : _type_, optional
			_description_, by default None
		repeats : int, optional
			_description_, by default 3
		detection_kwargs : dict, optional
			_description_, by default {}
		sim_kwargs : dict, optional
			_description_, by default {}
		fitting_parm : dict, optional
			_description_, by default {}
		'''
		self.radii = radii
		self.detection_kwargs = detection_kwargs
		self.sim_kwargs = sim_kwargs
		self.fitting_parm = fitting_parm
		self.repeats = repeats
		super().__init__(**sim_kwargs)
		self.blob_detector = None
		self._blob_detection_object(detection_kwargs=detection_kwargs,fitting_parm=fitting_parm)

	def _create_sim(self,radius):
		'''_summary_

		Parameters
		----------
		radius : _type_
			_description_

		Returns
		-------
		_type_
			_description_
		'''
		self.radius=radius
		sim_obj = self.simulate_point()
		return sim_obj

	def _repeat_sim(self,repeats,radius):
		'''_summary_

		Parameters
		----------
		repeats : _type_
			_description_
		radius : _type_
			_description_

		Returns
		-------
		_type_
			_description_
		'''
		repeat_obj = {}
		for i in range(repeats):
			repeat_obj[i+1] = self._create_sim(radius=radius)
		return repeat_obj

	def _blob_detection_object(self,detection_kwargs={},fitting_parm={}):
		'''_summary_

		Parameters
		----------
		detection_kwargs : dict, optional
			_description_, by default {}
		fitting_parm : dict, optional
			_description_, by default {}
		'''
		self.blob_detector = blob_detection.blob_detection(path=0,**detection_kwargs)
		self.blob_detector._update_fitting_parameters(kwargs=fitting_parm)

	def _mapdetection(self,radius,repeats):
		'''_summary_

		Parameters
		----------
		radius : _type_
			_description_
		repeats : _type_
			_description_

		Returns
		-------
		_type_
			_description_
		'''
		repeat_map = self._repeat_sim(repeats=repeats,radius=radius)
		found_map = {}
		for i,j in repeat_map.items():
			self.blob_detector.img = np.array(j[0])
			found_map[i] = self.blob_detector.detection(type = "bp")
		return found_map,repeat_map

	def _found_utils(self,found_spots,method="single"):
		'''_summary_

		Parameters
		----------
		found_spots : _type_
			_description_
		method : str, optional
			_description_, by default "single"

		Returns
		-------
		_type_
			_description_

		Raises
		------
		ValueError
			_description_
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
		'''_summary_

		Returns
		-------
		_type_
			_description_
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
		'''_summary_

		Parameters
		----------
		total_points : _type_
			_description_

		Returns
		-------
		_type_
			_description_
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






