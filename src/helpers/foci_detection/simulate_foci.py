import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.patches import Circle


class sim_foci():
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
		self.psf_sigma = 10 #~200nm if using 1 bin = 10 nm
		self.point_intensity = 1 # can change for multiple different intensity points
		self.noise = 0 #

		self.uniform_blob = kwargs.get("unifrom",False)
	
	def _define_space(self,**kwargs):
		b = ((self.max_x**2) - (np.pi*(self.radius)**2)*self.density_dif)/((self.max_x**2) - (np.pi*(self.radius)**2))
		return b/(self.max_x*self.max_x)
	
	def _makePoints(self):
		if self.uniform_blob:
			return generate_radial_points(total_points=self.total_points,\
										min_x=self.min_x,\
										max_x=self.max_x,\
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
	
def generate_radial_points(total_points,min_x,max_x,center,radius):

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






