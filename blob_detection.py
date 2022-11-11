import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter
import import_functions
from skimage.util import dtype
from scipy.ndimage import filters
from skimage.feature import blob
from scipy import spatial
class blob_detection:
	'''
	Parameters
	----------
	Path : string
		Full path of the image to be read
	median : bool
		if true apply a median filter to the image before blog detection
	threshold : float
		threshold for the blob detection
	min_sigma : float
		Minimum value of the gaussian sigma for the blobs
	max_sigma : float
		Maximum value of the gaussian sigma for the blobs
	num_sigma : int
		Eqidistant values between min_sigma and max_sigma to consider
	overlap : float
		Allowed overlap of identified blobs. If 1, full overlap is allowed

	Methods
	-------
	open_file()
		opens the file and applied media filter if true
		retuns an array
	detection()
		applies blob detection using np.blob_log
		returns array of blob attributes
	
	Notes
	-----
	theory: https://www.cse.psu.edu/~rtc12/CSE586/lectures/featureExtractionPart2_6pp.pdf
	https://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/lecture10_detector_descriptors_2015_notes.pdf
	
	'''
	def __init__(self,path,median = False,threshold = 0.0005,min_sigma = 1.0,max_sigma = 1.5,num_sigma = 500,overlap = 1.):
		self.path = path
		self.median = median
		self.threshold = threshold
		self.min_sigma = min_sigma
		self.max_sigma = max_sigma
		self.num_sigma = num_sigma
		self.overlap = overlap

	def open_file(self):
		'''
		Opens and retuns array of the image data

		TODO:
		-----
		Greyscale or not implimentation. 

		Returns
		-------
		array-like
			2D array of the image data
		'''
		file_gray = import_functions.read_file(self.path)
		return file_gray

	def detection(self,type = 'skimage'):
		'''
		Returns
		-------
		array-like
			returns attributes of the blobs
			[x,y,r]
			x : int
				x coordinate of the blob center
			y : int
				y coordinate of the blob center
			r : float
				radius of the blob 
		
		'''
		file = self.open_file()
		if type == 'skimage':
			blobs = blob.blob_log(file,threshold = self.threshold,min_sigma = self.min_sigma,max_sigma = self.max_sigma,num_sigma = self.num_sigma, overlap = self.overlap)
			blobs[:,2]*=np.sqrt(2) #converting the standard deviation of the gaussian fit to radius of the circle 
		elif type == "bp": 
			blobs = blob_detection.blob_logv2(file,threshold = self.threshold,min_sigma = self.min_sigma,max_sigma = self.max_sigma,num_sigma = self.num_sigma, overlap = self.overlap)
			blobs[:,2]*=np.sqrt(2) #converting the standard deviation of the gaussian fit to radius of the circle 
		return np.array(blobs) #blobs returns array of size 3 tuples (x,y,radius) defining the circle defining the spot
	
	@staticmethod
	def _prune_blobs(blobs_array, overlap, *, sigma_dim=1,**kwargs):
		"""Eliminated blobs with area overlap.

		Parameters
		----------
		blobs_array : ndarray
			A 2d array with each row representing 3 (or 4) values,
			``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
			where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
			and ``sigma`` is the standard deviation of the Gaussian kernel which
			detected the blob.
			This array must not have a dimension of size 0.
		overlap : float
			A value between 0 and 1. If the fraction of area overlapping for 2
			blobs is greater than `overlap` the smaller blob is eliminated.
		sigma_dim : int, optional
			The number of columns in ``blobs_array`` corresponding to sigmas rather
			than positions.

		Returns
		-------
		A : ndarray
			`array` with overlapping blobs removed.
		"""
		max_lap = kwargs.get("max_lap",None)
		sigma = blobs_array[:, -sigma_dim:].max()
		distance = 2 * sigma * np.sqrt(blobs_array.shape[1] - sigma_dim)
		tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
		pairs = np.array(list(tree.query_pairs(distance)))
		if len(pairs) == 0:
			return blobs_array
		else:
			for (i, j) in pairs:
				blob1, blob2 = blobs_array[i], blobs_array[j]
				if blob._blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
					# note: this test works even in the anisotropic case because
					# all sigmas increase together.
					if max_lap[-1] > max_lap[-1]:
						blob2[-1] = 0
					else:
						blob1[-1] = 0

		return np.stack([b for b in blobs_array if b[-1] > 0])
	@staticmethod
	def blob_logv2(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
				overlap=.5, log_scale=False, *, exclude_border=False,**kwargs):
		r"""Finds blobs in the given grayscale image. Adapted from the implimentation of skimage blob-log: 
		https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

		Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
		For each blob found, the method returns its coordinates and the standard
		deviation of the Gaussian kernel that detected the blob.

		Parameters
		----------
		image : 2D or 3D ndarray
			Input grayscale image, blobs are assumed to be light on dark
			background (white on black).
		min_sigma : scalar or sequence of scalars, optional
			the minimum standard deviation for Gaussian kernel. Keep this low to
			detect smaller blobs. The standard deviations of the Gaussian filter
			are given for each axis as a sequence, or as a single number, in
			which case it is equal for all axes.
		max_sigma : scalar or sequence of scalars, optional
			The maximum standard deviation for Gaussian kernel. Keep this high to
			detect larger blobs. The standard deviations of the Gaussian filter
			are given for each axis as a sequence, or as a single number, in
			which case it is equal for all axes.
		num_sigma : int, optional
			The number of intermediate values of standard deviations to consider
			between `min_sigma` and `max_sigma`.
		threshold : float, optional.
			The absolute lower bound for scale space maxima. Local maxima smaller
			than thresh are ignored. Reduce this to detect blobs with less
			intensities.
		overlap : float, optional
			A value between 0 and 1. If the area of two blobs overlaps by a
			fraction greater than `threshold`, the smaller blob is eliminated.
		log_scale : bool, optional
			If set intermediate values of standard deviations are interpolated
			using a logarithmic scale to the base `10`. If not, linear
			interpolation is used.
		exclude_border : tuple of ints, int, or False, optional
			If tuple of ints, the length of the tuple must match the input array's
			dimensionality.  Each element of the tuple will exclude peaks from
			within `exclude_border`-pixels of the border of the image along that
			dimension.
			If nonzero int, `exclude_border` excludes peaks from within
			`exclude_border`-pixels of the border of the image.
			If zero or False, peaks are identified regardless of their
			distance from the border.

		Returns
		-------
		A : (n, image.ndim + sigma) ndarray
			A 2d array with each row representing 2 coordinate values for a 2D
			image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
			When a single sigma is passed, outputs are:
			``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
			``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
			deviation of the Gaussian kernel which detected the blob. When an
			anisotropic gaussian is used (sigmas per dimension), the detected sigma
			is returned for each dimension.

		References
		----------
		.. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

		Examples
		--------
		>>> from skimage import data, feature, exposure
		>>> img = data.coins()
		>>> img = exposure.equalize_hist(img)  # improves detection
		>>> feature.blob_log(img, threshold = .3)
		array([[124.        , 336.        ,  11.88888889],
			[198.        , 155.        ,  11.88888889],
			[194.        , 213.        ,  17.33333333],
			[121.        , 272.        ,  17.33333333],
			[263.        , 244.        ,  17.33333333],
			[194.        , 276.        ,  17.33333333],
			[266.        , 115.        ,  11.88888889],
			[128.        , 154.        ,  11.88888889],
			[260.        , 174.        ,  17.33333333],
			[198.        , 103.        ,  11.88888889],
			[126.        , 208.        ,  11.88888889],
			[127.        , 102.        ,  11.88888889],
			[263.        , 302.        ,  17.33333333],
			[197.        ,  44.        ,  11.88888889],
			[185.        , 344.        ,  17.33333333],
			[126.        ,  46.        ,  11.88888889],
			[113.        , 323.        ,   1.        ]])

		Notes
		-----
		The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
		a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
		"""
		image = dtype.img_as_float(image)

		# if both min and max sigma are scalar, function returns only one sigma
		scalar_sigma = (
			True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
		)

		# Gaussian filter requires that sequence-type sigmas have same
		# dimensionality as image. This broadcasts scalar kernels
		if np.isscalar(max_sigma):
			max_sigma = np.full(image.ndim, max_sigma, dtype=float)
		if np.isscalar(min_sigma):
			min_sigma = np.full(image.ndim, min_sigma, dtype=float)

		# Convert sequence types to array
		min_sigma = np.asarray(min_sigma, dtype=float)
		max_sigma = np.asarray(max_sigma, dtype=float)

		if log_scale:
			# for anisotropic data, we use the "highest resolution/variance" axis
			standard_axis = np.argmax(min_sigma)
			start = np.log10(min_sigma[standard_axis])
			stop = np.log10(max_sigma[standard_axis])
			scale = np.logspace(start, stop, num_sigma)[:, np.newaxis]
			sigma_list = scale * min_sigma / np.max(min_sigma)
		else:
			scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
			sigma_list = scale * (max_sigma - min_sigma) + min_sigma

		# computing gaussian laplace
		# average s**2 provides scale invariance
		gl_images = [-filters.gaussian_laplace(image, s) * np.mean(s) ** 2
					for s in sigma_list]

		image_cube = np.stack(gl_images, axis=-1)

		exclude_border = blob._format_exclude_border(image.ndim, exclude_border)
		local_maxima = blob.peak_local_max(
			image_cube,
			threshold_abs=threshold,
			footprint=np.ones((3,) * (image.ndim + 1)),
			threshold_rel=0.0,
			exclude_border=exclude_border,
		)


		# Catch no peaks
		if local_maxima.size == 0:
			return np.empty((0, 3))

		#find the max of the laplacian for each peak found
		#figure out a way to vectorize it using slicing: https://numpy.org/doc/stable/user/basics.indexing.html
		max_lap = image_cube[local_maxima[:,0],local_maxima[:,1],local_maxima[:,2]]



		# Convert local_maxima to float64
		lm = local_maxima.astype(np.float64)

		# translate final column of lm, which contains the index of the
		# sigma that produced the maximum intensity value, into the sigma
		sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

		if scalar_sigma:
			# select one sigma column, keeping dimension
			sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

		# Remove sigma index and replace with sigmas
		lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

		sigma_dim = sigmas_of_peaks.shape[1]

		return blob_detection._prune_blobs(lm, overlap, sigma_dim=sigma_dim,max_lap = max_lap)

if __name__ == "__main__":
	os.chdir('..')
	path = 'DATA/new_days/20190527/rpoc_ez/gfp/rpoc_ez_2.tif'
	a = blob_detection(path,threshold=1e-3)
	b = a.detection(type = 'bp')