import csv
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter
from skimage.feature import blob_log

import import_functions


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

		Returns
		-------
		array-like
			2D array of the image data
		'''
		# file = np.array(cv2.imread(self.path))
		
		# if self.median == True:
		# 	file_gray = median_filter(input = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY),size = 20)
		# else:
		# 	file_gray = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
		file_gray = import_functions.read_file(self.path)
		return file_gray

	def detection(self):
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
		blobs = blob_log(file,threshold = self.threshold,min_sigma = self.min_sigma,max_sigma = self.max_sigma,num_sigma = self.num_sigma, overlap = self.overlap)
		blobs[:,2]*=np.sqrt(2) #converting the standard deviation of the gaussian fit to radius of the circle 
		return np.array(blobs) #blobs returns array of size 3 tuples (x,y,radius) defining the circle defining the spot
