'''
This file contains most functions used in Input/Output of different files and directories

Author: Baljyot Singh Parmar

'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from PIL import Image
import PIL.ImageOps
import sys
import glob as glob

class IO_run_analysis:
	
	def __init__(self) -> None:
		pass

	@staticmethod
	def _save_sptanalysis_data(pp,test):
		fmt = '%d', '%d', '%1.9f', '%1.9f', '%d'
		np.savetxt(pp[:-4] + '_sptsanalysis.csv',test,delimiter = "\t",fmt=fmt)

	@staticmethod
	def _load_superSegger(cd,_string):
		xy_frame_dir_names = []
		#load the data of segmented cells from SuperSegger (cell files)
		for root, subdirs, files in os.walk(cd + _string):
			for d in subdirs:
				if d[:2] == 'xy':
					xy_frame_dir_names.append(cd + _string+ '/' +d)
		return xy_frame_dir_names


def read_data(path, delimiter = ',',skiprow = 1):
	'''
	Parameters
	----------

	path : str
		path to the file to be read
	
	delimiter : str
		delimiter used in the file to seperate individual value

	skiprow : int
		number of rows to skip from the start of the file before reading

	Returns
	-------

	array-like
		array of the loaded data
	
	'''
	data = np.loadtxt(path,delimiter = delimiter,skiprows = skiprow)

	return data

def read_file(file_loc):
    '''
    Parameters
	----------

    file_loc : str
		path to the file

    Returns
	-------

    Array-like 
		the array is 2D array of the pixel locations
    '''
    img = mpimg.imread(file_loc)
    return img


'''
includes functions relating to IO of tiff images
'''

def combine_path(root,path):
	'''
	Parmaters
	---------
	root : str 
		path of the root directory
	path : str
		name of the file or directory to combine with root
	
	Returns
	-------
	string
		combined path given root and path	
	'''
	return os.path.join(root,path)
def find_image(path, ends_with = '.tif',full_path = False):
	'''
	Docstring for find_image:
	Find all files in a directory that end with a certain string

	Parameters
	----------
	path : string
		Full path of the directory in which to find the files. Not recursive find!
	ends_with : string, default = '.tif'
		Unique string to find files that end with this string
	full_path : bool
		if true return the full path of the file, else return just the file name
	Returns
	-------
	list
		list of file paths with the root provided in Parameters with contrainsts of ends_with
	'''
	file_paths = []
	for f in os.listdir(path):
		if not os.path.isfile(f):
			if f.endswith(ends_with):
				if full_path:
					file_path = os.path.join(path,f)
					file_paths.append(file_path)
				else: 
					file_path = f
					file_paths.append(file_path)
	return file_paths

def name_sorter(strings,keyword):
	'''
	Docstring for name_sorter:
	Find all the strings in a list that contain a keyword

	Parameters:
	-----------
	strings : list
		list of strings to search through
	keyword : string
		keyword to search for in the list of strings
	
	Returns:
	--------
	list
		list of strings that contain the keyword
	'''
	keyword_strings = []
	for s in strings:
		if keyword in s:
			keyword_strings.append(s)
	return keyword_strings

def find_files(path, extension, keyword = None):
    '''
    Docstring for find_files
    Finds files in a directory with a specific extension and keyword in the name

    Parameters:
    -----------
    path : str
        path to the directory where the files are located
    extension : str
        extension of the files to be found
    keyword : str    
        keyword to be searched in the file name
    Returns:
    --------
    files : list
        list of files that match the criteria
    '''
    #find all images in the directory using import functions
    files = find_image(path=path,ends_with=extension,full_path=True)
    #sort the files to get only ones conatining the word "RFP" for the flourescent protein
    files = name_sorter(strings=files,keyword=keyword)
    return files

#create a similar function for PIL.ImageOps.invert to convert 16-int unsigned images
def invert_I16u(img,array = False):
	'''
	Parameters
	----------
	img : PIL.Image object
		image object to invert
	array : bool
		if true return a numpy array of the inverted image, else return a PIL.Image object
	
	Returns
	-------
	PIL.Image object or numpy based on the boolean value of array
		inverted image
	'''
	convert_array = np.array(img)
	max_I16u = 65535
	inverted = max_I16u - convert_array
	convert_PIL = Image.fromarray(inverted)
	if array:
		return np.array(convert_PIL)
	else:
		return convert_PIL
def invert_img(path):
    '''
    Parameters
    ----------
    path : string
        full path of the image to invert
    
    Returns
    -------
    PIL Image object
        to save this image use object.save(new_path)
    '''

    img = Image.open(path)
    invert = invert_I16u(img)
    return invert


def save_img(object, path):
	'''
	Paramteres
	----------
	object : PIL.Image object
		image object
	path : str
		path to which this image should be saved
	'''
	object.save(path)
	return 


if __name__ == "__main__":
	print("This is running as the main script./n")
