import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from PIL import Image
import PIL.ImageOps
import sys
import glob as glob



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
			if f.endswith('.tif'):
				if full_path:
					file_path = os.path.join(path,f)
					file_paths.append(file_path)
				else: 
					file_path = f
					file_paths.append(file_path)
	return file_paths

#create a similar function for PIL.ImageOps.invert to convert 16-int unsigned images
def invert_I16u(img):
	'''
	Parameters
	----------
	img : PIL.Image object
		image object to invert
	
	Returns
	-------
	PIL.Image object
		inverted image
	'''
	convert_array = np.array(img)
	max_I16u = 65535
	inverted = max_I16u - convert_array
	convert_PIL = Image.fromarray(inverted)
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



def read_imag(path,fig = False,ax = False):
	if fig == False:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	ori_img = io.imread(path)
	print(np.shape(ori_img))
	print(ori_img)
	ax.imshow(ori_img)
	plt.show()
	return

# def rgb_to_grey(rgb_img):
#     '''Convert rgb image to greyscale'''
#     return rgb2gray(rgb_img)
    

if __name__ == "__main__":
	print("This is running as the main script./n")
