import numpy as np
import matplotlib.pylab as plt
import os
import sys
import glob 
from skimage import io
import plotting_functions
from matplotlib.patches import Circle

def read_drop_data(dir_):


	#load all data (x,y,rad,estimated_diameter,std)
	drops = []
	files = glob.glob(dir_ + '/**.csv')
	for i in range(len(files)):
		data = np.loadtxt(files[i],delimiter=",")
		drops.append(data)

	return [drops,files]

def import_images(files):
	

	img_data = []
	file = []
	for i in range(len(files)):
		img_data.append(io.imread(files[i][:-10] + '.tif'))
		file.append(files[i][:-10] + '.tif')


	return img_data,file


def save_plots(dir_):

	drops,files = read_drop_data(dir_)
	image_data,files_t = import_images(files)

	for i in range(len(image_data)):
		fig,ax = plt.subplots()
		ax.imshow(image_data[i],'gray')
		for j in range(len(drops[i])):
			
			if len(np.shape(drops[i])) < 2:
				cir = Circle((drops[i][0],drops[i][1]),drops[i][2],color = 'r',alpha = 1,fill=False)
			else:
				cir = Circle((drops[i][j][0],drops[i][j][1]),drops[i][j][2],color = 'r',alpha = 1,fill = False)
			ax.add_artist(cir)

		plt.savefig(files_t[i][:-4] + '_drop')
	return



