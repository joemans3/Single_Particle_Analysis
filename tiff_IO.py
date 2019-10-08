import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io

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

def rgb_to_grey(rgb_img):
    '''Convert rgb image to greyscale'''
    return rgb2gray(rgb_img)
    

if __name__ == "__main__":
	print("This is running as the main script./n")
