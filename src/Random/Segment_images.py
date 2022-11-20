import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from tifffile import imsave
import mahotas as mh 
import glob

stack_size = 8
path = "/Users/baljyot/Desktop/Baljyot_EXP_RPOC/Images_TM_Spots"


f_img_db = "/repImages"
bf_img_db = "/repImagesBF"

files_f = glob.glob(path + f_img_db + "/**")
files_bf = glob.glob(path + bf_img_db + "/**")

#########
#test
def test(thresh, imgage_i,file_n):
	test_16 = io.imread(files_f[file_n])[imgage_i]
	test_8 = (test_16/256).astype('uint8')
	gaus_fil = mh.gaussian_filter(test_8,thresh)

	rmax = mh.regmax(gaus_fil.astype('uint8'))
	seeds,num_spots = mh.label(rmax)
	T = mh.thresholding.otsu(gaus_fil.astype('uint8'))

	dist_0 = mh.distance(gaus_fil.astype('uint8') > T)
	dist = dist_0.max() - dist_0
	dist -= dist.min()
	dist = dist/float(dist.ptp())*255
	dist = dist.astype(np.uint8)

	blobs = mh.cwatershed(dist,seeds)

	# plt.imshow(blobs)
	# plt.show()

	mask = dist < 225
	mask_i = np.invert(mask)
	mask_f = (gaus_fil.astype('uint8') > T)*test_16




	# fig = plt.figure()
	# ax = fig.add_subplot(141)
	# ax1 = fig.add_subplot(142)
	# ax2 = fig.add_subplot(143)
	# ax3 = fig.add_subplot(144)
	# ax.imshow(test_8,'gray')
	# ax1.imshow(mask*test_8,'gray')
	# ax2.imshow(mask_i*test_8,'gray')
	# ax3.imshow((gaus_fil.astype('uint8') > T)*test_8,'gray')
	# fig.tight_layout()
	#plt.show()
	return mask_f

def tif_stack_conv(files):

	for i in range(len(files)):
		for j in range(stack_size):
			imsave(files[i][:-4] + '_m_{0}'.format(j+1) + '.tif',np.array(test(16,j,i)))

	return 


















