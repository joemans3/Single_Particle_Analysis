import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
from Analysis_functions import *
from plotting_functions import *
from scipy.ndimage import gaussian_filter
from tifffile import imsave
import matplotlib.pyplot as plt
import fbm_utility as fbm
from scipy.stats import powerlaw

'''
TAKEAWAYS:

1) small track lengths with same diffusion coefficient when simulated give an 
apparent diffusion coefficient that is much lower.

2) angles between track localizations is uniform which is not what we 
find in actual data where there is more emphasis on 180 deg.

3) 



'''


'''
DOCUMENTATION














'''



#test
def d_rand(l):

	n = l
	x = np.zeros(n)
	y = np.zeros(n)
	for i in range(1, n):
		val = np.random.randint(1, 4)
		if val == 1:
			x[i] = x[i - 1] + 1
			y[i] = y[i - 1]
		elif val == 2:
			x[i] = x[i - 1] - 1
			y[i] = y[i - 1]
		elif val == 3:
			x[i] = x[i - 1]
			y[i] = y[i - 1] + 1
		else:
			x[i] = x[i - 1]
			y[i] = y[i - 1] - 1

	return np.array([x,y])
















#random number from min - max
def random_scale(minv,maxv):

	return (maxv - minv) * np.random.random() + minv


#create a dynamic rectanglar grid

def make_grid(size):
	'''
	Takes size measurements of a rectantuglar grid and returns a zero-array of shape 'size'.
	'''
	#make sure size is a tuple
	if isinstance(size, (list,tuple,np.array,int)):
		grid = np.zeros(size)
	return 



#set boundaries for the trajectory of the tracks, um (microns)

y_min = 0.
y_max = 4.

x_min = 0.
x_max = 2.


#simulate subdiffusion



#Simulate FBM 

def crw(a,length,diff,start,time_end):

	holder_array = np.zeros((length+1,2)) 
	holder_array[0] = start

	for j in range(1,length):


		sigma = np.sqrt(2.*diff)
		good = True
		test_wall = 0

		while good:

			rx = np.random.rand()
			ry = np.random.rand()
			move_x = 0
			move_y = 0

			i = np.float(j)/time_end
			if (0<rx and rx<(a*i)):

				move_x += sigma
			if ((a*i**(a-1))<rx and rx<(2.*a*i**(a-1))):

				move_x -= sigma
			if ((2.*a*i**(a-1))<rx and rx<1):

				move_x += 0.

			if (0<=ry and ry<(a*i)):

				move_y += sigma
			if ((a*i**(a-1))<=ry and ry<(2.*a*i**(a-1))):
				
				move_y -= sigma
			if ((2.*a*i**(a-1))<=ry and ry<1):
				
				move_y += 0.


			test_wall = np.array(holder_array[j-1]) + np.array([move_x,move_y])
			#print(test_wall)
			if (not (test_wall[0] > x_max)) and (not (test_wall[0] < x_min)) and (not (test_wall[1] > y_max)) and (not (test_wall[1] < y_min)):
				#print("in")
				#print(test_wall)
				good = False

		holder_array[j] = test_wall

	return holder_array
# track generator takes track and global variables and outputs a trajectory of average expected length -> exponentially distributed


def track_generator(length, diff_coef, start_loc , bleach_time = 20, amount = 100, subdiffusion = False, hurst = 0.5):
	'''
	INPUT:
	length = average length of the tracks with exponential distribution -> from data
	diff_coed = average diffusion coefficient with gaussian coefficient -> from data
	start_loc = location of where the track will start (2-4 DNA loci, or randomly on this grid.) -> 2D-gaussain distribution around this.


	RETURN:
	1-D array of track localization (float)
	'''


	track_loc = []

	cent_loc = 0

	if start_loc == 1:
		cent_loc = (1*(x_max-x_min)/6.,2*(y_max-y_min)/6.)
		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1)
	if start_loc == 2:
		cent_loc = (2*(x_max-x_min)/6.,3*(y_max-y_min)/6.)
		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1)
	if start_loc == 3:
		cent_loc = (4*(x_max-x_min)/6.,4*(y_max-y_min)/6.)
		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1)
	if start_loc == 4:
		cent_loc = (5*(x_max-x_min)/6.,5*(y_max-y_min)/6.)
		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1) 
	if start_loc == 5:
		cent_loc = (3*(x_max-x_min)/6.,3*(y_max-y_min)/6.) 
		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1)
	if start_loc == 14:
		cent_loc = (1*(x_max-x_min)/4.,1*(y_max-y_min)/4.) 

		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1)
	if start_loc == 34:
		cent_loc = (3*(x_max-x_min)/4.,3*(y_max-y_min)/4.) 

		cent_loc+= np.array(cent_loc)*np.random.normal(0,min(x_max,y_max)*0.1)
	if start_loc == 'r':
		cent_loc = (random_scale(x_min,x_max),random_scale(y_min,y_max))

	holder = cent_loc
	for i in range(amount):

		rand_length = int(np.round(np.random.exponential(length)))
		if rand_length == 0:
			rand_length = 10
		holder_array = np.zeros((rand_length+1,2)) #2D

		holder_array += np.array([cent_loc for ll in range(len(holder_array))])



		sigma = np.sqrt(2.*diff_coef)

		if subdiffusion:
			print("sub")
			alpha = hurst*2.
			#holder_array = c_FBM(alpha,rand_length,diff_coef,cent_loc,1000.)



			var_fbm = fbm.get_fbm_sample(l=rand_length, h = hurst, d = 2, n = rand_length)

			x_vals = np.array(var_fbm[1][0])*np.sqrt(2.*diff_coef)
			y_vals = np.array(var_fbm[1][1])*np.sqrt(2.*diff_coef)
			#print(np.diff(np.array(var_fbm[1][0])))
			holder_array[:,0] += x_vals[:]
			holder_array[:,1] += y_vals[:]


		#old use: single step (doesnt work for subdiffusion)
		if not subdiffusion:
			print("notsub")
			for j in range(1,rand_length):



				sigma = np.sqrt(2.*diff_coef)
				good = True
				test_wall = 0

				while good:

					move_x = np.random.normal(loc = 0, scale = sigma)
					move_y = np.random.normal(loc = 0, scale = sigma)
					test_wall = np.array(holder_array[j-1]) + np.array([move_x,move_y])
					#print(test_wall)
					if (not (test_wall[0] > x_max)) and (not (test_wall[0] < x_min)) and (not (test_wall[1] > y_max)) and (not (test_wall[1] < y_min)):
						#print("in")
						#print(test_wall)
						good = False

				holder_array[j] = test_wall
		if bleach_time == 30:
			print("bleah")
			temp11 = d_rand(rand_length)
			holder_array[1:][:,0] = temp11[0]
			holder_array[1:][:,1] = temp11[1]


		# for j in range(1,rand_length):
		# 	sigma = np.sqrt(2.*diff_coef)
		# 	good = True
		# 	test_wall = 0

		# 	while good:
		# 		if subdiffusion:

		# 			var_fbm = fbm.get_fbm_sample(l=, h = hurst, d = 2)
					
		# 			move_x = var_fbm[1][0][0] #* np.random.normal(loc = 0, scale = sigma)
		# 			move_y = var_fbm[1][1][0] #* np.random.normal(loc = 0, scale = sigma)
					
		# 			test_wall = np.array(holder_array[j-1]) + np.array([move_x,move_y])
		# 			#print(test_wall)

		# 		# 	if (not (test_wall[0] > x_max)) and (not (test_wall[0] < x_min)) and (not (test_wall[1] > y_max)) and (not (test_wall[1] < y_min)):
		# 		# 		#print("in")
		# 		# 		#print(test_wall)
		# 		# 		good = False
		# 		# else:
		# 		# 	move_x = np.random.normal(loc = 0, scale = sigma)
		# 		# 	move_y = np.random.normal(loc = 0, scale = sigma)
		# 		# 	test_wall = np.array(holder_array[j-1]) + np.array([move_x,move_y])
		# 		# 	#print(test_wall)
		# 		# 	if (not (test_wall[0] > x_max)) and (not (test_wall[0] < x_min)) and (not (test_wall[1] > y_max)) and (not (test_wall[1] < y_min)):
		# 		# 		#print("in")
		# 		# 		#print(test_wall)
		# 		# 		good = False

		# 	holder_array[j] = test_wall

		if start_loc == 'r':
			cent_loc = (random_scale(x_min,x_max),random_scale(y_min,y_max))

		track_loc.append(holder_array)
	return track_loc


#in um^2/s
diffusion_coeff = np.array([0.01,0.1,1])

def run_flow(difs,amounts,places,lengths,movie_length,subdiffusion = False, hurst = 0.5):
	holder = []
	frame_movie = np.zeros((movie_length,int(100*x_max),int(100*y_max)))
	x_tracks = []
	y_tracks = []
	for i in range(len(difs)):
		test1 = track_generator(lengths[i],difs[i],places[i],amount = amounts[i],subdiffusion = subdiffusion[i],hurst = hurst[i], bleach_time = 20)
		
		msd_test1 = []
		msd_a_test1 = []
		ang_test1 = []
		len_test1 = []
		frames = []

		for j in range(len(test1)):

			msd_test1.append(MSD_tavg(test1[j][:,0],test1[j][:,1],1))
			msd_a_test1.append(track_decomp(test1[j][:,0],test1[j][:,1],range(1,len(test1[j][:,1])),3))
			ang_test1 += angle_trajectory_2d(test1[j][:,0],test1[j][:,1])
			len_test1.append(len(test1[j][:,0]))

			start_frame = np.random.randint(0,movie_length-len_test1[-1])
			frames.append(np.array([x for x in range(start_frame,len_test1[-1]+start_frame)]))



			if True:
				test1[j][:,0] = rescale(test1[j][:,0],x_min,x_max)
				test1[j][:,1] = rescale(test1[j][:,1],y_min,y_max)


				x_tracks.append(test1[j][:,0])
				y_tracks.append(test1[j][:,1])

			for k in range(len(test1[j])):
				#print(test1[j])
				#print(test1[j][:,k])

				#print(test1[j][:,k])
				if test1[j][k][0] > 2 or test1[j][k][0] < 0:
					print("x wrong")
					print(test1[j][k][0])
				if test1[j][k][1] > 4 or test1[j][k][1] < 0:
					print("y wrong")
					print(test1[j][k][1])
				try:
					intensity = 10
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])][int(100*test1[j][k][1])] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])][int(100*test1[j][k][1])+1] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])+1][int(100*test1[j][k][1])] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])-1][int(100*test1[j][k][1])] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])][int(100*test1[j][k][1])-1] += intensity

					frame_movie[frames[-1][k]][int(100*test1[j][k][0])+1][int(100*test1[j][k][1])+1] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])+1][int(100*test1[j][k][1])-1] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])-1][int(100*test1[j][k][1])-1] += intensity
					frame_movie[frames[-1][k]][int(100*test1[j][k][0])-1][int(100*test1[j][k][1])+1] += intensity
				except:
					continue

		# 	plt.plot(test1[i][:,0],test1[i][:,1])
		# 	plt.plot(test1[i][0][0],test1[i][0][1],'ro')
		# 	plt.plot(test1[i][-1][0],test1[i][-1][1],'bo')
		# plt.show()

		msd_test1a = np.array(msd_test1)
		msd_a_test1a = np.array(msd_a_test1)
		ang_test1a = np.array(ang_test1)
		len_test1a = np.array(len_test1)
		framesa = np.array(frames)
		holder.append([msd_test1a,ang_test1a,len_test1a,framesa,frame_movie,msd_a_test1a])

	return [holder,x_tracks,y_tracks]


#simulate tracks with the respective diffusion coefficients at 14, and 34.
#all_hold = run_flow([0.01,0.1,1],[50,50,50],[14,34,'r'],[10,10,10],500)
all_hold = run_flow([0.1],[200],[14],[100],1000,subdiffusion = [True], hurst = [0.2])
name = "0.001-0.01-0.1_100-100-100_r-r_10-10-10_1000_SD_0.5_test"
def add_noise(movie, base_int, gaussian_blurr_v = 1):

	movie *= base_int
	#movie += int(base_int/2.)
	movie *= np.random.rand(*np.shape(movie))
	new_movie = np.zeros(np.shape(movie))
	for i in range(len(movie)):

		test = gaussian_filter(movie[i], sigma = gaussian_blurr_v)
		new_movie[i] = test
		# plt.imshow(new_movie[i])
		# plt.draw()
		# plt.pause(0.2)
	return new_movie+np.random.rand(*np.shape(movie))

image = np.zeros(np.shape(all_hold[0][0][4]))


temp = []
for i in range(len(all_hold[0])):
	image+=all_hold[0][i][4]
	temp+=list(all_hold[0][i][5])
	plt.plot(np.log10(all_hold[0][i][0]),all_hold[0][i][2],'.')
plt.show()


# MSD_a_value_sim(temp)
# plt.show()


MSD_a_value_sim_all(temp,xy_data = [all_hold[1],all_hold[2]])
plt.show()
# MSD_a_value_all_ens(xy_data = [all_hold[1],all_hold[2]],sim = True,threshold = 10)
# plt.show()

# MSD_a_value_all_ens_sim(xy_data = [all_hold[1],all_hold[2]])
# plt.show()
new_image = add_noise(image,100,3)
#new_image = image
new_image = np.array(new_image, 'uint16')


os.mkdir(name)

imsave(name + '/' + '{0}_1_seg.tif'.format(name), new_image)



os.mkdir(name + '/' + 'segmented')

stack_step = 5.0



length = int(len(new_image)/stack_step)

hold_img = []
hold_name = []
for j in np.arange(stack_step):

    hold_img.append(np.std(new_image[int(j*length):int((j+1)*length)],axis=0))
    hold_name.append("{0}_".format(int(j+1)) + name + '_1_seg' '.tif')


for k in range(len(hold_img)):

    imsave(name + '/' + 'segmented' + '/' + hold_name[k],np.array(hold_img[k],'uint16'))



