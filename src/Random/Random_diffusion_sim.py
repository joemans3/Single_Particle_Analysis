import numpy as np 
import matplotlib.pyplot as plt 
import Analysis_functions as af 

#simulate a set of 2D random walk

size_step = 1.
time_step = 1.
mean_length = 15

#number of trajectories to simulate
trajectory_total = 10000
trajectory_container = []

for i in range(trajectory_total):
	#define the length of the trajectories to be exponentially distributed with mean 15 (time)
	good = True
	length = 0
	while good:

		length = np.random.exponential(mean_length)
		if length > 10:
			good = False

	trajectory = np.zeros((int(length),2))
	for j in range(1,len(trajectory)):
		#update the trajectory by randomly selecting the angle

		angle = np.random.random()*2.*np.pi
		

		dx = np.cos(angle)*size_step
		dy = np.sin(angle)*size_step

		trajectory[j] = trajectory[j-1] + np.array([dx,dy])

	trajectory_container.append(trajectory)



angle_container = []
for i in trajectory_container:
	angles_all = af.trajectory_angle(i[:,0],i[:,1])
	angle_container.append(angles_all)


if __name__ == "__main__":

	flatten_angle = af.flatten(angle_container)

	plt.hist(flatten_angle,bins = 100,alpha = 0.2)


	plt.show()

