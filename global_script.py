from trajectory_analysis_script import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
a= run_analysis("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/LACO_LACI","TB54_FAST")
a.read_parameters()
a.run_flow()

b = a.viable_drop_total
c = a.in_track_total 
d = a.segmented_drop_files

e1 = a.in_msd_all
e2 = a.io_msd_all
e3 = a.ot_msd_all

for i in range(len(e1)):
	for j in range(len(e1[i])):
		if len(e1[i][j]) != 0:
			plt.plot(e1[i][j],np.ones(len(e1[i][j]))*j,'b.')
		if len(e2[i][j]) != 0:
			plt.plot(e2[i][j],np.ones(len(e2[i][j]))*j,'y.')
		if len(e3[i][j]) != 0:
			plt.plot(e3[i][j],np.ones(len(e3[i][j]))*j,'r.')

	plt.xscale("log")
	plt.xlabel("MSD in um^2/s")
	plt.ylabel("Frame Subset (index from 0)")
	plt.legend(["Blue: in","Yellow: In/Out","Red: Out"])
	plt.show()

for i in range(len(b)):
	if len(d[i]) != 0:
		img = mpimg.imread(d[i][0])
		timg = plt.imshow(img)
	for j in range(len(b[i])):
		for l in range(len(c[i][j])):
			if len(c[i][j][l])!=0:
				temp = np.array(c[i][j][l])
				plt.plot(temp[0],temp[1])


		if (len(b[i][j])>0):
			for k in range(len(b[i][j])):
				circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c='r', alpha = 0.5)
	plt.show()


fraction_tick = [i for i in range(1,int(a.frame_total/a.frame_step)+1)]

create_box_plot(a.tmframe_occ,fraction_tick,y_label = "Fraction inside the drop",x_label = "Frame number",y_lim = (),title = "Percent Occupation of Track in Drop per Frame Over All Experiments")

for i in a.tmframe_occ:
    w_i = np.ones_like(i)/float(len(i))
    plt.hist(i,histtype = 'step',weights=w_i)
    plt.xlabel("Fraction inside the drop")
    plt.ylabel("Probability")
    plt.title("Percent Occupation of Track in Drop per Frame")
plt.show()
