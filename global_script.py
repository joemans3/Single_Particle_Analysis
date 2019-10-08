from trajectory_analysis_script import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import tensorflow as tf
import os
from plotting_functions import *
from import_functions import *



from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable


os.chdir("..")

# rp= run_analysis("DATA","RPOC")
# rp.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp.run_flow()
# total_rp = np.array(list(rp.i_d_tavg) + list(rp.io_d_tavg) + list(rp.o_d_tavg))

# rp_m9= run_analysis("DATA/rpoc_M9/20190515","rpoc_M9")
# rp_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp_m9.run_flow()
# total_rp_m9 = np.array(list(rp_m9.i_d_tavg) + list(rp_m9.io_d_tavg) + list(rp_m9.o_d_tavg))

rp_ez= run_analysis("DATA/new_days/20190527/rpoc_ez","rpoc_ez")
rp_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez.run_flow()
total_rp_ez = np.array(list(rp_ez.i_d_tavg) + list(rp_ez.io_d_tavg) + list(rp_ez.o_d_tavg))

rp_m9_2= run_analysis("DATA/new_days/20190524/rpoc_m9","rpoc_M9")
rp_m9_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_m9_2.run_flow()
total_rp_m9_2 = np.array(list(rp_m9_2.i_d_tavg) + list(rp_m9_2.io_d_tavg) + list(rp_m9_2.o_d_tavg))

# ll_m9_24 = run_analysis("DATA/new_days/20190524/laco_laci_M9","laco_laci_M9")
# ll_m9_24.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9_24.run_flow()
# total_ll_m9_24 = np.array(list(ll_m9_24.i_d_tavg) + list(ll_m9_24.io_d_tavg) + list(ll_m9_24.o_d_tavg))


rp_ez_h3 = run_analysis("DATA/new_days/20190527/rpoc_ez_hex_3","rpoc_ez_hex_3")
rp_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez_h3.run_flow()
total_rp_ez_h3 = np.array(list(rp_ez_h3.i_d_tavg) + list(rp_ez_h3.io_d_tavg) + list(rp_ez_h3.o_d_tavg))

rp_ez_h5 = run_analysis("DATA/rpoc_ez_hex_5","rpoc_ez_hex_5")
rp_ez_h5.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez_h5.run_flow()
total_rp_ez_h5 = np.array(list(rp_ez_h5.i_d_tavg) + list(rp_ez_h5.io_d_tavg) + list(rp_ez_h5.o_d_tavg))



rp_ez_h5_2 = run_analysis("DATA/rpoc_ez_hex_5_2","rpoc_ez_h_5")
rp_ez_h5_2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
rp_ez_h5_2.run_flow()
total_rp_ez_h5_2 = np.array(list(rp_ez_h5_2.i_d_tavg) + list(rp_ez_h5_2.io_d_tavg) + list(rp_ez_h5_2.o_d_tavg))





ll_ez= run_analysis("DATA/new_days/20190527/ll_ez","laco_laci_ez")
ll_ez.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll_ez.run_flow()
total_ll_ez = np.array(list(ll_ez.i_d_tavg) + list(ll_ez.io_d_tavg) + list(ll_ez.o_d_tavg))

# ll_m9= run_analysis("DATA/new_days/20190527/ll_m9","laco_laci_m9")
# ll_m9.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9.run_flow()
# total_ll_m9 = np.array(list(ll_m9.i_d_tavg) + list(ll_m9.io_d_tavg) + list(ll_m9.o_d_tavg))


# ll_m9n= run_analysis("DATA/laco_m9","laco_m9")
# ll_m9n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_m9n.run_flow()
# total_ll_m9n = np.array(list(ll_m9n.i_d_tavg) + list(ll_m9n.io_d_tavg) + list(ll_m9n.o_d_tavg))


# nh= run_analysis("DATA/nusa_ez_hex_5","nusa_ez_hex_5")
# nh.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# nh.run_flow()
# total_nh = np.array(list(nh.i_d_tavg) + list(nh.io_d_tavg) + list(nh.o_d_tavg))



# ll_ez_h3= run_analysis("DATA/new_days/20190527/ll_ez_hex_3","laco_laci_ez__hex_3")
# ll_ez_h3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# ll_ez_h3.run_flow()
# total_ll_ez_h3 = np.array(list(ll_ez_h3.i_d_tavg) + list(ll_ez_h3.io_d_tavg) + list(ll_ez_h3.o_d_tavg))


# rp2= run_analysis("DATA/RPOC_new","RPOC")
# rp2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp2.run_flow()
# total_rp2 = np.array(list(rp2.i_d_tavg) + list(rp2.io_d_tavg) + list(rp2.o_d_tavg))
# rp3= run_analysis("DATA/Files_RPOC","RPOC")
# rp3.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp3.run_flow()
# total_rp3 = np.array(list(rp3.i_d_tavg) + list(rp3.io_d_tavg) + list(rp3.o_d_tavg))
# rp1= run_analysis("DATA/Other_RPOC","rpoc")
# rp1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# rp1.run_flow()
# total_rp1 = np.array(list(rp1.i_d_tavg) + list(rp1.io_d_tavg) + list(rp1.o_d_tavg))
ll= run_analysis("DATA/LACO_LACI","TB54_FAST")
ll.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ll.run_flow()
total_ll = np.array(list(ll.i_d_tavg) + list(ll.io_d_tavg) + list(ll.o_d_tavg))
# n= run_analysis("DATA/newer_NUSA","NUSA")
# n.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n.run_flow()
# total_n = np.array(list(n.i_d_tavg) + list(n.io_d_tavg) + list(n.o_d_tavg))
# n1= run_analysis("DATA/New_NUSA","NUSA")
# n1.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n1.run_flow()
# total_n1 = np.array(list(n1.i_d_tavg) + list(n1.io_d_tavg) + list(n1.o_d_tavg))
# n2= run_analysis("DATA/Nusa_20190304","NUSA")
# n2.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
# n2.run_flow()
# total_n2 = np.array(list(n2.i_d_tavg) + list(n2.io_d_tavg) + list(n2.o_d_tavg))
a= run_analysis("DATA/Nusa_20190305","NUSA")
a.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
a.run_flow()
total_a = np.array(list(a.i_d_tavg) + list(a.io_d_tavg) + list(a.o_d_tavg))


###chephalexin
ceph_rpoc = run_analysis("DATA/chephalexin/20190731/rpoc_ez_50_ceph", "rpoc_ez_50ceph")
ceph_rpoc.read_parameters(minimum_percent_per_drop_in = 0.1,t_len_u = 50)
ceph_rpoc.run_flow()
total_ceph_rpoc = np.array(list(ceph_rpoc.i_d_tavg) + list(ceph_rpoc.io_d_tavg) + list(ceph_rpoc.o_d_tavg))




# b = a.viable_drop_total

# c = a.in_track_total 
# c1 = a.io_track_total 
# c2 = a.ot_track_total

# rg = a.in_radius_g
# rg1 = a.io_radius_g
# rg2 = a.ot_radius_g

# cp = a.in_msd_all
# cp1 = a.io_msd_all
# cp2 = a.ot_msd_all

# d = a.segmented_drop_files

total_log_msd = np.log10(np.array(a.i_d_tavg + a.io_d_tavg + a.o_d_tavg))


cvals  = [np.min(con_pix_si(total_log_msd,which = 'msd')),np.percentile(con_pix_si(total_log_msd,which = 'msd'),25), np.percentile(con_pix_si(total_log_msd,which = 'msd'),75), np.max(con_pix_si(total_log_msd,which = 'msd'))]
colors = ["green","red","violet","blue"]

norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(list(map(norm,cvals)), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


# e1 = a.in_msd_all
# e2 = a.io_msd_all
# e3 = a.ot_msd_all


def plot_msd_n(temp_a,list_number,label,title = " ",log_scale = False):
	for i in range(list_number):
		plt.hist(np.log10(temp_a[i]),label = label[i],alpha = 0.3,density = True)
	if log_scale:
		#plt.xscale("log")
		pass
	plt.xlabel("Log10 MSD in um^2/s")
	plt.title(title)
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.tight_layout()
	plt.show()

def plot_msd(temp_a,list_number,label,fig,ax,title = " ",log_scale = False):
	for i in range(list_number):
		ax.hist(np.log10(temp_a[i]),label = label[i],alpha = 0.3,density = True)
	if log_scale:
		#plt.xscale("log")
		pass
	ax.set_xlabel("Log10 MSD in um^2/s")
	ax.set_title(title)
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	return 




##################################################################################
#figure_1

fig = plt.figure(figsize = [10,10])

img_rpoc_seg ='/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez/segmented/4_rpoc_ez_5_seg.tif'
img_ll_seg = '/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/segmented/4_laco_laci_ez_6_seg.tif'

img_rp_ez = mpimg.imread(img_rpoc_seg)
img_ll_ez = mpimg.imread(img_ll_seg)


ax = fig.add_subplot(231)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)

ax3 = fig.add_subplot(234)
ax4 = fig.add_subplot(235)
ax5 = fig.add_subplot(236)




b = rp_ez.viable_drop_total
d = rp_ez.segmented_drop_files
c = rp_ez.in_track_total
c1 = rp_ez.io_track_total
c2 = rp_ez.ot_track_total
cp = rp_ez.in_msd_all
cp1 = rp_ez.io_msd_all
cp2 = rp_ez.ot_msd_all


which = "all"
show = False
line = True
scatter = True
good = 0
i = 4
img_ori = mpimg.imread(d[4][3])


cmap_all=plt.get_cmap('gray')


cmap_all.set_bad(color = 'white')

img = np.ma.masked_where(img_ori == 126,img_ori)


def masked(img):
    return np.ma.masked_where(img == 126,img)


timg = ax1.imshow(img,cmap=cmap_all,origin = "lower")
other_tim = ax2.imshow(img,cmap=cmap_all,origin = "lower")




rp_ez_im = ax.imshow(masked(img_rp_ez),origin = 'lower',cmap = cmap_all)
ll_ez_im = ax3.imshow(masked(img_ll_ez),origin = 'lower',cmap = cmap_all)



copy_array_in = np.zeros(np.shape(img))
copy_array_io = np.zeros(np.shape(img))
copy_array_ot = np.zeros(np.shape(img))
copy_array_all = np.zeros(np.shape(img))


random_choose_c = [0,3]
random_choose_c1 = [0,3]
random_choose_c2 = [0,3]
choose_b = np.random.randint(0,len(b[i]),2)

in_track_used = []
io_track_used = []
ot_track_used = []
for j in range(len(b[i])):


    for l in range(len(c[i][j])):
        if len(c[i][j][l])!=0:
            temp = np.array(c[i][j][l])
            copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            #plt.plot(temp[0],temp[1],'b-')
            if (which == "all" or which == "in") and scatter :
                ax2.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
            if (which == "all" or which == "in") and line:
                if (good == 0):
                    if len(temp[0]) != 0:
                        print(temp[0])
                        ax.plot(temp[0],temp[1],c = 'r')
                        in_track_used = [temp[0],temp[1]]
                        good = 1
                        print("good = {0},{1}".format(j,l))

    for l in range(len(c1[i][j])):
        if len(c1[i][j][l])!=0:
            temp = np.array(c1[i][j][l])
            copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            #plt.plot(temp[0],temp[1],'g-')
            if (which == "all" or which == "io") and scatter:
                ax2.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
            if (which == "all" or which == "io") and line and (j == 0) and (l == 0):
                ax.plot(temp[0],temp[1],c = 'b')
                io_track_used = [temp[0],temp[1]]

    for l in range(len(c2[i][j])):
        if len(c2[i][j][l])!=0:
            temp = np.array(c2[i][j][l])
            copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            #plt.plot(temp[0],temp[1],'r-')
            if (which == "all" or which == "out") and scatter:
                what = ax2.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
            if (which == "all" or which == "out") and line and (j == 0) and (l == 6):
                ax.plot(temp[0],temp[1],c = 'g')
                ot_track_used = [temp[0],temp[1]]
    # if (len(b[i][j])>0):
    #   for k in range(len(b[i][j])):
    #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
#ax2.colorbar()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(what,cax=cbar_ax)
ax2.annotate('$log\\left( \\frac{um^2}{s}\\right) $', xy=(2.9*ax2.bbox.width, 1.5*ax2.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
ax.annotate('RPOC', xy=(-1.0*ax.bbox.width, 0.65*ax.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
ax3.annotate('LacO \n LacI', xy=(-1.0*ax3.bbox.width, 0.65*ax3.bbox.height), xycoords="axes pixels", fontsize=15, weight = 'bold')
axins_in = zoomed_inset_axes(ax, 6, loc=1)
axins_in.plot(in_track_used[0],in_track_used[1],'r',lw = 1)
axins_in.get_xaxis().set_visible(False)
axins_in.get_yaxis().set_visible(False)
axins_io = zoomed_inset_axes(ax, 6, loc=7)
axins_io.plot(io_track_used[0],io_track_used[1],'b',lw = 1)
axins_io.get_xaxis().set_visible(False)
axins_io.get_yaxis().set_visible(False)
axins_ot = zoomed_inset_axes(ax, 6, loc=4)
axins_ot.plot(ot_track_used[0],ot_track_used[1],'g',lw = 1)
axins_ot.get_xaxis().set_visible(False)
axins_ot.get_yaxis().set_visible(False)
mark_inset(ax, axins_in, loc1=2, loc2=4, fc="none", ec="0.5")
mark_inset(ax, axins_io, loc1=2, loc2=4, fc="none", ec="0.5")
mark_inset(ax, axins_ot, loc1=3, loc2=1, fc="none", ec="0.5")
cont = ax1.contour(copy_array_all)







b = ll_ez.viable_drop_total
d = ll_ez.segmented_drop_files
c = ll_ez.in_track_total
c1 = ll_ez.io_track_total
c2 = ll_ez.ot_track_total
cp = ll_ez.in_msd_all
cp1 = ll_ez.io_msd_all
cp2 = ll_ez.ot_msd_all


which = "all"
show = False
line = True
scatter = True
good = 0
i = 5
img = mpimg.imread(d[5][3])
timg = ax4.imshow(masked(img),cmap=cmap_all,origin = "lower")
other_tim = ax5.imshow(masked(img),cmap=cmap_all,origin = "lower")
copy_array_in = np.zeros(np.shape(img))
copy_array_io = np.zeros(np.shape(img))
copy_array_ot = np.zeros(np.shape(img))
copy_array_all = np.zeros(np.shape(img))


random_choose_c = [0,3]
random_choose_c1 = [0,3]
random_choose_c2 = [0,3]
choose_b = np.random.randint(0,len(b[i]),2)

in_track_used = []
io_track_used = []
ot_track_used = []
for j in range(len(b[i])):


    for l in range(len(c[i][j])):
        if len(c[i][j][l])!=0:
            temp = np.array(c[i][j][l])
            copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            #plt.plot(temp[0],temp[1],'b-')
            if (which == "all" or which == "in") and scatter :
                ax5.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
            if (which == "all" or which == "in") and line:
                if (good == 0):
                    if len(temp[0]) != 0:
                        print(temp[0])
                        ax3.plot(temp[0],temp[1],c = 'r')
                        in_track_used = [temp[0],temp[1]]
                        good = 1
                        print("good = {0},{1}".format(j,l))

    for l in range(len(c1[i][j])):
        if len(c1[i][j][l])!=0:
            temp = np.array(c1[i][j][l])
            copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            #plt.plot(temp[0],temp[1],'g-')
            if (which == "all" or which == "io") and scatter:
                ax5.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
            if (which == "all" or which == "io") and line and (j == 0) and (l == 0):
                ax3.plot(temp[0],temp[1],c = 'b')
                io_track_used = [temp[0],temp[1]]

    for l in range(len(c2[i][j])):
        if len(c2[i][j][l])!=0:
            temp = np.array(c2[i][j][l])
            copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
            #plt.plot(temp[0],temp[1],'r-')
            if (which == "all" or which == "out") and scatter:
                ax5.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
            if (which == "all" or which == "out") and line and (j == 0) and (l == 6):
                ax3.plot(temp[0],temp[1],c = 'g')
                ot_track_used = [temp[0],temp[1]]
    # if (len(b[i][j])>0):
    #   for k in range(len(b[i][j])):
    #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)

axins_in = zoomed_inset_axes(ax3, 6, loc=4)
axins_in.plot(in_track_used[0],in_track_used[1],'r')
axins_in.set_xlim((np.min(in_track_used[0]) - 1, np.max(in_track_used[0]) + 1 ))
axins_in.set_ylim((np.min(in_track_used[1]) - 1, np.max(in_track_used[1]) + 1 ))
axins_in.get_xaxis().set_visible(False)
axins_in.get_yaxis().set_visible(False)
# axins_io = zoomed_inset_axes(ax3, 6, loc=7)
# axins_io.plot(io_track_used[0],io_track_used[1],'b')
# axins_io.get_xaxis().set_visible(False)
# axins_io.get_yaxis().set_visible(False)
axins_ot = zoomed_inset_axes(ax3, 6, loc=1)
axins_ot.plot(ot_track_used[0],ot_track_used[1],'g')
axins_ot.set_xlim((np.min(ot_track_used[0]) - 1, np.max(ot_track_used[0]) + 1 ))
axins_ot.set_ylim((np.min(ot_track_used[1]) - 1, np.max(ot_track_used[1]) + 1 ))
axins_ot.get_xaxis().set_visible(False)
axins_ot.get_yaxis().set_visible(False)
mark_inset(ax3, axins_in, loc1=3, loc2=1, fc="none", ec="0.5")
# mark_inset(ax3, axins_io, loc1=2, loc2=4, fc="none", ec="0.5")
mark_inset(ax3, axins_ot, loc1=2, loc2=4, fc="none", ec="0.5")
cont = ax4.contour(copy_array_all)



ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)

ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)

plt.show()



# plot_msd([a.i_d_tavg,ll.i_d_tavg,rp_ez.i_d_tavg,rp_m9_2.i_d_tavg],4,["Nusa","LACO_LACI","RPOC","RPOC_M9"])

# plot_msd([a.i_d_tavg,a.io_d_tavg,a.o_d_tavg,total_ll],4,["in","in/out","out","LACO_LACI"],"Nusa_201905")
# plot_msd([rp_m9.i_d_tavg,rp_m9.io_d_tavg,rp_m9.o_d_tavg,total_ll],4,["in","in/out","out","LACO_LACI"],"Rpoc_M9")

# plot_msd([rp_m9.i_d_tavg,rp_m9.io_d_tavg,rp_m9.o_d_tavg],3,["in","in/out","out"],"Rpoc_M9")

# plot_msd([rp.i_d_tavg,rp.io_d_tavg,rp.o_d_tavg],3,["in","in/out","out"],"First")	
# plot_msd([rp1.i_d_tavg,rp1.io_d_tavg,rp1.o_d_tavg],3,["in","in/out","out"],"RPOC_new")
# plot_msd([rp2.i_d_tavg,rp2.io_d_tavg,rp2.o_d_tavg],3,["in","in/out","out"],"Files_RPOC")
# plot_msd([rp3.i_d_tavg,rp3.io_d_tavg,rp3.o_d_tavg],3,["in","in/out","out"],"Other_RPOC")
# plot_msd([ll.i_d_tavg,ll.io_d_tavg,ll.o_d_tavg],3,["in","in/out","out"],"LACO_LACI")
# plot_msd([a.i_d_tavg,a.io_d_tavg,a.o_d_tavg],3,["in","in/out","out"],"Nusa_201905")
# plot_msd([n.i_d_tavg,n.io_d_tavg,n.o_d_tavg],3,["in","in/out","out"],"newer_NUSA")
# plot_msd([n1.i_d_tavg,n1.io_d_tavg,n1.o_d_tavg],3,["in","in/out","out"],"New_NUSA")
# plot_msd([n2.i_d_tavg,n2.io_d_tavg,n2.o_d_tavg],3,["in","in/out","out"],"Nusa_201904")

# plot_msd([a.i_d_tavg,n.i_d_tavg,n1.i_d_tavg,n2.i_d_tavg],4,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904"])
# plot_msd([rp.i_d_tavg,rp1.i_d_tavg,rp2.i_d_tavg,rp3.i_d_tavg],4,["First","RPOC_new","Files_RPOC","Other_RPOC"])
#plot_msd_n([a.o_d_tavg,n.o_d_tavg,n1.o_d_tavg,n2.o_d_tavg],4,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904"])
# plot_msd([rp_m9_2.i_d_tavg,rp_m9_2.io_d_tavg,rp_m9_2.o_d_tavg,total_ll],4,["in","in/out","out","control"],"Rpoc_m9_20190524")
# plot_msd([rp_ez.i_d_tavg,rp_ez.io_d_tavg,rp_ez.o_d_tavg,total_ll],4,["in","in/out","out","control"],"Rpoc_ez_20190527")

# #laco_laci com
# plot_msd([total_ll,total_ll_m9,total_ll_ez,total_ll_ez_h3],4,["Nic's","m9","ez","ez + hex 3%"])
#plot_msd_n([total_ll,ll_m9n.in_sorted_experiment[0] + ll_m9n.io_sorted_experiment[0] + ll_m9n.ot_sorted_experiment[0],total_ll_ez,total_ll_ez_h3],4,["Nic's","m9","ez","ez + hex 3%"])
# #rpoc_compare total
# plot_msd([total_rp,total_rp2,total_rp3,total_rp_ez,total_rp_m9,total_rp_m9_2,total_rp_ez_h3,total_ll],8,["First","second","third","new_ez","old_m9","new_m9","ez_hex 3%","control"])
# #in only
# plot_msd([rp.i_d_tavg, rp2.i_d_tavg,rp3.i_d_tavg,rp_ez.i_d_tavg,rp_m9.i_d_tavg,rp_m9_2.i_d_tavg,rp_ez_h3.i_d_tavg, total_ll],8,["First","second","third","new_ez","old_m9","new_m9","ez_hex 3%","control"])
#plot_msd_n([rp.i_d_tavg, rp2.i_d_tavg,rp3.i_d_tavg,rp_ez.i_d_tavg,rp_m9.i_d_tavg,rp_m9_2.i_d_tavg,rp_ez_h3.i_d_tavg, total_ll],8,["First","second","third","new_ez","old_m9","new_m9","ez_hex 3%","control"])
#plot_msd_n([rp_m9_2.i_d_tavg,rp_m9_2.io_d_tavg,rp_m9_2.o_d_tavg,total_ll],4,["in","in/out","out","control"],"Rpoc_m9_20190524")

#plot_msd([ll_m9n.i_d_tavg, ll_m9n.io_d_tavg,ll_m9n.o_d_tavg, total_ll],4,["in","io","out","control"],fig,ax)
#plot_msd([np.array(ll_m9n.in_sorted_experiment), np.array(ll_m9n.io_sorted_experiment),np.array(ll_m9n.ot_sorted_experiment), total_ll],4,["in","io","out","control"],fig,ax2)






#chephalexin rpoc no flophore
#plot_msd_n([ceph_rpoc.i_d_tavg,ceph_rpoc.io_d_tavg,ceph_rpoc.o_d_tavg,total_ll],4,["rpoc_in","rpoc_io","rpoc_out","ll_control"])



# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(len(rp_ez_h5.in_sorted_experiment)):
# 	ax.hist(np.log10(np.array(rp_ez_h5.in_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# ax.legend()

# plt.show()
# fig.clear()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(len(rp_ez_h5.io_sorted_experiment)):
# 	ax.hist(np.log10(np.array(rp_ez_h5.io_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# ax.legend()

# plt.show()
# fig.clear()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(len(rp_ez_h5.ot_sorted_experiment)):
# 	ax.hist(np.log10(np.array(rp_ez_h5.ot_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# ax.legend()

# plt.show()
# fig.clear()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(len(nh.in_sorted_experiment)):
# 	ax.hist(np.log10(np.array(nh.in_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# ax.legend()

# plt.show()
# fig.clear()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(len(nh.io_sorted_experiment)):
# 	ax.hist(np.log10(np.array(nh.io_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# ax.legend()

# plt.show()
# fig.clear()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(len(nh.ot_sorted_experiment)):
# 	ax.hist(np.log10(np.array(nh.ot_sorted_experiment[i])),label = "{0}".format(i),alpha = 0.5)
# ax.legend()

# plt.show()
# fig.clear()


# plot_msd([total_a,total_n,total_n1,total_n2],4,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904"])
# plot_msd([total_rp,total_rp1,total_rp2,total_rp3,total_rp_m9],4,["First","RPOC_new","Files_RPOC","RPOC_M9"])

# plot_msd([list(total_a)+list(total_n)+list(total_n1)+list(total_n2),list(total_rp)+list(total_rp1)+list(total_rp2)+list(total_rp3),list(total_ll),list(total_rp_m9)],4,["Nusa","RPOC","LACO_LACI","RPOC_M9"])
# plot_msd([list(a.i_d_tavg)+list(n.i_d_tavg)+list(n2.i_d_tavg),list(rp.i_d_tavg)+list(rp1.i_d_tavg)+list(rp2.i_d_tavg)+list(rp3.i_d_tavg),total_ll,list(rp_m9.i_d_tavg)],4,["Nusa","RPOC","LACO_LACI","RPOC_M9"])
# plot_msd([a.i_d_tavg,n.i_d_tavg,n1.i_d_tavg,n2.i_d_tavg,rp.i_d_tavg,rp1.i_d_tavg,rp2.i_d_tavg,rp3.i_d_tavg,total_ll],9,["Nusa_201905","newer_NUSA","New_NUSA","Nusa_201904","First","RPOC_new","Files_RPOC","Other_RPOC","LACO_LACI"])


# plt.hist(np.log10(a.i_d_tavg),label = "in",alpha = 0.3,density = True)
# plt.hist(np.log10(a.o_d_tavg),label = "out",alpha = 0.3,density= True)
# plt.hist(np.log10(a.io_d_tavg),label = "in/out",alpha = 0.3,density = True)
# plt.hist(np.log10(a.unrestricted_msd),label = "BASELINE",alpha = 0.3,density = True)
# #plt.xscale("log")
# plt.xlabel("Log10 MSD in um^2/s")
# plt.legend()
# plt.show()

# #plot_msd([rp_ez_h5.in_sorted_experiment[7],rp_ez_h5.io_sorted_experiment[7],rp_ez_h5.ot_sorted_experiment[7],total_ll],4,["in","in/out","out","control"],fig,ax3,"Rpoc_hex")
# #############
# #try plotting radius of gyration and msd with distinctions
# plt.scatter(np.log10(a.i_d_tavg),np.log10(rg),label = "in", alpha = 0.3)
# plt.scatter(np.log10(a.io_d_tavg),np.log10(rg1),label = "in/out", alpha = 0.3)
# plt.scatter(np.log10(a.o_d_tavg),np.log10(rg2),label = "out", alpha = 0.03)
# plt.legend()
# plt.xlabel("Log10 MSD in um^2/s")
# plt.ylabel("Log10 Radius of Gyration (um)")
# plt.title("Radius of Gyration vs MSD with Classification")
# plt.show()

# plt.scatter(np.log10(rp_m9.i_d_tavg),np.log10(rp_m9.in_radius_g),label = "in", alpha = 0.3)
# plt.scatter(np.log10(rp_m9.io_d_tavg),np.log10(rp_m9.io_radius_g),label = "in/out", alpha = 0.3)
# plt.scatter(np.log10(rp_m9.o_d_tavg),np.log10(rp_m9.ot_radius_g),label = "out", alpha = 0.03)
# plt.legend()
# plt.xlabel("Log10 MSD in um^2/s")
# plt.ylabel("Log10 Radius of Gyration (um)")
# plt.title("Radius of Gyration vs MSD with Classification")
# plt.show()


# #compare the in fraction with laco-laci and rpoc
# plt.scatter(np.log10(rp.i_d_tavg),np.log10(rp.in_radius_g),label = "in_RPOC", alpha = 0.1)
# plt.scatter(np.log10(ll.i_d_tavg),np.log10(ll.in_radius_g),label = "in_LACO_LACI", alpha = 0.3)
# plt.scatter(np.log10(a.i_d_tavg),np.log10(a.in_radius_g),label = "in_NUSA", alpha = 0.1)
# plt.legend()
# plt.xlabel("Log10 MSD in um^2/s")
# plt.ylabel("Log10 Radius of Gyration (um)")
# plt.title("Radius of Gyration vs MSD with Classification")
# plt.show()

# #compare the in fraction with laco-laci and rpoc
# plt.scatter(np.log10(rp.i_d_tavg),np.log10(rp.in_radius_g),label = "in_RPOC", alpha = 0.3)
# plt.scatter(np.log10(ll.i_d_tavg),np.log10(ll.in_radius_g),label = "in_LACO_LACI", alpha = 0.3)
# #plt.scatter(np.log10(a.i_d_tavg),np.log10(a.in_radius_g),label = "in_NUSA", alpha = 0.3)
# plt.scatter(np.log10(rp_m9.i_d_tavg),np.log10(rp_m9.in_radius_g),label = "in_RPOC_M9", alpha = 0.3)
# plt.legend()
# plt.xlabel("Log10 MSD in um^2/s")
# plt.ylabel("Log10 Radius of Gyration (um)")
# plt.title("Radius of Gyration vs MSD with Classification")
# plt.show()


# ###radius of gyration + end to end + msd
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(a.i_d_tavg),np.log10(rg),np.log10(a.in_ete),label = "in", alpha = 0.3)
# ax.scatter(np.log10(a.io_d_tavg),np.log10(rg1),np.log10(a.io_ete),label = "in/out", alpha = 0.3)
# ax.scatter(np.log10(a.o_d_tavg),np.log10(rg2),np.log10(a.ot_ete),label = "out", alpha = 0.03)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(con_pix_si(rp_ez.i_d_tavg,which ='msd')),np.log10(con_pix_si(rp_ez.in_radius_g,which = 'um')),np.log10(con_pix_si(rp_ez.in_ete,which = 'um')),label = "in", alpha = 0.3)
# ax.scatter(np.log10(con_pix_si(rp_ez.io_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.io_radius_g,which = 'um')),np.log10(con_pix_si(rp_ez.io_ete,which = 'um')),label = "in/out", alpha = 0.3)
# ax.scatter(np.log10(con_pix_si(rp_ez.o_d_tavg,which = 'msd')),np.log10(con_pix_si(rp_ez.ot_radius_g,which = 'um')),np.log10(con_pix_si(rp_ez.ot_ete,which = 'um')),label = "out", alpha = 0.03)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(rp.i_d_tavg),np.log10(rp.in_radius_g),np.log10(rp.in_ete),label = "in_RPOC", alpha = 0.1)
# ax.scatter(np.log10(ll.i_d_tavg),np.log10(ll.in_radius_g),np.log10(ll.in_ete),label = "in_LACO_LACI", alpha = 0.3)
# ax.scatter(np.log10(a.i_d_tavg),np.log10(a.in_radius_g),np.log10(a.in_ete),label = "in_NUSA", alpha = 0.1)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(np.log10(total_rp_ez),np.log10(np.array(rp_ez.in_radius_g + rp_ez.io_radius_g + rp_ez.ot_radius_g)),np.log10(np.array(rp_ez.in_ete + rp_ez.io_ete + rp_ez.ot_ete)),label = "RPOC", alpha = 0.1)
# ax.scatter(np.log10(total_ll),np.log10(np.array(ll.in_radius_g + ll.io_radius_g + ll.ot_radius_g)),np.log10(np.array(ll.in_ete + ll.io_ete + ll.ot_ete)),label = "LACO_LACI", alpha = 0.3)
# ax.scatter(np.log10(total_a),np.log10(np.array(a.in_radius_g + a.io_radius_g + a.ot_radius_g)),np.log10(np.array(a.in_ete + a.io_ete + a.ot_ete)),label = "NUSA", alpha = 0.1)
# ax.set_xlabel("Log10 MSD in um^2/s")
# ax.set_ylabel("Log10 Radius of Gyration (um)")
# ax.set_zlabel("Log10 End to End Distance (um)")
# #ax.set_title("Radius of Gyration vs MSD vs End to End Distance with Classification")
# ax.legend()
# plt.show()




# total_len = len(a.i_d_tavg) + len(a.io_d_tavg) + len(a.o_d_tavg)
# in_len = float(len(a.i_d_tavg))/total_len
# io_len = float(len(a.io_d_tavg))/total_len
# out_len = float(len(a.o_d_tavg))/total_len


#Dapi controls
#rpoc_noflorophore + dapi at 0.1 ug/ml
path_rp_dapi = '/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/dapi_2/20190820/rpoc_nofl_dapi_0.1'
#hup_mCherry + dapi at 0.1 ug/ml
path_hup_mCherry_dapi = '/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA/dapi_2/20190820/hup_mcherry_dapi_0.1'


#read dapi+rpoc_nofl
rp_dapi_files = glob.glob(path_rp_dapi + "/" + "rpoc_fapi_" + "**.tif")
#read hup_mcherry + dapi
hup_mcherry_files = glob.glob(path_hup_mCherry_dapi + "/" + "hup_mcherry_" + "**.tif")


# for i in rp_dapi_files:
# 	img_read = read_imag(i,show = 0)
# 	contour_intens(img_read,perc = 99)
# for i in hup_mcherry_files:
# 	img_read = read_imag(i,show = 0)
# 	contour_intens(img_read,perc = 99)
'''
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
'''
# drop_color = ["y","b","r","g","m"]
# def overall_plot2D(op, which = "all"):

# 	b = op.viable_drop_total
# 	d = op.segmented_drop_files
# 	c = op.in_track_total
# 	c1 = op.io_track_total
# 	c2 = op.ot_track_total
# 	cp = op.in_msd_all
# 	cp1 = op.io_msd_all
# 	cp2 = op.ot_msd_all

# 	for i in range(len(b)):

# 		if len(d[i]) != 0:
# 			img = mpimg.imread(d[i][0])
# 			timg = plt.imshow(img,cmap=plt.get_cmap('gray'))

# 		for j in range(len(b[i])):
# 			if which == "all" or which == "in":
# 				for l in range(len(c[i][j])):
# 					if len(c[i][j][l])!=0:
# 						temp = np.array(c[i][j][l])
# 						#plt.plot(temp[0],temp[1],'b-')
# 						plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
# 			if which == "all" or which == "io":
# 				for l in range(len(c1[i][j])):
# 					if len(c1[i][j][l])!=0:
# 						temp = np.array(c1[i][j][l])
# 						#plt.plot(temp[0],temp[1],'g-')
# 						plt.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
# 			if which == "all" or which == "out":

# 				for l in range(len(c2[i][j])):
# 					if len(c2[i][j][l])!=0:
# 						temp = np.array(c2[i][j][l])
# 						#plt.plot(temp[0],temp[1],'r-')
# 						plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

# 			# if (len(b[i][j])>0):
# 			# 	for k in range(len(b[i][j])):
# 			# 		circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# 		plt.colorbar()
# 		#plt.savefig("Frame_{0}".format(i))
# 		plt.show()
# 	return






















# def overall_plot2D_contour(op, which = "all", scatter = 0, line = 0):

# 	b = op.viable_drop_total
# 	d = op.segmented_drop_files
# 	c = op.in_track_total
# 	c1 = op.io_track_total
# 	c2 = op.ot_track_total
# 	cp = op.in_msd_all
# 	cp1 = op.io_msd_all
# 	cp2 = op.ot_msd_all

# 	for i in range(len(b)):

# 		if len(d[i]) != 0:
# 			img = mpimg.imread(d[i][0])
# 			timg = plt.imshow(img,cmap=plt.get_cmap('gray'),origin = "lower")
# 			copy_array_in = np.zeros(np.shape(img))
# 			copy_array_io = np.zeros(np.shape(img))
# 			copy_array_ot = np.zeros(np.shape(img))
# 			copy_array_all = np.zeros(np.shape(img))


# 		random_choose_c = [0,3]
# 		random_choose_c1 = [0,3]
# 		random_choose_c2 = [0,3]
# 		choose_b = np.random.randint(0,len(b[i]),2)
# 		for j in range(len(b[i])):

# 			for l in range(len(c[i][j])):
# 				if len(c[i][j][l])!=0:
# 					temp = np.array(c[i][j][l])
# 					copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
# 					copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
# 					#plt.plot(temp[0],temp[1],'b-')
# 					if (which == "all" or which == "in") and scatter :
# 						plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
# 					if (which == "all" or which == "in") and line and (l in random_choose_c) and (j in choose_b):
# 						plt.plot(temp[0],temp[1],c = 'r')

# 			for l in range(len(c1[i][j])):
# 				if len(c1[i][j][l])!=0:
# 					temp = np.array(c1[i][j][l])
# 					copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
# 					copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
# 					#plt.plot(temp[0],temp[1],'g-')
# 					if (which == "all" or which == "io") and scatter:
# 						plt.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
# 					if (which == "all" or which == "in") and line and l in random_choose_c1:
# 						plt.plot(temp[0],temp[1],c = 'b')

# 			for l in range(len(c2[i][j])):
# 				if len(c2[i][j][l])!=0:
# 					temp = np.array(c2[i][j][l])
# 					copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
# 					copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
# 					#plt.plot(temp[0],temp[1],'r-')
# 					if (which == "all" or which == "out") and scatter:
# 						plt.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
# 					if (which == "all" or which == "in") and line and l in random_choose_c2:
# 						plt.plot(temp[0],temp[1],c = 'g')
# 			# if (len(b[i][j])>0):
# 			# 	for k in range(len(b[i][j])):
# 			# 		circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# 		plt.contour(copy_array_all)
# 		plt.colorbar()
# 		#plt.savefig("Frame_{0}".format(i))
# 		plt.show()
# 	return


# def animate(i,ax):
#     # azimuth angle : 0 deg to 360 deg
#     ax.view_init(elev=i, azim=i*4)

#     return 



# def overall_plot3D(op,which = "all"):

# 	b = op.viable_drop_total
# 	d = op.segmented_drop_files
# 	c = op.in_track_total
# 	c1 = op.io_track_total
# 	c2 = op.ot_track_total
# 	cp = op.in_msd_all
# 	cp1 = op.io_msd_all
# 	cp2 = op.ot_msd_all


# 	for i in range(len(b)):
# 		fig = plt.figure()
# 		ax = fig.add_subplot(111,projection = '3d')
# 		if len(d[i]) != 0:
# 			img = mpimg.imread(d[i][0])
# 			#timg = ax2.imshow(img, cmap=plt.get_cmap('gray'))
		
# 		for j in range(len(b[i])):
# 			if which == "all" or which == "in":
# 				for l in range(len(c[i][j])):
# 					if len(c[i][j][l])!=0:
# 						temp = np.array(c[i][j][l])
# 						#plt.plot(temp[0],temp[1],'b-')
# 						im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
# 			if which == "all" or which == "io":
# 				for l in range(len(c1[i][j])):
# 					if len(c1[i][j][l])!=0:
# 						temp = np.array(c1[i][j][l])
# 						#plt.plot(temp[0],temp[1],'g-')
# 						im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
# 			if which == "all" or which == "out":
# 				for l in range(len(c2[i][j])):
# 					if len(c2[i][j][l])!=0:
# 						temp = np.array(c2[i][j][l])
# 						#plt.plot(temp[0],temp[1],'r-')
# 						im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

# 			# if (len(b[i][j])>0):
# 			# 	for k in range(len(b[i][j])):
# 			# 		circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
# 		#fig.colorbar(im,ax=ax3)
# 		#plt.savefig("Frame_{0}".format(i))
# 		#fig.show()
# 		ani = animation.FuncAnimation(fig, animate,fargs = [ax],frames=180, interval=50)
# 		fn = op.wd + "{0}".format(i)
# 		ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
# 		ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)
# 	return


# def spacialplot_msd(op):
# 	x = op.all_tracks_x
# 	y = op.all_tracks_y
# 	z = op.all_msd
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(x, y, z, marker = '.',alpha = 0.3)
# 	plt.show()

# 	return

#overall_plot(rp2.viable_drop_total,rp2.segmented_drop_files,rp2.in_track_total ,rp2.io_track_total ,rp2.ot_track_total,rp2.in_msd_all, rp2.io_msd_all, rp2.ot_msd_all)

# def other_plot(op):
# 	fraction_tick = [i for i in range(1,int(op.frame_total/op.frame_step)+1)]
# 	create_box_plot(op.tmframe_occ,fraction_tick,y_label = "Fraction inside the drop",x_label = "Frame number",y_lim = (),title = "Percent Occupation of Track in Drop per Frame Over All Experiments")

# 	for i in op.tmframe_occ:

# 		w_i = np.ones_like(i)/float(len(i))
# 		plt.hist(i,histtype = 'step',weights=w_i)
# 		plt.xlabel("Fraction inside the drop")
# 		plt.ylabel("Probability")
# 		plt.title("Percent Occupation of Track in Drop per Frame")
# 	plt.show()
# 	return
#other_plot(rp2)
































