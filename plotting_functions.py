import numpy as np
import matplotlib.pylab as plt
from sklearn import mixture
import matplotlib.image as mpimg
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Analysis_functions import *


def create_box_plot(box_data,tick_list,y_label = "",x_label = "",y_lim = (),title = ""):
    ticks = tick_list
    plt.boxplot(box_data,positions = list(range(1,len(tick_list)+1)), notch = True, showfliers = False)
    for i in range(1,len(tick_list)+1):
        y = box_data[i-1]
        x = np.random.normal(i, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.2)
    try:
        plt.ylim(y_lim) 
    except:
        print("Warning: y_lim not valid")
    plt.xticks(range(1, len(ticks) * 1 + 1, 1), ticks)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()
        
    return


def GMM_utility(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = ()):
    
    data = np.array(data)
    
    p_thresh = np.percentile(data,inclusion_thresh)
    inds = ((data>=p_thresh[0]) & (data<=p_thresh[1]))
    data = data[inds]
    
    gmix = mixture.GMM(n_components=n, covariance_type='diag')
    if log:
        (results,bins) = np.histogram(np.log10(data),density='true',bins=biners)
    else:
        (results,bins) = np.histogram(data,density='true',bins=biners)


    data_arr = np.zeros((len(data),2))
    data_arr[:,0] = np.random.normal(1, 0.04, size=len(data))
    if log:
        data_arr[:,1] = np.log10(data)
    else: 
        data_arr[:,1] = data
    if verbose:
        plt.plot(data_arr[:,1],data_arr[:,0],'r.')
        plt.ylim((0,2))
        plt.title(title_1d)
        plt.xlabel(x_label)
        plt.show()
    gmix.fit(data_arr)
    
    if log:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1])))
        print("Fitted Mean(normal): {0} +/- {1}".format(np.exp(gmix.means_[:,1]),np.exp(gmix.means_[:,1])*np.sqrt(gmix.covars_[:,1])))
    else:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1])))
    max_r = np.max(results)
    plt.plot(np.diff(bins)+bins[:len(bins)-1],results)
    
    for i in gmix.means_:
        plt.axvline(x=i[1],color='red')
    plt.title(title_2d)
    plt.xlabel(x_label)
    plt.ylabel(y_label_2d)
    try:
        plt.xlim(x_limit)
    except:
        print("Warning: x_limit is invalid")
    plt.show()
    
    return 


def BGMM_utility(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = ()):
    
    data = np.array(data)
    
    p_thresh = np.percentile(data,inclusion_thresh)
    inds = ((data>=p_thresh[0]) & (data<=p_thresh[1]))
    data = data[inds]
    
    bgmix = mixture.BayesianGaussianMixture(n_components=n, covariance_type='diagonal')
    if log:
        (results,bins) = np.histogram(np.log10(data),density='true',bins=biners)
    else:
        (results,bins) = np.histogram(data,density='true',bins=biners)


    data_arr = np.zeros((len(data),2))
    data_arr[:,0] = np.random.normal(1, 0.04, size=len(data))
    if log:
        data_arr[:,1] = np.log10(data)
    else: 
        data_arr[:,1] = data
    if verbose:
        plt.plot(data_arr[:,1],data_arr[:,0],'r.')
        plt.ylim((0,2))
        plt.title(title_1d)
        plt.xlabel(x_label)
        plt.show()
    bgmix.fit(data_arr)
    
    if log:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1])))
        print("Fitted Mean(normal): {0} +/- {1}".format(np.exp(gmix.means_[:,1]),np.exp(gmix.means_[:,1])*np.sqrt(gmix.covars_[:,1])))
    else:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1])))
    max_r = np.max(results)
    plt.plot(np.diff(bins)+bins[:len(bins)-1],results)
    
    for i in gmix.means_:
        plt.axvline(x=i[1],color='red')
    plt.title(title_2d)
    plt.xlabel(x_label)
    plt.ylabel(y_label_2d)
    try:
        plt.xlim(x_limit)
    except:
        print("Warning: x_limit is invalid")
    plt.show()
    
    return 




def read_imag(path,fig = False,ax = False, show = True):
    ori_img = io.imread(path)
    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(ori_img)
    
    if show:
        plt.show()
    return ori_img


def contour_intens(img,fig = False,ax = False, show = True, seg = True,perc = 95):
    grey_img = rgb_to_grey(img)
    normed_grey_img = grey_img * (grey_img>np.percentile(grey_img,perc))
    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    image_plot = ax.imshow(grey_img)
    contour_img_plot = ax.contour(normed_grey_img)
    fig.colorbar(image_plot)

    if show:
        plt.show()

    return



def spacialplot_msd(op,fig = False, ax = False, show = True):
    x = op.all_tracks_x
    y = op.all_tracks_y
    z = op.all_msd
    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker = '.',alpha = 0.3)

    if show:
        plt.show()

    return





def other_plot(op,fig = False, ax = False, show = True):

    fraction_tick = [i for i in range(1,int(op.frame_total/op.frame_step)+1)]
    create_box_plot(op.tmframe_occ,fraction_tick,y_label = "Fraction inside the drop",x_label = "Frame number",y_lim = (),title = "Percent Occupation of Track in Drop per Frame Over All Experiments")

    for i in op.tmframe_occ:

        w_i = np.ones_like(i)/float(len(i))
        if fig == False:
            plt.hist(i,histtype = 'step',weights=w_i)
            plt.xlabel("Fraction inside the drop")
            plt.ylabel("Probability")
            plt.title("Percent Occupation of Track in Drop per Frame")
        else:
            ax.hist(i,histtype = 'step',weights=w_i)
            ax.xlabel("Fraction inside the drop")
            ax.ylabel("Probability")
            ax.title("Percent Occupation of Track in Drop per Frame")
    if show:
        plt.show()
    return



def animate(i,ax):
    # azimuth angle : 0 deg to 360 deg
    ax.view_init(elev=i, azim=i*4)

    return 




def overall_plot3D(op,which = "all", fig = False, ax = False,cmap = 'warm', save = True):
    is_fig = fig

    b = op.viable_drop_total
    d = op.segmented_drop_files
    c = op.in_track_total
    c1 = op.io_track_total
    c2 = op.ot_track_total
    cp = op.in_msd_all
    cp1 = op.io_msd_all
    cp2 = op.ot_msd_all


    for i in range(len(b)):
        if fig == False:
            is_fig = plt.figure()
            ax = fig.add_subplot(111,projection = '3d')
        if len(d[i]) != 0:
            img = mpimg.imread(d[i][0])
            #timg = ax2.imshow(img, cmap=plt.get_cmap('gray'))
        
        for j in range(len(b[i])):
            if which == "all" or which == "in":
                for l in range(len(c[i][j])):
                    if len(c[i][j][l])!=0:
                        temp = np.array(c[i][j][l])
                        #plt.plot(temp[0],temp[1],'b-')
                        im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
            if which == "all" or which == "io":
                for l in range(len(c1[i][j])):
                    if len(c1[i][j][l])!=0:
                        temp = np.array(c1[i][j][l])
                        #plt.plot(temp[0],temp[1],'g-')
                        im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
            if which == "all" or which == "out":
                for l in range(len(c2[i][j])):
                    if len(c2[i][j][l])!=0:
                        temp = np.array(c2[i][j][l])
                        #plt.plot(temp[0],temp[1],'r-')
                        im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

            # if (len(b[i][j])>0):
            #   for k in range(len(b[i][j])):
            #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
        #fig.colorbar(im,ax=ax3)
        #plt.savefig("Frame_{0}".format(i))
        #fig.show()
        if save:
            ani = animation.FuncAnimation(fig, animate,fargs = [ax],frames=180, interval=50)
            fn = op.wd + "{0}".format(i)
            ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
            ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)
    return



def overall_plot2D_contour(op, which = "all", scatter = 0, line = 0, fig = False, ax = False,cmap = 'warm', show = True, delay = False):

    is_fig = fig

    b = op.viable_drop_total
    d = op.segmented_drop_files
    c = op.in_track_total
    c1 = op.io_track_total
    c2 = op.ot_track_total
    cp = op.in_msd_all
    cp1 = op.io_msd_all
    cp2 = op.ot_msd_all

    for i in range(len(b)):
        if is_fig == False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if len(d[i]) != 0:
            img = mpimg.imread(d[i][0])
            timg = plt.imshow(img,cmap=plt.get_cmap('gray'),origin = "lower")
            copy_array_in = np.zeros(np.shape(img))
            copy_array_io = np.zeros(np.shape(img))
            copy_array_ot = np.zeros(np.shape(img))
            copy_array_all = np.zeros(np.shape(img))


        random_choose_c = [0,3]
        random_choose_c1 = [0,3]
        random_choose_c2 = [0,3]
        choose_b = np.random.randint(0,len(b[i]),2)
        for j in range(len(b[i])):


            for l in range(len(c[i][j])):
                if len(c[i][j][l])!=0:
                    temp = np.array(c[i][j][l])
                    copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
                    copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
                    #plt.plot(temp[0],temp[1],'b-')
                    if (which == "all" or which == "in") and scatter :
                        ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
                    if (which == "all" or which == "in") and line and (l in random_choose_c) and (j in choose_b):
                        ax.plot(temp[0],temp[1],c = 'r')

            for l in range(len(c1[i][j])):
                if len(c1[i][j][l])!=0:
                    temp = np.array(c1[i][j][l])
                    copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
                    copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
                    #plt.plot(temp[0],temp[1],'g-')
                    if (which == "all" or which == "io") and scatter:
                        ax.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
                    if (which == "all" or which == "io") and line and l in random_choose_c1:
                        ax.plot(temp[0],temp[1],c = 'b')

            for l in range(len(c2[i][j])):
                if len(c2[i][j][l])!=0:
                    temp = np.array(c2[i][j][l])
                    copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
                    copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
                    #plt.plot(temp[0],temp[1],'r-')
                    if (which == "all" or which == "out") and scatter:
                        ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
                    if (which == "all" or which == "out") and line and l in random_choose_c2:
                        ax.plot(temp[0],temp[1],c = 'g')
            # if (len(b[i][j])>0):
            #   for k in range(len(b[i][j])):
            #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
        cont = ax.contour(copy_array_all)
        fig.colorbar(cont)
        #plt.savefig("Frame_{0}".format(i))
        if show:
            plt.show()
    return


def overall_plot2D(op, which = "all", fig = False, ax = False,cmap = 'warm', show = True):
    drop_color = ["y","b","r","g","m"]
    is_fig = fig

    b = op.viable_drop_total
    d = op.segmented_drop_files
    c = op.in_track_total
    c1 = op.io_track_total
    c2 = op.ot_track_total
    cp = op.in_msd_all
    cp1 = op.io_msd_all
    cp2 = op.ot_msd_all

    for i in range(len(b)):
        if is_fig == False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if len(d[i]) != 0:
            img = mpimg.imread(d[i][0])
            timg = plt.imshow(img,cmap=plt.get_cmap('gray'))

        for j in range(len(b[i])):
            if which == "all" or which == "in":
                for l in range(len(c[i][j])):
                    if len(c[i][j][l])!=0:
                        temp = np.array(c[i][j][l])
                        #plt.plot(temp[0],temp[1],'b-')
                        ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
            if which == "all" or which == "io":
                for l in range(len(c1[i][j])):
                    if len(c1[i][j][l])!=0:
                        temp = np.array(c1[i][j][l])
                        #plt.plot(temp[0],temp[1],'g-')
                        ax.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
            if which == "all" or which == "out":

                for l in range(len(c2[i][j])):
                    if len(c2[i][j][l])!=0:
                        temp = np.array(c2[i][j][l])
                        #plt.plot(temp[0],temp[1],'r-')
                        ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

            # if (len(b[i][j])>0):
            #   for k in range(len(b[i][j])):
            #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
        ax.colorbar()
        #plt.savefig("Frame_{0}".format(i))
        if show:
            plt.show()
    return






#plot a histogram of angles in polar coordinates for each fraction
def hist_polar(data, fig = False, ax_n = False, bin_n = 10, show = True, include_ = True, align = 'edge'): #align can also be 'center'
    '''Helper function to plot histogram of angles in deg on polar coordinates.'''

    bins = np.linspace(0.0, 2.0 * np.pi, bin_n + 1.0)

    #convert to radians
    temp = d_to_rad(data)
    if include_:
        angle_rad = temp
    else:
        angle_rad = temp[temp<3.14]

    print(angle_rad)
    n, _ = np.histogram(angle_rad, bins)

    width = 2.0 * np.pi / bin_n

    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')

    else:
        ax = fig.add_subplot(ax_n)


    bars = ax.bar(bins[:bin_n], n, width=width, bottom=0.0,align = align)

    for bar in bars:
        bar.set_alpha(0.5)

    if show:
        plt.show()

    return angle_rad




def create_circular_mask(h, w, center=None, radius=None):

    ''' h,w are the dimensions of the image'''
    
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask






