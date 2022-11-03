import PIL
from joblib import PrintTime
import numpy as np
from skimage.color import rgb2gray
import math
import sys
from scipy.optimize import curve_fit
from sklearn import mixture
import matplotlib.pyplot as plt
from PIL import Image
import re

def rt_to_xy(r,theta):
    y = r*np.sin(theta)
    x = r*np.cos(theta)
    return np.array([x,y])


def pad_array(subarray, shape, top_left_coord, pad = 0):
    '''
    Parameters
    ----------
    subarray : 2D array-like
        array to pad 
    shape : tuple, list, array-like of length 2
        2D shape of the full array
    top_left_coord : list, array-like of length 2
        coordinate of the top left corner of the subarray in the full array of shape
    
    Returns
    -------
    array-like 2D
        returns the full array of with size shape entries of the subarray are inputted relative to top_left_coord
        padded with 0s
    '''
    try: 
        full_array = np.zeros(shape) + pad
    except:
        PrintTime("shape is not the correct type")

    shape_sub = np.shape(subarray)
    full_array[top_left_coord[1]-1:top_left_coord[1]+shape_sub[0]-1,top_left_coord[0]-1:top_left_coord[0]+shape_sub[1]-1] = subarray

    return full_array
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key, reverse=False)

def subarray2D(arr,mask,full = True,transpose = True):
    '''
    Gives a new array from a 2D defined by mask. Assumes mask is [n,m]

    Parameters
    ----------
    arr : 2D numpy array-like
        original array to be zoomed
    mask : 2D numpy array-like
        2D mask defining the corners of the box to make the subarray
    full : bool
        if True return the full size array with 0 entry anywhere not in the subarray
        else return a new array defined by the mask
    transpose : bool 
        if true transpose the mask before subindexing
        else use mask as is

    Returns
    -------
    numpy array-like
        subarray defined using the mask.
        array is same shape as original with 0 values outside subarray if full = True
    
    Notes
    -----
    Assumes mask is the same shape or smaller than the input array
    '''
    if not transpose:
        min_x = math.ceil(np.min(mask[:,0]))
        max_x = math.ceil(np.max(mask[:,0]))
        min_y = math.ceil(np.min(mask[:,1]))
        max_y = math.ceil(np.max(mask[:,1]))
    else:
        min_y = math.ceil(np.min(mask[:,0]))
        max_y = math.ceil(np.max(mask[:,0]))
        min_x = math.ceil(np.min(mask[:,1]))
        max_x = math.ceil(np.max(mask[:,1]))

    if full == False:
        return arr[min_x:max_x,min_y:max_y]
    else:
        arr_copy = np.zeros(arr.shape) 
        arr_copy[min_x:max_x,min_y:max_y] = arr[min_x:max_x,min_y:max_y]
        return arr_copy 

def flatten(t):
    '''
    function to flatten a list of any dimension (arbitrary sublist dimension)

    Parameters
    ----------
    t : list
        list of any size
    
    Returns
    -------
    list
        flattened list along all dimensions of t.
    '''
    return [item for sublist in t for item in sublist]



# displacemnt cum distribution

def cum_sum(data,binz = 10):
  count, bins = np.histogram(data,bins = binz)
  pdf = count/sum(count)
  cdf = np.cumsum(pdf)
  return [cdf,bins]


def rescale(x,a,b):

    x = np.array(x)
    max_x = np.max(x)
    min_x = np.min(x)
    a = np.float(a)
    b = np.float(b)
    return np.array((((b-a)*(x-min_x)))/np.array((max_x - min_x))) + a




#fit for one gaussian for displacements
def gaus1D(x,a,b,c):
    return a*np.exp(-(x-b)/(2.*(c**2)))

def gaus2D(x,a,b,c,a1,b1,c1):
    return a*np.exp(-(x-b)/(2.*(c**2))) + a1*np.exp(-(x-b1)/(2.*(c1**2)))

def fit_MSD(t,p_0,p_1):
    return p_0 * (t**(p_1))

def dif_dis(x,y):
    c = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return c

def dist(x,y,c1,c2):
    '''Distance(s) x,y away from a point c1,c2 in 2D'''
    try:
        tx=np.abs(c1-np.array(x))
        ty=np.abs(c2-np.array(y))

        temp=np.sqrt((tx)**2 + (ty)**2)
        return temp
    except:
        return np.nan


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2 = (1,0)):
    """ Returns the angle in radians between vectors 'v1' and 'v2': over 0-2pi
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.atan2(np.linalg.det([v1_u,v2_u]),np.dot(v1_u,v2_u))

def angle_multi(v1):
    angles = []
    for i in v1:
        angles.append(angle_between(i))
    return angles












def MSD_tavg1(x,y,f,f_inc = False):
    if f_inc == True:
        return np.mean((np.diff(dist(np.array(x)[1:],np.array(y)[1:],np.array(x)[0],np.array(y)[0])/np.diff(f)))**2)/4.
    else:
        return np.mean(np.diff(dist(np.array(x)[1:],np.array(y)[1:],np.array(x)[0],np.array(y)[0]))**2)/4.


def MSD_tavg(x,y,f,f_inc = False):
    
    dists = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        dists[i] = dist(x[i],y[i],x[i+1],y[i+1])
    if f_inc == True:
        return np.mean((np.diff(dists/np.diff(f)))**2)/4.
    else:
        return np.mean((np.diff(dists))**2)/4.
    
def MSD_tavg_single(x,f,f_inc = False):
    if f_inc == True:
        return np.mean((np.diff(x/f))**2)/4.
    else:
        return np.mean((np.diff(x))**2)/4.
    
    
    

def gaussian_fit(x,p0,p1,p2):
    return ((np.sqrt(2.*np.pi*p0))**-1)*np.exp(-((x-p1)**2)/(2*p0)) + p2



##################################################################################################################################
#implimenting MLE method for detecting diffusion coeff/ velocity change in single tracks as outlined in: Detection of Velocity and Diffusion Coefficient Change Points in Single-Particle Trajectories, Yin et al. 2018

def prop_vel(x,frame_rate,n):
    
    return (1./(n*frame_rate))*np.sum(np.diff(x))

def prop_diff_c(x,frame_rate,n):
    
    return (1./(2*n*frame_rate))*np.sum((np.diff(x) - prop_vel(x,frame_rate,n)*frame_rate)**2)

def ll_0(x,n,frame_rate):
    
    return 0.5*(n*np.log10(prop_diff_c(x,frame_rate,n)))

def log_likelihood_k(x,n,k,frame_rate):
    
    return ll_0(x,n,frame_rate) - ll_0(x[:k],k,frame_rate) - ll_0(x[k+1:n],n-k,frame_rate)
    
    
    
def MLE_decomp(x,y,frame_rate):
    N = len(x)
    pros_k = list(range(1,N))
    hold_prop_kx = np.zeros(len(pros_k)+1)
    hold_prop_ky = np.zeros(len(pros_k)+1)
    
    #log-likelihood linear in x,y
    for i in range(len(pros_k)):
        hold_prop_kx[i+1] = 2.*log_likelihood_k(x,N,pros_k[i],frame_rate)
        hold_prop_ky[i+1] = 2.*log_likelihood_k(y,N,pros_k[i],frame_rate)
    
    max_x = np.sqrt(np.max(hold_prop_kx))
    max_y = np.sqrt(np.max(hold_prop_ky))
    
    return 

def cm_periodic(x,y,sizeN = 1):
    #transform x,y to -pi <-> pi
    xpi=x*2.*np.pi/sizeN
    ypi=y*2.*np.pi/sizeN
    #find the geometric mean (all points have weighting factor of 1)
    xpi_meanc=np.mean(np.cos(xpi))
    xpi_means=np.mean(np.sin(xpi))
    
    ypi_meanc=np.mean(np.cos(ypi))
    ypi_means=np.mean(np.sin(ypi))
    
    
    
    #transform back to x,y space
    thetax=np.arctan2(-xpi_means,-xpi_meanc) + np.pi
        
    thetay=np.arctan2(-ypi_means,-ypi_meanc) + np.pi

    xcm=sizeN*thetax/(2.*np.pi)
    ycm=sizeN*thetay/(2.*np.pi)
    
    return np.array([xcm,ycm])

def cm_normal(x,y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.array([mean_x,mean_y])



def radius_of_gyration(x,y):
    x = np.array(x)
    y = np.array(y)

    cm_x,cm_y = cm_normal(x,y)
    r_m = np.sqrt(cm_x**2 + cm_y**2)
    #convert to radial units
    r = np.sqrt(x**2 + y**2)

    return np.mean(np.sqrt((r-r_m)**2))

#end to end distance
def end_distance(x,y):
    x = np.array(x)
    y = np.array(y)

    return np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)







def track_decomp(x,y,f,max_track_decomp):
    #takes tracks and finds MSD for various timestep conditions.
    
    #return array-like: 
    #msd = msd values at all tau values considered
    #popt = fitted parameters on MSD equation
    #pcov = covariance matrix of fit
    
    max_decomp = np.floor(len(x)/max_track_decomp)
    tau = list(range(1,int(max_decomp+1.0)))
    msd = []
    for i in tau:
        if i < len(x):
            n_x = np.array(x)[::i]
            n_y = np.array(y)[::i]
            n_f = np.array(f)[::i]
            msd.append(MSD_tavg(n_x,n_y,n_f))
        
    #popt , pcov = curve_fit(fit_MSD,tau,np.array(msd),p0=[1,1],maxfev=10000)
    
    
    return np.array(msd)




def fit_MSD(t,p_0,p_1):
    return p_0 * (t**(p_1)) 


def fit_MSD_Linear(t,p_0,p_1):
    return t*p_1 + p_0


#fill list of lists with nan to get symmetic array
def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def ens_MSD(x,y,tau):

    msd = np.nan

    if tau < len(x[0]):
        x = np.array(x)
        y = np.array(y)

        x1 = x[:,0]
        y1 = y[:,0]
        x2 = x[:,tau]
        y2 = y[:,tau]
        #print(np.nanvar(dist(x1,y1,x2,y2)))
        msd = np.nanmean((dist(x1,y1,x2,y2)**2)/4.)

    return msd


def MSD_a_value_all_ens(xy_data, lengths = False, threshold = 3, plot_avg = True, plot_all = True, plot_dist = False, plot_dist_log = True, plot_box = False, verbose = False, sim = False):
    
    if sim == True:
        x = xy_data[0]
        y = xy_data[1]
    else:
        x = []
        y = []
        for i in xy_data:
            for j in range(len(i)):
                x.append(i[j][0])
                y.append(i[j][1])

    x_pad = boolean_indexing(x)
    y_pad = boolean_indexing(y)




    msd = []
    for i in range(1,len(x_pad[0])):
        msd.append(ens_MSD(x_pad,y_pad,i))


    tau = range(1,len(x_pad[0]))


    dist_a = []
    dist_d = []

    # msd_var = msd

    # msd_new = []

    # for i in range(len(msd_var)):
    #     msd_new += msd_var[i]
    # msd = msd_new

    if lengths != False:
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)

        ax2.hist(lengths)
        ax2.set_xlabel("Track Length")
        ax2.set_ylabel("Count")
    else:
        fig = plt.figure()
        ax = fig.add_subplot()

    popt, pcov = curve_fit(fit_MSD,np.array(tau[:threshold]),np.array(msd[:threshold]),p0=[1,1],maxfev=1000000)
    dist_a = popt[1]
    dist_d = popt[0]
    ax.plot(tau,msd,'-')

    ax.plot(np.arange(1,15),fit_MSD(np.arange(1,15),10.0,1.0),'k--')
    #ax.plot(np.arange(1,15),fit_MSD(np.arange(1,15),0.0000001,0.0),'k--')

            

    ax.set_xlabel("tau (-1)")
    ax.set_ylabel("MSD")
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.show()
    dists = [dist_a,dist_d]

    if plot_dist:
        ax1.hist(dist_a,bins=50)
        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Count")
        plt.show()

    if (plot_dist & plot_box):
        
        
        ticks = ["Alpha", "Diffusion Coeff."]
        for i in range(1,len(dists)+1):
            plt.boxplot(dists[i-1])
            y = dists[i-1]
            x = np.random.normal(1, 0.04, size=len(y))
            plt.plot(x, y, 'r.', alpha=0.2)
            #if i == 1:
            #   plt.ylim((-2,2))
            
            plt.ylabel("Distributions of Fitted {0} Without Averaging Tracks".format(ticks[i-1]))
            plt.show()
        

    return [dists,msd,tau]

def MSD_a_value_all(msd, xy_data, lengths = False, threshold = 3, plot_avg = True, plot_all = True, plot_dist = True, plot_dist_log = True, plot_box = False, verbose = False, sim = False):
    #use self.in_msd_track etc. for msd input
    dist_a = []
    dist_d = []
    if sim == True:
            x = xy_data[0]
            y = xy_data[1]
    else:
        x = []
        y = []
        for i in xy_data:
            for j in range(len(i)):
                x.append(i[j][0])
                y.append(i[j][1])

    x_pad = boolean_indexing(x)
    y_pad = boolean_indexing(y)




    msd_e = []
    for i in range(1,len(x_pad[0])):
        msd_e.append(ens_MSD(x_pad,y_pad,i))


    tau = range(1,len(x_pad[0]))
    popt1, pcov1 = curve_fit(fit_MSD,np.array(tau[:threshold]),np.array(msd_e[:threshold]),p0=[1,1],maxfev=1000000)

    print(popt1,np.sqrt(np.diag(pcov1)))


    # msd_var = msd

    # msd_new = []

    # for i in range(len(msd_var)):
    #     msd_new += msd_var[i]
    # msd = msd_new

    if lengths != False:
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)

        ax2.hist(lengths)
        ax2.set_xlabel("Track Length")
        ax2.set_ylabel("Count")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)
    #print(msd)
    for i in range(len(msd)):
        # if i ==0:
        #     print(msd[i])
       #print("msdi")
        #print(msd[i])
        for j in range(len(msd[i])):
            if len(msd[i][j]) > 3:

                #print(msd[i][j])
                temp_mean = msd[i][j]

                temp_mean = temp_mean[~np.isnan(temp_mean)]
                #print(len(np.array(range(1,len(temp_mean)+1)[:threshold])))
                #print(len(np.array(temp_mean[:threshold])))
                popt, pcov = curve_fit(fit_MSD,np.array(range(1,len(temp_mean)+1)[:threshold]),np.array(temp_mean[:threshold]),p0=[1,1],maxfev=1000000)
                dist_a.append(popt[1])
                dist_d.append(popt[0])
                ax.plot(range(1,len(temp_mean)+1),temp_mean,'-')

                #ax.plot(range(1,len(temp_mean)+1),fit_MSD(range(1,len(temp_mean)+1),popt[0],popt[1]),'g',lw=3)
    ax.plot(np.arange(1,15),fit_MSD(np.arange(1,15),10.0,1.0),'k--')
    #ax.plot(np.arange(1,15),fit_MSD(np.arange(1,15),0.0000001,0.0),'k--')

    ax.plot(tau,msd_e,'b-',linewidth = 3,label = "Ensemble = {0}".format(popt1[1]))
  
    ax.legend()
    ax.set_xlabel("tau (-1)")
    ax.set_ylabel("MSD")
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.show()
    dists = [dist_a,dist_d]

    if plot_dist:
        ax1.hist(dist_a,bins=50)
        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Count")
        plt.show()

    if (plot_dist & plot_box):
        
        
        ticks = ["Alpha", "Diffusion Coeff."]
        for i in range(1,len(dists)+1):
            plt.boxplot(dists[i-1])
            y = dists[i-1]
            x = np.random.normal(1, 0.04, size=len(y))
            plt.plot(x, y, 'r.', alpha=0.2)
            #if i == 1:
            #   plt.ylim((-2,2))
            
            plt.ylabel("Distributions of Fitted {0} Without Averaging Tracks".format(ticks[i-1]))
            plt.show()
        
    print(np.mean(dist_a),np.std(dist_a))
    return [dists,msd_e,popt1]


def MSD_a_value_sim_all(msd,xy_data, lengths = False, threshold = 10, plot_avg = True, plot_all = True, plot_dist = True, plot_dist_log = True, plot_box = False, verbose = False):

    x = xy_data[0]
    y = xy_data[1]




    x_pad = boolean_indexing(x)
    y_pad = boolean_indexing(y)

    
    tau = list((range(1,len(x_pad[0]))))
    msd1 = []
    for i in tau:
        msd1.append(ens_MSD(x_pad,y_pad,i))


    dist_a = []
    dist_d = []

    if lengths != False:
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax1 = fig.add_subplot(312)
        ax2 = fig.add_subplot(313)

        ax2.hist(lengths)
        ax2.set_xlabel("Track Length")
        ax2.set_ylabel("Count")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

    for i in range(len(msd)):
        if len(msd[i][0]) >3:

            temp_mean = msd[i][0]
            temp_mean = temp_mean[~np.isnan(temp_mean)]
            #print(len(np.array(range(1,len(temp_mean)+1)[:threshold])))
            #print(len(np.array(temp_mean[:threshold])))
            popt, pcov = curve_fit(fit_MSD,np.array(range(1,len(temp_mean)+1)[:threshold]),np.array(temp_mean[:threshold]),p0=[1,1],maxfev=1000000)
            dist_a.append(popt[1])
            dist_d.append(popt[0])
            ax.plot(range(1,len(temp_mean)+1),temp_mean,'-')

            #ax.plot(range(1,len(temp_mean)+1),fit_MSD(range(1,len(temp_mean)+1),popt[0],popt[1]),'g',lw=3)
    ax.plot(np.arange(1,150),fit_MSD(np.arange(1,150),0.1,1.0),'k--')
    ax.plot(np.arange(1,150),fit_MSD(np.arange(1,150),0.01,1.0),'k--')
    ax.plot(np.arange(1,150),fit_MSD(np.arange(1,150),0.0000001,0.0),'k--')

    ax.plot(tau,msd1,'b-',linewidth = 3)
    popt, pcov = curve_fit(fit_MSD,np.array(tau)[:threshold],np.array(msd1[:threshold]),p0=[1,1],maxfev=1000000)
    print(popt,np.sqrt(np.diag(pcov)))
    ax.set_xlabel("tau (-1)")
    ax.set_ylabel("MSD")
    ax.set_xscale('log')
    ax.set_yscale('log')
    #.show()
    dists = [dist_a,dist_d]

    if plot_dist:
        ax1.hist(dist_a)
        ax1.set_xlabel("Alpha")
        ax1.set_ylabel("Count")
        #plt.show()

    if (plot_dist & plot_box):
        
        
        ticks = ["Alpha", "Diffusion Coeff."]
        for i in range(1,len(dists)+1):
            plt.boxplot(dists[i-1])
            y = dists[i-1]
            x = np.random.normal(1, 0.04, size=len(y))
            plt.plot(x, y, 'r.', alpha=0.2)
            #if i == 1:
            #   plt.ylim((-2,2))
            
            plt.ylabel("Distributions of Fitted {0} Without Averaging Tracks".format(ticks[i-1]))
            #plt.show()
        
    print(np.mean(dist_a),np.std(dist_a))
    return dists


def track_decomp_single(x,f,max_track_decomp):
    #takes tracks and finds MSD for various timestep conditions.
    
    #return array-like: 
    #msd = msd values at all tau values considered
    #popt = fitted parameters on MSD equation
    #pcov = covariance matrix of fit
    
    max_decomp = np.floor(len(x)/max_track_decomp)
    tau = list(range(1,int(max_decomp+1.0)))
    msd = []
    for i in tau:
        if i < len(x):
            n_x = np.array(x)[::i]
            n_f = np.array(f)[::i]
            msd.append(MSD_tavg_single(n_x,n_f))
        
    #popt , pcov = curve_fit(fit_MSD,tau,np.array(msd),p0=[1,1],maxfev=10000)
    
    
    return np.array(msd)



def rgb_to_grey(rgb_img):
    '''Convert rgb image to greyscale'''
    return rgb2gray(rgb_img)






def cumsum(x,y):
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    
    dr = np.sqrt(dx**2 + dy**2)
    
    return dr






def dot(a,b):
    return a[0]*b[0] + a[1]*b[1]





#define a function to calculate the angle between 3 points in 2D

def angle2D(X,Y):
    
    #find the two vectors defined by the 3 points
    vec_a = np.array([X[0] - X[1], Y[0] - Y[1]])
    vec_b = np.array([X[2] - X[1], Y[2] - Y[1]])

    #define the dot product
    dot_ab = np.dot(vec_a,vec_b)

    #define the angle using the definition of the dot product and the angle between the two vectors
    angle = np.arccos(dot_ab/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b)))

    #return the angle in degrees
    #make condition if the angle is undefined to return 0 angle
    if np.isnan(angle):
        return 0

    return rad_to_d(angle)



#define a utility function which takes trajectory data and outputs the angles between all the vectors defined in the trajectory

def trajectory_angle(X,Y):

    #check to see if the input of X is actual in the right form (list of coordinates)
    #if it is not then assume it is a nested list and do the angle calculation for the inner list
    if isinstance(X[0],list):
        nested_angle = [[] for i in X]
        for i in range(len(X)):
            for j in range(len(X[i])-2):

                x = np.array([X[i][j],X[i][j+1],X[i][j+2]])
                y = np.array([Y[i][j],Y[i][j+1],Y[i][j+2]])

                nested_angle[i].append(angle2D(x,y))
        return nested_angle
    else:
        angles = []
        for i in range(len(X)-2):

            x = np.array([X[i],X[i+1],X[i+2]])
            y = np.array([Y[i],Y[i+1],Y[i+2]])

            angles.append(angle2D(x,y))
        return np.array(angles)




























# def ang(a,b):

#     ''' takes input as tuple of tuples of X,Y.'''

#     la = [(a[0][0]-a[1][0]), (a[0][1]-a[1][1])]
#     lb = [(b[0][0]-b[1][0]), (b[0][1]-b[1][1])]

#     dot_ab = dot(la,lb)

#     ma = dot(la,la)**0.5
#     mb = dot(lb,lb)**0.5

#     a_cos = dot_ab/(ma*mb)
#     try:
#         angle = math.acos(dot_ab/(mb*ma))
#     except:
#         angle = 0
#     ang_deg = math.degrees(angle)%360

#     if ang_deg-180>=0:
#         return 360 - ang_deg
#     else:
#         return ang_deg




# #get the angle between a series of connected lines (trajectories); N lines = N-1



# def angle_trajectory_2d(x,y,ref = True):
#     ''' Takes input (x,y) of a series of arrays or one array of 
#     trajectorie(s) and returns a series or one array of angles in 2D.
    
#     INPUTS:

#     x,y (array-like): series of arrays or one array of trajectorie(s).

#     ref (boolian): If True, return an extra angle which is the angle between the first line in the set and a verticle line
    
#     RETURN:

#     Array-like: A series or one array of angles in 2D depending on shape of x,y.


#     '''
   
#     if isinstance(x[0],list):
#         angle_list = [[] for i in x]
#         for i in range(len(x)):
#             for j in range(len(x[i])-2):

#                 angle_list[i].append(ang((((x[i],y[i]),(x[i+1],y[i+1])),((x[i+1],y[i+1]),(x[i+2],y[i+2])))))
#         return angle_list

#     else:
#         return [ang(((x[i],y[i]),(x[i+1],y[i+1])),((x[i+1],y[i+1]),(x[i+2],y[i+2]))) for i in range(len(x)-2)]

# def angle_trajectory_3d(x,y,z,ref = True):
#     ''' Takes input (x,y,z) of a series of arrays or one array of 
#     trajectorie(s) and returns a series or one array of angles in 3D.
    
#     INPUTS:

#     x,y (array-like): series of arrays or one array of trajectorie(s).

#     ref (boolian): If True, return an extra angle which is the angle between the first line in the set and a verticle line
    
#     RETURN:

#     Array-like: A series or one array of angles in 2D depending on shape of x,y,z.


#     '''
   
#     if isinstance(x[0],list):
#         angle_list = [[] for i in x]
#         for i in range(len(x)):
#             for j in range(len(x[i])-2):

#                 angle_list[i].append(ang((((x[i],y[i],z[i]),(x[i+1],y[i+1],z[i+1])),((x[i+1],y[i+1],z[i+1]),(x[i+2],y[i+2],z[i+2])))))
#         return angle_list

#     else:
#         return [ang(((x[i],y[i],z[i]),(x[i+1],y[i+1],z[i+1])),((x[i+1],y[i+1],z[i+1]),(x[i+2],y[i+2],z[i+2]))) for i in range(len(x)-2)]




#convert degrees to rad
def d_to_rad(deg_):
    return np.array(deg_)*np.pi/180.0

#convert rad to deg
def rad_to_d(rad_):
    return np.array(rad_)*180.0/np.pi


def con_pix_si(data, con_nm = 0.130,con_ms = 20.,which = 0):

    if which == 0:
        return data

    if which == 'msd':
        return (1000./con_ms)*(con_nm**2)*np.array(data)

    if which == 'um':
        return (con_nm)*np.array(data)







def GMM_utility2(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = (),ax = 0):
    
    data = np.array(data)
    weights_1 = np.ones_like(data)/float(len(data))
    p_thresh = np.percentile(data,inclusion_thresh)
    inds = ((data>=p_thresh[0]) & (data<=p_thresh[1]))
    data = data[inds]
    
    gmix = mixture.GaussianMixture(n_components=n, covariance_type='diag')
    if log:
        (results,bins) = np.histogram(np.log10(data),weights=weights_1,bins=biners)
    else:
        (results,bins) = np.histogram(data,weights=weights_1,bins=biners)


    data_arr = np.zeros((len(data),2))
    data_arr[:,0] = np.random.normal(1, 0.04, size=len(data))
    if log:
        data_arr[:,1] = np.log10(data)
    else: 
        data_arr[:,1] = data
    if verbose:
        ax.plot(data_arr[:,1],data_arr[:,0],'r.')
        ax.set_ylim((0,2))
        ax.set_title(title_1d)
        ax.set_xlabel(x_label)
    gmix.fit(data_arr)
    
    if log:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1])))
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,0],np.sqrt(gmix.covars_[:,0])))
        print("Fitted Mean(normal): {0} +/- {1}".format(np.exp(gmix.means_[:,1]),np.exp(gmix.means_[:,1])*np.sqrt(gmix.covars_[:,1])))
    else:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1])))
    max_r = np.max(results)
    ax.plot(np.diff(bins)+bins[:len(bins)-1],results)
    for i in gmix.means_:
        ax.axvline(x=i[1],color='red')
    ax.set_title(title_2d)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_2d)
    try:
        ax.set_xlim(x_limit)
    except:
        print("Warning: x_limit is invalid")
    
    return 


def create_box_plot(box_data,tick_list,y_label = "",x_label = "",y_lim = (),title = ""):
    ticks = tick_list
    plt.boxplot(box_data,positions = range(1,len(tick_list)+1))
    for i in range(1,len(tick_list)+1):
        y = box_data[i-1]
        x = np.random.normal(i, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.2)
    try:
        plt.ylim(y_lim) 
    except:
        print("Warning: y_lim not valid")
    plt.xticks(xrange(1, len(ticks) * 1 + 1, 1), ticks)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()
        
    return






def GMM_utility(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = ()):
    import matplotlib.pyplot as plt



    data = np.array(data)
    weights_1 = np.ones_like(data)/float(len(data))
    p_thresh = np.percentile(data,inclusion_thresh)
    inds = ((data>=p_thresh[0]) & (data<=p_thresh[1]))
    data = data[inds]
    
    gmix = mixture.GaussianMixture(n_components=n, covariance_type='diag')
    if log:
        (results,bins) = np.histogram(np.log10(data),weights=weights_1,bins=biners)
    else:
        (results,bins) = np.histogram(data,weights=weights_1,bins=biners)


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
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covariances_[:,1])))
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,0],np.sqrt(gmix.covariances_[:,0])))
        print("Fitted Mean(normal): {0} +/- {1}".format(10**(gmix.means_[:,1]),10**(gmix.means_[:,1])*np.sqrt(gmix.covariances_[:,1])))
    else:
        print("Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covariances_[:,1])))
    max_r = np.max(results)


    #figure setup
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)




    ax.plot(np.diff(bins)+bins[:len(bins)-1],results)
    for i in gmix.means_:
        ax.axvline(x=i[1],color='red')
    ax.set_title(title_2d)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_2d)
    try:
        ax.set_xlim(x_limit)
    except:
        print("Warning: x_limit is invalid")
    fig.savefig("gmm_{0}.svg".format(title_1d))
    plt.show()
    
    return 

def norm_weights(data):
  weights = np.ones_like(data)/float(len(data))
  return weights

def log_p(which):
  plt.hist(np.log10(which))
  plt.show()
  m, s = stats.norm.fit(np.log10(which))
  return 10**(m),10**(s)

def run_gmm_all(which,n):
  GMM_utility(which,n,log = True)
  return

def nor(which):
  print([np.mean(which),np.std(which)])
  return 


####For out tajectories, calculate the average or minimum distance away from the drops

def distance_from_drop_OUT(data_set,cm_distance = False, minimum_distance = True, plot_it = True):
    #input should be an instance of run_analysis for the dataset being analysed
    #this will go over every movie in that dataset and calculate the distances between the trajectorys listed in Trajectory_Collection for every drop identified (only OUT)
    #by default it will calculate the minimum distance away from said drop of the trajectory


    total_dist_away = []
    all_msd_total = []
    for key,value in data_set.Movie.items():
        droplet_distance_away = []
        msd_total = []
        for key_i,value_i in value.Trajectory_Collection.items():
            
            #store the coordinates and radius of the droplet
            droplet_x, droplet_y, droplet_radius = data_set.Movie[key].Drop_Collection[key_i]
            per_droplet = []
            msd_per_drop = []
            #iterate over the OUT_Trajectory_Collection
            for k in value_i.OUT_Trajectory_Collection:

                track_x = k.X
                track_y = k.Y
                distance_drop_center = con_pix_si(dist(track_x,track_y,droplet_x,droplet_y), which = 'um')
                if cm_distance:
                    k.distance_from_OUT = np.mean(distance_drop_center)
                    per_droplet.append(np.mean(distance_drop_center))
                    msd_per_drop.append(k.MSD_total_um)
                elif minimum_distance:
                    k.distance_from_OUT = np.min(distance_drop_center)
                    per_droplet.append(np.min(distance_drop_center))
                    msd_per_drop.append(k.MSD_total_um)
            droplet_distance_away.append(per_droplet)
            msd_total.append(msd_per_drop)

            if plot_it:
                plt.scatter(per_droplet ,np.log10(msd_per_drop), alpha = 0.5,label = "Droplet: {0} Radius = {1}".format(key_i,droplet_radius))
                plt.xlim((0,5))
                plt.xlabel("Minimum Distance from Drop")
                plt.ylabel("MSD of Out Trajectory")
                plt.legend()

        plt.show()        
        total_dist_away.append(droplet_distance_away)
        all_msd_total.append(msd_total)
    return [total_dist_away, all_msd_total]



def pairCorrelationFunction_2D(x, y, S, rMax, dr):
    """Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius rMax drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller rMax...or write some code to handle edge effects! ;)
    
    Paramaters
    ----------
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        rMax            outer diameter of largest annulus
        dr              increment for increasing radius of annulus

    Returns
    -------
    a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles

    Notes
    -----
    Implimentation taken from: https://github.com/cfinch/Shocksolution_Examples/blob/master/PairCorrelation/paircorrelation.py
    """
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram
    # Number of particles in ring/area of ring/number of reference particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Find particles which are close enough to the box center that a circle of radius
    # rMax will not cross any edge of the box
    bools1 = x > rMax
    bools2 = x < (S - rMax)
    bools3 = y > rMax
    bools4 = y < (S - rMax)
    interior_indices, = where(bools1 * bools2 * bools3 * bools4)
    num_interior_particles = len(interior_indices)

    if num_interior_particles < 1:
        raise  RuntimeError ("No particles found for which a circle of radius rMax\
                will lie entirely within a square of side length S.  Decrease rMax\
                or increase the size of the square.")

    edges = arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = zeros([num_interior_particles, num_increments])
    radii = zeros(num_increments)
    numberDensity = len(x) / S**2

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = sqrt((x[index] - x)**2 + (y[index] - y)**2)
        d[index] = 2 * rMax

        (result, bins) = histogram(d, bins=edges, normed=False)
        g[p, :] = result/numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = mean(g[:, i]) / (pi * (rOuter**2 - rInner**2))

    return (g_average, radii, interior_indices)