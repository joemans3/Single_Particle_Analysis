import numpy as np
from skimage.color import rgb2gray
import math
import sys

def fit_MSD(t,p_0,p_1):
    return p_0 * (t**(p_1))

def dif_dis(x,y):
    c = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return c

def dist(x,y,c1,c2):
    '''Distance(s) x,y away from a point c1,c2 in 2D'''
    tx=np.abs(c1-x)
    ty=np.abs(c2-y)

    temp=np.sqrt((tx)**2 + (ty)**2)
    return temp

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
10
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

def ang(a,b):

    ''' takes input as tuple of tuples of X,Y.'''

    la = [(a[0][0]-a[1][0]), (a[0][1]-a[1][1])]
    lb = [(b[0][0]-b[1][0]), (b[0][1]-b[1][1])]

    dot_ab = dot(la,lb)

    ma = dot(la,la)**0.5
    mb = dot(lb,lb)**0.5

    a_cos = dot_ab/(ma*mb)
    try:
        angle = math.acos(dot_ab/(mb*ma))
    except:
        angle = math.acos(round(dot_ab/(mb*ma)))

    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        return 360 - ang_deg
    else:
        return ang_deg



#get the angle between a series of connected lines (trajectories); N lines = N-1



def angle_trajectory_2d(x,y,ref = True):
    ''' Takes input (x,y) of a series of arrays or one array of 
    trajectorie(s) and returns a series or one array of angles in 2D.
    
    INPUTS:

    x,y (array-like): series of arrays or one array of trajectorie(s).

    ref (boolian): If True, return an extra angle which is the angle between the first line in the set and a verticle line
    
    RETURN:

    Array-like: A series or one array of angles in 2D depending on shape of x,y.


    '''
   
    if isinstance(x[0],list):
        angle_list = [[] for i in x]
        for i in range(len(x)):
            for j in range(len(x[i])-2):

                angle_list[i].append(ang((((x[i],y[i]),(x[i+1],y[i+1])),((x[i+1],y[i+1]),(x[i+2],y[i+2])))))
        return angle_list

    else:
        return [ang(((x[i],y[i]),(x[i+1],y[i+1])),((x[i+1],y[i+1]),(x[i+2],y[i+2]))) for i in range(len(x)-2)]

def angle_trajectory_3d(x,y,z,ref = True):
    ''' Takes input (x,y,z) of a series of arrays or one array of 
    trajectorie(s) and returns a series or one array of angles in 3D.
    
    INPUTS:

    x,y (array-like): series of arrays or one array of trajectorie(s).

    ref (boolian): If True, return an extra angle which is the angle between the first line in the set and a verticle line
    
    RETURN:

    Array-like: A series or one array of angles in 2D depending on shape of x,y,z.


    '''
   
    if isinstance(x[0],list):
        angle_list = [[] for i in x]
        for i in range(len(x)):
            for j in range(len(x[i])-2):

                angle_list[i].append(ang((((x[i],y[i],z[i]),(x[i+1],y[i+1],z[i+1])),((x[i+1],y[i+1],z[i+1]),(x[i+2],y[i+2],z[i+2])))))
        return angle_list

    else:
        return [ang(((x[i],y[i],z[i]),(x[i+1],y[i+1],z[i+1])),((x[i+1],y[i+1],z[i+1]),(x[i+2],y[i+2],z[i+2]))) for i in range(len(x)-2)]




#convert degrees to rad
def d_to_rad(deg_):
    return np.array(deg_)*np.pi/180.0

#convert rad to deg
def rad_to_d(rad_):
    return np.array(rad_)*180.0/np.pi


def con_pix_si(data, con_nm = 0.130,con_ms = 20,which = 0):

    if which == 0:
        return data

    if which == 'msd':
        return (1000./20.)*(con_nm**2)*np.array(data)

    if which == 'um':
        return (con_nm)*np.array(data)








