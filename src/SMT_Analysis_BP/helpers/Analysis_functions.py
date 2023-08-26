'''
This module contains a collection of functions for analysis

Some of the functions are old and not used anymore, but I am keeping them here for now
Author: Baljyot Singh Parmar
'''
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from joblib import PrintTime
from scipy.optimize import curve_fit
from skimage.color import rgb2gray
from sklearn import mixture
import copy
from scipy.spatial import ConvexHull
#import gmm
from sklearn.mixture import GaussianMixture
from src.SMT_Analysis_BP.helpers.decorators import deprecated

#curve fitting utility functions
def non_linear_curvefit(func,xdata,ydata,p0=None,method='lm',bounds=None):
    '''
    Docstring for non_linear_curvefit
    This function fits a curve to the data using scipy.optimize.curve_fit
    
    Parameters:
    -----------
    func : function
        The function to be fitted to the data
    xdata : numpy array
        The x data
    ydata : numpy array
        The y data
    p0 : numpy array, optional (default=None)
        The initial guess for the parameters
    method : str, optional (default='lm')
        The method to be used for the curve fitting, can be 'lm' or 'trf' or 'dogbox'
    bounds : tuple, optional (default=None)
        The lower and upper bounds for the parameters
    
    Returns:
    --------
    tuple
        The fitted parameters and the covariance matrix
    '''
    #check if the method is valid
    if method not in ['lm','trf','dogbox']:
        raise ValueError("Method not supported.")
    #check if func is a function
    if not callable(func):
        raise ValueError("func is not a function.")
    #check if xdata and ydata are same shape
    if not np.shape(xdata) == np.shape(ydata):
        raise ValueError("xdata and ydata are not the same shape.")
    #check if p0 has the same length as the number of parameters in func
    if not len(p0) == func.__code__.co_argcount-1:
        raise ValueError("p0 has the wrong length.")
    #check if bounds is of shape (len(p0),2)
    if not np.shape(bounds) == (2,len(p0)) and bounds is not None:
        raise ValueError("bounds has the wrong shape.")
    #if bounds are provided, use method 'trf'
    if bounds is not None:
        method = 'trf'
    #if the number of data points is less than the number of parameters, use method 'trf'
    if len(xdata) < len(p0):
        method = 'trf'
    
    #perform the curve fitting
    popt, pcov = curve_fit(func, xdata, ydata, p0=p0, method=method, bounds=bounds)
    return popt, pcov
    
def linear_fitting(xdata,ydata,deg=1):
    '''
    Docstring for linear_fitting
    This function perfroms a linear fit to the data using polyfit

    Parameters:
    -----------
    xdata : numpy array
        The x data
    ydata : numpy array
        The y data
    deg : int, optional (default=1)
        The degree of the polynomial to be fitted
    
    Returns:
    --------
    numpy array with 2 elements
        1. The polynomial coefficients
        2. The covariance matrix
    '''
    #check if xdata and ydata are same shape
    if not np.shape(xdata) == np.shape(ydata):
        raise ValueError("xdata and ydata are not the same shape.")
    #check if deg is an integer
    if not isinstance(deg,int):
        raise ValueError("deg is not an integer.")
    #check if deg is greater than 0
    if not deg > 0:
        raise ValueError("deg is not greater than 0.")
    
    #perform the linear fitting
    popt,pcov = np.polyfit(xdata,ydata,deg,cov=True,full=False)
    return np.array([popt,pcov])


#find the diffusion coefficient given a time and distance
def find_diffusion_coefficient(time,distance,dim):
    '''
    Docstring for find_diffusion_coefficient
    This function finds the diffusion coefficient given a time and distance, this is just a simple calculation:
    D = (1/(2*dim))*(distance)^2/time

    Parameters:
    -----------
    time : array-like or int or float
        The time
    distance : array-like or int or float
        The distance
    dim : int
        The dimensionality of the system
    
    Returns:
    --------
    float or numpy array
        The diffusion coefficient, units are based on the units of the input parameters (time and distance, ie: um^2/s)
    '''
    #if the time and distances are lists convert to numpy arrays
    if isinstance(time,list):
        time = np.array(time)
    if isinstance(distance,list):
        distance = np.array(distance)

    #check if time and distance are arrays or numbers
    if not isinstance(time,(int,float,np.ndarray)) or not isinstance(distance,(int,float,np.ndarray)):
        raise ValueError("time and distance must be arrays or numbers.")
    #if they are arrays, check if they are the same shape
    if isinstance(time,np.ndarray) and isinstance(distance,np.ndarray):
        if not np.shape(time) == np.shape(distance):
            raise ValueError("time and distance are not the same shape.")
    #if the dim is a number but time and distance are arrays, convert dim to an array of the same shape as time and distance
    if isinstance(dim,(int,float)) and isinstance(time,np.ndarray) and isinstance(distance,np.ndarray):
        dim = np.ones(np.shape(time))*dim
    
    #calculate the diffusion coefficient
    D = (distance**2)/(time*2*dim) 
    return D

def find_static_localization_error_MSD(sigma,dim):
    '''Docstring for find_static_localization_error_MSD
    Given the isotropic gaussian scale (sigma), this function finds the static localization error (sigma_loc) using the equation:
    sigma_loc = 2n*(sigma)^2

    Parameters:
    -----------
    sigma : array-like or int or float
        The isotropic gaussian scale
    dim : int
        The dimensionality of the system
    
    Returns:
    --------
    float or numpy array
        The static localization error
    '''
    #if sigma is a list convert to numpy array
    if isinstance(sigma,list):
        sigma = np.array(sigma)
    #check if sigma is an array or number
    if not isinstance(sigma,(int,float,np.ndarray)):
        raise ValueError("sigma must be an array or number.")
    #calculate the static localization error
    sigma_loc = 2*dim*(sigma**2)
    return sigma_loc

#function to assign a random starting point in a range
def _random_starting_point(start,end):
    return np.random.randint(start,end)


def bin_ndarray(ndarray, new_shape, operation='sum'):
    '''
    Docstring for bin_ndarray
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    
    Parameters:
    -----------
    ndarray : numpy array
        The array to be binned
    new_shape : tuple
        The shape of the output array
    operation : str, optional (default='sum')
        The operation to be performed on the pixels, can be 'sum' or 'mean'
    
    Returns:
    --------
    numpy array
        The binned array of shape new_shape
    
    Raises:
    -------
    ValueError
        If the operation is not 'sum' or 'mean'
    ValueError
        If the number of dimensions of the input array does not match the length of the new_shape tuple
 
    Examples:
    ---------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
        [[ 22  30  38  46  54]
        [102 110 118 126 134]
        [182 190 198 206 214]
        [262 270 278 286 294]
        [342 350 358 366 374]]
    '''
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def bin_img(img,bin=2,operation='sum'):
    '''
    Docstring for bin_img

    Parameters:
    -----------
    img : numpy array
        The image to be binned
    bin : int, optional (default=2)
        The binning factor, this is the number of pixels to be binned together for each axis
    operation : str, optional (default='sum')
        The operation to be performed on the pixels, can be 'sum' or 'mean'

    Returns:
    --------
    numpy array
        The binned image of shape (img.shape[0]//bin,img.shape[1]//bin)
    '''
    if operation == 'sum':
        return bin_ndarray(img, new_shape=(img.shape[0]//bin,img.shape[1]//bin), operation='sum')
    elif operation == 'mean':
        return bin_ndarray(img, new_shape=(img.shape[0]//bin,img.shape[1]//bin), operation='mean')
    else:
        raise ValueError("Operation not supported.")

#convert a (N,3) array to a (N,2) array by removing the z coordinate
def convert_3d_to_2d(a):
    b = np.zeros((np.shape(a)[0],2))
    b[:,0] = a[:,0]
    b[:,1] = a[:,1]
    return b

def squared_mean_difference(a):
    # check if the input is empty
    if a is None:
        return 0
    # check if the input is a numpy array
    if not isinstance(a, np.ndarray):
        return 0
    # check if the input is of length 0
    if len(a) == 0:
        return 0
    # check if the input is of length 1
    if len(a) == 1:
        return a[0]
    # check if the input contains any nan
    if np.any(np.isnan(a)):
        return np.nan
    # check if the input contains any inf
    if np.any(np.isinf(a)):
        return np.inf
    # calculate the square root of the sum of the squares of the input divided by the length of the input
    return np.sqrt(np.sum(a**2))/len(a)


# is a point inside a circle
def point_inside_circle2D(circle,point):
    '''Check if a point is inside a circle

    Parameters:
    -----------
    circle : tuple
        (x,y,radius) of the circle
    point : tuple
        (x,y) of the point
    
    Returns:
    --------
    bool
        True if the point is inside the circle, False otherwise
    
    Raises:
    -------
    TypeError
        If circle or point are not tuples
    TypeError
        If circle or point are not tuples of length 3 and 2 respectively
    TypeError
        If circle or point are not tuples of numbers
    ValueError
        If the radius of the circle is not positive
    
    Examples:
    ---------
    >>> point_inside_circle2D((0,0,1),(0,0))
    True
    >>> point_inside_circle2D((0,0,1),(0,1))
    False

    Notes:
    ------
    1. This function is not vectorized, so it will not work with numpy arrays
    2. This function is not robust to floating point errors
    '''

    circle_x,circle_y,rad = circle
    x,y = point

    if not isinstance(circle_x,(int,float)) or \
       not isinstance(circle_y,(int,float)) or \
       not isinstance(rad,(int,float)):
        raise TypeError("circle parameters must be numbers")
    if not isinstance(x,(int,float)) or \
       not isinstance(y,(int,float)):
        raise TypeError("point parameters must be numbers")
    if not rad > 0:
        raise ValueError("radius must be positive")

    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True
    else:
        return False


@deprecated("Sometimes this breaks, use with caution. Use array slicing instead (ex: a[:,[0,2,1,3]] to switch the 2nd and 3rd columns)")
def reshape_col2d(arr,permutations):
    '''
    Docstring for reshape_col2d
    This function reshapes a 2D array by permuting the columns in the order specified by permutations

    Parameters:
    -----------
    arr : numpy array
        The array to be reshaped
    permutations : list of integers
        The permutations to be applied to the columns of arr
    
    Returns:
    --------
    numpy array
        The reshaped array

    NOTES:
    ------
    Sometimes this breaks i have no idea why. Use with caution
    
    '''
    # Check that permutations is a list of integers 
    if not isinstance(permutations,list):
        raise TypeError('permutations must be a list')
    if not all([isinstance(i,int) for i in permutations]):
        raise TypeError('permutations must be a list of integers')
    # Check that permutations is a permutation of np.arange(len(permutations))
    idx = np.empty_like(permutations)
    idx[permutations] = np.arange(len(permutations))
    arr[:]=arr[:, idx]
    return arr

def range_distance(a,b):
    '''
    Docstring for range_distance
    
    Parameters:
    -----------
    a : float
        The first number
    b : float
        The second number
    
    Returns:
    --------
    float
        The distance between the two numbers
    '''
    #find the max of the two
    max = a if a>b else b
    #find the min of the two
    min = a if a<b else b
    #find the distance between the two
    distance = np.abs(max-min)
    #return the distance
    return distance


def rt_to_xy(r,theta):
    '''
    Docstring for rt_to_xy

    Parameters:
    -----------
    r : float
        The radial coordinate
    theta : float
        The angular coordinate
    
    Returns:
    --------
    tuple
        The x and y coordinates
    
    '''
    # Check to see if r and theta are floats
    if type(r) != float or type(theta) != float:
        raise TypeError('r and theta must be floats')
    # Check to see if r and theta are arrays
    if type(r) == np.ndarray or type(theta) == np.ndarray:
        raise ValueError('r and theta must be floats')
    # Check to see if r is negative
    if r<0:
        raise ValueError('r must be positive')
    # Check to see if theta is between 0 and 2pi
    if theta < 0 or theta > 2*np.pi:
        raise ValueError('theta must be between 0 and 2*pi')
    # If all of the above checks pass, convert to cartesian coordinates
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    # Return the coordinates as an array
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
        return
    shape_sub = np.shape(subarray)
    if top_left_coord[0] + shape_sub[0] > shape[0]:
        PrintTime("The subarray does not fit in the full array along the x axis")
        return
    if top_left_coord[1] + shape_sub[1] > shape[1]:
        PrintTime("The subarray does not fit in the full array along the y axis")
        return
    full_array[top_left_coord[1]-1:top_left_coord[1]+shape_sub[0]-1,top_left_coord[0]-1:top_left_coord[0]+shape_sub[1]-1] = subarray

    return full_array
def sorted_alphanumeric(data):
    # Function to convert text to int if text is a digit, else convert to lowercase
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # Function to split the text into a list of digits and non-digits
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    # Sort the list of data using the alphanum_key function
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
        
    # Check that mask is within the bounds of arr
    if (min_x < 0) or (max_x > arr.shape[0]) or (min_y < 0) or (max_y > arr.shape[1]):
        raise ValueError('Mask is outside the bounds of the array.')
    # Check that mask is not empty
    if (max_x - min_x) < 1 or (max_y - min_y) < 1:
        raise ValueError('Mask is empty.')

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
    try:
        return [item for sublist in t for item in sublist]
    except TypeError:
        return [t]

def rescale_range(x,min_x,max_x,a,b):
    '''https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range'''
    if min_x >= max_x:
        raise ValueError("min_x={} is not less than max_x={}".format(min_x,max_x))
    if a >= b:
        raise ValueError("a={} is not less than b={}".format(a,b))
    return ((b-a)*(x - min_x)/(max_x - min_x)) + a

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
    a = np.float(a)  # type: ignore
    b = np.float(b)  # type: ignore
    if min_x == max_x:
        if min_x == 0:
            return np.array([a] * len(x))
        else:
            return np.array([a] * len(x)) + np.array([b-a] * len(x)) * x
    else:
        return np.array((((b-a)*(x-min_x)))/np.array((max_x - min_x))) + a




#fit for one gaussian for displacements
def gaus1D(x,a,b,c):
    return a*np.exp(-(x-b)/(2.*(c**2)))

def gaus2D(x,a,b,c,a1,b1,c1):
    return a*np.exp(-(x-b)/(2.*(c**2))) + a1*np.exp(-(x-b1)/(2.*(c1**2)))


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




#######MSD calculations for sim data format of dict ###

def MSD_tau_utility(x,y,tau=1,permutation=True):
    '''Documentation for MSD_tau_utility

    Parameters:
    -----------
    x : array
        x positions of the data
    y : array
        y positions of the data
    tau : int
        time lag for the MSD calculation
    permutation : bool
        if permutation == True then the MSD is calculated for all possible permutations of the data
        if permutation == False then the MSD is calculated for the data in the order it is given
    
    Returns:
    --------
    displacements : array, shape (n,2)
        array of displacements 
    
    
    '''

    #if permutation == True then the MSD is calculated for all possible permutations of the data
    #if permutation == False then the MSD is calculated for the data in the order it is given
    if permutation == True:
        displacements = _msd_tau_utility_all(x,y,tau)
    else:
        #dont use this condition, its wrong
        displacements = _msd_tau_utility_single(x,y,tau)

    return displacements
    
def _msd_tau_utility_all(x,y,tau):
    ''' Documentation for _msd_tau_utility_all

    Parameters:
    -----------
    x : array
        x positions of the data
    y : array
        y positions of the data
    tau : int
        time lag for the MSD calculation

    Returns:
    --------
    displacements : array, shape (n,2)
        array of displacements for all possible permutations of the data
    
    Notes:
    ------
    For the theory behind this see https://web.mit.edu/savin/Public/.Tutorial_v1.2/Concepts.html#A1
    '''
    #find the total displacements possible, from https://web.mit.edu/savin/Public/.Tutorial_v1.2/Concepts.html#A1
    total_displacements = len(x) - tau
    #create an array to store the displacements
    displacements = np.zeros((total_displacements,2))
    #loop through the displacements
    for i in range(total_displacements):
        #calculate the displacements
        #make sure that i+tau is less than the length of the data
        if i+tau < len(x):
            displacements[i] = np.array([x[i+tau]-x[i],y[i+tau]-y[i]])
    #return the displacements as (x,y) pairs
    return displacements

def _msd_tau_utility_single(x,y,tau): 
    #dont use this, its just to show this doesn't work as well as the permutation method
    x_dis = np.diff(x[::tau])
    y_dis = np.diff(y[::tau])
    #return the displacements as (x,y) pairs
    return np.array([x_dis,y_dis]).T

def MSD_tau(x,y,permutation=True):
    '''Documentation for MSD_tau

    Parameters:
    -----------
    x : array
        x positions of the data
    y : array
        y positions of the data
    permutation : bool
        if permutation == True then the MSD is calculated for all possible permutations of the data
        if permutation == False then the MSD is calculated for the data in the order it is given
    
    Returns:
    --------
    displacements : dict
        dictionary of displacements for each time lag, key = time lag, value = array of displacements, shape (n,2)
    
    '''

    #find the maximum time lag possible
    max_tau = len(x)-1
    #create a dictionary to store the displacements for each time lag
    displacements = {}
    #loop through the time lags
    for tau in range(1,max_tau+1):
        #calculate the displacements for each time lag
        displacements[tau] = MSD_tau_utility(x,y,tau,permutation)
    #return the displacements
    return displacements

def MSD_Tracks(tracks,permutation=True,return_type="msd_curves",verbose=False,conversion_factor=None):
    '''Documentation for MSD_Tracks

    Parameters:
    -----------
    tracks : dict
        dictionary of tracks, key = track ID, value = [[x,y],...] of coordinates
    permutation : bool (default = True, don't change this)
        if permutation == True then the MSD is calculated for all possible permutations of the data
        if permutation == False then the MSD is calculated for the data in the order it is given
    return_type : str (default = "msd_curves")
        if return_type == "msd_curves" then the function returns the MSD curves for each track (ensemble MSD curve)
        if return_type == "displacements" then the function returns the displacements for each track
        if return_type == "both" then the function returns both the MSD curves and the displacements for each track
    verbose : bool (default = False)
        if verbose == True then returns the raw ensemble MSD displacement for each tau aswell as the MSD curves noted above
        TODO: this is annoying and should be separated into a different function but is not yet
    conversion_factor : float (default = None)
        if conversion_factor != None then the coordinates are converted to the desired units before the MSD is calculated
    
    Returns:
    --------
    return_dict : dict
        dictionary of MSD curves for each track, key = track ID, value = dictionary of displacements for each time lag, key = time lag, value = array of displacements, shape (n,2)
    
    '''
    #create a dictionary to store the MSD curves for each track
    ensemble_msd = {}
    #create a dictionary to store the displacements for each track
    tracks_displacements = {}
    #loop through the tracks
    for key,value in tracks.items():

        #convert the coordinates based on the conversion factor
        if conversion_factor != None:
            value *= conversion_factor
        #calculate the displacements for each track
        disp = MSD_tau(value[:,0],value[:,1],permutation)
        tracks_displacements[key] = disp
        #unify the ensemble MSD curve dictionary with disp
        ensemble_msd = dic_union_two(ensemble_msd,disp)
    if verbose == True:
        ensemble_msd_copy = copy.deepcopy(ensemble_msd)
    #update the ensemble MSD curve dictionary
    ensemble_msd,errors_ensemble_msd = msd_avgerage_utility(ensemble_msd)
    return_dict = {"msd_curves":[ensemble_msd,errors_ensemble_msd],"displacements":tracks_displacements}
    if verbose == True:
        if return_type == "both":
            return return_dict,ensemble_msd_copy
        else:
            return return_dict[return_type],ensemble_msd_copy
    else:
        if return_type == "both":
            return return_dict
        else:
            return return_dict[return_type]

def msd_avgerage_utility(displacements):
    '''Documentation for _msd_avgerage_utility

    Parameters: 
    -----------
    displacements : dict
        dictionary of displacements for each time lag, key = time lag, value = array of displacements, shape (n,D), D is the dimension of the data
    
    Returns:
    --------
    msd : dict
        dictionary of the MSD for each time lag, key = time lag, value = array of MSD values, shape (n,)
    error_msd : dict (this is the standard error of the mean of the MSD)
        dictionary of the error in the MSD for each time lag, key = time lag, value = array of error in the MSD values, shape (n,) 
    
    '''
    #create a dictionary to store the MSD for each time lag
    msd = {}
    error_msd = {}
    #loop through the time lags
    for key,value in displacements.items():
        #calculate the MSD for each time lag
        #the MSD is the average of the squared displacements
        #the squared displacements are the sum of the squared components of the displacements
        #divide by the number of dimensions to get the average of the squared displacements
        msd[key] = np.mean(np.sum(np.array(value)**2,axis=1))
        #calculate the error in the MSD for each time lag
        #the error in the MSD is the standard deviation of the standard error of the mean of the squared displacements
        #the standard error of the mean of the squared displacements is the standard deviation of the squared displacements divided by the square root of the number of displacements
        error_msd[key] = np.std(np.sum(np.array(value)**2,axis=1))/np.sqrt(len(value))
    #return the MSD
    return [msd,error_msd]

def dic_union_two(dic_1,dic_2):
    '''Documentation for dic_union_two
    Find the unique union of two dictionaries, assumes the values are lists

    Parameters:
    -----------
    dic_1 : dict
        dictionary 1
    dic_2 : dict
        dictionary 2
    
    Returns:
    --------
    dic_union : dict
        dictionary of the unique union of the two dictionaries
    
    Notes:
    ------
    The values of the dictionaries must be lists for the list concatenation to work properly
    This function turns the values of the dictionaries into lists, if arrays are needed then the values must be converted back to arrays

    '''
    #create a dictionary to store the union
    dic_union = {}
    #loop through the keys in the first dictionary
    for key,value in dic_1.items():
        #check if the key is in the second dictionary
        if key in dic_2:
            #if the key is in both dictionaries then add the values together
            dic_union[key] = list(value) + list(dic_2[key])
        else:
            #if the key is only in the first dictionary then add the value to the union
            dic_union[key] = value
    #loop through the keys in the second dictionary
    for key,value in dic_2.items():
        #check if the key is in the first dictionary
        if key in dic_1:
            #if the key is only in the second dictionary then add the value to the union
            dic_union[key] = list(value) + list(dic_1[key])
        else:
            #if the key is only in the second dictionary then add the value to the union
            dic_union[key] = list(value)
    #return the union
    return dic_union


######Track percent identity functions######

def point_per_frame_difference(true_points_per_frame,extracted_points_per_frame):
    ''' Documentation for point_per_frame_difference
    Finds the closest point in the extracted points per frame to the true points per frame and returns the difference between the two points
    for each such point in each frame
    
    Parameters:
    -----------
    true_points_per_frame : dict
        dictionary of the true points per frame, key = frame number, value = array of points, shape (n,2)
    extracted_points_per_frame : dict
        dictionary of the extracted points per frame, key = frame number, value = array of points, shape (n,2)

    Returns:
    --------
    '''
    #extracted points per frame may be less than true points per frame so we need to find the closest point in extracted points per frame to each point in true points per frame
    

    return 

def percent_error(true,estimate,abs=True):
    ''' Documentation for percent_error

    Parameters:
    -----------
    true : float
        true value
    estimate : float
        estimated value
    abs : bool (default = True)
        if abs == True then the absolute value of the percent error is returned

    Returns:
    --------
    percent_error : float
        percent error between the true and estimated values
    
    Notes:
    ------
    1. Assumes that the true and estimated values are floats
    2. Assumes that the true value is not zero
    3. Assumes that the true value is not negative
    4. Assumes that the true value is not NaN
    5. Assumes that the estimated value is not NaN


    '''
    #check if the true and estimated values are floats or ints
    if not isinstance(true,(float,int)):
        raise TypeError("The true value must be a float or int")
    if not isinstance(estimate,(float,int)):
        raise TypeError("The estimated value must be a float or int")  
    #check if the true value is zero
    if true == 0:
        raise ValueError("The true value cannot be zero")
    #check if the true value is negative
    if true < 0:    
        raise ValueError("The true value cannot be negative")
    
    #calculate the percent error
    if abs == True:
        percent_error = np.abs((true-estimate)/true)
    else:
        percent_error = (true-estimate)/true
    #return the percent error
    return percent_error*100

def _point_identity(point_true,track_estimate,distance_threshold):
    ''' Documentation for _point_identity

    Parameters:
    -----------
    point_true : array
        true point, shape (,2)
    track_estimate : array
        estimated track, shape (n,2)
    distance_threshold : float
        distance threshold for the point to be considered a match
    
    Returns:
    --------
    1 : int
        if the point is within the distance threshold
    0 : int
        if the point is not within the distance threshold

    Notes:
    ------
    1. Assumes that the tracks and points are formatted as [[x,y,T],[x,y,T],...,[x,y,T]] numpy arrays and [x,y,T] for a single point
    '''
    #find the point in the track_estimate that is closest to the point_true in terms of euclidean distance
    #only search for the tracks that has the same T
    min_temp =np.sqrt(np.sum((track_estimate[track_estimate[:,2] == point_true[2]] - point_true)**2,axis=1))
    if min_temp.size == 0:
        return 0
    else:
        min_dist = np.min(min_temp)
    #check if the minimum distance is less than the distance threshold
    if min_dist < distance_threshold:
        #if the minimum distance is less than the distance threshold then return 1
        return 1
    else:
        #if the minimum distance is greater than the distance threshold then return 0
        return 0

def identity_tracks(track_true,track_estimate,**kwargs):
    ''' Documentation for identity_tracks

    Parameters:
    -----------
    track_true : array
        true track, shape (n,2)
    track_estimate : array
        estimated track, shape (m,2), m can be different than n
    threshold : float
        distance threshold for the point to be considered a match
    
    Returns:
    --------
    mean_point_identity : float
        mean percent identity between the points in the true track and the estimated track
    length_error : float
        percent error between the lengths of the tracks
    
    Notes:
    ------
    1. Assumes that the tracks and points are formatted as [[x,y,T],[x,y,T],...,[x,y,T]] numpy arrays and [x,y,T] for a single point
    2. Assumes that the true track is not empty
    3. Assumes that the estimated track is not empty
    '''
    #for each point in the true track find the percent identity between the true point and the estimated track points
    mean_point_identity = np.mean(np.array([_point_identity(point_true,track_estimate,kwargs.get("threshold",1)) for point_true in track_true]))
    #find the percent error between the lengths of the tracks
    length_error = percent_error(len(track_true),len(track_estimate),abs=False)

    return mean_point_identity,length_error

def identity_track_matrix(tracks_true,tracks_estimate,verbose=True,**kwargs):
    ''' Documentation for identity_track_matrix
    This creates all unique combinations of tracks and then finds the identity between each pair

    Parameters:
    -----------
    tracks_true : dict
        dict of true tracks, shape (n,2). Key is the track number, value is the track
    tracks_estimate : dict
        dict of estimated tracks, shape (m,2). Key is the track number, value is the track
    verbose : bool
        if True then return all the identity matrices, if False then only return the per track results
    
    Returns:
    --------
    max_identity : array
        array of the maximum identity between each true track and the estimated tracks
    lengh_error : array
        array of the percent error between the lengths of each true track and the estimated tracks
    identity_matrix : array
        matrix of the identity between each pair of tracks
    length_error_matrix : array
        matrix of the percent error between the lengths of each pair of tracks
    
    Notes:
    ------

    '''
    
    #create a matrix to store the identity between each pair of tracks
    identity_matrix = np.zeros((len(tracks_true),len(tracks_estimate)))
    #create a matrix to store the percent error between the lengths of each pair of tracks
    length_error_matrix = np.zeros((len(tracks_true),len(tracks_estimate)))

    #make a matrix to store the lengths of the true tracks and the estimated tracks
    length_true = np.array([len(track) for _,track in tracks_true.items()])
    length_estimate = np.array([len(track) for _,track in tracks_estimate.items()])

    #loop through the true tracks
    for i,(_,track_true) in enumerate(tracks_true.items()):
        #loop through the estimated tracks
        for j,(_,track_estimate) in enumerate(tracks_estimate.items()):
            #find the identity between the true track and the estimated track
            identiy_error,length_error = identity_tracks(track_true,track_estimate,**kwargs)
            #store the identity error in the identity matrix
            identity_matrix[i,j] = identiy_error
            #store the length error in the length error matrix
            length_error_matrix[i,j] = length_error


    #for each true track find the estimated track that has the maximum identity, find all multiple occurances the indices where the max identity is
    max_val_per_row = np.max(identity_matrix,axis=1)

    #all the below is probably overkill because the max identity very rarely has multiple occurances
    #however to be safe I am going to keep it in, this is very inefficient for scaling of the track numbers; any better way to do this?
    #if this ends up being too slow i am adding the old way to do it in a comment
    #max_index = np.argmax(identity_matrix,axis=1)
    #max_identity = identity_matrix[np.arange(len(tracks_true)),max_index]
    #min_length_error = length_error_matrix[np.arange(len(tracks_true)),max_index]

    #find all the indices where the max identity is
    max_identity_row = np.array([np.where(identity_matrix[i] == max_val_per_row[i])[0] for i in range(len(tracks_true))])
    #for each row find the index what minimizes the length error
    min_length_error = np.array([np.min(length_error_matrix[i,max_identity_row[i]]) for i in range(len(tracks_true))])
    #find the index of the minimum length error
    min_length_error_index = np.array([np.where(length_error_matrix[i,max_identity_row[i]] == min_length_error[i])[0][0] for i in range(len(tracks_true))])
    #find the index of the maximum identity
    max_identity = np.array([max_identity_row[i][min_length_error_index[i]] for i in range(len(tracks_true))])

    #find the maximum identity value
    max_identity_value = np.array([identity_matrix[i,max_identity[i]] for i in range(len(tracks_true))])
    #find the length error
    length_error = np.array([length_error_matrix[i,max_identity[i]] for i in range(len(tracks_true))])


    #if verbose is False jsut return the identiy_matrix[max_identity] and length_error as a dict
    if verbose == False:    
        return {'max_identity':max_identity_value,
                'length_error':length_error,
                "true_track_lengths":length_true,
                "estimate_track_lengths":length_estimate[max_identity]}

    #if verbose is True then return the max_identity and length_error as a dict and the identity matrix and length error matrix
    else:
        return {'max_identity':max_identity_value,
                'length_error':length_error,
                'identity_matrix':identity_matrix,
                'length_error_matrix':length_error_matrix,
                "true_track_lengths":length_true,
                "estimate_track_lengths":length_estimate[max_identity]}

def point_error_detection(true_points,extracted_points,threshold=0.5):
    '''Docstring for point_error_detection
    Calculate the error between the true points and the extracted points per frame

    Parameters:
    -----------
    true_points : dict
        dictionary of true points where the keys are the frame numbers and the values are the true points in that frame
    extracted_points : dict
        dictionary of extracted points where the keys are the frame numbers and the values are the extracted points in that frame
    threshold : float
        threshold for the distance between the true points and the extracted points
    
    Returns:
    --------
    percent_detected : float
        percent of the true points that are detected

    '''
    #make a counter to store the number of points that are detected
    detected_points = 0
    #find the total amount of points across all frames
    total_points = np.sum([len(points) for _,points in true_points.items()])
    #loop through the true points
    for frame_number,true_points in true_points.items():
        #if the frame number is in the extracted points
        if frame_number in extracted_points.keys():
            #loop through the true points
            for point in true_points:
                #find the distances between the true point and all the extracted points,
                #if the min distance is less than the threshold then add one to the detected points
                min_temp =np.sqrt(np.sum((extracted_points[frame_number] - point)**2,axis=1))
                if min_temp.size == 0:
                    continue
                if np.min(min_temp) < threshold:
                    detected_points += 1
    #return the percent of points detected
    return 100*detected_points/total_points

def points_per_frame_convert(tracks):
    ''' Docstring for points_per_frame_convert
    Take a track dictionary and convert it to a points per frame dictionary where the keys are the frame numbers and the values are the points in that frame
    
    Parameters:
    -----------
    tracks : dict
        dictionary of tracks where the keys are the track numbers and the values are the tracks

    Returns:
    --------
    points_per_frame : dict
        dictionary of points per frame where the keys are the frame numbers and the values are the points in that frame
    '''
    #create a dictionary to store the points per frame
    points_per_frame = {}
    #loop through the tracks
    for _,track in tracks.items():
        #loop through the points in the track
        for point in track:
            #if the frame number is not in the points per frame dictionary add it
            if point[-1] not in points_per_frame:
                points_per_frame[point[-1]] = []
            #add the point to the points per frame dictionary 
            points_per_frame[point[-1]].append(point[:-1])
    #make sure the points per frame dictionary is a numpy array of structure: [[x,y],[x,y],...] for each frame
    for frame_number,points in points_per_frame.items():
        points_per_frame[frame_number] = np.array(points)
    #return the points per frame dictionary
    return points_per_frame

def points_per_frame_bulk_sort(x:np.ndarray,y:np.ndarray,t:np.ndarray)->dict:
    '''
    Utility function, shouldn't be needed.

    Sorts points by frame number. The output should be a dict with keys of frame numbers and values of points in that frame.

    Parameters:
    -----------
    x : np.ndarray
        x coordinates of points
    y : np.ndarray
        y coordinates of points
    t : np.ndarray
        frame numbers of points

    Returns:
    --------
    points_per_frame : dict
        dictionary of points per frame where the keys are the frame numbers and the values are the points in that frame
    '''
    #make sure x,y,t are of same length
    assert len(x) == len(y) == len(t)
    #convert the values of t into integers
    t = t.astype(int)
    points_frame = {}
    for i in range(len(t)):
        if t[i] not in points_frame.keys():
            points_frame[t[i]] = []
        points_frame[t[i]].append([x[i],y[i]])
    for frame_number,points in points_frame.items():
        points_frame[frame_number] = np.array(points)
    return points_frame

def convert_point_pairs(tracks):
    '''Docstring for convert_point_pairs
    Convert a track dictionary to a point pair dictionary where the keys is a combination of the two frames of the points and the values are the point pairs
    point pairs are defined as consecutive points in a track.
    eg: track = [[x1,y1,frame1],[x2,y2,frame2],[x3,y3,frame3]]
        point_pairs = {"frame1,frame2":[[x1,y1,frame1],[x2,y2,frame2]],"frame1,frame2":[[x2,y2,frame2],[x3,y3,frame3]]}
    
    Parameters:
    -----------
    tracks : dict, keys are track numbers and values are tracks as defined above
        dictionary of tracks where the keys are the track numbers and the values are the tracks
    
    Returns:
    --------
    point_pairs : dict, keys is a combination of the two frames of the points are point pairs as defined above
        dictionary of point pairs where the keys is a combination of the two frames of the points and the values are the point pairs
    
    '''
    #create a dictionary to store the point pairs
    point_pairs = {}
    #loop through the tracks
    for _,track in tracks.items():
        #loop through the points in the track
        for i in range(len(track)-1):
            #if the point pair is not in the point pairs dictionary add it
            if str(track[i][-1])+","+str(track[i+1][-1]) not in point_pairs:
                point_pairs[str(track[i][-1])+","+str(track[i+1][-1])] = []
            #add the point pair to the point pairs dictionary
            point_pairs[str(track[i][-1])+","+str(track[i+1][-1])].append([track[i],track[i+1]])
    
    #make sure each value is a numpy array of structure: [[[x1,y1,frame1],[x2,y2,frame2]],[[x1,y1,frame1],[x2,y2,frame2]]]
    for key,point_pair in point_pairs.items():
        point_pairs[key] = np.array(point_pair)
    #return the point pairs dictionary
    return point_pairs

def point_pair_error_detection(true_point_pairs,extracted_point_pairs,threshold=0.5):
    '''Docstring for point_pair_error_detection
    Calculate the error between the true point pairs and the extracted point pairs per frame
    point_pairs = {"frame1,frame2":[[x1,y1,frame1],[x2,y2,frame2]],"frame1,frame2":[[x2,y2,frame2],[x3,y3,frame3]]}
    
    Parameters:
    -----------
    true_point_pairs : dict, keys is a combination of the two frames of the points are point pairs as defined above
        dictionary of true point pairs where the keys is a combination of the two frames of the points and the values are the point pairs
    extracted_point_pairs : dict, keys is a combination of the two frames of the points are point pairs as defined above
        dictionary of extracted point pairs where the keys is a combination of the two frames of the points and the values are the point pairs
    threshold : float
        threshold for the distance between the true point pairs and the extracted point pairs
    
    Returns:
    --------
    percent_detected : float
        percent of the true point pairs that are detected
    mismatch_error : float
        percent of true detected point pairs over the total amount of extracted pairs
    '''
    #make a counter to store the number of point pairs that are detected
    detected_point_pairs = 0
    #find the total amount of point pairs across all frames
    total_point_pairs = np.sum([len(point_pairs) for _,point_pairs in true_point_pairs.items()])
    #find the total amount of point pairs found in the extracted point pairs
    total_extracted_point_pairs = np.sum([len(point_pairs) for _,point_pairs in extracted_point_pairs.items()])

    #make a deep copy of the extracted point pairs
    extracted_point_pairs_copy = copy.deepcopy(extracted_point_pairs)

    #loop through the true point pairs
    for frame_number,true_point_pair in true_point_pairs.items():
        #if the frame number is in the extracted point pairs
        if frame_number in extracted_point_pairs_copy.keys():
            #loop through the true point pairs
            for point_pair in true_point_pair:
                #find the distances between the true point pair and all the extracted point pairs,
                #if the min distance is less than the threshold then add one to the detected point pairs
                min_temp = np.sqrt(np.sum((extracted_point_pairs_copy[frame_number] - point_pair)**2,axis=2))
                #if there are no extracted point pairs then continue (should not happen)
                if min_temp.size == 0:
                    continue
                #find the minimum distance between the true point pair and the extracted point pairs
                #min_temp should look like a (n,2) array where n is the number of extracted point pairs 
                # and the 2 is the distances between the true point pair and the extracted point pairs
                #then you need to sum the distances across the 2 axis to get a (n,) array of the distances for 2 points
                #so the threshold is multiplied by 2 because there are 2 points in the true point pair for comparison
                if np.min(np.sum(min_temp,axis=1)) < threshold*2:
                    detected_point_pairs += 1
                    #remove the point pair from the extracted point pairs
                    extracted_point_pairs_copy[frame_number] = np.delete(extracted_point_pairs_copy[frame_number],np.argmin(np.sum(min_temp,axis=1)),axis=0)
    #return the percent of point pairs detected
    return 100*detected_point_pairs/total_point_pairs, 100*np.abs(total_extracted_point_pairs-detected_point_pairs)/total_extracted_point_pairs


#density calculations 

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

def radius_of_gyration(*args)->float:
    '''Determine the radius of gyration of a particle given its x,y coordinates.
    If only one argument is given, it is assumed to be a 2D array of x,y coordinates.
    If two arguments are given, they are assumed to be x,y coordinates.

    Parameters:
    -----------
    *args : array-like
        if one argument is given, it is assumed to be a 2D array of x,y coordinates. (N,2)
        if two arguments are given, they are assumed to be x,y coordinates. (N,),(N,)

    Returns:
    --------
    r_g: float
        radius of gyration of particles
    
    Raises:
    -------
    ValueError
        if the number of arguments is not 1 or 2
    
    '''
    if len(args) == 1:
        x = args[0][:,0]
        y = args[0][:,1]
    elif len(args) == 2:
        x = np.array(args[0])
        y = np.array(args[1])
    else:
        raise ValueError('Input should be (N,2) array of x,y coordinates or two arrays of x,y coordinates (N,), (N,)')

    #find center of mass
    cm_x,cm_y = cm_normal(x,y)
    r_m = np.sqrt(cm_x**2 + cm_y**2)
    #convert to radial units
    r = np.sqrt(x**2 + y**2)

    return np.mean(np.sqrt((r-r_m)**2))

def area_points_per_frame(points_per_frame:dict,area_func:callable = radius_of_gyration)->dict:
    '''
    Parameters:
    -----------
    points_per_frame : dict
        dictionary of points per frame, keys are the frame number and the values are the points in that frame of shape (n,2) given 2D data
    area_func : callable
        function to calculate the area of the points per frame, default is radius of gyration
    Returns:
    --------
    area_per_frame : dict
        dictionary of area per frame, keys are the frame number and the values are the area of the points in that frame

    '''
    area_per_frame = {}
    for frame_number,points in points_per_frame.items():
        area_per_frame[frame_number] = area_func(points)
    return area_per_frame

def convex_hull_area(points):
    '''Docstring for convex_hull_area
    Calculate the convex hull area of a set of points
    
    Parameters:
    -----------
    points : numpy array of shape (n,2)
        points to calculate the convex hull area
    
    Returns:
    --------
    hull_points : numpy array of shape (n,2)
        convex hull area of the input points
    '''
    #check if the points are atlest 3 points
    if len(points) < 3:
        #return None
        return 0
    #calculate the convex hull of the points
    try:
        hull = ConvexHull(points)
    except:
        return 0
    #return the volume of the convex hull
    return hull.volume




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

def fit_MSD_loc_err(t,p_0,p_1,p_2):
    return p_0 * (t**(p_1)) + p_2

def fit_MSD(t,p_0,p_1,p_2):
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

from numpy import arange, histogram, mean, pi, sqrt, zeros
def centered_pairCorrelation_2D(x, y, center, rMax, dr, **kwargs):
    edges = arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = zeros([1, num_increments])
    radii = zeros(num_increments)
    xy_dist = dist(x,y,center[0],center[1])
    num_in_radius = len(np.where(xy_dist <= rMax)[0])
    numberDensity = num_in_radius / pi*(rMax**2)
    # Compute pairwise correlation for each interior particle
    for p in range(1):
        d = sqrt((center[0] - x)**2 + (center[1] - y)**2)

        (result, _) = histogram(d, bins=edges, normed=False)
        g[p, :] = result/numberDensity
    g_average = zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = mean(g[:, i]) / (pi * (rOuter**2 - rInner**2))
    return (g_average, radii, center)

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


###########################
#utility functions for GMM calculations
###########################
def GMM_1D(data:np.ndarray|list, n_components:int, **kwargs)->tuple:
    ''' Docstring for GMM_1D
    Uses sklearn.mixture.GaussianMixture to fit a 1D Gaussian Mixture Model to data
    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html for more details

    Parameters:
    -----------
    data: np.ndarray|list
        data to be fit
    n_components: int
        number of components to fit to data
    **kwargs:
        additional keyword arguments to pass to GaussianMixture
    
    Returns:
    --------
    tuple:
        (means, covariances, weights, gmm_model)

    '''
    #reshape data
    data = np.array(data).reshape(-1,1)
    #fit GMM
    gmm = GaussianMixture(n_components=n_components, **kwargs).fit(data)
    #get means, covariances, and weights
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_
    #return tuple
    return (means, covariances, weights, gmm)


























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


# def create_box_plot(box_data,tick_list,y_label = "",x_label = "",y_lim = (),title = ""):
#     ticks = tick_list
#     plt.boxplot(box_data,positions = range(1,len(tick_list)+1))
#     for i in range(1,len(tick_list)+1):
#         y = box_data[i-1]
#         x = np.random.normal(i, 0.04, size=len(y))
#         plt.plot(x, y, 'r.', alpha=0.2)
#     try:
#         plt.ylim(y_lim) 
#     except:
#         print("Warning: y_lim not valid")
#     plt.xticks(xrange(1, len(ticks) * 1 + 1, 1), ticks)
#     plt.ylabel(y_label)
#     plt.xlabel(x_label)
#     plt.title(title)
#     plt.show()
        
#     return






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

# def log_p(which):
#   plt.hist(np.log10(which))
#   plt.show()
#   m, s = stats.norm.fit(np.log10(which))
#   return 10**(m),10**(s)

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
                distance_drop_center = con_pix_si(dist(track_x,track_y,droplet_x,droplet_y), which = 'um')  # type: ignore
                if cm_distance:
                    k.distance_from_OUT = np.mean(distance_drop_center)  # type: ignore
                    per_droplet.append(np.mean(distance_drop_center))  # type: ignore
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
    from numpy import arange, histogram, mean, pi, sqrt, where, zeros

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