'''
This module contains functions for generating fractional brownian motion samples and computing the mean squared displacement

Functions:
----------
get_fbm_sample(l=1,h=0.5,d=1,n=1)
    Generates a sample of fractional brownian motion
compute_msd_np(xy, t, t_step)
    Computes the mean squared displacement for a given sample

Author: Baljyot Singh Parmar

'''

import numpy as np
import pandas as pd
from fbm import FBM


def get_fbm_sample(l=1,h=0.5,d=1,n=1):
    '''
    Generates a sample of fractional brownian motion 
    Theory: https://en.wikipedia.org/wiki/Fractional_Brownian_motion
    Implementation is using the fbm package: https://pypi.org/project/fbm/
    Default values are for testing purposes only
    

    Parameters:
    -----------
    l : float,int
        end time (from 0)
    h : float,int  (0 < h < 1)
        hurst parameter, must be between 0 and 1
    d : int 
        dimensions (x,y,z .... ) for one realization, must be greater than 0
    n : int
        even intervals from 0,l, must be greater than 0
    
    Returns:
    --------
    list of lists of numpy arrays, where the first list is the time values for each sample, and the second list is the samples themselves

    Raises:
    -------
    TypeError
        If any of the parameters are not of the correct type
    ValueError
        If any of the parameters are not of the correct value

    Notes:
    ------
    1. The number of samples is equal to the number of dimensions
	'''

    #in the following checks we make sure to print the value of the parameter that is incorrect
    if not isinstance(l, (float,int)):
        raise TypeError("Please enter a valid type for length parameter, you entered: ", type(l))
    if not isinstance(h, (float,int)):
        raise TypeError("Please enter a valid type for hurst parameter, you entered: ", type(h))
    if not isinstance(d, int):
        raise TypeError("Please enter a valid type for dimensions parameter, you entered: ", type(d))
    if not isinstance(n, (float,int)):
        raise TypeError("Please enter a valid type for intervals parameter, you entered: ", type(n))
    if l < 1:
        raise ValueError("Please enter a valid value for length parameter, you entered: ", l)
    if h < 0 or h > 1:
        raise ValueError("Please enter a valid value for hurst parameter, you entered: ", h)
    if d < 1:
        raise ValueError("Please enter a valid value for dimensions parameter, you entered: ", d)
    if n < 1:
        raise ValueError("Please enter a valid value for intervals parameter, you entered: ", n)

    f = FBM(length = l, hurst = h,n = n)
    samples = []
    sample_t = []
    for i in range(d):
        fbm_sample = f.fbm()
        fgn_sample = f.fgn()
        t_values = f.times()  
        if n == 1 and n != l:

            samples.append(fbm_sample[1:])
            sample_t.append(t_values[1:])

        else:
            samples.append(fbm_sample[:-1])
            sample_t.append(t_values[:-1])
    return [sample_t[0], samples]

	
def track_gen_util(hurst,track_num,track_length,diffusion_coefficient):
    # Generate a track with given hurst parameter and track length for each track
    num_tracks = track_num
    track_length = track_length
    hurst = hurst
    track_dict = {}
    for i in range(num_tracks):
        track = get_fbm_sample(1,hurst,2,track_length)
        _,track_2d = track
        #remap the 2d track from [[x1,x2,x3...],[y1,y2,y3...]] to [[x1,y1],[x2,y2],[x3,y3]...]
        track_2d = np.transpose(track_2d)*np.sqrt(2*diffusion_coefficient) + 100 #shift to avoid -ve values
        track_dict[i+1] = track_2d
    return track_dict

def combine_track_dict_util(*dicts):
    #assume agrs are dictionaries of the tracks
    track_dict_list = [i for i in dicts]
    combined_dict = {}
    track_counter = 1
    for track_dict in track_dict_list:
        for track in track_dict.values():
            combined_dict[track_counter] = track
            track_counter += 1
    return combined_dict
