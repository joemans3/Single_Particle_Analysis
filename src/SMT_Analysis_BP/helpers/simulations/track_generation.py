import src.SMT_Analysis_BP.helpers.simulations.fbm_utility as fbm
import numpy as np

def track_gen_util(hurst:float,track_num:int,track_length:int,diffusion_coefficient:float,dim:int=2)->dict:
    ''' Utility function to generate tracks using fbm_utility.py in a dictionary format. Keys are track numbers and values are the tracks themselves.

    Parameters:
    -----------
    hurst : float
        Hurst parameter of the track, one unique hurst for each track
    track_num : int
        Number of tracks to generate
    track_length : int
        Length of each track, here all tracks are of the same length
    diffusion_coefficient : float
        Diffusion coefficient of the track. In units of (1 space^2/ 1 time) unit. One unique diffusion coefficient for each track.
    dim : int
        Dimension of the track
    
    Returns:
    --------
    dict
        Dictionary of tracks with keys as track numbers and values as the tracks themselves
    '''
    # Generate a track with given hurst parameter and track length for each track
    num_tracks = track_num
    track_length = track_length
    hurst = hurst
    track_dict = {}
    for i in range(num_tracks):
        track = fbm.get_fbm_sample(track_length,hurst,dim,track_length)
        _,track_2d = track
        #remap the 2d track from [[x1,x2,x3...],[y1,y2,y3...]] to [[x1,y1],[x2,y2],[x3,y3]...]
        track_2d = np.transpose(track_2d)*np.sqrt(4*diffusion_coefficient) + 100 #shift to avoid -ve values
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