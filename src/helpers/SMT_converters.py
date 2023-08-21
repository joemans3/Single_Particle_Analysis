import numpy as np
import glob as glob
from src.helpers import Analysis_functions as af

#make a function to take track data in a dict format dict = {track_id:[[x0,y0,frame0],[x1,y1,frame,1],...],...} and convert it to the format required for SMAUG analysis
#format for SMAUG analysis is [track_id,time_step_number,placeholder,x,y] see https://github.com/BiteenMatlab/SMAUG
#the time_step_number is the consecutive frame number starting from 0

def convert_track_data_SMAUG_format(track_data:dict)->list:
    #lets first get the number of tracks
    num_tracks = len(track_data.keys())
    #now we create a placeholder list to store the data
    data_convert = []
    track_IDS = 1
    #now we loop through the tracks
    for track_id in track_data.keys():
        #we get the data for the track
        track = track_data[track_id]
        #we loop through the track and append the data to the placeholder list
        for i in range(len(track)):
            data_convert.append([track_IDS,i+1,i+1,track[i][0],track[i][1]])
        #we increment the track id
        track_IDS += 1
    #now we return the data
    return data_convert


#converter for NOBIAS data style: https://github.com/BiteenMatlab/NOBIAS
#the goal is to use the displacment functions from Analysis_functions.py to get the displacment data for each tau,
#right now i believe the displacements for NOBIAS are only for tau =1 but we can store the whole set for later use.
#ill need to make a dir for the storing and then have the main file for tau=1 and the rest as aux files


def convert_track_data_NOBIAS_format_global(track_data:dict):
    ''' Docstring for convert_track_data_NOBIAS_format_global
    This should be run in the background to gain all the tau permutations for posterity but the main function should be convert_track_data_NOBIAS_format
    for a single tau. The other funtions will be described later.
    
    Parameters:
    -----------
    track_data: dict
        dict of track data in the format {track_id:[[x0,y0,frame0],[x1,y1,frame1],...],...}
    
    Returns:
    --------
    track_data_NOBIAS: dict
        dict of the form {"TAU_VAL":{"obs":[[x1-x0,x2-x1,...],[y1-y0,y2-y1,...]],"TrID":[track_id1,track_id1,...]}}
        "TAU_VAL" is the tau value for the displacements in "obs" (this will start at 1)
        "obs" is 2 x T where T is the number of frames and 2 is the dimension of the data (x,y)
        "TrID" is a list of track ids that correspond to the displacements in "obs"
        For example:
        {"obs":[[x1-x0,x2-x1,...],[y1-y0,y2-y1,...]],"TrID":[1,1,...]}
        this means that the displacements in "obs" are for track id 1


    Notes:
    ------
    This function is used to convert track data in the format {track_id:[[x0,y0,frame0],[x1,y1,frame1],...],...}
    to the format required for NOBIAS analysis
    '''

    
    return