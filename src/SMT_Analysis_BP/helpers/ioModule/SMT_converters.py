import os
import numpy as np
import glob as glob
from SMT_Analysis_BP.helpers.analysisFunctions import MSD_Utils as af

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

def convert_track_data_NOBIAS_format_global(track_data:dict,max_tau:int = 1)->list:
    ''' Docstring for convert_track_data_NOBIAS_format_global
    This should be run in the background to gain all the tau permutations for posterity but the main function should be convert_track_data_NOBIAS_format
    for a single tau. The other funtions will be described later.
    
    Parameters:
    -----------
    track_data: dict
        dict of track data in the format {track_id:[[x0,y0,frame0],[x1,y1,frame1],...],...}
    max_tau: int
        the maximum tau to use for the NOBIAS analysis

    Returns:
    --------
    track_data_NOBIAS: list of dict
        dict of the form [{"obs":[[x1-x0,x2-x1,...],[y1-y0,y2-y1,...]],"TrID":[track_id1,track_id1,...]},...]
        Each element in the list is a dict of the displacements for a given tau
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

    #we can use the MSD_Tracks function from Analysis_functions.py even if it does other needless things
    disp_track_tau = af.MSD_Tracks(track_data,return_type="both")["displacements"]
    #this is in the form: trackID -> tau_for_track -> [total_disp,2] where 2 is just assuming 2D data
    #we need to convert this to the format required for NOBIAS but also perserve the non useable taus (>1)
    nobias_dict_out_Collection = []
    for i in range(1,max_tau+1):
        nobias_dict_out_Collection.append(_convert_track_data_NOBIAS_format_tau(disp_track_tau,i))

    return nobias_dict_out_Collection

#util function for convert_track_data_NOBIAS_format_global that formats the track data for a specific tau
def _convert_track_data_NOBIAS_format_tau(displacement_dict,tau,include_cor_obs:bool = False):
    ''' Docstring for _convert_track_data_NOBIAS_format_tau
    This is a utility function for convert_track_data_NOBIAS_format_global that formats the track data for a specific tau

    Parameters:
    -----------
    displacement_dict: dict
        dict of the form {track_id:{tau:[[x1-x0,x2-x1,...],[y1-y0,y2-y1,...]],...},...}
        This is the output from the MSD_Tracks function in Analysis_functions.py
    tau: int
        the tau to use for the NOBIAS analysis
    include_cor_obs: bool
        whether to include the correlation observations in the output

    Returns:
    --------
    nobias_dict_out: dict
        dict of the form {"obs":[[x1-x0,x2-x1,...],[y1-y0,y2-y1,...]],"TrID":[track_id1,track_id1,...]}
        "obs" is 2 x T where T is the number of frames and 2 is the dimension of the data (x,y)
        "TrID" is a list of track ids that correspond to the displacements in "obs"

    Notes:
    ------
    For the "cor_obs" key, i have no idea what the values for this are and as such i am leaving it out. But can be implemented later if needed
    '''
    nobias_dict_out = {
        "obs":[],
        "TrID":[],
        "cor_obs":[]
    }
    track_ID_counter = 1
    for track_id in displacement_dict.keys():
        if tau in displacement_dict[track_id].keys():
            len_disp = len(displacement_dict[track_id][tau])
            track_IDs = [track_ID_counter]*len_disp
            if len(nobias_dict_out["obs"]) == 0:
                nobias_dict_out["obs"] = np.array(displacement_dict[track_id][tau]).T
                nobias_dict_out["TrID"] = track_IDs
            else:
                nobias_dict_out["obs"] = np.concatenate((nobias_dict_out["obs"],np.array(displacement_dict[track_id][tau]).T),axis=1)
                nobias_dict_out["TrID"] = nobias_dict_out["TrID"] + track_IDs
        track_ID_counter += 1
    if include_cor_obs==False:
        nobias_dict_out.pop("cor_obs")
    nobias_dict_out["TrID"] = np.array(nobias_dict_out["TrID"])
    return nobias_dict_out


class IO_run_analysis:

	def __init__(self) -> None:
		pass

	@staticmethod
	def _save_sptanalysis_data(pp,test):
		fmt = '%d', '%d', '%1.9f', '%1.9f', '%d'
		np.savetxt(pp[:-4] + '_sptsanalysis.csv',test,delimiter = "\t",fmt=fmt)

	@staticmethod
	def _load_superSegger(cd,_string):
		xy_frame_dir_names = []
		#load the data of segmented cells from SuperSegger (cell files)
		for root, subdirs, files in os.walk(os.path.join(cd,_string)):
			for d in subdirs:
				if d[:2] == 'xy':
					xy_frame_dir_names.append(os.path.join(cd,_string,d))
		return xy_frame_dir_names
