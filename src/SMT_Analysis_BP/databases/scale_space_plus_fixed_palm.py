'''
Helper script to do the reconstruction and scale space blob detection along with the DBSCAN clustering on fixed cell PALM data.
This is different from scale_space_plus_database_tracked.py since it inforces a cell mask for each cell in a movie and also ixpects a different file structure.


############################################################################################################
File structure:
---------------
The file structure is as follows:

<main_path>
    /Movies
        /Movie_<01>
            /Cell_<cell_number>
                /localizations.csv
                /mask.tif

This is the base file structure for the PALM data. The script will look for the localizations and mask files in the above structure.

The localizations file needs to be in the form of TRACKMATE output from the GUI. TODO make it more general.

The objective of this script is to add the following files to the above structure: (new files and directories have a # next to them)

<main_path>
    /Movies
        /Movie_<01>
            /Cell_<cell_number>
                /localizations.csv
                /mask.tif
                /reconstruction.tif #
                /uniform_reconstruction.tif #
                /normal_scale_projection.tif #
                /reconstruction_parameters.json #
                    /Analysis #
                        /scale_space_plus_blob_fitted.csv #
                        /scale_space_plus_blob_scale.csv #
                        /DBSCAN_clusters.csv #
                        /analysis_parameters.json #

'''


from typing import Any
import numpy as np
import pandas as pd
import os
import json
import glob
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
    import matplotlib.pyplot as plt
from src.SMT_Analysis_BP.helpers.scale_space_plus import SM_reconstruction_masked,MASK_VALUE,BOUNDING_BOX_PADDING,CONVERSION_TYPES,RANDOM_SEED
from src.SMT_Analysis_BP.helpers.blob_detection import residuals_gaus2d
from src.SMT_Analysis_BP.helpers.clustering_methods import perfrom_DBSCAN_Cluster,scale_space_plus_blob_detection,rescale_scale_space_blob_detection

CORRECTION_FACTOR=1.

LOCALIZATION_UNIQUE_TYPE = "first" #or "mean" is the other option



class Reconstruct_Live_PALM_DATASETS:
    def __init__(self,
                 cd,
                 blob_parameters,
                 fitting_parameters,
                 rescale_pixel_size = 10,
                 pixel_size = 130,
                 loc_error = 30,
                 include_all = True
                 ):

        self.cd = cd
        self.blob_parameters = blob_parameters
        self.fitting_parameters = fitting_parameters
        self.rescale_pixel_size = rescale_pixel_size
        self.pixel_size = pixel_size
        self.loc_error = loc_error
        self.include_all = include_all
        self._check_directory_structure()
        self._store_parameters()


    def _print_message(self):
        #print a welcome message with the parameters
        message = '''
        Welcome to the reconstruction script for PALM data. The parameters are:
        rescale_pixel_size: {0} nm
        pixel_size: {1} nm
        loc_error: {2} nm
        include_all: {3}
        '''.format(self.rescale_pixel_size,self.pixel_size,self.loc_error,self.include_all)
        #print the stored parameters in a pretty way since they are a dict. Dont include the path_structure_dict
        new_state_parameters = self.state_parameters.copy()
        new_state_parameters.pop("path_structure_dict")
        second_message = json.dumps(new_state_parameters,indent=4)
        print("#"*100)
        print(message)
        print("#"*100)
        print("Full parameters: ")
        print(second_message)
        print("#"*100)
        print("Starting reconstruction and analysis... \n Ill see you soon bby")
        #print the _save_reconstruction_parameters
    def reconstruct(self):
        '''
        Reconstruct the localizations in each cell in each movie
        '''
        #get the path structure dict
        path_structure_dict = self.path_structure_dict
        #print the welcome message
        self._print_message()
        #loop through the movies
        for movie_ID in path_structure_dict.keys():
            #loop through the cells
            for cell_ID in path_structure_dict[movie_ID]["cells"].keys():
                #get the localizations and mask paths
                localizations_path = path_structure_dict[movie_ID]["cells"][cell_ID]["localizations_path"]
                mask_path = path_structure_dict[movie_ID]["cells"][cell_ID]["mask_path"]
                #load the localizations
                localizations_df = load_localizations(localizations_path)
                #get the unique localizations if include_all is False
                if self.include_all == False:
                    unique_localizations = get_unique_localizations(localizations_df,unique_loc_type=LOCALIZATION_UNIQUE_TYPE)
                    localizations_df = unique_localizations
                #get the dims of the mask image
                mask_img = plt.imread(mask_path)
                mask_img_dim = mask_img.shape
                #correct the x,y values of the localizations using the correction factor
                localizations_corrected = localizations_df[['x','y']].to_numpy()/CORRECTION_FACTOR
                #make localization errors in the same length as the localizations
                loc_error_arr = np.ones(len(localizations_corrected))*self.loc_error

                #create the reconstruction object
                reconstruction_obj = SM_reconstruction_masked(mask_img_dim,self.pixel_size,self.rescale_pixel_size)
                #reconstruct space 
                reconstruction = reconstruction_obj.make_reconstruction(localizations=localizations_corrected,
                                                                        localization_error=loc_error_arr,
                                                                        masked_img=mask_img)
                #make the uniform reconstruction
                uniform_reconstruction = reconstruction_obj.make_uniform_reconstruction(localizations=localizations_corrected,
                                                                                        localization_error=loc_error_arr,
                                                                                        masked_img=mask_img)
                #now make a new object but with the original pixel size as the rescale pixel size 
                reconstruction_normal_scale_obj = SM_reconstruction_masked(mask_img_dim,self.pixel_size,self.pixel_size)
                #make the normal scale projection
                normal_scale_projection = reconstruction_normal_scale_obj.make_reconstruction(localizations=localizations_corrected,
                                                                                                        localization_error=self.pixel_size,
                                                                                                        masked_img=mask_img)
                #save the reconstruction, uniform reconstruction and normal scale projection
                reconstruction_obj.saving_image(img=reconstruction,
                                                full_path=path_structure_dict[movie_ID]["cells"][cell_ID]["reconstruction_path"])
                reconstruction_obj.saving_image(img=uniform_reconstruction,
                                                full_path=path_structure_dict[movie_ID]["cells"][cell_ID]["uniform_reconstruction_path"])
                reconstruction_obj.saving_image(img=normal_scale_projection,
                                                full_path=path_structure_dict[movie_ID]["cells"][cell_ID]["normal_scale_projection_path"])
                
                #now do the scale space blob detection
                print("#"*100)
                blobs = scale_space_plus_blob_detection(reconstruction,self.blob_parameters,self.fitting_parameters,show=True)
                #we need to rescale the blobs to the original image space
                blobs_rescaled = rescale_scale_space_blob_detection(blobs=blobs,
                                                                    rescaling_func=reconstruction_obj.coordinate_conversion,
                                                                    type_of_convertion='RC_to_Original')
                #now we need to save the blobs
                np.savetxt(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["scale_space_plus_blob_fitted_path"],blobs_rescaled["Fitted"],delimiter=',')
                np.savetxt(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["scale_space_plus_blob_scale_path"],blobs_rescaled["Scale"],delimiter=',')
                print("#"*100)
                #now we need to do the DBSCAN clustering
                print("#"*100)
                try:
                    cluster_labels,cluster_centers,cluster_radii = perfrom_DBSCAN_Cluster(localizations=localizations_corrected,
                                                                                        D=2*self.loc_error/self.pixel_size,
                                                                                        minP=5,
                                                                                        show=True)
                except:
                    cluster_labels = np.zeros(len(localizations_corrected))
                    cluster_centers = np.zeros((1,2))
                    cluster_radii = np.zeros(1)
                    print("DBSCAN clustering failed for {0}".format(path_structure_dict[movie_ID]["cells"][cell_ID]["localizations_path"]))
                print("#"*100)
                #now we need to save the DBSCAN clusters
                np.savetxt(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["DBSCAN_clusters_path"],np.hstack((cluster_centers,cluster_radii.reshape(-1,1))),delimiter=',')
                #now we need to save the analysis parameters(this is the same as the reconstruction parameters)
                self._save_reconstruction_parameters(full_path=path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["analysis_parameters_path"])
                #save the reconstruction parameters
                self._save_reconstruction_parameters(full_path=path_structure_dict[movie_ID]["cells"][cell_ID]["reconstruction_parameters_path"])
        print("#"*100)
        print("Reconstruction and analysis complete. Come again soon! \n ill be waiting for you (ill be sad until then)! :) ")
        print("#"*100)
                

    def _store_parameters(self):
        try:
            fitting_parameters = self.fitting_parameters.copy()
            fitting_parameters["radius_func"] = fitting_parameters["radius_func"].__name__
            fitting_parameters["residual_func"] = fitting_parameters["residual_func"].__name__
        except:
            print("something broke buddy")
            pass

        params = {
            "rescale_pixel_size":self.rescale_pixel_size,
            "pixel_size":self.pixel_size,
            "loc_error":self.loc_error,
            "include_all":self.include_all,
            "CORRECTION_FACTOR":CORRECTION_FACTOR,
            "LOCALIZATION_UNIQUE_TYPE":LOCALIZATION_UNIQUE_TYPE,
            "MASK_VALUE":MASK_VALUE,
            "BOUNDING_BOX_PADDING":BOUNDING_BOX_PADDING,
            "RANDOM_SEED":RANDOM_SEED,
            "blob_parameters":blob_parameters,
            "fitting_parameters":fitting_parameters,
            "path_structure_dict":self.path_structure_dict
        }
        self._state_parameters = params

    
    def _save_reconstruction_parameters(self,full_path:str)->None:

        params = self.state_parameters

        with open(full_path,"w") as f:
            json.dump(params,f,indent=4)
    
    def _check_directory_structure(self)->None:
        #check if the directory structure is correct for this analysis
        #check if cd exists
        if not os.path.isdir(self.cd):
            raise ValueError("The directory {0} does not exist".format(self.cd))
        #check if the Movies directory exists
        if not os.path.isdir(os.path.join(self.cd,"Movies")):
            raise ValueError("The directory {0} does not exist".format(os.path.join(self.cd,"Movies")))
        #check if inside the Movies directory there are directories that start with Movie_
        movie_dirs = glob.glob(os.path.join(self.cd,"Movies","Movie_*"))
        if len(movie_dirs) == 0:
            raise ValueError("There are no directories that start with Movie_ in {0}".format(os.path.join(self.cd,"Movies")))
        #check if inside the Movie_ directories there are directories that start with Cell_
        for movie_dir in movie_dirs:
            cell_dirs = glob.glob(os.path.join(movie_dir,"Cell_*"))
            if len(cell_dirs) == 0:
                raise ValueError("There are no directories that start with Cell_ in {0}".format(movie_dir))
            #check if inside the Movie_<number>_cell_ directories there are localizations and mask files
            for cell_dir in cell_dirs:
                if len(glob.glob(os.path.join(cell_dir,"localizations.csv"))) == 0:
                    raise ValueError("There are no localizations.csv files in {0}".format(cell_dir))
                if len(glob.glob(os.path.join(cell_dir,"mask.tif"))) == 0:
                    raise ValueError("There are no mask.tif files in {0}".format(cell_dir))
        if self._check_if_previous_analysis_has_occured():
            printing_warning ='''WARNING: It seems like the analysis has already occured. \n This is now overwriting the previous analysis.'''
            print("#"*100)
            print(printing_warning)
            print("#"*100)
    def _check_if_previous_analysis_has_occured(self)->None:
        #check if analysis has already occured by checking if the reconstruction_parameters.json file exists
        #get the path structure dict
        path_structure_dict = self.path_structure_dict
        #loop through the movies
        truth = 0
        for movie_ID in path_structure_dict.keys():
            #loop through the cells
            for cell_ID in path_structure_dict[movie_ID]["cells"].keys():
                #get the reconstruction parameters path
                reconstruction_parameters_path = path_structure_dict[movie_ID]["cells"][cell_ID]["reconstruction_parameters_path"]
                #check if the file exists
                if os.path.isfile(reconstruction_parameters_path):
                    truth += 1
                else:
                    return False
        if truth>0:
            return True
    @property
    def path_structure_dict(self):
        if hasattr(self,"_path_structure_dict"):
            return self._path_structure_dict
        else:
            structured_dict_paths = {}
            # for each key is the movie ID which has a dict of cell IDs, and path (of the movie ID) as the value
            #for each cell ID there is a dict of the localizations and mask paths

            #get the movie directories
            movie_dirs = glob.glob(os.path.join(self.cd,"Movies","Movie_*"))
            for movie_dir in movie_dirs:
                #get the movie ID as the * in Movie_*
                movie_ID = os.path.basename(movie_dir).split("_")[1]
                #get the cell directories
                cell_dirs = glob.glob(os.path.join(movie_dir,"Cell_*"))
                #initialize the dict for the movie ID
                structured_dict_paths[movie_ID] = {"path":movie_dir,"cells":{}}
                for cell_dir in cell_dirs:
                    #get the cell ID as the * in Movie_<number>_cell_*
                    cell_ID = os.path.basename(cell_dir).split("_")[-1]
                    #initialize the dict for the cell ID
                    structured_dict_paths[movie_ID]["cells"][cell_ID] = {"path":cell_dir}
                    #get the localizations and mask paths
                    localizations_path = glob.glob(os.path.join(cell_dir,"localizations.csv"))[0]
                    mask_path = glob.glob(os.path.join(cell_dir,"mask.tif"))[0]
                    #add the localizations and mask paths to the dict
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["localizations_path"] = localizations_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["mask_path"] = mask_path
                    
                    #now make the new directories and file paths (even though we will not save the files yet)
                    #make the Analysis directory
                    analysis_dir = os.path.join(cell_dir,"Analysis")
                    if not os.path.isdir(analysis_dir):
                        os.mkdir(analysis_dir)
                    #make the new files as shown in the docstring
                    reconstruction_path = os.path.join(cell_dir,"reconstruction.tif")
                    uniform_reconstruction_path = os.path.join(cell_dir,"uniform_reconstruction.tif")
                    normal_scale_projection_path = os.path.join(cell_dir,"normal_scale_projection.tif")
                    reconstruction_parameters_path = os.path.join(cell_dir,"reconstruction_parameters.json")
                    scale_space_plus_blob_fitted_path = os.path.join(analysis_dir,"scale_space_plus_blob_fitted.csv")
                    scale_space_plus_blob_scale_path = os.path.join(analysis_dir,"scale_space_plus_blob_scale.csv")
                    DBSCAN_clusters_path = os.path.join(analysis_dir,"DBSCAN_clusters.csv")
                    analysis_parameters_path = os.path.join(analysis_dir,"analysis_parameters.json")
                    #add the new file paths to the dict in the proper place with the name as the key and the full path as the value
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["reconstruction_path"] = reconstruction_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["uniform_reconstruction_path"] = uniform_reconstruction_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["normal_scale_projection_path"] = normal_scale_projection_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["reconstruction_parameters_path"] = reconstruction_parameters_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["Analysis"] = {"scale_space_plus_blob_fitted_path":scale_space_plus_blob_fitted_path,
                                                                                    "scale_space_plus_blob_scale_path":scale_space_plus_blob_scale_path,
                                                                                    "DBSCAN_clusters_path":DBSCAN_clusters_path,
                                                                                    "analysis_parameters_path":analysis_parameters_path}
            self._path_structure_dict = structured_dict_paths
            return self._path_structure_dict
    @property
    def state_parameters(self):
        return self._state_parameters
def load_localizations(localizations_path):
    '''
    Load the localizations from the localizations.csv file
    '''
    colnames = ['track_ID','x','y','frame','intensity']
    df = pd.read_csv(localizations_path,usecols=(2,4,5,8,12),delimiter=',',skiprows=4,names=colnames) #this can be changed depending on the format of the localizations.csv file
    return df

def get_unique_localizations(localizations_df:pd.DataFrame,unique_loc_type:str="first")->pd.DataFrame:
    '''
    For each unique track_ID get the first localization or the mean value of all the localizations

    Parameters:
    -----------
    localizations_df: pd.DataFrame
        The dataframe of the localizations
    unique_loc_type: str
        The type of unique localization to get. Can be either "first" or "mean"
    '''
    if unique_loc_type == "first":
        unique_localizations = localizations_df.groupby("track_ID").first()
    elif unique_loc_type == "mean":
        unique_localizations = localizations_df.groupby("track_ID").mean()
    else:
        raise ValueError("The unique_loc_type can be either first or mean")
    return unique_localizations















####testing

if __name__ == '__main__':
    global_path = '/Users/baljyot/Documents/SMT_Movies/testing_SM_recon'



    blob_parameters = {
        "threshold": 3e-2, \
        "overlap": 0, \
        "median": False, \
        "min_sigma": 4/np.sqrt(2), \
        "max_sigma": 20/np.sqrt(2), \
        "num_sigma": 30, \
        "detection": 'bp', \
        "log_scale": False
        }
    fitting_parameters = {
        "mask_size":5,
        "plot_fit":False,
        "fitting_image":"LAP",
        "radius_func":np.mean,#identity,
        "residual_func":residuals_gaus2d,
        "sigma_range":4,
        "centroid_range":3,
        "height_range":1
        }

    sm_rec = Reconstruct_Live_PALM_DATASETS(cd = global_path,
                                            blob_parameters = blob_parameters,
                                            fitting_parameters = fitting_parameters,
                                            rescale_pixel_size = 10,
                                            pixel_size = 130,
                                            loc_error = 30,
                                            include_all = False
                                            )
    sm_rec.reconstruct()