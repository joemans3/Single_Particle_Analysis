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
from abc import ABC, abstractmethod
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts/src')
    import matplotlib.pyplot as plt
from SMT.SMT_Analysis_BP.helpers.analysisFunctions.scale_space_plus import SM_reconstruction_masked,MASK_VALUE,BOUNDING_BOX_PADDING,CONVERSION_TYPES,RANDOM_SEED
from SMT.SMT_Analysis_BP.helpers.clusterMethods.blob_detection import residuals_gaus2d
from SMT.SMT_Analysis_BP.helpers.clusterMethods.clustering_methods import perfrom_DBSCAN_Cluster,scale_space_plus_blob_detection

CORRECTION_FACTOR=1.

LOCALIZATION_UNIQUE_TYPE = "mean" #or "mean" is the other option

XY_NAMES = ['x','y']#['x [px]','y [px]']#

def load_localizations(localizations_path,skiprows=4):
    '''
    Load the localizations from the localizations.csv file
    '''
    colnames = ['track_ID','x','y','frame','intensity']
    df = pd.read_csv(localizations_path,usecols=(2,4,5,8,12),delimiter=',',skiprows=skiprows,names=colnames) #this can be changed depending on the format of the localizations.csv file
    return df
def load_localizations_TS(localizations_path,skiprows=0):
    '''
    Load the localizations from the localizations.csv file (ThunderSTORM format)
    '''
    columns = ['id','frame','x [nm]','y [nm]','sigma [nm]','intensity [photon]','offset [photon]','bkgstd [photon]','chi2','uncertainty [nm]','detections','y [px]','x [px]','sigma [px]']
    df = pd.read_csv(localizations_path,delimiter=',',names=columns,skiprows=skiprows) #this can be changed depending on the format of the localizations.csv file
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
def get_unique_localizations_TS(localizations_df:pd.DataFrame,unique_loc_type:str="first")->pd.DataFrame:
    '''
    For each unique track_ID get the first localization or the mean value of all the localizations

    Parameters:
    -----------
    localizations_df: pd.DataFrame
        The dataframe of the localizations
    unique_loc_type: str
        The type of unique localization to get. Can be either "first" or "mean"
    '''
    unique_localizations = localizations_df
    return unique_localizations
def rescale_scale_space_blob_detection(blobs,rescaling_func:callable,**kwargs):
    #get the fitted blobs
    fitted_blobs = blobs["Fitted"]
    #get the scale-space blobs
    scale_space_blobs = blobs["Scale"]
    type_of_convertion = kwargs.get("type_of_convertion")

    #now we need to rescale the fitted blobs using the rescaling function
    fitted_holder = np.zeros_like(fitted_blobs)
    scale_holder = np.zeros_like(scale_space_blobs)
    for i in range(len(fitted_blobs)):
        #get the radius
        radius_fitted = np.mean(fitted_blobs[i][2:4])
        #get the center
        center_fitted = fitted_blobs[i][0:2]
        #get the radius for scale
        radius_scale = np.mean(scale_space_blobs[i][2])
        #get the center for scale
        center_scale = scale_space_blobs[i][0:2]
        #rescale the fitted blobs
        #the function should take in the centers,radius, and a string which is supplied by the kwargs

        center_fitted_scaled,radius_fitted_scaled = rescaling_func(center_fitted,radius_fitted,type_of_convertion)
        center_scale_scaled,radius_scale_scaled = rescaling_func(center_scale,radius_scale,type_of_convertion)

        #now we need to put the scaled values into the holder
        fitted_holder[i][0:2] = center_fitted_scaled
        fitted_holder[i][2:4] = radius_fitted_scaled
        scale_holder[i][0:2] = center_scale_scaled
        scale_holder[i][2] = radius_scale_scaled
    
    #now we need to put the fitted blobs back into the blobs dictionary
    blobs["Fitted"] = fitted_holder
    blobs["Scale"] = scale_holder
    return blobs


class Reconstruct_Masked_PALM_DATASETS(ABC):
    def __init__(self,
                 cd:str,
                 blob_parameters:dict,
                 fitting_parameters:dict,
                 rescale_pixel_size = 10,
                 pixel_size = 130,
                 loc_error = 30,
                 include_all = True,
                 unique_localization_getter:callable = get_unique_localizations):

        self.cd = cd
        self.blob_parameters = blob_parameters
        self.fitting_parameters = fitting_parameters
        self.rescale_pixel_size = rescale_pixel_size
        self.pixel_size = pixel_size
        self.loc_error = loc_error
        self.include_all = include_all
        self._unique_localization_getter = unique_localization_getter
        self._check_directory_structure()
        self._store_parameters()   
    @abstractmethod
    def _load_localizations(self):
        raise NotImplementedError("This is an abstract method and needs to be implemented in the child class")
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
                localizations_df = self._load_localizations(localizations_path=localizations_path)
                #get the unique localizations if include_all is False
                if self.include_all == False:
                    unique_localizations = self._unique_localization_getter(localizations_df,unique_loc_type=LOCALIZATION_UNIQUE_TYPE)
                    localizations_df = unique_localizations
                #save the localizations as a csv file with molecule_loc.csv as the name with the last column being the area of the mask
                mask_img11 = plt.imread(mask_path)
                mask_img1 = np.copy(mask_img11)
                #find all pixels not 0 
                mask_img1[mask_img1!=0] = 1
                #get the area of the mask
                area = np.sum(mask_img1)
                #add the area to the localizations df
                localizations_df["area"] = area*((self.pixel_size/1000.)**2)
                localizations_df.to_csv(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["molecule_loc_path"],index=False)
                #get the dims of the mask image
                mask_img = plt.imread(mask_path)
                mask_img_dim = mask_img.shape
                #correct the x,y values of the localizations using the correction factor
                localizations_corrected = localizations_df[XY_NAMES].to_numpy()/CORRECTION_FACTOR
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
                
                print("#"*100)
                print("Reconstruction complete for {0}".format(path_structure_dict[movie_ID]["cells"][cell_ID]["localizations_path"]))
                print("#"*100)

                #now do the scale space blob detection
                print("#"*100)
                blobs = scale_space_plus_blob_detection(reconstruction,self.blob_parameters,self.fitting_parameters,show=True)


                #we need to rescale the blobs to the original image space
                blobs_rescaled = rescale_scale_space_blob_detection(blobs=blobs,
                                                                    rescaling_func=reconstruction_obj.coordinate_conversion,
                                                                    type_of_convertion='RC_to_Original')
                #find the number of localizations in each blob 
                blobs_rescaled["num_localizations"] = np.zeros(len(blobs_rescaled["Scale"]))
                blobs_rescaled["num_localizations_scale"] = np.zeros(len(blobs_rescaled["Scale"]))
                for i in range(len(blobs_rescaled["Scale"])):
                    blobs_rescaled["num_localizations"][i] = 0
                    for j in range(len(localizations_corrected)):
                        #check if it inside the circular blob
                        if np.sqrt((localizations_corrected[j][0]-blobs_rescaled["Fitted"][i][0])**2 + (localizations_corrected[j][1]-blobs_rescaled["Fitted"][i][1])**2) <= blobs_rescaled["Fitted"][i][2]:
                            blobs_rescaled["num_localizations"][i] += 1
                        if np.sqrt((localizations_corrected[j][0]-blobs_rescaled["Fitted"][i][0])**2 + (localizations_corrected[j][1]-blobs_rescaled["Fitted"][i][1])**2) <= blobs_rescaled["Scale"][i][2]:
                            blobs_rescaled["num_localizations_scale"][i] += 1
                #now we need to save the blobs
                #add the number of localizations to the blobs_rescaled dict
                #add the loc column to the blobs_rescaled dict for the fitted blobs and scale blobs
                blobs_rescaled["Fitted"] = np.hstack((blobs_rescaled["Fitted"],blobs_rescaled["num_localizations"].reshape(-1,1))) 
                blobs_rescaled["Scale"] = np.hstack((blobs_rescaled["Scale"],blobs_rescaled["num_localizations_scale"].reshape(-1,1)))
                np.savetxt(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["scale_space_plus_blob_fitted_path"],blobs_rescaled["Fitted"],delimiter=',')
                np.savetxt(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["scale_space_plus_blob_scale_path"],blobs_rescaled["Scale"],delimiter=',')
                print("#"*100)
                #now we need to do the DBSCAN clustering
                print("#"*100)
                try:
                    cluster_labels,cluster_centers,cluster_radii,loc_in_cluster = perfrom_DBSCAN_Cluster(localizations=localizations_corrected,
                                                                                        D=(self.loc_error)/self.pixel_size,
                                                                                        minP=5,
                                                                                        show=True)

                except:
                    cluster_labels = np.zeros(len(localizations_corrected))
                    cluster_centers = np.zeros((1,2))
                    cluster_radii = np.zeros(1)
                    loc_in_cluster = np.zeros(1)
                    print("DBSCAN clustering failed for {0}".format(path_structure_dict[movie_ID]["cells"][cell_ID]["localizations_path"]))
                print("#"*100)
                #now we need to save the DBSCAN clusters
                np.savetxt(path_structure_dict[movie_ID]["cells"][cell_ID]["Analysis"]["DBSCAN_clusters_path"],np.hstack((cluster_centers,cluster_radii.reshape(-1,1),loc_in_cluster.reshape(-1,1))),delimiter=',')
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
                    molecule_loc_path = os.path.join(cell_dir,"molecule_loc.csv")
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
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["Analysis"]["molecule_loc_path"] = molecule_loc_path
            self._path_structure_dict = structured_dict_paths
            return self._path_structure_dict
    @property
    def state_parameters(self):
        return self._state_parameters

class Reconstruct_Fixed_PALM_DATASETS(Reconstruct_Masked_PALM_DATASETS):
    def __init__(self, cd, blob_parameters, fitting_parameters, rescale_pixel_size=10, 
                 pixel_size=130, loc_error=30, include_all=True,
                 localization_loader:callable = load_localizations, unique_localization_getter:callable = get_unique_localizations):
        super().__init__(cd, blob_parameters, fitting_parameters, rescale_pixel_size, pixel_size, loc_error, include_all,unique_localization_getter)
        self._localization_loader = localization_loader
    def _load_localizations(self,**kwargs):
        #if skiprows is not in kwargs then set it to 4
        if "skiprows" not in kwargs.keys():
            kwargs["skiprows"] = 1
        df = self._localization_loader(**kwargs)
        return df

class Reconstruct_Tracked_PALM_DATASETS_with_mask(Reconstruct_Masked_PALM_DATASETS):
    '''
    For the masked version of the tracked PALM data, this is different from scale_space_plus_database_tracked.segmentation_scale_space \n
    since it inforces a cell mask for each cell in a movie and also expects a different file structure. (Cellpose mask or other)
    File structure is the same as the Fixed cell PALM data.
    '''
    def __init__(self, cd, blob_parameters, fitting_parameters, rescale_pixel_size=10, 
                 pixel_size=130, loc_error=30, include_all=True,
                 localization_loader:callable = load_localizations, unique_localization_getter:callable = get_unique_localizations):
        super().__init__(cd, blob_parameters, fitting_parameters, rescale_pixel_size, pixel_size, loc_error, include_all,unique_localization_getter)
        self._localization_loader = localization_loader
    def _load_localizations(self,**kwargs):
        #if skiprows is not in kwargs then set it to 1
        if "skiprows" not in kwargs.keys():
            kwargs["skiprows"] = 1
        df = self._localization_loader(**kwargs)
        return df
    def reconstructTime(self,subsample_frequency:int=500,total_frames:int=5000):
        '''
        For each cell we will now make a reconstruction (only, no scale and dbscan analysis) for subsampling
        '''
        #get the path structure dict
        path_structure_dict = self.path_structure_dict
        #print the welcome message
        self._print_message()
        #loop through the movies
        for movie_ID in path_structure_dict.keys():
            #loop through the cells
            for cell_ID in path_structure_dict[movie_ID]["cells"].keys():
                print("movie_ID: {0}, cell_ID: {1}".format(movie_ID,cell_ID))
                #get the localizations and mask paths
                localizations_path = path_structure_dict[movie_ID]["cells"][cell_ID]["localizations_path"]
                mask_path = path_structure_dict[movie_ID]["cells"][cell_ID]["mask_path"]
                #make a new directory for the subsampled recon. called subsampled_reconstruction
                subsampled_reconstruction_dir = os.path.join(path_structure_dict[movie_ID]["cells"][cell_ID]["path"],"subsampled_reconstruction_sampling_{0}".format(subsample_frequency))
                if not os.path.isdir(subsampled_reconstruction_dir):
                    os.mkdir(subsampled_reconstruction_dir)
                #load the localizations
                localizations_df = self._load_localizations(localizations_path=localizations_path)
                #get the unique localizations if include_all is False
                if self.include_all == False:
                    unique_localizations = self._unique_localization_getter(localizations_df,unique_loc_type=LOCALIZATION_UNIQUE_TYPE)
                    localizations_df = unique_localizations
                
                global_reconstruction = []
                global_uniform_reconstruction = []
                global_normal_scale_projection = []

                #now we need to subsample the localizations based on the subsample_frequency
                subsample_frames = np.arange(0,total_frames,subsample_frequency)
                for sampler in range(len(subsample_frames)):
                    #get the localizations in frames subsample_frames[i] to subsample_frames[i+1]
                    try:
                        localizations_subsampled = localizations_df[(localizations_df["frame"]>=subsample_frames[sampler]) & (localizations_df["frame"]<subsample_frames[sampler+1])]
                    except IndexError:
                        #this means that we are at the last frame
                        localizations_subsampled = localizations_df[(localizations_df["frame"]>=subsample_frames[sampler])]
                    #save the localizations as a csv file with molecule_loc.csv as the name 
                    localizations_subsampled.to_csv(os.path.join(subsampled_reconstruction_dir,"molecule_loc_{0}.csv".format(sampler)),index=False)
                    #get the dims of the mask image
                    mask_img = plt.imread(mask_path)
                    mask_img_dim = mask_img.shape
                    #correct the x,y values of the localizations using the correction factor
                    localizations_corrected = localizations_subsampled[XY_NAMES].to_numpy()/CORRECTION_FACTOR
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
                    #add the reconstruction to the global reconstruction
                    global_reconstruction.append(reconstruction)
                    global_uniform_reconstruction.append(uniform_reconstruction)
                    global_normal_scale_projection.append(normal_scale_projection)
                
                #now save as a .tif file the global_reconstruction, global_uniform_reconstruction, global_normal_scale_projection
                global_reconstruction = np.array(global_reconstruction)
                global_uniform_reconstruction = np.array(global_uniform_reconstruction)
                global_normal_scale_projection = np.array(global_normal_scale_projection)
                #save the reconstruction, uniform reconstruction and normal scale projection
                reconstruction_obj.saving_image(img=global_reconstruction,
                                                full_path=os.path.join(subsampled_reconstruction_dir,"reconstruction.tif"))
                reconstruction_obj.saving_image(img=global_uniform_reconstruction,
                                                full_path=os.path.join(subsampled_reconstruction_dir,"uniform_reconstruction.tif"))
                reconstruction_obj.saving_image(img=global_normal_scale_projection,
                                                full_path=os.path.join(subsampled_reconstruction_dir,"normal_scale_projection.tif"))
                print("#"*100)
                print("Reconstruction complete for {0}".format(path_structure_dict[movie_ID]["cells"][cell_ID]["localizations_path"]))
                print("#"*100)
                    



        
    


####testing

if __name__ == '__main__':
    global_path = [
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_rif_fixed_2"
    ]





    blob_parameters = {
        "threshold": 2e-1, \
        "overlap": 0, \
        "median": False, \
        "min_sigma": 3/np.sqrt(2), \
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
        "centroid_range":2,
        "height_range":1
        }

    for i in range(len(global_path)):
        # sm_rec = Reconstruct_Tracked_PALM_DATASETS_with_mask(cd = global_path[i],
        #                                         blob_parameters = blob_parameters,
        #                                         fitting_parameters = fitting_parameters,
        #                                         rescale_pixel_size = 10,
        #                                         pixel_size = 130,
        #                                         loc_error = 30,
        #                                         include_all = True)
        #                                         #localization_loader = load_localizations_TS,
        #                                         #unique_localization_getter = get_unique_localizations_TS)
        # sm_rec.reconstructTime(subsample_frequency=1000,total_frames=5000)
        sm_rec = Reconstruct_Fixed_PALM_DATASETS(cd = global_path[i],
                                        blob_parameters = blob_parameters,
                                        fitting_parameters = fitting_parameters,
                                        rescale_pixel_size = 10,
                                        pixel_size = 130,
                                        loc_error = 20,
                                        include_all = False)
                                        #localization_loader = load_localizations_TS,
                                        #unique_localization_getter = get_unique_localizations_TS)
        sm_rec.reconstruct()
