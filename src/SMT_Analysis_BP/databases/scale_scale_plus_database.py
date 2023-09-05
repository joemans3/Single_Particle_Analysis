import numpy as np
import pandas as pd
import os
import json
import glob
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
    import matplotlib.pyplot as plt
from src.SMT_Analysis_BP.helpers.scale_space_plus import SM_reconstruction_image
from src.SMT_Analysis_BP.helpers.blob_detection import blob_detection,residuals_gaus2d
from src.SMT_Analysis_BP.helpers.Analysis_functions import reshape_col2d
#import DBSCAN
from sklearn.cluster import DBSCAN
#import convex hull
from scipy.spatial import ConvexHull
CORRECTION_FACTOR=1.
'''
Helper script to create a new folder in the directory with the name segmented_scale_space_plus
This assumes an Analysis or Analysis_new folder exists in the directory that contains the raw SMT data for each movie

The structure of the folder is as follows:

cd
|---Analysis
|   |---t_string_i_seg.tif_spots.csv
|   |---...
|---Analysis_new
|   |---t_string_i_seg.tif_spots.csv
|   |---...

Here t_string is the unique identifier of the dataset and it the same for all movies in the dataset. 
i is the movie number and it is unique for each movie in the dataset.

The script will create a new folder in the directory called segmented_scale_space_plus and add in the following structure:

cd
|---segmented_scale_space_plus
|   |---j_t_string_i_seg.tif
|   |---...
|   |---params.json
|   |---Analysis
|   |   |---j_t_string_i_seg.tif_spots.csv
|   |   |---...

Here j is the segmentation number (5000 frame total movie segmented 5 ways is j=1,2,3,4,5)
t_string is the unique identifier of the dataset and it the same for all movies in the dataset.
i is the movie number and it is unique for each movie in the dataset.
params.json is a json file that contains the parameters used to create the segmentation and reconstruction and the detection parameters
The Analysis folder inside the segmented_scale_space_plus folder contains the identified spots for each movie in the dataset and segmentation number


EX: for a dataset with 2 movies and 5000 frames each segmented 5 ways, the folder structure will be:

cd
|---Analysis
|   |---t_string_1_seg.tif_spots.csv
|   |---t_string_2_seg.tif_spots.csv
|   |---...
|---Analysis_new
|   |---t_string_1_seg.tif_spots.csv
|   |---t_string_2_seg.tif_spots.csv
|   |---...
|---segmented_scale_space_plus
|   |---1_t_string_1_seg.tif
|   |---1_t_string_2_seg.tif
|   |---2_t_string_1_seg.tif
|   |---2_t_string_2_seg.tif
|   |---3_t_string_1_seg.tif
|   |---3_t_string_2_seg.tif
|   |---4_t_string_1_seg.tif
|   |---4_t_string_2_seg.tif
|   |---5_t_string_1_seg.tif
|   |---5_t_string_2_seg.tif
|   |---params.json
|   |---Analysis
|   |   |---1_t_string_1_seg.tif_spots.csv
|   |   |---1_t_string_2_seg.tif_spots.csv
|   |   |---2_t_string_1_seg.tif_spots.csv
|   |   |---2_t_string_2_seg.tif_spots.csv
|   |   |---3_t_string_1_seg.tif_spots.csv
|   |   |---3_t_string_2_seg.tif_spots.csv
|   |   |---4_t_string_1_seg.tif_spots.csv
|   |   |---4_t_string_2_seg.tif_spots.csv
|   |   |---5_t_string_1_seg.tif_spots.csv
|   |   |---5_t_string_2_seg.tif_spots.csv
'''

class segmentation_scale_space:
    def __init__(self,cd,t_string,blob_parameters,fitting_parameters,img_dim,rescale_pixel_size = 10,type_analysis_file = "new",total_frames = 5000,subframes = 5,pixel_size = 130,loc_error = 30,include_all = True):

        self.cd = cd
        self.t_string = t_string
        self.blob_parameters = blob_parameters
        self.fitting_parameters = fitting_parameters
        self.img_dim = img_dim
        self.rescale_pixel_size = rescale_pixel_size
        self.type_analysis_file = type_analysis_file
        self.total_frames = total_frames
        self.subframes = subframes
        self.pixel_size = pixel_size
        self.loc_error = loc_error
        self.include_all = include_all
        
    def main_run(self):
        #get the self.SMT_Analysis_path
        SMT_Analysis_path = self.SMT_Analysis_path
        #get the self.SM_reconstruction_Analysis_path
        SM_reconstruction_Analysis_path = self.SM_reconstruction_Analysis_path
        #get the self.segmented_scale_space_plus_path
        segmented_scale_space_plus_path = self.segmented_scale_space_plus_path

        #get the names of all the movie SMT data files
        SMT_data = glob.glob(os.path.join(SMT_Analysis_path,'*_seg.tif_spots.csv'))
        #for each movie SMT data file load the data
        for i in range(len(SMT_data)):
            #find the string between self.t_string_ and _seg.tif_spots.csv
            movie_ID = SMT_data[i].split('_seg.tif_spots.csv')[0].split(self.t_string+'_')[1]
            print(movie_ID,SMT_data[i])
            #load the data
            if self.type_analysis_file == "new":
                #the column names are not correct so rename them
                #track_ID, x,y,frame,intensity
                colnames = ['track_ID','x','y','frame','intensity']
                df = pd.read_csv(SMT_data[i],usecols=(2,4,5,8,12),delimiter=',',skiprows=4,names=colnames)
            elif self.type_analysis_file == "old":
                #the column names are not correct so rename them
                #track_ID, frame, x,y,intensity
                colnames = ['track_ID','frame','x','y','intensity']
                df = pd.read_csv(SMT_data[i],delimiter=',',names=colnames)

            for j in range(1,self.subframes+1):
                #get the segmentation number
                seg_num = str(j)
                #get the name of the file
                file_name = seg_num+'_'+self.t_string+'_'+movie_ID+'_seg'
                print(file_name)
                #get the frame numbers
                frame_numbers_lower = (j-1)*self.total_frames/self.subframes
                frame_numbers_upper = j*self.total_frames/self.subframes
                #get the data for the frame numbers
                df_sub = df[(df['frame'] >= frame_numbers_lower) & (df['frame'] < frame_numbers_upper)]

                if self.include_all == False:
                    #we need to only take the first x,y for each unique track_ID
                    #get the unique track_ID
                    unique_track_ID = df_sub['track_ID'].unique()
                    #get the first x,y for each unique track_ID
                    df_sub = df_sub[df_sub['track_ID'].isin(unique_track_ID)]
                    #get the first x,y for each unique track_ID
                    df_sub = df_sub.groupby('track_ID').head(1)
                

                #get the localizations
                localizations = df_sub[['x','y']].to_numpy()/CORRECTION_FACTOR
                #get the localizations error
                localization_error = np.ones(len(localizations))*self.loc_error
                #create the reconstruction image
                img = SM_reconstruction_image(self.img_dim,self.pixel_size,self.rescale_pixel_size)
                img_space = img.make_reconstruction(localizations,localization_error)
                #save the image
                img.saving_image(segmented_scale_space_plus_path,file_name,'tif')
                #save the data
                #df_sub.to_csv(os.path.join(SM_reconstruction_Analysis_path,file_name+'_spots.csv'),index=False)
                blobs = scale_space_plus_blob_detection(img_space,self.blob_parameters,self.fitting_parameters)
                #we need to rescale the blobs to the original image space
                blobs["Fitted"] = blobs["Fitted"]/(self.pixel_size/img.rescale_pixel_size)
                blobs["Scale"] = blobs["Scale"]/(self.pixel_size/img.rescale_pixel_size)
                #save the data in the Analysis folder
                np.savetxt(os.path.join(SM_reconstruction_Analysis_path,file_name+'.tif_spots.csv'),blobs["Scale"],delimiter=',')

                #perform DBSCAN clustering on the localizations
                try:
                    cluster_labels,cluster_centers,cluster_radii = perfrom_DBSCAN_Cluster(localizations,D=2*self.loc_error/self.pixel_size,minP=5)#self.loc_error/self.pixel_size,minP=5)
                except:
                    cluster_labels = np.zeros(len(localizations))
                    cluster_centers = np.zeros((1,2))
                    cluster_radii = np.zeros(1)
                    print('DBSCAN failed for '+file_name)
                print(cluster_radii)
                #save the data in the Analysis_DBSCAN folder
                np.savetxt(os.path.join(self.SM_DBSCAN_Analysis_Path,file_name+'.tif_spots.csv'),np.hstack((cluster_centers,cluster_radii.reshape(-1,1))),delimiter=',')
                #lets visualize the clusters
                fig,ax = plt.subplots()
                ax.imshow(img_space,cmap='gray')
                #plot the localizations
                ax.scatter(localizations[:,0]*13,localizations[:,1]*13,s=1,c='b')

                #make a circle with the radius of the blob
                for i in range(len(cluster_centers)):
                    #get the radius
                    radius = cluster_radii[i]
                    #get the center
                    center = cluster_centers[i]
                    #get the circle
                    circle = plt.Circle(center,radius,color='r',fill=False)
                    #add the circle to the axis
                    ax.add_patch(circle)
                plt.show()

                
        #save the parameters
        self._save_parameters()
        return

    def _save_parameters(self):
        #get the collection of all parameters used
        params = {
            't_string':self.t_string,
            'blob_parameters':self.blob_parameters,
            'img_dim':self.img_dim,
            'type_analysis_file':self.type_analysis_file,
            'total_frames':self.total_frames,
            'subframes':self.subframes,
            'pixel_size':self.pixel_size,
            'loc_error':self.loc_error,
            'include_all':self.include_all,
            'SMT_Analysis_path':self.SMT_Analysis_path,
            'segmented_scale_space_plus_path':self.segmented_scale_space_plus_path,
            'SM_reconstruction_Analysis_path':self.SM_reconstruction_Analysis_path
        }
        #get the file path
        file_path = os.path.join(self.segmented_scale_space_plus_path,'params.json')
        #save the parameters
        with open(file_path,'w') as f:
            json.dump(params,f)
        return






    @property
    def SMT_Analysis_path(self):
        #check if the SMT_Analysis_path is set
        if not hasattr(self,'_SMT_Analysis_path'):
            #check if the cd is a directory
            if not os.path.isdir(self.cd):
                raise ValueError('The cd is not a directory')
            #check if the Analysis folder exists
            if self.type_analysis_file == "new":
                if not os.path.exists(os.path.join(self.cd,'Analysis_new')):
                    raise ValueError('The Analysis_new folder does not exist in the directory')
                #set the SMT_Analysis_path
                self._SMT_Analysis_path = os.path.join(self.cd,'Analysis_new')
            elif self.type_analysis_file == "old":
                if not os.path.exists(os.path.join(self.cd,'Analysis')):
                    raise ValueError('The Analysis folder does not exist in the directory')
                #set the SMT_Analysis_path
                self._SMT_Analysis_path = os.path.join(self.cd,'Analysis')
        return self._SMT_Analysis_path
    @property
    def segmented_scale_space_plus_path(self):
        #check if the segmented_scale_space_plus_path is set
        if not hasattr(self,'_segmented_scale_space_plus_path'):
            #check if the cd is a directory
            if not os.path.isdir(self.cd):
                raise ValueError('The cd is not a directory')
            #check if the segmented_scale_space_plus folder exists
            if not os.path.exists(os.path.join(self.cd,'segmented_scale_space_plus')):
                #make the folder
                os.makedirs(os.path.join(self.cd,'segmented_scale_space_plus'))
            #set the segmented_scale_space_plus_path
            self._segmented_scale_space_plus_path = os.path.join(self.cd,'segmented_scale_space_plus')
        return self._segmented_scale_space_plus_path
    @property
    def SM_reconstruction_Analysis_path(self):
        #check if the SM_reconstruction_Analysis_path is set
        if not hasattr(self,'_SM_reconstruction_Analysis_path'):
            #get the self.segmented_scale_space_plus_path
            segmented_scale_space_plus_path = self.segmented_scale_space_plus_path
            #check if the Analysis folder exists
            if not os.path.exists(os.path.join(segmented_scale_space_plus_path,'Analysis')):
                #make the folder
                os.makedirs(os.path.join(segmented_scale_space_plus_path,'Analysis'))
            #set the SM_reconstruction_Analysis_path
            self._SM_reconstruction_Analysis_path = os.path.join(segmented_scale_space_plus_path,'Analysis')
        return self._SM_reconstruction_Analysis_path
    @property
    def SM_DBSCAN_Analysis_Path(self):
        #check if the SM_DBSCAN_Analysis_Path is set
        if not hasattr(self,'_SM_DBSCAN_Analysis_Path'):
            #get the self.segmented_scale_space_plus_path
            segmented_scale_space_plus_path = self.segmented_scale_space_plus_path
            #check if the Analysis folder exists
            if not os.path.exists(os.path.join(segmented_scale_space_plus_path,'Analysis_DBSCAN')):
                #make the folder
                os.makedirs(os.path.join(segmented_scale_space_plus_path,'Analysis_DBSCAN'))
            #set the SM_DBSCAN_Analysis_Path
            self._SM_DBSCAN_Analysis_Path = os.path.join(segmented_scale_space_plus_path,'Analysis_DBSCAN')
        return self._SM_DBSCAN_Analysis_Path
    
def scale_space_plus_blob_detection(img,blob_parameters,fitting_parameters):

    blob_class = blob_detection(
        img,
        threshold = blob_parameters.get("threshold",1e-4),
        overlap = blob_parameters.get("overlap",0.5),
        median=blob_parameters.get("median",False),
        min_sigma=blob_parameters.get("min_sigma",1),
        max_sigma=blob_parameters.get("max_sigma",2),
        num_sigma=blob_parameters.get("num_sigma",500),
        logscale=blob_parameters.get("log_scale",False),
        verbose=True)
    blob_class._update_fitting_parameters(kwargs=fitting_parameters)
    blob = blob_class.detection(type = blob_parameters.get("detection",'bp'))
    fitted = blob["Fitted"]
    scale = blob["Scale"]
    blob["Fitted"] = reshape_col2d(fitted,[1,0,2,3])
    blob["Scale"] = reshape_col2d(scale,[1,0,2])
    blobs = blob
    fig,ax = plt.subplots()
    ax.imshow(img,cmap='gray')
    #make a circle with the radius of the blob
    for i in range(len(blobs["Fitted"])):

        #get the radius
        radius = fitting_parameters["radius_func"](blobs["Fitted"][i][2:4])
        #get the center
        center = blobs["Fitted"][i][0:2]
        #get the circle
        circle = plt.Circle(center,radius,color='r',fill=False)
        #add the circle to the axis
        ax.add_patch(circle)
    plt.show()
    print(blobs["Fitted"])
    print(blobs["Scale"])

    return blobs

def perfrom_DBSCAN_Cluster(localizations,D,minP):
    '''
    Parameters:
    -----------
    localizations: np.ndarray
        Numpy array of the localizations in the form [[x,y],...]
    D: float, in the units of the localizations
        The maximum distance between two samples for one to be considered as in the neighborhood of the other
    minP: int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    --------
    cluster_labels: np.ndarray
        Numpy array of the cluster labels in the form [0,0,1,1,2,2,...]
    cluster_centers: np.ndarray
        Numpy array of the cluster centers in the form [[x,y],...]
    cluster_radii: np.ndarray
        Numpy array of the cluster radii in the form [r1,r2,...]
    '''
    #get the DBSCAN object
    db = DBSCAN(eps=D,min_samples=minP)
    #fit the data
    db.fit(localizations)
    #get the labels
    cluster_labels = db.labels_
    #get the unique labels without -1
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])
    #get the cluster centers
    cluster_centers = np.zeros((len(unique_labels),2))
    #get the cluster radii
    cluster_radii = np.zeros(len(unique_labels))
    #loop over the unique labels
    for i in range(len(unique_labels)):
        #get the cluster label
        cluster_label = unique_labels[i]
        #get the cluster
        cluster = localizations[cluster_labels == cluster_label]
        #get the convex hull
        hull = ConvexHull(cluster)
        #get the cluster center
        cluster_centers[i] = np.mean(cluster[hull.vertices],axis=0)
        #get the cluster radius
        cluster_radii[i] = np.mean(np.linalg.norm(cluster[hull.vertices]-cluster_centers[i],axis=1))
    return cluster_labels,cluster_centers,cluster_radii


#lets to a main run
if __name__ == '__main__':
    #make a batch for a set of cds
    CORRECTION_FACTOR = 0.13
    cds = [
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/12/rpoc_m9_2"
    ]           
    t_strings = [
        "rpoc_ez"
    ]

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

    img_dims = [
        (196,196)
    ]

    rescale_pixel_size = [
        10
    ]
    type_analysis_file = [
        "new"
    ]
    total_frames = [
        5000
    ]
    subframes = [
        5
    ]
    pixel_size = [
        130
    ]
    loc_error = [
        30
    ]
    include_all = [
        False
    ]

    #loop through each cd
    for i in range(len(cds)):
        #make the batch
        batch = segmentation_scale_space(cds[i],
                                         t_strings[i],
                                         blob_parameters,
                                         fitting_parameters,
                                         img_dims[i],
                                         rescale_pixel_size[i],
                                         type_analysis_file[i],
                                         total_frames[i],
                                         subframes[i],
                                         pixel_size[i],
                                         loc_error[i],
                                         include_all[i])
        #run the batch
        batch.main_run()
