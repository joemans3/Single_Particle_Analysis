import numpy as np
import pandas as pd
import os
from lmfit import Parameters, minimize
from skimage.feature import peak_local_max
from skimage.io import imread
from matplotlib import pyplot as plt
import time

from typing import List

##############################################################################################################
#define some gaussian models for fitting
def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset,height,kwargs ={}):
	''' 2d gaussian function no theta'''
	return (height)*np.exp(-(((cen_x-x)/sig_x)**2 + ((cen_y-y)/sig_y)**2)/2.0) + offset

def gaussian2D_theta(x, y, cen_x, cen_y, sig_x, sig_y, theta, offset, height, kwargs={}):
    ''' 2d gaussian function with theta'''
    a = (np.cos(theta)**2)/(2*sig_x**2) + (np.sin(theta)**2)/(2*sig_y**2)
    b = -(np.sin(2*theta))/(4*sig_x**2) + (np.sin(2*theta))/(4*sig_y**2)
    c = (np.sin(theta)**2)/(2*sig_x**2) + (np.cos(theta)**2)/(2*sig_y**2)
    return (height)*np.exp(-(a*(x-cen_x)**2 + 2*b*(x-cen_x)*(y-cen_y) + c*(y-cen_y)**2)) + offset

def gaussian1D(x, cen_x, sig_x, offset, height, kwargs={}):
    ''' 1d gaussian function'''
    return (height)*np.exp(-(((cen_x-x)/sig_x)**2)/2.0) + offset


##############################################################################################################
#define some residual functions for fitting
def gaussian2D_residual(params, x, y, data, kwargs={}):
    ''' 2d gaussian residual function'''
    cen_x = params['cen_x']
    cen_y = params['cen_y']
    sig_x = params['sig_x']
    sig_y = params['sig_y']
    offset = params['offset']
    height = params['height']
    model = gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset, height, kwargs)
    return model - data

def gaussian2D_theta_residual(params, x, y, data, kwargs={}):
    ''' 2d gaussian residual function with theta'''
    cen_x = params['cen_x']
    cen_y = params['cen_y']
    sig_x = params['sig_x']
    sig_y = params['sig_y']
    theta = params['theta']
    offset = params['offset']
    height = params['height']
    model = gaussian2D_theta(x, y, cen_x, cen_y, sig_x, sig_y, theta, offset, height, kwargs)
    return model - data

def gaussian1D_residual(params, x, data, kwargs={}):
    ''' 1d gaussian residual function'''
    cen_x = params['cen_x']
    sig_x = params['sig_x']
    offset = params['offset']
    height = params['height']
    model = gaussian1D(x, cen_x, sig_x, offset, height, kwargs)
    return model - data





##############################################################################################################
#define functions to find candidate peaks in a signal
#for a 2D signal (image)
def find_peaks(image,**kwargs):
    '''
    Finds candidate peaks in a 2D image using skimage.feature.peak_local_max
    
    Parameters:
    -----------
    image: 2D numpy array
        image to find peaks in
    kwargs: dict
        dictionary of keyword arguments to pass to peak_local_max
        See skimage.feature.peak_local_max for more details
        Some useful kwargs:
            min_distance: int
                minimum distance in pixels between peaks
            threshold_abs: int
                minimum intensity of peaks
            exclude_border: int
                number of pixels to exclude from the border of the image
    '''
    #find peaks
    peaks = peak_local_max(image,**kwargs)
    return peaks

#save dataframe
def save_dataframe(df:pd.DataFrame,path:str):
    '''
    Saves a dataframe to a csv file
    
    Parameters:
    -----------
    df: pandas.DataFrame
        dataframe to save
    path: str
        path to save the dataframe to
    '''
    df.to_csv(path,index=False)


class SM_Localization_Image:
    def __init__(self,image,candidate_peaks:np.ndarray=None) -> None:
        self.plot_fit_results = False
        #make sure image is a numpy array of 2D shape
        assert isinstance(image,np.ndarray) and len(image.shape)==2
        self._initialize_properties(image)
        if candidate_peaks is None:
            self._find_candidate_peaks()
        else:
            self.candidate_peaks = candidate_peaks

    def _initialize_properties(self,image:np.ndarray):
        self._image = image
        self._img_shape = image.shape
        self._candidate_peaks = None
        self._fit_results = None
    
    def _update_lmfit_model(self,model:callable):
        self.lmfit_model = model
    
    def _find_candidate_peaks(self,**kwargs):
        '''
        Private method to find candidate peaks in the image

        Parameters:
        -----------
        kwargs: dict
            dictionary of keyword arguments to pass to find_peaks

        Returns:
        --------
        candidate_peaks: numpy array
            array of candidate peaks
        
        Keyword Arguments:
        ------------------
        min_distance: int
            minimum distance in pixels between peaks
        threshold_abs: int
            minimum intensity of peaks
        exclude_border: int
            number of pixels to exclude from the border of the image
        Others are passed to skimage.feature.peak_local_max
        '''
        find_peaks_kwargs_default = {'min_distance':2,'threshold_abs':np.mean(self.image)+np.std(self.image)*7.,'exclude_border':5}
        #if the passed kwargs are not in the default kwargs, add them
        for key in kwargs.keys():
            if key not in find_peaks_kwargs_default.keys():
                find_peaks_kwargs_default[key] = kwargs[key]
        #find peaks
        self.candidate_peaks = find_peaks(self.image,**find_peaks_kwargs_default)
    
    def _fit_peaks(self,fit_model_residual:callable,fit_model_params:dict,fit_model_params_bounds:dict):
        '''
        Fits the peaks in the image to the passed model
        
        Parameters:
        -----------

        fit_model_residual: callable
            callable residual function for the model
        fit_model_params: dict
            dictionary of initial parameters for the model
        fit_model_params_bounds: dict
            dictionary of bounds for the parameters of the model
        '''
        #initialize fit results
        fit_results = []
        #loop over all peaks
        for peak in self.candidate_peaks:
            #get the region of interest around the peak
            roi = self._get_roi_around_peak(peak)
            #update the initial parameters for the model using the peak location and the roi values
            fit_model_params.update({'cen_x':peak[0],'cen_y':peak[1],'height':roi.max(),'offset':roi.min()})
            #fit the model to the roi
            fit_results.append(self._fit_model_to_roi(roi,peak,fit_model_residual,fit_model_params,fit_model_params_bounds))
            #plot the fit results along with the roi
            if self.plot_fit_results:
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                #wireframe plot of the roi, use the x,y coordinates of the roi as the x,y coordinates of the plot
                ax.plot_wireframe(*self._get_roi_coordinates(roi,peak),
                                roi,
                                color='green')
                #plot the fit results using a wireplot with the gaussian model
                ax.plot_wireframe(*self._get_roi_coordinates(roi,peak),
                                gaussian2D(*self._get_roi_coordinates(roi,peak),**fit_results[-1].params.valuesdict()),
                                color='red')
                ax.set_xlabel('x (pixels)')
                ax.set_ylabel('y (pixels)')
                ax.set_zlabel('Intensity (a.u.)')
                ax.set_title('Peak at ({},{})'.format(peak[0],peak[1]))
                #legend
                ax.legend(['Data','Fit'])
                plt.show()

        #update the fit results
        self.fit_results = fit_results

    def _get_roi_around_peak(self,peak:np.ndarray,roi_size:int=5):
        '''
        Gets the region of interest around a peak
        
        Parameters:
        -----------
        peak: numpy array
            peak to get the roi around
        '''
        #get the roi around the peak
        roi = self.image[peak[0]-roi_size:peak[0]+roi_size+1,peak[1]-roi_size:peak[1]+roi_size+1]
        return roi
    
    def _fit_model_to_roi(self,roi:np.ndarray,peak:np.ndarray,fit_model_residual:callable,fit_model_params:dict,fit_model_params_bounds:dict):
        '''
        Fits the passed model to the passed roi
        
        Parameters:
        -----------
        roi: numpy array
            region of interest to fit the model to
        peak: numpy array
            peak to fit the model to
        fit_model_residual: callable
            callable residual function for the model
        fit_model_params: dict
            dictionary of initial parameters for the model
        fit_model_params_bounds: dict
            dictionary of bounds for the parameters of the model
        
        Returns:
        --------
        fit_results: lmfit.minimizer.MinimizerResult
            results of the fit in the form of a lmfit MinimizerResult object
        '''
        #get the initial parameters for the model
        params = self._get_model_params(fit_model_params,fit_model_params_bounds)
        #get the x and y coordinates of the roi
        x,y = self._get_roi_coordinates(roi,peak)
        #fit the model to the roi
        return self._fit_model_to_data(fit_model_residual,params,x,y,roi)
    
    def _get_model_params(self,fit_model_params:dict,fit_model_params_bounds:dict):
        '''
        Gets the initial parameters for the model
        
        Parameters:
        -----------
        fit_model_params: dict
            dictionary of initial parameters for the model
        fit_model_params_bounds: dict
            dictionary of bounds for the parameters of the model
        '''
        #initialize parameters
        params = Parameters()
        #loop over all parameters
        for key in fit_model_params.keys():
            #add the parameter to the model
            params.add(key,value=fit_model_params[key],min=fit_model_params_bounds[key][0],max=fit_model_params_bounds[key][1])
        return params
    
    def _get_roi_coordinates(self,roi:np.ndarray,peak:np.ndarray):
        '''
        Gets the x and y coordinates of the roi
        
        Parameters:
        -----------
        roi: numpy array
            region of interest to fit the model to
        peak: numpy array
            peak to fit the model to
        '''
        #get the x and y coordinates of the roi using meshgrid but make sure the values are the x,y values in the image
        x,y = np.meshgrid(np.arange(peak[0]-roi.shape[0]//2,peak[0]+roi.shape[0]//2+1),np.arange(peak[1]-roi.shape[1]//2,peak[1]+roi.shape[1]//2+1))
        return x,y

    def _fit_model_to_data(self,fit_model_residual:callable,params:Parameters,x:np.ndarray,y:np.ndarray,data:np.ndarray):
        '''
        Fits the passed model to the passed data
        
        Parameters:
        -----------
        fit_model_residual: callable
            callable residual function for the model
        params: Parameters
            parameters for the model
        x: numpy array
            x coordinates of the data
        y: numpy array
            y coordinates of the data
        data: numpy array
            data to fit the model to
        
        Returns:
        --------
        fit_results: lmfit.minimizer.MinimizerResult
            results of the fit
        '''
        #fit the model to the data
        fit_results = minimize(fit_model_residual,params,args=(x,y,data))
        return fit_results
    
    def _convert_fit_result_pandas(self)->pd.DataFrame:
        '''
        Converts the fit results to a pandas dataframe with each row corresponding to a peak, 
        and columns corresponding to the fit parameters and extra columns for the fit errors from the least squares fit

        Returns:
        --------
        fit_results_df: pandas.DataFrame
            dataframe of the fit results
        '''
        #initialize the dataframe with the fit parameters as column names and each row corresponding to a peak
        fit_results_df = pd.DataFrame(columns=list(self.fit_parameters['fit_model_params'].keys())+["Image_Background_Mean","Image_Background_Std"])
        #loop over all the results
        for result in self.fit_results:
            #get the parameters
            params = result.params.valuesdict()
            #convert the errors to a dictionary
            errors = {key+'_err':result.params[key].stderr for i,key in enumerate(params.keys())}
            #add the errors to the parameters dictionary
            params.update(errors)
            #add the background mean and std to the parameters dictionary
            params.update({'Image_Background_Mean':self.img_background_mean,'Image_Background_Std':self.img_background_std})
            #add the parameters to the dataframe using concat
            fit_results_df = pd.concat([fit_results_df,pd.DataFrame(params,index=[0])],ignore_index=True)

        #update the dataframe
        self._fit_results_df = fit_results_df
        return fit_results_df
    
    def _prune_fit_results(self,params_to_prune:list,params_thresholds:list):
        '''
        Prune the fit results based on the passed parameters and thresholds
        
        Parameters:
        -----------
        params_to_prune: list of keys of the parameters to prune
            list of parameters to prune
        params_thresholds: list of tuples
            list of tuples of the form (lower_threshold,upper_threshold) for each parameter in params_to_prune

        Returns:
        --------
        fit_results_pruned: list
            list of pruned fit results
        Also updates the fit_results property
        '''
        #initialize the pruned fit results
        fit_results_pruned = []
        #loop over all the fit results
        for result in self.fit_results:
            #get the parameters
            params = result.params.valuesdict()
            #get the values of the parameters to prune
            params_to_prune_values = [params[key] for key in params_to_prune]
            #check if the parameters are within the thresholds
            if all([params_to_prune_values[i]>=params_thresholds[i][0] and params_to_prune_values[i]<=params_thresholds[i][1] for i in range(len(params_to_prune_values))]):
                #if they are, append the fit result to the pruned fit results
                fit_results_pruned.append(result)
        #update the fit results
        self.fit_results = fit_results_pruned
        return fit_results_pruned
    #run the main logic of the class
    def fit(self,fit_model_residual:callable,fit_model_params:dict,fit_model_params_bounds:dict,prune_direction:dict=None,**kwargs):
        '''
        Finds candidate peaks in the image and fits them to the passed model, supports only gaussian models for now
        
        Parameters:
        -----------
        fit_model_residual: callable
            callable residual function for the model
        fit_model_params: dict
            dictionary of initial parameters for the model
        fit_model_params_bounds: dict
            dictionary of bounds for the parameters of the model
        prune_direction: dict, optional
            dictionary of parameters to prune and the direction to prune them in
            This is a dictionary of the form {parameter_name:prune_direction} where prune_direction is a tuple of the form (lower_threshold,upper_threshold)
            the parameter_name is a string matching the key in the fit_model_params dictionary
        kwargs: dict
            dictionary of keyword arguments to pass to find_peaks
        '''
        #store the fit parameters
        self.fit_parameters = {'fit_model_residual':fit_model_residual,'fit_model_params':fit_model_params,'fit_model_params_bounds':fit_model_params_bounds}
        #find the candidate peaks
        self._find_candidate_peaks(**kwargs)
        #fit the peaks
        self._fit_peaks(fit_model_residual,fit_model_params,fit_model_params_bounds)
        #prune the fit results
        if prune_direction is not None:
            self._prune_fit_results(list(prune_direction.keys()),list(prune_direction.values()))
        #return the fit results
        return self.fit_results


    @property
    def image(self):
        return self._image
    @property
    def img_shape(self):
        return self._img_shape
    @property
    def img_background_mean(self):
        if not hasattr(self,'_image_background_mean'):
            self._image_background_mean = self.image.mean()
        return self._image_background_mean
    @property
    def img_background_std(self):
        if not hasattr(self,'_image_background_std'):
            self._image_background_std = self.image.std()
        return self._image_background_std
    
    @property
    def candidate_peaks(self)->np.ndarray:
        return self._candidate_peaks
    @candidate_peaks.setter
    def candidate_peaks(self,peaks:np.ndarray):
        self._candidate_peaks = peaks
    
    @property
    def fit_results(self)->list:
        return self._fit_results
    @fit_results.setter
    def fit_results(self,fit_results:list):
        self._fit_results = fit_results

    @property
    def lmfit_model(self):
        return self._lmfit_model
    @lmfit_model.setter
    def lmfit_model(self,model):
        self._lmfit_model = model

    @property
    def fit_parameters(self):
        return self._fit_parameters
    @fit_parameters.setter
    def fit_parameters(self,fit_parameters):
        self._fit_parameters = fit_parameters

    @property
    def fit_results_df(self):
        return self._convert_fit_result_pandas()


class SM_localization_movie:
    '''
    Extension of the SM_Localization_Image class to handle movies in which each frame is a 2D image and is fit to the same model
    contains utility function to convert to a final pandas dataframe
    '''
    def __init__(self,movie) -> None:
        #make sure movie is a numpy array of 3D shape
        assert isinstance(movie,np.ndarray) and len(movie.shape)==3
        self._initialize_properties(movie)

    def _initialize_properties(self,movie:np.ndarray)->None:
        self._movie = movie
        self._movie_shape = movie.shape

    def _convert_fit_result_pandas(self)->pd.DataFrame:
        '''
        Converts the fit results to a pandas dataframe with each row corresponding to a peak, 
        and columns corresponding to the fit parameters and extra columns for the fit errors from the least squares fit
        We also add 2 more columns for the frame number and a unique peak number in the frame to identify each peak (Spot_ID)

        Returns:
        --------
        fit_results_df: pandas.DataFrame
            dataframe of the fit results
        '''
        #initialize the dataframe with the fit parameters as column names and each row corresponding to a peak
        fit_results_df = pd.DataFrame(columns=list(self.fit_parameters['fit_model_params'].keys())+["Image_Background_Mean","Image_Background_Std","Frame","Spot_ID"])
        #loop over all the fit results in each frame
        for frame_num,frame_fit_results in enumerate(self.fit_results):
            #loop over all the results
            for peak_num,result in enumerate(frame_fit_results):
                #get the parameters
                params = result.params.valuesdict()
                #convert the errors to a dictionary
                errors = {key+'_err':result.params[key].stderr for i,key in enumerate(params.keys())}
                #add the errors to the parameters dictionary
                params.update(errors)
                #add the background mean and std to the parameters dictionary
                params.update({'Image_Background_Mean':self.movie_background_mean[frame_num],'Image_Background_Std':self.movie_background_std[frame_num]})
                #add the frame number and peak number to the parameters dictionary
                params.update({'Frame':frame_num,'Spot_ID':peak_num})
                #add the parameters to the dataframe using concat
                fit_results_df = pd.concat([fit_results_df,pd.DataFrame(params,index=[0])],ignore_index=True)
        #update the dataframe
        self._fit_results_df = fit_results_df
        return fit_results_df

    def _save_dataframe(self,path:str)->None:
        '''
        Saves the fit results dataframe to a csv file
        
        Parameters:
        -----------
        path: str
            path to save the dataframe to (needs to be an absolute path)
        '''
        #make sure that the directory exists in which the name is to be saved
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        #save the dataframe
        save_dataframe(self.fit_results_df,path)

    def fit(self,fit_model_residual:callable,fit_model_params:dict,fit_model_params_bounds:dict,prune_direction:dict=None,candidate_peaks_per_frame:List[np.ndarray]=None,**kwargs):
        '''
        Finds candidate peaks in the image and fits them to the passed model, supports only gaussian models for now
        
        Parameters:
        -----------
        fit_model_residual: callable
            callable residual function for the model
        fit_model_params: dict
            dictionary of initial parameters for the model
        fit_model_params_bounds: dict
            dictionary of bounds for the parameters of the model
        prune_direction: dict, optional
            dictionary of parameters to prune and the direction to prune them in
            This is a dictionary of the form {parameter_name:prune_direction} where prune_direction is a tuple of the form (lower_threshold,upper_threshold)
            the parameter_name is a string matching the key in the fit_model_params dictionary
        candidate_peaks_per_frame: numpy array, optional
            numpy array of candidate peaks for each frame, if not passed, the peaks are found using find_peaks 
            These need to be [x,y] coordinates
        kwargs: dict
            dictionary of keyword arguments to pass to find_peaks
        '''
        #store the fit parameters
        self._fit_parameters = {'fit_model_residual':fit_model_residual,'fit_model_params':fit_model_params,'fit_model_params_bounds':fit_model_params_bounds}
        #initialize the fit results
        self._fit_results = []
        #loop over all frames in the movie
        if candidate_peaks_per_frame is not None:
            for indx,frame in enumerate(self.movie):
                #initialize the SM_Localization_Image class for each frame
                sm_localization_image = SM_Localization_Image(frame,candidate_peaks_per_frame[indx])
                #fit the model to the frame
                self._fit_results.append(sm_localization_image.fit(fit_model_residual,fit_model_params,fit_model_params_bounds,prune_direction,**kwargs))
        else:
            start_time = time.time()
            for frame in self.movie:
                #initialize the SM_Localization_Image class for each frame
                sm_localization_image = SM_Localization_Image(frame)
                #fit the model to the frame
                self._fit_results.append(sm_localization_image.fit(fit_model_residual,fit_model_params,fit_model_params_bounds,prune_direction,**kwargs))
            print("Time to fit frame: {} seconds".format(time.time()-start_time))
        #convert the fit results to a pandas dataframe
        #time the conversion
        start_time = time.time()
        self._fit_results_df = self._convert_fit_result_pandas()
        print("Time to convert fit results to pandas dataframe: {} seconds".format(time.time()-start_time))
        return self.fit_results_df

    @property
    def movie(self)->np.ndarray:
        return self._movie
    
    @property
    def movie_shape(self)->tuple:
        if hasattr(self,'_movie_shape'):
            return self._movie_shape
        else:
            self._movie_shape = self.movie.shape
            return self._movie_shape
    @property
    def movie_background_mean(self)->float:
        if not hasattr(self,'_movie_background_mean'):
            self._movie_background_mean = self.movie.mean(axis=(1,2))
        return self._movie_background_mean
    @property
    def movie_background_std(self)->float:
        if not hasattr(self,'_movie_background_std'):
            self._movie_background_std = self.movie.std(axis=(1,2))
        return self._movie_background_std
    @property
    def fit_parameters(self)->dict:
        return self._fit_parameters
    
    @property
    def fit_results(self)->list:
        return self._fit_results

    @property
    def fit_results_df(self)->pd.DataFrame:
        return self._fit_results_df



##############################################################################################################
#tests
if __name__ == "__main__":
        
    path = "/Volumes/Baljyot_HD/SMT_Olympus/Baljyot_temp/20190527/rpoc_ez/Movie/rpoc_ez_6.tif"
    image = imread(path)

    #initialize the SM_Localization_movie class
    sm_localization_movie = SM_localization_movie(image)
    #fit the movie to a 2d gaussian
    fit_results_df = sm_localization_movie.fit(
        gaussian2D_residual,
        {'cen_x':0,'cen_y':0,'sig_x':1,'sig_y':1,'offset':0,'height':1},
        {'cen_x':(-np.inf,np.inf),'cen_y':(-np.inf,np.inf),'sig_x':(0,np.inf),'sig_y':(0,np.inf),'offset':(-np.inf,np.inf),'height':(0,np.inf)},
        prune_direction={'sig_x':(0.7,2),'sig_y':(0.7,2)})
    #save the dataframe
    sm_localization_movie._save_dataframe("/Volumes/Baljyot_HD/SMT_Olympus/Baljyot_temp/20190527/rpoc_ez/Movie/rpoc_ez_6_fit_results.csv")

