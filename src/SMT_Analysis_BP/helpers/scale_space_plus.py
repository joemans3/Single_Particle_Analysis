'''
Suite of functions and classes to perfrom the scale space plus procedure to create the reconstruction image for scale space analysis.

'''
#for testing add src to the path
if __name__=="__main__":
	import sys
	sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')

import numpy as np
import pandas as pd
import skimage as skimage
import os
from src.SMT_Analysis_BP.helpers.pickle_util import PickleUtil
from src.SMT_Analysis_BP.helpers.simulate_foci import get_gaussian
from src.SMT_Analysis_BP.helpers.Analysis_functions import rescale_range
import matplotlib.pyplot as plt
KEY_IMAGE = {
    'png':skimage.io.imsave,
    'jpg':skimage.io.imsave,
    'tif':skimage.io.imsave,
    'svg':skimage.io.imsave
    }

class SM_reconstruction_image:

    def __init__(self,img_dims_normal,pixel_size_normal=130,rescale_pixel_size=10) -> None:
        self.img_dims_normal = img_dims_normal #in pixels
        self.pixel_size_normal = pixel_size_normal #in nm
        self.rescale_pixel_size = rescale_pixel_size #in nm
        self.img_dims = [int(i*self.pixel_size_normal/self.rescale_pixel_size) for i in self.img_dims_normal]        
        self.domain = [np.arange(0,self.img_dims[0],1.),np.arange(0,self.img_dims[1],1.)]


    def make_reconstruction(self,localizations,localization_error):
        '''
        Parameters:
        -----------
        localizations: np.ndarray
            Array of localizations in the form of (x,y) in pixels (original scale)
        localization_error: np.ndarray or float
            Array of localization error in nm or a scalar
        
        Returns:
        --------
        img_space: np.ndarray
            Image space of the reconstruction
        '''
        #create a pandas dataframe for the localizations and localization error
        #check if the localization error is a scalar or an array of the same length as the localizations
        if np.isscalar(localization_error):
            localization_error = np.ones(len(localizations))*localization_error
        elif len(localization_error) != len(localizations):
            raise ValueError('The length of the localization error must be the same as the length of the localizations')
        self.df_localizations = pd.DataFrame({'x':localizations[:,0],'y':localizations[:,1],'localization_error':localization_error})
        #reformat to be in the form of a collection of (x,y)
        self.df_localizations = self.df_localizations[['x','y','localization_error']]
        #rescale the localizations
        self.df_localizations[['x','y']] = self.df_localizations[['x','y']]*self.pixel_size_normal/self.rescale_pixel_size
        #loop over the localizations
        self.img_space = np.zeros(self.img_dims)

        for i in range(len(self.df_localizations)):
            x = self.df_localizations.iloc[i]['x']
            y = self.df_localizations.iloc[i]['y']
            loc_error_val = self.df_localizations.iloc[i]['localization_error']/(self.rescale_pixel_size)
            sigma_shape = np.ones(2)*loc_error_val
            
            #we want to only sample a 10x10 pixel area around the localization and then remap it to the original image space
            #make the new domain range scale with this localization error/ pixel size ratio
            domain_max = int(10*loc_error_val)
            domain = [np.arange(0,domain_max,1.),np.arange(0,domain_max,1.)]
            #lets get the x,y in this new domain
            x_scaled = rescale_range(x,*[0,self.img_dims[0]],*[0,domain_max])
            y_scaled = rescale_range(y,*[0,self.img_dims[1]],*[0,domain_max])
            #get the gaussian
            temp_scape = get_gaussian(mu = [x_scaled,y_scaled],sigma = sigma_shape,domain = domain)
            #now we have a 20x20 matrix but we need to embed it into the orginal self.img_space
            #we can find the bottom corner of the 20x20 matrix in the self.img_space
            x_bottom = int(x - domain_max/2.)
            y_bottom = int(y - domain_max/2.)
            #now we can embed the 20x20 matrix into the self.img_space
            try:
                self.img_space[x_bottom:x_bottom+domain_max,y_bottom:y_bottom+domain_max] = self.img_space[x_bottom:x_bottom+domain_max,y_bottom:y_bottom+domain_max] + np.array(temp_scape)
            except:
                print("The localization is too close to the edge of the image space. Edge cases are not implimented yet so the localization is not included in the reconstruction.")
        return self.img_space.T

    def saving_image(self,full_path,name,type):
        '''
        Parameters:
        -----------
        full_path: str
            Full path to save the image
        name: str
            Name of the image
        type: str
            Type of the image. Supported:
                - png
                - jpg
                - tif
                - svg
        '''
        #check if the type is supported
        if type not in KEY_IMAGE.keys():
            raise ValueError('The type of the image is not supported')
        #check if the path exists
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        #make the name by joining the strings
        name = os.path.join(full_path,name+'.'+type)
        #save the image
        skimage.io.imsave(name,self.img_space.T)

        # #now we need to save the object of the localizations
        # #create a PickleUtil object
        # pickle_util = PickleUtil()
        # #save the object
        # pickle_util.save(path=full_path, name=name, docs='This is the reconstruction image data as a pickle file: x,y,loc_error in the scaled image space', obj=self.df_localizations)


    
    def _make_uniform_localizations(self): #TODO impliment this
        return


    @property
    def img_dims(self):
        return self._img_dims
    @img_dims.setter
    def img_dims(self,img_dims):
        self._img_dims = img_dims
    @property
    def img_space(self):
        if not hasattr(self,'_img_space'):
            self._img_space = np.zeros(self.img_dims)
        return self._img_space
    @img_space.setter
    def img_space(self,img_space):
        self._img_space = img_space
    @property
    def domain(self):
        if not hasattr(self,'_domain'):
            self._domain = [np.arange(0,self.img_dims[0],1.),np.arange(0,self.img_dims[1],1.)]
        return self._domain
    @domain.setter
    def domain(self,domain):
        self._domain = domain
    @property
    def df_localizations(self):
        return self._df_localizations
    @df_localizations.setter
    def df_localizations(self,df_localizations):
        self._df_localizations = df_localizations
    @property
    def total_localizations(self):
        return len(self.df_localizations)