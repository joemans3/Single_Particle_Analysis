'''
Module focused on extracting features from a mask image.

Main properties focused on extracting:
    - Area
    - Bounding Box
    - Centroid
    - r_offset (bottom left corner of bounding box)
'''
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage import io

class extract_mask_properties:
    def __init__(self,masked_image,invert_axis) -> None:
        '''
        Parameters:
        -----------
        masked_image: numpy array
            Masked image
        invert_axis: bool
            If True, inverts the axis of the masked image
        '''
        self.masked_image = masked_image
        if invert_axis:
            #make the x = y and y = x
            self.masked_image = np.transpose(self.masked_image)
        self.extract_properties()
        self._populate_properties()
    def extract_properties(self):
        # Extracting the region properties of the masked image
        self.region_properties = measure.regionprops(self.masked_image)
    def _populate_properties(self):
        # Populating the properties
        self._area = self.region_properties[0].area
        self._bounding_box = self.region_properties[0].bbox
        self._centroid = self.region_properties[0].centroid
        self._r_offset = self.region_properties[0].bbox[0:2]
        self._longest_axis = self.region_properties[0].major_axis_length
        self._shortest_axis = self.region_properties[0].minor_axis_length
        self._coordinates = self.region_properties[0].coords
        self._orientation = self.region_properties[0].orientation

    def __dict__(self):
        return {'area':self.area,'bounding_box':self.bounding_box,'centroid':self.centroid,'r_offset':self.r_offset,'longest_axis':self.longest_axis,'shortest_axis':self.shortest_axis,'coordinates':self.coordinates,'orientation':self.orientation}

    @property
    def region_properties(self):
        '''
        Returns the region properties of the masked image
        '''
        return self._region_properties
    @region_properties.setter
    def region_properties(self,region_properties):
        self._region_properties = region_properties

    @property
    def area(self):
        '''
        Returns the area of the mask
        '''
        return self._area
    
    @property
    def bounding_box(self):
        '''
        Returns the bounding box of the mask as a numpy array but in the form of the coordinates of the bounding box (bottom left corner, top right corner, etc.)
        '''
        #regionprops output is the min_X,min_Y,max_X,max_Y
        #we want to convert this to [[min_X,min_Y],[max_X,max_Y]]
        return np.array([[self._bounding_box[0],self._bounding_box[1]],[self._bounding_box[2],self._bounding_box[3]]])

    @property
    def centroid(self):
        '''
        Returns the centroid of the mask
        '''
        return self._centroid
    
    @property
    def r_offset(self):
        '''
        Returns the r_offset of the mask
        '''
        return self._r_offset
    
    @property
    def longest_axis(self):
        '''
        Returns the longest axis of the mask
        '''
        return self._longest_axis
    
    @property
    def shortest_axis(self):
        '''
        Returns the shortest axis of the mask
        '''
        return self._shortest_axis

    @property
    def coordinates(self):
        '''
        Returns the coordinates of the mask
        '''
        return self._coordinates

    @property
    def orientation(self):
        '''
        Returns the orientation of the mask
        '''
        return self._orientation


##############################################################################################################
#Testing

if __name__ == '__main__':
    path = '/Users/baljyot/Documents/SMT_Movies/testing_SM_recon/Movies/Movie_1/Cell_1/mask.tif'
    mask = io.imread(path)
    prop_obj = extract_mask_properties(mask)
    print(prop_obj.area)
    print(prop_obj.bounding_box)
    print(prop_obj.centroid)
    print(prop_obj.r_offset)

    #lets plot to see the results
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(mask)
    ax[0].set_title('Mask')
    ax[1].imshow(mask)
    ax[1].set_title('Mask with properties')
    ax[1].add_patch(plt.Rectangle((prop_obj.bounding_box[1],prop_obj.bounding_box[0]),prop_obj.bounding_box[3]-prop_obj.bounding_box[1],prop_obj.bounding_box[2]-prop_obj.bounding_box[0],fill=False,color='r'))
    ax[1].scatter(prop_obj.centroid[1],prop_obj.centroid[0],color='r',label='centroid')
    #plot the r_offset
    ax[1].scatter(prop_obj.r_offset[1],prop_obj.r_offset[0],color='g',label='r_offset')
    ax[1].legend()
    plt.show()
