'''
Helper script to find all the masks in a directory and format them into a directory structure

The directory structure is as follows:

-parent_dir
    -full_path_to_mask_dir

    -Movies
        -Movie_1
            -IMAGEJ_ROI.zip
            -Cell_1
                -mask.tif

This is repeated for each movie defined by the number of mask files found. 
The cells are determined by the number of unique masks found in each mask file.

The IMAGEJ_ROI.zip file is the ROI file that works for input into ImageJ for further analysis
'''

if __name__ == '__main__':
    import sys
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from skimage import measure
import skimage.io as ioski
import cv2
from cellpose import utils,io
from src.SMT_Analysis_BP.helpers.Analysis_functions import sorted_alphanumeric
#remove warnings
import warnings
warnings.filterwarnings("ignore")

class movie_mask_directory_structure_manager:
    def __init__(self,full_path_to_mask_dir) -> None:
        self._full_path_to_mask_dir = full_path_to_mask_dir
        self._find_mask_files()
        self._make_directory_structure()

    def _find_mask_files(self)->None:
        '''Finds the mask files (right now these are .npy files) in the directory, defined by cellpose output
        This sets the mask_files attribute
        '''
        #determine if the directory exists
        if not os.path.isdir(self.full_path_to_mask_dir):
            raise ValueError('The directory does not exist')
        self._mask_files = sorted_alphanumeric(glob.glob(os.path.join(self.full_path_to_mask_dir,'*.npy')))
        print(self.mask_files)
        if len(self.mask_files) == 0:
            raise ValueError('No mask files found in the directory')
    def _make_directory_structure(self)->None:
        '''Makes the directory structure for the Movies, cells and saves each mask inside it
        this follows from the strucure described above
        '''

        path_structure = {}
        #find the parent directory of the directory containing the masks
        parent_dir = os.path.dirname(self.full_path_to_mask_dir)
        path_structure['parent_dir'] = parent_dir
        path_structure['mask_containing_dir'] = self.full_path_to_mask_dir

        #store the movie directory
        path_structure['movie_dir'] = {}
        path_structure['movie_dir']['movies'] = {}
        path_structure['movie_dir']["Path"] = os.path.join(parent_dir,'Movies')
        #make the movie directory
        if not os.path.isdir(path_structure['movie_dir']["Path"]):
            os.mkdir(path_structure['movie_dir']["Path"])
        
        for movie in range(len(self.mask_files)):
            #add to the path structure
            path_structure['movie_dir']['movies'][movie+1] = {}
            path_structure['movie_dir']['movies'][movie+1]['mask_file_path'] = self.mask_files[movie]
            path_structure['movie_dir']['movies'][movie+1]["path"] = os.path.join(path_structure['movie_dir']["Path"],'Movie_{}'.format(movie+1))
            #make the movie directory
            if not os.path.isdir(path_structure['movie_dir']['movies'][movie+1]["path"]):
                os.mkdir(path_structure['movie_dir']['movies'][movie+1]["path"])

            #obtain the cell masks
            cell_masks = obtain_masks_npy(self.mask_files[movie])
            #store the cell directory
            path_structure['movie_dir']['movies'][movie+1]['cells'] = {}
            #store the number of cells
            path_structure['movie_dir']['movies'][movie+1]['num_cells'] = len(cell_masks)
            for cells,cell_mask in cell_masks.items():
                #add to the path structure
                path_structure['movie_dir']['movies'][movie+1]['cells'][cells] = {}
                path_structure['movie_dir']['movies'][movie+1]['cells'][cells]['mask'] = cell_mask
                #save the directory structure
                path_structure['movie_dir']['movies'][movie+1]['cells'][cells]['path'] = os.path.join(path_structure['movie_dir']['movies'][movie+1]['path'],'Cell_{}'.format(cells))
                #make the cell directory
                if not os.path.isdir(path_structure['movie_dir']['movies'][movie+1]['cells'][cells]['path']):
                    os.mkdir(path_structure['movie_dir']['movies'][movie+1]['cells'][cells]['path'])
                #save the mask
                ioski.imsave(os.path.join(path_structure['movie_dir']['movies'][movie+1]['cells'][cells]['path'],'mask.tif'),cell_mask)
            #TODO impliment the ROI file saving
            #save the ROI files to the Movie directory with name Movie_{}.zip by taking the total mask (all cells) and saving it as a ROI file
            #save the ROI file
            io.save_rois(read_npy_file(self.mask_files[movie])['masks'],os.path.join(path_structure['movie_dir']['movies'][movie+1]["path"],"Movie_{}".format(movie+1)))
        self._directory_structure_paths = path_structure


    @property
    def full_path_to_mask_dir(self):
        return self._full_path_to_mask_dir
    @property
    def mask_files(self):
        return self._mask_files
    @property
    def directory_structure_paths(self):
        return self._directory_structure_paths

def read_npy_file(path,correct_dtype=True):
    '''
    Reads the npy file from the path
    '''
    if correct_dtype:
        return np.load(path,allow_pickle=True).item()
    else:
        return np.load(path,allow_pickle=True)

def obtain_masks_npy(npy_output_file)->dict:
    '''
    Obtains the masks from the npy file
    '''
    mask = read_npy_file(npy_output_file)
    mask = mask['masks']
    cell_masks = {}
    for cell in range(len(np.unique(mask))):
        #ignore 0 
        if cell == 0:
            continue
        cell_masks[cell] = (mask == cell)*255
        #convert to 8 bit
        cell_masks[cell] = cell_masks[cell].astype(np.uint8)
    return cell_masks




if __name__ == '__main__':
    ##############################################################################################################
    #testing

    # path = "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/gfp/laco_laci_ez_6_seg.npy"
    # mask = read_npy_file(path)
    # mask = mask['masks']

    path = [
        '/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/gfp',
        
        '/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/new_days/20190527/rpoc_ez/gfp',
        '/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Other_RPOC/_gfp',
        '/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/RPOC_new/_gfp',

        '/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/rpoc_M9/20190515/gfp',
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/12/rpoc_m9_2/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/12/rpoc_m9/gfp",

        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/15/rp_ez_hex5/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/15/rp_ez_hex5_2/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/16/rp_ez_hex5/gfp",

        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/20200215/ll_m9/gfp",
        
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/16/ll_hex5/gfp"

    ]

    # path = [
    #     "/Volumes/Baljyot_HD/SMT_Olympus/Fixed/20230920/MG16655_1_P_PFA_30_min/Sorted/BF",
    #     "/Volumes/Baljyot_HD/SMT_Olympus/Fixed/20230920/MG16655_3_P_glyaxol_30_min/Sorted/BF",
    #     "/Volumes/Baljyot_HD/SMT_Olympus/Fixed/20230920/MG16655_24_04_PFA_Glu_30_min/Sorted/BF",
    #     "/Volumes/Baljyot_HD/SMT_Olympus/Fixed/20230920/MG16655_Live/Sorted/BF"
    # ]

    for path in path:
        mask_dir = movie_mask_directory_structure_manager(path)




