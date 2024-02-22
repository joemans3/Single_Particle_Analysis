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
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts/src')

import numpy as np
import os
import glob
import skimage.io as ioski
from cellpose import utils,io
from SMT.SMT_Analysis_BP.helpers.analysisFunctions.Analysis_functions import sorted_alphanumeric
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
        base_files = sorted_alphanumeric(glob.glob(os.path.join(self.full_path_to_mask_dir,'*.tif')))
        if len(base_files) != 0:
            _reorder_file_names(base_files)
        self._base_files = sorted_alphanumeric(glob.glob(os.path.join(self.full_path_to_mask_dir,'*.tif')))
        mask_files = sorted_alphanumeric(glob.glob(os.path.join(self.full_path_to_mask_dir,'*.npy')))
        #check if there are any mask files
        if len(mask_files) != 0:
            _reorder_file_names(mask_files,end_string='_seg')
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
    @property
    def base_files(self):
        return self._base_files
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

def _reorder_file_names(file_paths,end_string=None)->None:
    '''Reorders the names of the files to start from 1->end.
    Given a set of files in the directory, it will reorder them to start from 1->end
    so <name>_20.tif will be <name>_1.tif, <name>_21.tif will be <name>_2.tif and so on
    This is to rename the file 
    '''
    #sort the file paths
    file_paths = sorted_alphanumeric(file_paths)
    #get the string common to all the files for the base name
    base_name = os.path.basename(file_paths[0])
    if end_string is not None:
        base_name = base_name[:base_name.rfind(end_string)]
    #find the text before the last "_"
    base_name = base_name[:base_name.rfind("_")]

    #rename the files
    for i,file_path in enumerate(file_paths):
        #get the file name
        file_name = os.path.basename(file_path)
        #get the file extension
        file_extension = os.path.splitext(file_name)[1]
        #rename the file
        if end_string is None:
            os.rename(file_path,os.path.join(os.path.dirname(file_path),base_name+"_"+str(i+1)+file_extension))
        else:
            os.rename(file_path,os.path.join(os.path.dirname(file_path),base_name+"_"+str(i+1)+end_string+file_extension))




    


if __name__ == '__main__':
    ##############################################################################################################
    #testing

    # path = "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/new_days/20190527/ll_ez/gfp/laco_laci_ez_6_seg.npy"
    # mask = read_npy_file(path)
    # mask = mask['masks']

    # path = [
    #     "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/16/ll_m9/gfp"


    # ]
    # for path in path:
    #     mask_dir = movie_mask_directory_structure_manager(path)


    path = [
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231015/ll_ez/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/ll_hex5_fixed/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_ez_fixed/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_ez_fixed_2/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_hex5_fixed/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_hex5_fixed_2/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_m9_fixed/TS/gfp",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_rif_fixed/TS/gfp"
        ]
    for path in path:
        mask_dir = movie_mask_directory_structure_manager(path)
