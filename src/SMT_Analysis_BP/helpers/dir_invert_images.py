'''
This script takes input at runtime to a directory where images are kept to invert them and save them in a new directory

Author: Baljyot Singh Parmar
'''
import os
import numpy as np 
from PIL import Image
import PIL.ImageOps
import sys
import glob as glob

if __name__ == "__main__":
    sys.path.append('/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts')
    try:
        dir_path = str(sys.argv[1])
        if not isinstance(dir_path,str):
            dir_path = str(dir_path)

    except:
        dir_path = input("Full path of directory holding files to invert: ")
        if not isinstance(dir_path,str):
            dir_path = str(dir_path)

import src.SMT_Analysis_BP.helpers.import_functions as import_functions


def save_invert_images(path,**kwargs):
    '''
    Makes directory to store inverted images and put files there
    '''

    #make new directory inside the provided dir_path to store the inverted images
    invert_path = kwargs.get("invert_path",os.path.join(path,'Inverted_Images'))
    if not os.path.exists(invert_path):
        os.makedirs(invert_path)
    image_paths = import_functions.find_image(path,ends_with = '.tif',full_path = True)
    inverted_imgs = list(map(import_functions.invert_img,image_paths))
    
    
    image_paths = import_functions.find_image(path,ends_with = '.tif',full_path = False)

    for i in range(len(inverted_imgs)):
        new_path = import_functions.combine_path(invert_path,image_paths[i])
        inverted_imgs[i].save(new_path)
    return 

if __name__ == "__main__":
    if dir_path == "":
        dir_path = [
            "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/16/ll_hex5/gfp",
            "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/15/rp_ez_hex5/gfp",
            "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/15/rp_ez_hex52/gfp",
            "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/12/rpoc_m9/gfp",
            "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/12/rpoc_m9_2/gfp"
        ]
        for i in dir_path:
            save_invert_images(i)
    else:
        save_invert_images(dir_path)
    


