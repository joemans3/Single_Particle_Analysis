import numpy as np
import pylab as py
import os
import sys
import glob
from skimage import io
import tkinter as tk
import tkinter.filedialog as fd


SEGMENT_TYPE = {
    "mean":np.mean,
    "median":np.median,
    "max":np.max,
    "min":np.min,
    "sum":np.sum,
    "std":np.std
}
STACK_STEP = 5
CONVERT_NAME = 1
DIR_PATH = "/Volumes/Baljyot_HD/SMT_Olympus/MG16655_Controls/MG16655_AutoFluo_Controls/Live/20230926/MG16655_hex5_5min_Live/Movie"


#check if the directory exists
if not os.path.exists(DIR_PATH):
    print("The directory does not exist")
    sys.exit()
#set the root directory to DIR_PATH
root = tk.Tk()
filez = fd.askopenfilenames(parent=root, title='Choose a file', initialdir=DIR_PATH)
#check if the user has chosen any files
if len(filez) == 0:
    print("No files were chosen")
    sys.exit()
#close the root window
root.destroy()

#get the root directory of the chosen files
root_dir = os.path.dirname(filez[0])

#make a directory named Segemented_mean in the save_dir
if not os.path.exists(os.path.join(root_dir,"Segmented_mean")):
    os.makedirs(os.path.join(root_dir,"Segmented_mean"))

#sort the files based on the number right before the extension
sorted_a_files = sorted(filez,key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))

for i in range(len(sorted_a_files)):
    temp_im = io.imread(sorted_a_files[i])
    temp_del = temp_im[:]

    length = int(len(temp_del)/STACK_STEP)

    hold_img = []
    hold_name = []
    for j in np.arange(STACK_STEP):
        movie_name = sorted_a_files[i].split("/")[-1].split(".")[0]
        #change the name of the file to be the sorted index starting at 1, and add "_seg" before the extension
        if CONVERT_NAME:
            updated_movie_name = movie_name.split("_")[:-1]
            updated_movie_name.append(str(int(i+1)))
            updated_movie_name = "_".join(updated_movie_name)
        else:
            updated_movie_name = movie_name.split("_")[:-1]
            updated_movie_name.append(str(int(movie_name.split("_")[-1])+1))
            updated_movie_name = "_".join(updated_movie_name)

        hold_img.append(SEGMENT_TYPE["mean"](temp_del[int(j*length):int((j+1)*length)],axis=0))
        hold_name.append(os.path.join(root_dir,"Segmented_mean",str(int(j+1)) + "_" + updated_movie_name + "_seg.tif"))

    for k in range(len(hold_img)):
        io.imsave(hold_name[k],np.array(hold_img[k],dtype = "uint16"))




