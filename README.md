

#This is a short doc for the SMT project.
-----------------------------------------

Author: Baljyot Singh Parmar
Affiliation at the time of writing: McGill University, Canada. Weber Lab



## 1. Installation
-------------------
1. Make sure you have anaconda installed: <https://www.anaconda.com/download>
2. Download or clone this repository.
3. In the conda prompt, navigate to the folder where you downloaded this repository using : **cd "path_to_folder"**
4. Using the **SMT_env_BP.yml** file, create a new environment using: **conda env create -f SMT_env.yml**
5. Activate the environment using: **conda activate SMT_env_BP**
6. Install the extra pip packages using: **pip install -r requirements.txt**
7. Since tensflow and tensorflow-probability are platform dependent we need to install inidividually.
    - Try the conda install method: **conda install tensorflow**, and **conda install tensorflow-probability**
    - If the above method fails, try the pip install method: **pip install tensorflow**, and **pip install tensorflow-probability**

8. TODO add a test script to check if the installation is successful.

9. You need to download the latest stable release of MATLAB. Use McGill account.
10. Install the bioformats package for MATLAB and follow their install instructions to set the environment variables: <https://docs.openmicroscopy.org>. 
11. Install the most stable release of Cellpose for python <https://github.com/MouseLand/cellpose/tree/main/README.md>. I would recommend setting up another conda environment for only Cellpose so that it doesn't mess up the SMT_env_BP.  

## 2. Preparing the data
----------------------------

### 2.1. Converting Olympus .vsi files to .tif files
-----------------------------------------------------
- The Olympus .vsi files are not supported by the python libraries. So we need to convert them to .tif files. (this is not entirely true, we can use the python-bioformats library to read the .vsi files, but it is very slow and the API is not very user friendly) To do this we will use the Bioformats tool in Matlab. 
    - Download the Bioformats tool from: <https://www.openmicroscopy.org/bio-formats/downloads/>
    - Install the tool in Matlab using the instructions given in the link above. (you will need to set the environment variable in Matlab to access the bioformats library, or manually add path every time.)
    - Open the **src/SMT_Analysis/Matlab/batch_convert_single_fluorescence_BW.m** file in Matlab and edit the following lines:
        - **line 2**: change the initial file to the first .vsi file in the folder.
        - **line 3**: Repeat but now replace the numbering term with "\*" which will be used as a wildcard for finding all files with the basename which is not "\*".
        - **line 19**: Same as in line 3 but now replace with "\*" with "%d" which will be used to name the output files. 
    - Please note that this takes a long time and scales $O(n^{2})$ with the number of frames in a single file, and $mO(n^{2})$ with the number of files in the folder. Go drink coffee or find a better way to do this. 

### 2.2. Pre-Processing the data
---------------------------------
#### 2.2.1 Extracting Movies, GFP and BF files
- Depending on how you did your acquisition the files may be different. For SMT in my method you will generally have 4 files per frame of view.

    - **BF**: This will likely be the first image.
    - **random_file**: This will likely be the second image.
    - **Movie**: This will likely be the third image (.tif).
    - **GFP**: This will likely be the fourth image.

- Make 3 folders to store the files in. **bf**, **gfp**, **Movie**, and copy the files to the respective folders. 
- **DO NOT MOVE THE ACTUAL DATA. ONLY COPY IT! This applies for all the processing steps later so that we don't overwrite something on accident.**
    - Disk space is cheap, time (and your sanity) is not.

#### 2.2.2. Extracting the ROI for Cells in a Movie
- I prefer to use SuperSegger or CellPose to extract the ROI. The issue with SuperSegger is that it is written in Matlab and is not very extendable and does not allow manual adjustments. CellPose is written in python and is very extendable and allows manual adjustments. So ill just focus on CellPose here. However, the analysis code has API for reading formatting outputs from either one.
- Download the Cellpose for python and use the GUI to save .npy files for the mask using the gfp images (or the bf).
- Now that you have a folder with the .npy files run the script: **src/SMT_Analysis_BP/databases/format_mask_structure.py** 
    - You need to tell it which path the .npy files are in and it will save the masks and cells in a new folder called **Movies** in the parent directory.
    - If running as a script, change the path variable to your path after the "if \_\_name\_\_ == '\_\_main\_\_':" line. Else interface with the API if calling from another script.

#### 2.2.3. Extracting the Time Average Projections.
- Run the file **src/SMT_Analysis_BP/helpers/tifanal.py** with the directory path changed to the **Movie** path. This will create a folder called **Segmented_mean** in the directory. Move this folder to the main parent path, such that the folder is in the same scope as the **Movies** folder.

#### 2.2.4. Performing the Tracking.
- You will need TrackMate plugin in ImageJ/FIJI for this.
- I would use the default settings in the file **src/SMT_Analysis_BP/fiji/BP_TRACKING.py**. Change the directory path (DIRECTORY_MOVIES) to the **Movies** folder and change the value for the BASE_MOVIE_NAME to the base name of the set of movies in Movies. (this is the string that does not include the changing number in the file names).
- You should open this script in FIJI/ImageJ unless you have told your interpreter to use the FIJI executable.
- Run the script; it will create a folder called **Analysis_new**. Move this folder to the parent directory so that it is in the same scope as the **Movies** folder.

##### Final Directory Structure
Your directory for the dataset should look something like this at the very least:

```
.
├── Analysis_new
├── Movies
├── Segmented_mean
├── gfp
```
Not having this minimal structure will limit the functionality of the API. You can add more folders to this structure but the API will not be able to read them unless explicitly told to do so.






        




