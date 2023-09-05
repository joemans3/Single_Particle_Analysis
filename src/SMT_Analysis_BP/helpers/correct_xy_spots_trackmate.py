'''
This is a script to correct the output of trackmate which sometimes outputs x,y values in um in the *.tif_spots.csv spots files in the Analysis_new directory for some datasets.

The analysis pipeline assumes it is in pixels and so this script converts the x,y values to pixels. using the defined pixel size in the analysis pipeline.



'''
import pandas as pd
import os
import sys
import numpy as np


paths_to_correct = [
    "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/12/rpoc_m9_2/Analysis_new"
]

pixel_size = 0.13 #in um

def correct_csv(csv_path):
    #the first 4 lines are the header with the first line as the column names
    #read the csv using the first line as the column names skip lines number 2,3,4
    df = pd.read_csv(csv_path,header=0)
    #skip the first 4 lines
    df_cor = df.iloc[4:]
    #convert the x,y values to pixels
    df_cor["POSITION_X"] = df_cor["POSITION_X"]/pixel_size
    df_cor["POSITION_Y"] = df_cor["POSITION_Y"]/pixel_size
    print(df_cor.head())

def find_csvs(path,endswith):
    csvs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(endswith):
                csvs.append(os.path.join(root, file))
    return csvs


if __name__ == "__main__":
    for path in paths_to_correct:
        csvs = find_csvs(path,".tif_spots.csv")
        for csv in csvs:
            correct_csv(csv)