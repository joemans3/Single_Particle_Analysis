import numpy as np
import glob as glob

#make a function to take track data in a dict format dict = {track_id:[[x0,y0,frame0],[x1,y1,frame,1],...],...} and convert it to the format required for SMAUG analysis
#format for SMAUG analysis is [track_id,time_step_number,placeholder,x,y] see https://github.com/BiteenMatlab/SMAUG
#the time_step_number is the consecutive frame number starting from 0

def convert_track_data_SMAUG_format(track_data:dict)->list:
    #lets first get the number of tracks
    num_tracks = len(track_data.keys())
    #now we create a placeholder list to store the data
    data_convert = []
    track_IDS = 1
    #now we loop through the tracks
    for track_id in track_data.keys():
        #we get the data for the track
        track = track_data[track_id]
        #we loop through the track and append the data to the placeholder list
        for i in range(len(track)):
            data_convert.append([track_IDS,i+1,i+1,track[i][0],track[i][1]])
        #we increment the track id
        track_IDS += 1
    #now we return the data
    return data_convert




