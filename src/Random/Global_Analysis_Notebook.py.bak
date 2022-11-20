
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib as plt 

from trajectory_analysis_script import *

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, IntSlider
import ipywidgets as widgets
'''
frame_step = 1000 #change manual
frame_total = 5000 #change manual
#cutoff for track length
t_len_l = 10 #change manual #people use 5


t_len_u = 100 #change manual #100 #use 20
MSD_avg_threshold = 0.001
#upper and lower "bound proportion" threshold for determining if in_out track or out only.
upper_bp = 0.99
lower_bp = 0.50
max_track_decomp = 1.0
frames = int(frame_total/frame_step)
conversion_p_nm = 130.
minimum_tracks_per_drop = 3
'''
def define_variables(frame_step,frame_total,t_len_l,t_len_u,MSD_avg_threshold,upper_bp,lower_bp,max_track_decomp,
                     conversion_p_nm,minimum_tracks_per_drop):
    
    frame_step = frame_step #change manual
    frame_total = frame_total #change manual
    #cutoff for track length
    t_len_l = t_len_l #change manual #people use 5
    t_len_u = t_len_u #change manual #100 #use 20
    MSD_avg_threshold = MSD_avg_threshold
    #upper and lower "bound proportion" threshold for determining if in_out track or out only.
    upper_bp = upper_bp
    lower_bp = lower_bp
    max_track_decomp = max_track_decomp
    conversion_p_nm = conversion_p_nm
    minimum_tracks_per_drop = minimum_tracks_per_drop
    return

#first rpoc
rpoc_1 = run_analysis("/Users/baljyot/Desktop/Baljyot_EXP_RPOC/DATA","RPOC")
print(rpoc_1.frames)
#rpoc_1.read_parameters
interact_manual(rpoc_1.read_parameters,frame_step = FloatSlider(min=100,max=1000,step=100,value=1000),
                frame_total = FloatSlider(min=0,max=5000,step=1000,value=5000),
                t_len_l = FloatSlider(min=1,max=50,step=1,value=10),
                t_len_u = FloatSlider(min=20,max=100,step=10,value=20),
                MSD_avg_threshold = 0.001,
                upper_bp = FloatSlider(min=0,max=1,step=0.05,value=0.95),
                lower_bp = FloatSlider(min=0,max=1,step=0.05,value=0.50),
                max_track_decomp = 1.0, 
                conversion_p_nm = 130,
                minimum_tracks_per_drop = FloatSlider(min=0,max=10,step=1,value=3))


# In[ ]:

print(rpoc_1.wd)


# In[ ]:



