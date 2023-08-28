'''
Collection of structured data for utility. Mainly used within trajectory_analysis_script.py and scale_scale_plus_database.py
'''
import os
import sys

####### FOLDER STRUCTURES ########
SEGMENTATION_FOLDER_TYPES = {
    "TRACKMATE": "Segmented",
    "Scale": "Segmented_mean",
    "Fitted": "Segmented_mean",
    "SCALE_SPACE_PLUS": "segmented_scale_space_plus",
    "DBSCAN": "segmented_scale_space_plus"
}

ANALYSIS_FOLDER_TYPES = {
    "TRACKMATE": "Analysis",
    "Scale": "Analysis",
    "Fitted": "Analysis",
    "SCALE_SPACE_PLUS": "Analysis",
    "DBSCAN": "Analysis_DBSCAN"
}
####### FOLDER STRUCTURES END########

