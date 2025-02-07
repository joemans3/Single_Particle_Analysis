"""
Collection of structured data for utility. Mainly used within trajectory_analysis_script.py and scale_scale_plus_database.py
"""

####### FOLDER STRUCTURES ########
SEGMENTATION_FOLDER_TYPES = {
    "TRACKMATE": "Segmented",
    "Scale": "Segmented_mean",
    "Fitted": "Segmented_mean",
    "SCALE_SPACE_PLUS": "segmented_scale_space_plus",
    "DBSCAN": "segmented_scale_space_plus",
}

ANALYSIS_FOLDER_TYPES = {
    "TRACKMATE": "Analysis",
    "Scale": "Analysis",
    "Fitted": "Analysis",
    "SCALE_SPACE_PLUS": "Analysis",
    "DBSCAN": "Analysis_DBSCAN",
}

LOADING_DROP_BLOB_TYPES = {
    "TRACKMATE": True,
    "Scale": False,
    "Fitted": False,
    "SCALE_SPACE_PLUS": True,
    "DBSCAN": True,
}

####### FOLDER STRUCTURES END########
