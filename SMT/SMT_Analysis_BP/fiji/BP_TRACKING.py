import sys
from ij import IJ
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import (
    SparseLAPTrackerFactory,
)  # trackmate developer change it from time to time, if you update find the current
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer
from fiji.plugin.trackmate.gui.displaysettings.DisplaySettings import TrackMateObject
from fiji.plugin.trackmate.gui.displaysettings.DisplaySettings import TrackDisplayMode
import csv
import os
import time
import glob
import re

# 2.0, 100 for 100ms (1.5, 60)
# smt (4,10)
# Path for Directory with Movies
DIRECTORY_MOVIES = "/Volumes/Baljyot_HD/SMT_Olympus/RIF_TREATMET_LIVE/20230528/Movie"

MOVIE_BASE_NAME = "nusa_rif"

# Name of subdirectory in DIRECTORY_MOVIES where the results will be saved
DIRECTORY_SAVE = "Analysis_new"

# define some global variables, this is in the spatial units of the image (pixel)
LINKING_PARAMETERS = {
    "LINKING_MAX_DISTANCE": 5.0,
    "GAP_CLOSING_MAX_DISTANCE": 5.0,
    "SPLITTING_MAX_DISTANCE": 0.0,
    "MERGING_MAX_DISTANCE": 0.0,
    "MAX_FRAME_GAP": 0,
    "ALLOW_TRACK_SPLITTING": False,
    "ALLOW_TRACK_MERGING": False,
}
LOCALIZATION_PARAMETERS = {
    "DETECTOR_FACTORY": LogDetectorFactory(),
    "DO_SUBPIXEL_LOCALIZATION": True,
    "RADIUS": 2.0,
    "TARGET_CHANNEL": 1,
    "THRESHOLD": 10.0,
    "DO_MEDIAN_FILTERING": True,
}
FILTERING_PARAMETERS = {
    "QUALITY": None,
    "MIN_NUMBER_SPOTS_IN_TRACK": None,
    "MAX_NUMBER_SPOTS_IN_TRACK": None,
}
"""LABEL	ID	TRACK_ID	QUALITY	POSITION_X	POSITION_Y	POSITION_Z	POSITION_T	FRAME	
RADIUS	VISIBILITY	MANUAL_SPOT_COLOR	MEAN_INTENSITY_CH1	MEDIAN_INTENSITY_CH1	MIN_INTENSITY_CH1	
MAX_INTENSITY_CH1	TOTAL_INTENSITY_CH1	STD_INTENSITY_CH1	CONTRAST_CH1	SNR_CH1"""
OUTPUT_COLUMNS = {
    "LABEL": True,
    "ID": True,
    "TRACK_ID": True,
    "QUALITY": True,
    "POSITION_X": True,
    "POSITION_Y": True,
    "POSITION_Z": True,
    "POSITION_T": True,
    "FRAME": True,
    "RADIUS": True,
    "VISIBILITY": True,
    "MANUAL_SPOT_COLOR": True,
    "MEAN_INTENSITY_CH1": True,
    "MEDIAN_INTENSITY_CH1": True,
    "MIN_INTENSITY_CH1": True,
    "MAX_INTENSITY_CH1": True,
    "TOTAL_INTENSITY_CH1": True,
    "STD_INTENSITY_CH1": True,
    "CONTRAST_CH1": True,
    "SNR_CH1": True,
}
# column order
OUTPUT_COLUMNS_ORDER = [
    "LABEL",
    "ID",
    "TRACK_ID",
    "QUALITY",
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "POSITION_T",
    "FRAME",
    "RADIUS",
    "VISIBILITY",
    "MANUAL_SPOT_COLOR",
    "MEAN_INTENSITY_CH1",
    "MEDIAN_INTENSITY_CH1",
    "MIN_INTENSITY_CH1",
    "MAX_INTENSITY_CH1",
    "TOTAL_INTENSITY_CH1",
    "STD_INTENSITY_CH1",
    "CONTRAST_CH1",
    "SNR_CH1",
]

PAD_ROWS = 3


# We have to do the following to avoid errors with UTF8 chars generated in
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding("utf-8")


# Get currently selected image
# imp = WindowManager.getCurrentImage()

#########################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################


def main():
    dir_containing_tif_movies = DIRECTORY_MOVIES
    file_tif_paths_in_dir = sorted_alphanumeric(
        glob.glob(os.path.join(dir_containing_tif_movies, "*.tif"))
    )

    # check if the directory exists
    if not os.path.exists(dir_containing_tif_movies):
        sys.exit("Directory does not exist: {0}".format(dir_containing_tif_movies))

    print("Working in directory: {0}".format(dir_containing_tif_movies))
    # check if the subdirectory exists
    dir_save = os.path.join(dir_containing_tif_movies, DIRECTORY_SAVE)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
        print("Created directory: {0}".format(dir_save))

    for i in range(len(file_tif_paths_in_dir)):
        file_path_full = file_tif_paths_in_dir[i]
        tracking_plus_save(dir_containing_tif_movies, dir_save, file_path_full, i + 1)
    return 0


#########################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################


def tracking_plus_save(directory_path, dir_save, file_path, file_counter):
    # find the file name without the full path
    file_name_original = os.path.basename(file_path)
    # remove the extension
    file_name_original = os.path.splitext(file_name_original)[0]
    # print that you are working on this file
    print(
        "Working on file: {0} in directory: {1}".format(
            file_name_original, directory_path
        )
    )

    file_name = file_name_original[: len(MOVIE_BASE_NAME) + 1] + "{0}".format(
        file_counter
    )
    # old name and new name print
    print("Old name: {0}".format(file_name_original))
    print("New name: {0}".format(file_name))

    # create a path for the analysis file
    file_name_save = os.path.join(dir_save, file_name + "_seg.tif_spots.csv")

    imp = IJ.openImage(file_path)
    dims = imp.getDimensions()  # default order: XYCZT
    if dims[4] == 1:
        imp.setDimensions(*[dims[2], dims[4], dims[3]])
    # imp.show()
    #########################################################################################################################################################################################################################################################
    #########################################################################################################################################################################################################################################################
    #########################################################################################################################################################################################################################################################

    # -------------------------
    # Instantiate model object
    # -------------------------

    model = Model()

    # Set logger
    model.setLogger(Logger.IJ_LOGGER)

    # ------------------------
    # Prepare settings object
    # ------------------------

    settings = Settings(imp)
    # Configure detector
    settings.detectorFactory = LOCALIZATION_PARAMETERS["DETECTOR_FACTORY"]
    settings.detectorSettings = {
        "DO_SUBPIXEL_LOCALIZATION": LOCALIZATION_PARAMETERS["DO_SUBPIXEL_LOCALIZATION"],
        "RADIUS": LOCALIZATION_PARAMETERS["RADIUS"],
        "TARGET_CHANNEL": LOCALIZATION_PARAMETERS["TARGET_CHANNEL"],
        "THRESHOLD": LOCALIZATION_PARAMETERS["THRESHOLD"],
        "DO_MEDIAN_FILTERING": LOCALIZATION_PARAMETERS["DO_MEDIAN_FILTERING"],
    }

    # Configure tracker
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    settings.trackerSettings["LINKING_MAX_DISTANCE"] = LINKING_PARAMETERS[
        "LINKING_MAX_DISTANCE"
    ]
    settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = LINKING_PARAMETERS[
        "GAP_CLOSING_MAX_DISTANCE"
    ]
    settings.trackerSettings["SPLITTING_MAX_DISTANCE"] = LINKING_PARAMETERS[
        "SPLITTING_MAX_DISTANCE"
    ]
    settings.trackerSettings["MERGING_MAX_DISTANCE"] = LINKING_PARAMETERS[
        "MERGING_MAX_DISTANCE"
    ]
    settings.trackerSettings["MAX_FRAME_GAP"] = LINKING_PARAMETERS["MAX_FRAME_GAP"]
    settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = LINKING_PARAMETERS[
        "ALLOW_TRACK_SPLITTING"
    ]
    settings.trackerSettings["ALLOW_TRACK_MERGING"] = LINKING_PARAMETERS[
        "ALLOW_TRACK_MERGING"
    ]

    # Add the analyzers for some spot features.
    # Here we decide brutally to add all of them.
    settings.addAllAnalyzers()

    if FILTERING_PARAMETERS["QUALITY"] is not None:
        settings.addSpotFilter("QUALITY", FILTERING_PARAMETERS["QUALITY"], True)
    if FILTERING_PARAMETERS["MIN_NUMBER_SPOTS_IN_TRACK"] is not None:
        settings.addTrackFilter(
            "NUMBER_SPOTS", FILTERING_PARAMETERS["MIN_NUMBER_SPOTS_IN_TRACK"], True
        )
    if FILTERING_PARAMETERS["MAX_NUMBER_SPOTS_IN_TRACK"] is not None:
        settings.addTrackFilter(
            "NUMBER_SPOTS", FILTERING_PARAMETERS["MAX_NUMBER_SPOTS_IN_TRACK"], False
        )

    # print(str(settings))

    # ----------------------
    # Instantiate trackmate
    # ----------------------

    trackmate = TrackMate(model, settings)
    trackmate.getModel().getLogger().log(settings.toStringFeatureAnalyzersInfo())
    trackmate.computeSpotFeatures(True)
    trackmate.computeEdgeFeatures(True)
    trackmate.computeTrackFeatures(True)
    # ------------
    # Execute all checks
    # ------------

    ok = trackmate.checkInput()
    if not ok:
        print("sysErrorInput")
        return
    ok = trackmate.process()
    if not ok:
        print("sysErrorProcess")
        return

    # Read the default display settings.
    ds = DisplaySettingsIO.readUserDefault()

    # With the line below, we state that we want to color tracks using
    # a numerical feature defined for TRACKS, and that has they key 'TRACK_DURATION'.
    ds.setTrackColorBy(TrackMateObject.TRACKS, "TRACK_DURATION")

    # With the line below, we state that we want to color spots using
    # a numerical feature defined for SPOTS, and that has they key 'QUALITY'.
    ds.setSpotColorBy(TrackMateObject.SPOTS, "QUALITY")

    # Now we want to display tracks as comets or 'dragon tails'. That is:
    # tracks should fade in time.
    ds.setTrackDisplayMode(TrackDisplayMode.LOCAL_BACKWARD)
    # ----------------
    # Display results
    # ----------------
    selectionModel = SelectionModel(model)

    displayer = HyperStackDisplayer(model, selectionModel, imp, ds)
    # ----------------------
    # Display the displayer
    # ----------------------
    displayer.render()
    displayer.refresh()

    spt_m = model.getSpots()
    tracks_found = model.getTrackModel().trackIDs(True)
    tot_spts = spt_m.getNSpots(True)
    print(model)
    print("Spot total: ", spt_m.getNSpots(True))
    print("Tracks total: ", len(model.getTrackModel().trackIDs(True)))

    if not (tot_spts > 0):
        print("No spots found")
        return
    if not tracks_found:
        print("No Tracks found")
        return

    # # The feature model, that stores edge and track features.
    fm = model.getFeatureModel()

    # make the header row
    header_row = [header_name for header_name in OUTPUT_COLUMNS_ORDER]
    collection_spots_per_track = [header_row]

    # add the header row again given the constant pad_row
    for pad_rows in range(PAD_ROWS):
        collection_spots_per_track.append(header_row)

    # Iterate over all the tracks that are visible.
    for id in model.getTrackModel().trackIDs(True):
        # Get all the spots of the current track.
        track = model.getTrackModel().trackSpots(id)
        spot_counter = 0
        for spot in track:
            sid = spot.ID()
            # Fetch spot features directly from spot.
            spot_label = spot_counter
            spot_ID = sid
            spot_counter += 1
            track_ID = id
            spot_quality = spot.getFeature("QUALITY")
            spot_x = spot.getFeature("POSITION_X")
            spot_y = spot.getFeature("POSITION_Y")
            spot_z = spot.getFeature("POSITION_Z")
            spot_t = spot.getFeature("POSITION_T")
            spot_frame = spot.getFeature("FRAME")
            spot_radius = spot.getFeature("RADIUS")
            spot_visibility = spot.getFeature("VISIBILITY")
            spot_manual_spot_color = spot.getFeature("MANUAL_SPOT_COLOR")
            spot_mean_intensity_ch1 = spot.getFeature("MEAN_INTENSITY_CH1")
            spot_median_intensity_ch1 = spot.getFeature("MEDIAN_INTENSITY_CH1")
            spot_min_intensity_ch1 = spot.getFeature("MIN_INTENSITY_CH1")
            spot_max_intensity_ch1 = spot.getFeature("MAX_INTENSITY_CH1")
            spot_total_intensity_ch1 = spot.getFeature("TOTAL_INTENSITY_CH1")
            spot_std_intensity_ch1 = spot.getFeature("STD_INTENSITY_CH1")
            spot_contrast_ch1 = spot.getFeature("CONTRAST_CH1")
            spot_snr_ch1 = spot.getFeature("SNR_CH1")
            # add to the collection
            # add in the order of the output columns
            collection_spots_per_track.append(
                [
                    spot_label,
                    spot_ID,
                    track_ID,
                    spot_quality,
                    spot_x,
                    spot_y,
                    spot_z,
                    spot_t,
                    spot_frame,
                    spot_radius,
                    spot_visibility,
                    spot_manual_spot_color,
                    spot_mean_intensity_ch1,
                    spot_median_intensity_ch1,
                    spot_min_intensity_ch1,
                    spot_max_intensity_ch1,
                    spot_total_intensity_ch1,
                    spot_std_intensity_ch1,
                    spot_contrast_ch1,
                    spot_snr_ch1,
                ]
            )

    with open(file_name_save, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(collection_spots_per_track)
    IJ.log("Success")
    imp.close()


def sorted_alphanumeric(data):
    # Function to convert text to int if text is a digit, else convert to lowercase
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # Function to split the text into a list of digits and non-digits
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    # Sort the list of data using the alphanum_key function
    return sorted(data, key=alphanum_key, reverse=False)


if True:
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
