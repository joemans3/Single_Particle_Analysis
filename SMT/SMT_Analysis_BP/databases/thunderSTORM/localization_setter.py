import glob
import os
import pandas as pd
from skimage.io import imread

if __name__ == "__main__":
    import sys

    sys.path.append(
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/Scripts/src"
    )

from SMT.SMT_Analysis_BP.helpers.analysisFunctions.Analysis_functions import (
    sorted_alphanumeric,
)
from SMT.SMT_Analysis_BP.helpers.analysisFunctions.features_from_mask import (
    extract_mask_properties,
)
from shapely.geometry import Point, Polygon

RELATIVE_ANALYSIS_FOLDER = "TS_Analysis"
BOUNDING_BOX_EXTRA_BORDER = 2


class path_structure:
    def __init__(self, path):
        self.cd = path

    @property
    def path(self):
        return self.cd

    @property
    def path_structure_dict(self):
        if hasattr(self, "_path_structure_dict"):
            return self._path_structure_dict
        else:
            structured_dict_paths = {}
            # for each key is the movie ID which has a dict of cell IDs, and path (of the movie ID) as the value
            # for each cell ID there is a dict of the localizations and mask paths

            # get the movie directories
            movie_dirs = glob.glob(os.path.join(self.cd, "Movies", "Movie_*"))
            for movie_dir in movie_dirs:
                # get the movie ID as the * in Movie_*
                movie_ID = os.path.basename(movie_dir).split("_")[1]
                # get the cell directories
                cell_dirs = glob.glob(os.path.join(movie_dir, "Cell_*"))
                # initialize the dict for the movie ID
                structured_dict_paths[movie_ID] = {"path": movie_dir, "cells": {}}
                for cell_dir in cell_dirs:
                    # get the cell ID as the * in Movie_<number>_cell_*
                    cell_ID = os.path.basename(cell_dir).split("_")[-1]
                    # initialize the dict for the cell ID
                    structured_dict_paths[movie_ID]["cells"][cell_ID] = {
                        "path": cell_dir
                    }
                    # get the localizations and mask paths
                    try:
                        localizations_path = glob.glob(
                            os.path.join(cell_dir, "localizations.csv")
                        )[0]
                        # if exists then remove the file
                        os.remove(localizations_path)
                    except IndexError:
                        # since it doesn't exist we can store a name and path for it to make later
                        localizations_path = os.path.join(cell_dir, "localizations.csv")
                    mask_path = glob.glob(os.path.join(cell_dir, "mask.tif"))[0]
                    # add the localizations and mask paths to the dict
                    structured_dict_paths[movie_ID]["cells"][cell_ID][
                        "localizations_path"
                    ] = localizations_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["mask_path"] = (
                        mask_path
                    )

                    # now make the new directories and file paths (even though we will not save the files yet)
                    # make the Analysis directory
                    analysis_dir = os.path.join(cell_dir, "Analysis")
                    if not os.path.isdir(analysis_dir):
                        os.mkdir(analysis_dir)
                    # make the new files as shown in the docstring
                    reconstruction_path = os.path.join(cell_dir, "reconstruction.tif")
                    uniform_reconstruction_path = os.path.join(
                        cell_dir, "uniform_reconstruction.tif"
                    )
                    normal_scale_projection_path = os.path.join(
                        cell_dir, "normal_scale_projection.tif"
                    )
                    reconstruction_parameters_path = os.path.join(
                        cell_dir, "reconstruction_parameters.json"
                    )
                    scale_space_plus_blob_fitted_path = os.path.join(
                        analysis_dir, "scale_space_plus_blob_fitted.csv"
                    )
                    scale_space_plus_blob_scale_path = os.path.join(
                        analysis_dir, "scale_space_plus_blob_scale.csv"
                    )
                    DBSCAN_clusters_path = os.path.join(
                        analysis_dir, "DBSCAN_clusters.csv"
                    )
                    analysis_parameters_path = os.path.join(
                        analysis_dir, "analysis_parameters.json"
                    )
                    # add the new file paths to the dict in the proper place with the name as the key and the full path as the value
                    structured_dict_paths[movie_ID]["cells"][cell_ID][
                        "reconstruction_path"
                    ] = reconstruction_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID][
                        "uniform_reconstruction_path"
                    ] = uniform_reconstruction_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID][
                        "normal_scale_projection_path"
                    ] = normal_scale_projection_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID][
                        "reconstruction_parameters_path"
                    ] = reconstruction_parameters_path
                    structured_dict_paths[movie_ID]["cells"][cell_ID]["Analysis"] = {
                        "scale_space_plus_blob_fitted_path": scale_space_plus_blob_fitted_path,
                        "scale_space_plus_blob_scale_path": scale_space_plus_blob_scale_path,
                        "DBSCAN_clusters_path": DBSCAN_clusters_path,
                        "analysis_parameters_path": analysis_parameters_path,
                    }
            self._path_structure_dict = structured_dict_paths
            return self._path_structure_dict


class mask_localization:
    def __init__(self, path):
        self.cd = path
        # make the path structure object
        self.path_structure = path_structure(self.cd).path_structure_dict

    def make_loc(self):
        # run over the movie IDs
        for movie_ID in self.path_structure.keys():
            # get the analysis file path for this movie ID
            movie_analysis_path = self.analysis_loc_files[int(movie_ID) - 1]
            # read the analysis file with the column names in the first row
            movie_analysis_main = pd.read_csv(movie_analysis_path, header=0)
            # make a copy of the main analysis file
            movie_analysis = movie_analysis_main.copy()
            # convert the y [nm], x [nm] sigma [nm] to y [px], x [px], sigma [px] with the conversion factor of 130 nm/px
            movie_analysis["y [px]"] = movie_analysis["y [nm]"] / 130.0
            movie_analysis["x [px]"] = movie_analysis["x [nm]"] / 130.0
            movie_analysis["sigma [px]"] = movie_analysis["sigma [nm]"] / 130.0
            # keep the header names
            header_names = movie_analysis.columns
            # get the cell IDs for this movie ID
            for cell_ID in self.path_structure[movie_ID]["cells"].keys():
                # get the mask image path for this cell ID
                mask_path = self.path_structure[movie_ID]["cells"][cell_ID]["mask_path"]
                # read the mask image
                mask = imread(mask_path)
                # make all values >1 to 1
                mask[mask > 1] = 1
                # get the features from the mask
                mask_features = extract_mask_properties(mask, invert_axis=True)
                # get the bounding box
                bounding_box = mask_features.bounding_box
                # make the polygon from the bounding box
                poly_cord = [
                    (bounding_box[0][0], bounding_box[0][1]),
                    (bounding_box[0][0], bounding_box[1][1]),
                    (bounding_box[1][0], bounding_box[1][1]),
                    (bounding_box[1][0], bounding_box[0][1]),
                ]
                poly = Polygon(poly_cord)
                # get the coordinates of the localizations
                coordinates = movie_analysis[["x [px]", "y [px]"]].values
                # make the points from the coordinates
                points = [Point(xy) for xy in coordinates]
                # make a list of the localizations in the mask
                localizations_in_mask = []
                # run over the points
                for point_id in range(len(points)):
                    # check if the point is in the polygon
                    if points[point_id].within(poly):
                        # add the point and the data for that point in the main analysis file to the list
                        localizations_in_mask.append(movie_analysis.iloc[point_id])
                # make a dataframe from the list of localizations in the mask
                localizations_in_mask = pd.DataFrame(
                    localizations_in_mask, columns=header_names
                )
                # save the localizations in the mask to the cell directory in the localizations.csv file
                localizations_in_mask.to_csv(
                    self.path_structure[movie_ID]["cells"][cell_ID][
                        "localizations_path"
                    ],
                    index=False,
                )

    @property
    def analysis_folder(self):
        return RELATIVE_ANALYSIS_FOLDER

    @property
    def analysis_loc_files(self):
        if hasattr(self, "_analysis_loc_files"):
            return self._analysis_loc_files
        else:
            # find all files in the analysis folder with .csv extension
            analysis_loc_files = glob.glob(
                os.path.join(self.cd, self.analysis_folder, "*.csv")
            )
            # sort the files
            analysis_loc_files = sorted_alphanumeric(analysis_loc_files)
            self._analysis_loc_files = analysis_loc_files
            return self._analysis_loc_files


if __name__ == "__main__":
    paths = [
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_rif_fixed/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_m9_fixed/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_hex5_fixed_2/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_hex5_fixed/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_ez_fixed_2/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/rp_ez_fixed/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231017/ll_hex5_fixed/TS",
        "/Users/baljyot/Documents/CODE/GitHub_t2/Baljyot_EXP_RPOC/DATA/Fixed_100ms/20231015/ll_ez/TS",
    ]
    for path in paths:
        ml = mask_localization(path)
        ml.make_loc()
