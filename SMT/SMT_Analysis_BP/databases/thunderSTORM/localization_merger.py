"""
ThunderSTORM localization merger  TODO THIS DOES NOT WORK YET
=============================================================

This module contains the function to merge localizations from ThunderSTORM .csv file.
The structure of the .csv file is the following:
id,	frame,	x [nm],	y [nm],	sigma [nm],	intensity [photon],	offset [photon],	bkgstd [photon],	chi2,	uncertainty [nm]

id: the id of the localization
frame: the frame number
x [nm]: the x coordinate of the localization in nanometers
y [nm]: the y coordinate of the localization in nanometers
sigma [nm]: the sigma of the localization in nanometers
intensity [photon]: the intensity of the localization in photon
offset [photon]: the offset of the localization in photon
bkgstd [photon]: the background standard deviation of the localization in photon
chi2: the chi2 of the localization
uncertainty [nm]: the uncertainty of the localization in nanometers

"""

COL_NAMES_THUNDERSTORM = [
    "id",
    "frame",
    "x [nm]",
    "y [nm]",
    "sigma [nm]",
    "intensity [photon]",
    "offset [photon]",
    "bkgstd [photon]",
    "chi2",
    "uncertainty [nm]",
]
NEW_COL_NAMES = [
    "id",
    "frame",
    "x [nm]",
    "y [nm]",
    "sigma [nm]",
    "intensity [photon]",
    "offset [photon]",
    "bkgstd [photon]",
    "chi2",
    "uncertainty [nm]",
    "merged",
]


import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances


def merge_localization(
    tracks: pd.DataFrame, max_dist=1, max_frame_gap=1
) -> pd.DataFrame:
    """Merge localizations into one localization given a maximum distance and a maximum frame gap.
    The function merge_localization() takes as input a DataFrame containing the localizations to be merged and returns a DataFrame containing the merged localizations.
    There is a new column at the end called 'merged' which contains the number of the merged localization produced for each new localization.

    Parameters:
    -----------
    tracks : pandas.DataFrame
        DataFrame containing the localizations to be merged. (this is the output of ThunderSTORM .csv file, see doc for details)
    max_dist : float (nm - nanometers, default = 1)
        Maximum distance between two localizations to be merged (in nanometers)
    max_frame_gap : int (default = 0, no frame gap)
        Maximum frame gap between two localizations to be merged (in frames)

    Returns:
    --------
    merged_tracks : pandas.DataFrame
        DataFrame containing the merged localizations with a new column 'merged' containing the number of the merged localization produced for each new localization.
    """
    # Check if tracks is a pandas DataFrame and contains the required columns
    if not isinstance(tracks, pd.DataFrame):
        raise TypeError("tracks must be a pandas DataFrame")
    if not set(COL_NAMES_THUNDERSTORM).issubset(tracks.columns):
        raise ValueError(
            "tracks must contain the following columns: "
            + ", ".join(COL_NAMES_THUNDERSTORM)
        )

    # create a temporary dataframe to store the merged localizations
    merged_tracks = pd.DataFrame(columns=NEW_COL_NAMES)

    # store the id, frame, x and y columns in numpy arrays to speed up the computation using vectorization
    id = tracks["id"].to_numpy()
    frame = tracks["frame"].to_numpy()
    x = tracks["x [nm]"].to_numpy()
    y = tracks["y [nm]"].to_numpy()
    sigma = tracks["sigma [nm]"].to_numpy()
    intensity = tracks["intensity [photon]"].to_numpy()
    offset = tracks["offset [photon]"].to_numpy()
    bkgstd = tracks["bkgstd [photon]"].to_numpy()
    chi2 = tracks["chi2"].to_numpy()
    uncertainty = tracks["uncertainty [nm]"].to_numpy()

    # Continue from the last line
    # Create a 2D array with x, y coordinates and frame
    coords = np.array(list(zip(x, y, frame)))

    # Define a custom distance metric that takes into account both the spatial distance and the frame gap
    def custom_metric(coord1, coord2):
        spatial_dist = distance.euclidean(coord1[:2], coord2[:2])
        frame_gap = abs(coord1[2] - coord2[2])
        return spatial_dist if frame_gap <= max_frame_gap else np.inf

    def sim_affinity(X):
        return pairwise_distances(X, metric=custom_metric)

    # Use Agglomerative Clustering to cluster the localizations
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=max_dist,
        linkage="single",
        affinity=sim_affinity,
    ).fit(coords)

    # Get the labels of the clusters
    labels = clustering.labels_
    print("done clustering")
    # For each cluster, merge the localizations by averaging their positions
    for label in set(labels):
        mask = labels == label
        # take the first localization of the cluster as the id of the merged localization
        merged_id = id[mask][0]
        merged_frame = frame[mask][0]
        merged_x = x[mask].mean()
        merged_y = y[mask].mean()
        merged_sigma = sigma[mask].mean()
        merged_intensity = intensity[mask].sum()
        merged_offset = offset[mask].mean()
        merged_bkgstd = bkgstd[mask].mean()
        merged_chi2 = chi2[mask].mean()
        merged_uncertainty = uncertainty[mask].mean()
        num_localizations = mask.sum()

        # Add the merged localization to the merged_tracks DataFrame using concat
        merged_tracks = pd.concat(
            [
                merged_tracks,
                pd.DataFrame(
                    [
                        [
                            merged_id,
                            merged_frame,
                            merged_x,
                            merged_y,
                            merged_sigma,
                            merged_intensity,
                            merged_offset,
                            merged_bkgstd,
                            merged_chi2,
                            merged_uncertainty,
                            num_localizations,
                        ]
                    ],
                    columns=NEW_COL_NAMES,
                ),
            ],
            ignore_index=True,
        )

    return merged_tracks
