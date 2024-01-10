'''
Documentation for the simulate_foci.py file.
This file contains the class for simulating foci in space.

It contains the following classes:

Classes:
--------
1. sim_foci: Class for simulating foci in space.
2. Track_generator: Class for generating tracks for the foci in space.
3. sim_focii: Class for simulating multiple space maps with foci in space and detecting the foci in the space maps.

Functions:
----------
1. tophat_function_2d: Function for generating the tophat function for the space map.
2. generate_points: Function for generating the points in the space map.
3. generate_radial_points: Function for generating the points in the space map with radial symmetry.
4. generate_spherical_points: Function for generating the points in the space map with spherical symmetry.
5. radius_spherical_cap: Function for calculating the radius of the spherical cap.
6. get_gaussian: Function for generating the gaussian function for the psf.

Author: Baljyot Singh Parmar
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.patches import Circle
import SMT_Analysis_BP.helpers.clusterMethods.blob_detection as blob_detection
import SMT_Analysis_BP.helpers.analysisFunctions.Analysis_functions as Analysis_functions
import SMT_Analysis_BP.helpers.simulations.fbm_utility as fbm
import SMT_Analysis_BP.helpers.simulations.condensate_movement as condensate_movement
