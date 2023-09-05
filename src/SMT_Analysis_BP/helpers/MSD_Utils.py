import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Type
from src.SMT_Analysis_BP.helpers import Analysis_functions as af
from src.SMT_Analysis_BP.helpers import decorators as dec
from src.SMT_Analysis_BP.helpers import smallestenclosingcircle as sec
from src.SMT_Analysis_BP.Parameter_Store import global_params as gparms
from src.SMT_Analysis_BP.databases import trajectory_analysis_script as tas

#radius of confinment fucntion
def radius_of_confinement(t,r_sqr,D,loc_msd):
    return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 4*(loc_msd**2)

#radius of confinment fucntion
def radius_of_confinement_xy(t,r_sqr,D,loc_msd_x,loc_msd_y):
    return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 4*(loc_msd_x**2) + 4*(loc_msd_y**2)

#power law function with independent x and y
def power_law_xy(t,alpha_x,alpha_y,D,loc_msd_x,loc_msd_y):
    return 4*(loc_msd_x**2) + 4*(loc_msd_y**2) + 4.*D*t**(alpha_x+alpha_y)

#power law function with r_sqr
def power_law(t,alpha,D,loc_msd):
    return 4*(loc_msd**2) + 4.*D*t**(alpha)


#lets create a generic MSD analysis class which will store all the MSD analysis for a single dataset
#using encapsulation we will utilize smaller functions to do the analysis
class MsdDatabaseUtil:
    def __init__(self,data_set_RA:Type(tas.run_analysis)) -> None:
        self.data_set = data_set_RA
        self.track_dict_bulk = self.data_set._convert_to_track_dict_bulk() #see tas.run_analysis._convert_to_track_dict_bulk
        
        pass




    @property 
    def data_set(self):
        return self._data_set
    @data_set.setter
    def data_set(self,data_set):
        if hasattr(self,'_data_set'):
            raise Exception('data_set already set for {0} data at {1}'.format(self.parameters_dict["t_string"],self.parameters_dict["wd"]))
        self._data_set = data_set
    @property
    def parameters_dict(self):
        if not hasattr(self,'_parameters_dict'):
            self._parameters_dict = self._data_set.parameter_storage
        return self._parameters_dict
    @property
    def track_dict_bulk(self):
        return self._track_dict_bulk
    @track_dict_bulk.setter
    def track_dict_bulk(self,track_dict_bulk):
        #do not allow track_dict_bulk to be set twice
        if hasattr(self,'_track_dict_bulk'):
            raise Exception('track_dict_bulk already set for {0} data at {1}'.format(self.parameters_dict["t_string"],self.parameters_dict["wd"]))
        self._track_dict_bulk = track_dict_bulk

