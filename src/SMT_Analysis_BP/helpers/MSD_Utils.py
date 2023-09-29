import numpy as np
from abc import ABC, abstractmethod
from src.SMT_Analysis_BP.helpers import Analysis_functions as af
from src.SMT_Analysis_BP.databases import trajectory_analysis_script as tas

class MSD_Calculation_abc(ABC):
    '''
    Abstract base class for MSD Calculations
    Will only inforce pixel size, units and frame length, units
    '''
    def __init__(self,pixel_size:float,frame_length:float,pixel_unit:str,frame_unit:str) -> None:
        self._pixel_size = pixel_size
        self._frame_length = frame_length
        self._pixel_unit = pixel_unit
        self._frame_unit = frame_unit
        #print a warning
        self._initialized_print_warning()

    def _initialized_print_warning(self):
        print_statement = '''
        ##############################################################################################################
        #You have initialized a Calculation class with the following parameters:
        #pixel_size: {}
        #frame_length: {}
        #pixel_unit: {}
        #frame_unit: {}
        '''
        print(print_statement.format(self.pixel_size,self.frame_length,self.pixel_unit,self.frame_unit))

    @property
    def pixel_size(self):
        return self._pixel_size
    @property
    def frame_length(self):
        return self._frame_length
    @property
    def pixel_unit(self):
        return self._pixel_unit
    @property
    def frame_unit(self):
        return self._frame_unit

    @abstractmethod
    def _build_MSD_Tracks(self):
        '''
        Does the actual building of the MSD_Tracks to create the storage objects
        '''
        raise NotImplementedError("This is an abstract method and needs to be implemented in the child class")
    
    @property
    @abstractmethod
    def combined_store(self):
        raise NotImplementedError("This is an abstract method and needs to be implemented in the child class")
    @property
    @abstractmethod
    def individual_store(self):
        raise NotImplementedError("This is an abstract method and needs to be implemented in the child class")

class MsdDatabaseUtil:
    '''
    lets create a generic MSD analysis class which will store all the MSD analysis for a single dataset or multiple datasets
    using encapsulation we will utilize smaller functions to do the analysis
    '''
    def __init__(self,data_set_RA:list|None=None,pixel_to_um:float=0.13,frame_to_seconds:float=0.02,**kwargs):
        '''
        Parameters:
        -----------
        data_set_RA: list
            list of tas.run_analysis objects
        pixel_to_um: float
            pixel to um conversion
        frame_to_seconds: float
            frame to seconds conversion
        '''
        #check if the data_set_RA is None, if not then it should be a list of tas.run_analysis objects
        if data_set_RA is not None:
            self.data_set_RA = data_set_RA
        
        #set the pixel to um conversion
        self.pixel_to_um = pixel_to_um
        #set the frame to seconds conversion
        self.frame_to_seconds = frame_to_seconds

    def _get_data_RA(self,dataset:list):
        #we need to make sure that we already dont have a data_set_RA defined
        if hasattr(self,"_data_set_RA"):
            #user should initialize a new class with the data_set_RA
            raise ValueError("data_set_RA is already defined, please initialize a new class")
        else:
            #don't need to check if its the right format since the setter does that
            self.data_set_RA = dataset
    

    @property
    def data_set_RA(self):
        #this is a read only property once it is set
        return self._data_set_RA
    @data_set_RA.setter
    def data_set_RA(self,data_set_RA:list):
        #this will only be set once, so make sure it is not defined already
        if hasattr(self,"_data_set_RA"):
            raise ValueError("data_set_RA can only be set once")
        else:
            #check if the data_set_RA is None, if not then it should be a list of tas.run_analysis objects
            #this is redundant due to the init function but it is here for completeness
            if isinstance(data_set_RA,list):
                self._data_set_RA = data_set_RA
            else:
                raise ValueError("data_set_RA must be a list of tas.run_analysis objects")

    @property
    def data_set_number(self):
        return len(self.data_set_RA) if self.data_set_RA is not None else None
    @property
    def data_set_names(self):
        return [i.my_name for i in self.data_set_RA] if self.data_set_RA is not None else []
    
    @property
    def data_set_parameters(self):
        return [i.parameter_storage for i in self.data_set_RA] if self.data_set_RA is not None else []

    @property
    def track_dict_bulk_list(self):
        if hasattr(self,"_track_dict_bulk_list"):
            return self._track_dict_bulk_list
        else:
            #set the attribute
            self._track_dict_bulk_list = [i._convert_to_track_dict_bulk() for i in self.data_set_RA]
            return self._track_dict_bulk_list

    @property
    def combined_track_dict_bulk(self):
        if hasattr(self,"_combined_track_dict_bulk"):
            return self._combined_track_dict_bulk
        else:
            #set the attribute
            self._combined_track_dict_bulk = combine_track_dicts(self.track_dict_bulk_list)
            return self._combined_track_dict_bulk

    @property
    def pixel_to_um(self):
        return self._pixel_to_um
    @pixel_to_um.setter
    def pixel_to_um(self,pixel_to_um:float):
        self._pixel_to_um = pixel_to_um
    
    @property
    def frame_to_seconds(self):
        return self._frame_to_seconds
    @frame_to_seconds.setter
    def frame_to_seconds(self,frame_to_seconds:float):
        self._frame_to_seconds = frame_to_seconds

class MSD_storage:
    def __init__(self,name,track_class_type,storage_data) -> None:
        self.name = name
        self.track_class_type = track_class_type
        self.storage_data = storage_data
        #store the data
        self._store_data()

    def _store_data(self):
        for i in self.storage_data.keys():
            setattr(self,i,self.storage_data[i])
    ##############################################################################################################
    #ensemble storage
    @property
    def ensemble_MSD(self):
        return self._ensemble_MSD
    @ensemble_MSD.setter
    def ensemble_MSD(self,ensemble_MSD):
        self._ensemble_MSD = ensemble_MSD
    @property
    def ensemble_MSD_error(self):
        return self._ensemble_MSD_error
    @ensemble_MSD_error.setter
    def ensemble_MSD_error(self,ensemble_MSD_error):
        self._ensemble_MSD_error = ensemble_MSD_error
    @property
    def ensemble_displacement(self):
        return self._ensemble_displacement
    @ensemble_displacement.setter
    def ensemble_displacement(self,ensemble_displacement):
        self._ensemble_displacement = ensemble_displacement
    ##############################################################################################################
    #track storage
    @property
    def track_MSD(self):
        return self._track_MSD
    @track_MSD.setter
    def track_MSD(self,track_MSD):
        self._track_MSD = track_MSD
    @property
    def track_MSD_error(self):
        return self._track_MSD_error
    @track_MSD_error.setter
    def track_MSD_error(self,track_MSD_error):
        self._track_MSD_error = track_MSD_error
    @property
    def track_displacement(self):
        return self._track_displacement
    @track_displacement.setter
    def track_displacement(self,track_displacement):
        self._track_displacement = track_displacement
    
    ##############################################################################################################
    #we need to have a datatype which converts the displacement which are (n,d) where d is the dimension to a single r = distance
    @property
    def track_displacement_r(self):
        if hasattr(self,"_track_displacement_r"):
            return self._track_displacement_r
        else:
            #lets convert the track_displacement to a single r
            #the structure is dict-> track_ID_key -> tau_key -> (n,d)
            #we need to convert this to a single r but perserve the structure
            #make a copy of the track_displacement
            copy_track_displacement = self.track_displacement.copy()
            temp_dict = {}
            for i in copy_track_displacement.keys():
                track_taus = copy_track_displacement[i]
                temp_dict[i] = {}
                for j in track_taus.keys():
                    #get the displacement
                    displacement = np.array(track_taus[j])
                    #get the distance
                    distance = np.sqrt(np.sum(displacement**2,axis=1))
                    #store the distance
                    temp_dict[i][j] = distance
            #store the temp_dict
            self._track_displacement_r = temp_dict
            return self._track_displacement_r
    @property
    def ensemble_displacement_r(self):
        if hasattr(self,"_ensemble_displacement_r"):
            return self._ensemble_displacement_r
        else:
            #lets convert the ensemble_displacement to a single r
            #the structure is dict-> tau_key -> (n,d)
            #we need to convert this to a single r but perserve the structure
            #make a copy of the track_displacement
            copy_ensemble_displacement = self.ensemble_displacement.copy()
            temp_dict = {}
            for i in copy_ensemble_displacement.keys():
                #get the displacement
                displacement = np.array(copy_ensemble_displacement[i])
                #get the distance
                distance = np.sqrt(np.sum(displacement**2,axis=1))
                #store the distance
                temp_dict[i] = distance
            #store the temp_dict
            self._ensemble_displacement_r = temp_dict
            return self._ensemble_displacement_r

class MSD_Calculations(MSD_Calculation_abc):
    def __init__(self,data_set_RA:list,pixel_to_um:float=0.13,frame_to_seconds:float=0.02,frame_units:str="s",pixel_units:str="um",**kwargs) -> None:
        '''
        Parameters:
        -----------
        data_set_RA: list
            list of tas.run_analysis objects
        pixel_to_um: float
            pixel to um conversion
        frame_to_seconds: float
            frame to seconds conversion
        frame_units: str
            frame units
        pixel_units: str
            pixel units
        
        KWARGS:
        -------
        min_track_length: int, Default = 1
            minimum track length to be considered. 
            This is passed on to the af.MSD_Tracks() function
        '''
        #initialize the MSD_Calculation_abc class
        super().__init__(pixel_size=pixel_to_um,
                        frame_length=frame_to_seconds,
                        pixel_unit=pixel_units,
                        frame_unit=frame_units
                        )
        
        #list has to be made up of objects of type tas.run_analysis
        self.MSD_dataset = MsdDatabaseUtil(data_set_RA=data_set_RA,pixel_to_um=pixel_to_um,frame_to_seconds=frame_to_seconds)
        #build the MSD_Tracks
        self._build_MSD_Tracks(**kwargs)

    def _build_MSD_Tracks_combined(self,**kwargs):
        self._combined_store = {}
        #lets build the MSD_Tracks() for the combined data
        for i in tas.TRACK_TYPES:
            #get the track_dict_bulk
            track_dict_bulk_temper = self.MSD_dataset.combined_track_dict_bulk[i]
            #build the MSD_Tracks
            MSD_Tracks_i,_ = af.MSD_Tracks(track_dict_bulk_temper,
                                        conversion_factor=self.MSD_dataset.pixel_to_um,
                                        tau_conversion_factor=self.MSD_dataset.frame_to_seconds,
                                        **kwargs)
            ensemble_MSD,ensemble_MSD_error,track_MSD,track_MSD_error = MSD_Tracks_i["msd_curves"]
            ensemble_displacements,track_displacements = MSD_Tracks_i["displacements"]
            #name a name for the storage using the combined names of all the datasets with a " / " in between
            name = " / ".join(self.MSD_dataset.data_set_names)
            #create the storage object
            storage_data = {"ensemble_MSD":ensemble_MSD,
                            "ensemble_MSD_error":ensemble_MSD_error,
                            "ensemble_displacement":ensemble_displacements,
                            "track_MSD":track_MSD,
                            "track_MSD_error":track_MSD_error,
                            "track_displacement":track_displacements}
            #create the storage object
            storage_object = MSD_storage(name,i,storage_data)
            #store the storage object
            self._combined_store[i] = storage_object

    def _build_MSD_Tracks_individual(self,**kwargs):
        individual_store = {
            k:{} for k,l in enumerate(self.MSD_dataset.data_set_names)
        }
        for k,l in enumerate(self.MSD_dataset.data_set_names):
            for i in tas.TRACK_TYPES:
                #get the track_dict_bulk
                track_dict_bulk_temper = self.MSD_dataset.track_dict_bulk_list[k][i]
                #build the MSD_Tracks
                MSD_Tracks_i,_ = af.MSD_Tracks(track_dict_bulk_temper,
                                            conversion_factor=self.MSD_dataset.pixel_to_um,
                                            tau_conversion_factor=self.MSD_dataset.frame_to_seconds,
                                            **kwargs)
                ensemble_MSD,ensemble_MSD_error,track_MSD,track_MSD_error = MSD_Tracks_i["msd_curves"]
                ensemble_displacements,track_displacements = MSD_Tracks_i["displacements"]
                name = self.MSD_dataset.data_set_names[k]
                #create the storage object
                storage_data = {"ensemble_MSD":ensemble_MSD,
                                "ensemble_MSD_error":ensemble_MSD_error,
                                "ensemble_displacement":ensemble_displacements,
                                "track_MSD":track_MSD,
                                "track_MSD_error":track_MSD_error,
                                "track_displacement":track_displacements}
                #create the storage object
                storage_object = MSD_storage(name,i,storage_data)
                #store the storage object
                individual_store[k][i] = storage_object
        #store the individual store
        self._individual_store = individual_store

    def _build_MSD_Tracks(self,**kwargs):
        '''
        Parameters:
        -----------
        KWARGS:
        -------
        Passed to af.MSD_Tracks()
        '''
        self._build_MSD_Tracks_individual(**kwargs)
        self._build_MSD_Tracks_combined(**kwargs)
    
    @property
    def combined_store(self):
        if hasattr(self,"_combined_store"):
            return self._combined_store
        else:
            #run the _build_MSD_Tracks() function
            self._build_MSD_Tracks()
            print("You should not be here. I have built the MSD calculations for you but you should be running that before accessing the combined_store")
            return self._combined_store
    @combined_store.setter
    def combined_store(self,combined_store):
        raise ValueError("combined_store cannot be set")

    @property
    def individual_store(self):
        if hasattr(self,"_individual_store"):
            return self._individual_store
        else:
            #run the _build_MSD_Tracks() function
            self._build_MSD_Tracks()
            print("You should not be here. I have built the MSD calculations for you but you should be running that before accessing the individual_store")
            return self._individual_store
    @individual_store.setter
    def individual_store(self,individual_store):
        raise ValueError("individual_store cannot be set")

class MSD_Calculations_Track_Dict(MSD_Calculation_abc):
    
    #this is a similar class to the MSD_Calculations class but it takes in a track_dict instead of a data_set_RA
    def __init__(self,track_dict,**kwargs) -> None:
        '''
        Parameters:
        -----------
        track_dict: dict
            dict of tracks in the form: 
            {
            track_ID: [[x,y],...,[x,y]],
            .
            .
            .
            }
        KWARGS:
        -------
        pixel_to_um: float
            pixel to um conversion
        frame_to_seconds: float
            frame to seconds conversion
        frame_units: str
            frame units
        pixel_units: str
            pixel units
        '''
        #initialize the MSD_Calculation_abc class
        super().__init__(pixel_size=kwargs.get("pixel_to_um",0.13),
                        frame_length=kwargs.get("frame_to_seconds",0.02),
                        pixel_unit=kwargs.get("pixel_units","um"),
                        frame_unit=kwargs.get("frame_units","s")
                        )
        #make sure that the track_dict is a dict
        if isinstance(track_dict,dict):
            #set the track_dict
            self.track_dict = track_dict
        else:
            raise ValueError("track_dict must be a dict")
        self.pixel_to_um = kwargs.get("pixel_to_um",0.13)
        self.frame_to_seconds = kwargs.get("frame_to_seconds",0.02)
        #build the MSD_Tracks
        self._build_MSD_Tracks()

    def _build_MSD_Tracks(self,**kwargs):
        #build the MSD_Tracks
        MSD_tracks,_ = af.MSD_Tracks(
            self.track_dict, 
            conversion_factor=self.pixel_to_um, 
            tau_conversion_factor=self.frame_to_seconds,
            **kwargs
            )
        ensemble_MSD,ensemble_MSD_error,track_MSD,track_MSD_error = MSD_tracks["msd_curves"]
        ensemble_displacements,track_displacements = MSD_tracks["displacements"]
        #create the storage object
        storage_data = {"ensemble_MSD":ensemble_MSD,
                        "ensemble_MSD_error":ensemble_MSD_error,
                        "ensemble_displacement":ensemble_displacements,
                        "track_MSD":track_MSD,
                        "track_MSD_error":track_MSD_error,
                        "track_displacement":track_displacements}
        #create the storage object
        storage_object = MSD_storage("track_dict","ALL",storage_data)
        #store the storage object
        self._storage = storage_object

    @property
    def combined_store(self):
        Warning("This is a track_dict and does not have a combined_store, returning the individual_store variable")
        return self._storage
    @combined_store.setter
    def combined_store(self,combined_store):
        raise ValueError("combined_store cannot be set")
    @property
    def individual_store(self):
        return self._storage
    @individual_store.setter
    def individual_store(self,individual_store):
        raise ValueError("individual_store cannot be set")


#radius of confinment fucntion
def radius_of_confinement(t,r_sqr,D,loc_msd):
    return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 4*(loc_msd**2)

#radius of confinment fucntion
def radius_of_confinement_xy(t,r_sqr,D,loc_msd_x,loc_msd_y):
    return (r_sqr**2)*(1.-np.exp(-4*D*t/(r_sqr**2))) + 4*(loc_msd_x**2) + 4*(loc_msd_y**2)

#power law function with independent x and y
def power_law_xy(t,alpha,D,loc_msd_x,loc_msd_y):
    return 4*(loc_msd_x**2) + 4*(loc_msd_y**2) + 4.*D*t**(alpha)

#power law function with r_sqr
def power_law(t,alpha,D,loc_msd):
    return 4*(loc_msd**2) + 4.*D*t**(alpha)

def linear_MSD_fit(t,a,b):
    '''
    linear fit function
    expects t to be scaled with log10, and returns msd output in log10
    b = log10(4*D)
    a = alpha
    t = log10(tau)
    '''
    return b + a*t

def combine_track_dicts(dicts):
    '''each dict is going to contain 4 dicts of name "IN","IO","OUT","ALL"
    we need to keep this strucutre for the final combined dict'''
    combined_dict = {"IN":{},"IO":{},"OUT":{},"ALL":{}}
    #iterate over the dicts
    track_counter = 1
    for i in dicts:
        #iterate over the keys of the dicts
        for j in i.keys():
            #iterate over the keys of the dicts
            for k in i[j].keys():
                #change the key of the dict
                combined_dict[j][str(track_counter)] = i[j][k]
                track_counter += 1
    return combined_dict


        
