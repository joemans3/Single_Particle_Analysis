import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import src.helpers.Analysis_functions as af
import src.helpers.fbm_utility as fbm
import src.helpers.simulate_foci as sf
import skimage as skimage
from PIL import Image
import pickle
import src.helpers.decorators as decorators
import src.helpers.probability_functions as pf

class Simulate_cells(sf.Track_generator):
    ''' MRO is used to inherit from the Track_generator class in simulate_foci.py'''
    def __init__(self,cell_params={},global_params={}) -> None:
        '''
        cell_params : dict, Default = {}
            dictionary of cell parameters
        global_params : dict, Default = {}
            dictionary of global parameters
        
        List of cell parameters:
        ------------------------
        diffusion_coefficients : list, Default = [], units = um^2/s
            list of diffusion coefficients for each track
        initials : list, Default = []
            list of initial positions for each track, must be same as length of diffusion_coefficients
        num_tracks : int, Default = 0
            number of tracks to be simulated, must be same as length of diffusion_coefficients
        track_type : str, Default = 'fbm'
            type of track to be simulated
        hursts : list, Default = []
            list of hurst exponents for each track. Must be of same length as diffusion_coefficients
        dims : tuple, Default = (100,100)
            dimensions of the space to be simulated
        movie_frames : int, Default = 500
            number of frames to be simulated
        cell_space : np.ndarray, Default = None
            Space designated for the cell. If None, a space is generated with dimensions dims.
            Make the assumption that it is a square space. So needs [min_x,max_x]. The second dimension is the same.

        
        List of global parameters:
        --------------------------
        track_length_mean : int, Default = None
            mean track length for the tracks to be simulated
        track_distribution : str, Default = None
            distribution of track lengths
        exposure_time : int, Default = None
            exposure time for the SMT experiment in ms
        frame_time : int, Default = None
            frame time for the SMT experiment in ms
        pixel_size : int, Default = None
            pixel size for the SMT experiment in nm
        axial_function : str, Default = 'ones'
            axial function for the SMT experiment. Options are 'ones' and 'exponential'
            This controls the axial (z) effect on the PSF (Ones is no axial effect, exponential is axial effect with a decay)
        
        Full list see the docstring for Track_generator and sim_foci class in simulate_foci.py 

        
        
        
        '''
        #if axial_function is not in global_params, set it to "ones"
        if 'axial_function' not in global_params:
            global_params['axial_function'] = 'ones'
        self.cell_params = cell_params #dictionary of cell parameters
        self.global_params = global_params #dictionary of global parameters
        self.units = self._define_units(pixel_size=global_params['pixel_size'],
                                        frame_time=global_params['frame_time'])
        cell_params['diffusion_coefficients']=self._update_units(unit=cell_params['diffusion_coefficients'],
                                                                orig_type='um^2/s',
                                                                update_type='pix^2/ms') #this converts to pix^2/(20)ms
        super().__init__(global_params,global_params)

        pass
    
    def _define_units(self,pixel_size,frame_time,update=False):
        ''' Docstring for _define_units: define the units for the simulation
        
        Parameters:
        -----------
        pixel_size : int
            pixel size in nm
        frame_time : int
            frame time in ms
        update : bool, Default = False
            whether to update the units dictionary
        
        Returns:
        --------
        units : dict
            dictionary of units for simulation
        '''
        units = {'pixel_size':pixel_size,'frame_time':frame_time}
        if update:
            self.units = units
        return units
    
    def _update_units(self,unit,orig_type,update_type):
        ''' Docstring for _update_units: update the unit from one type to another
        
        Parameters:
        -----------
        unit : int
            unit to be updated
        orig_type : str
            original type of unit
        update_type : str
            type to update unit to
        '''
        if orig_type == 'nm':
            if update_type == 'pix':
                return unit/self.units['pixel_size']
        elif orig_type == 'pix':
            if update_type == 'nm':
                return unit*self.units['pixel_size']
        elif orig_type == 'ms':
            if update_type == 's':
                return unit/1000.
        elif orig_type == 's':
            if update_type == 'ms':
                return unit*1000.
        elif orig_type == 'um^2/s':
            if update_type == 'pix^2/ms':
                return unit*((1e6)/self.units['pixel_size']**2)/(1000./self.units['frame_time']) #this converts to pix^2/(20)ms
        return 
    
    def _define_space(self,dims=(100,100),movie_frames=500):
        ''' Docstring for _define_space: make the empty space for simulation
        
        Parameters:
        -----------
        dims : tuple, Default = (100,100)
            dimensions of the space to be simulated
        movie_frames : int, Default = 500
            number of frames to be simulated
        
        Returns:
        --------
        space : array-like, shape = (movie_frames,dims[0],dims[1])
            empty space for simulation
        '''
        space = np.zeros((movie_frames,dims[0],dims[1]))
        return space

    def _create_cell_tracks(self,num_tracks,diffusion_coefficients,initials,
                            simulation_cube,hursts,track_type='fbm',mean_track_length=None,
                            track_length_distribution=None,
                            exposure_time=None):
        ''' Docstring for _create_cell_tracks: create the tracks for the cell

        Parameters:
        -----------
        diffusion_coefficients : list,  the units are pix^2/ms
            list of diffusion coefficients for each cell
        initials : list
            list of initial positions for each cell
        num_tracks : int
            number of tracks to be simulated
        track_type : str, Default = 'fbm'
            type of track to be simulated, can be 'fbm' or 'constant'
        simulation_cube : array-like
            empty space for simulation
        hursts : list
            list of hurst exponents for each track. Must be of same length as diffusion_coefficients
        mean_track_length : int, Default = None
            mean track length for the tracks to be simulated
        track_length_distribution : str, Default = None
            distribution of track lengths
        exposure_time : int, Default = None
            exposure time for the SMT experiment in ms
        
        Returns:
        --------
        tracks : list
            list of tracks for each cell
        points_per_track : list
            list of number of points in each frame
        
        '''
        if diffusion_coefficients is None:
            diffusion_coefficients = self.cell_params['diffusion_coefficients']
        if initials is None:
            initials = self.cell_params['initials']
        if num_tracks is None:
            num_tracks = self.cell_params['num_tracks']
        if track_type is None:
            track_type = self.cell_params['track_type']

        #get the mean track length
        if mean_track_length is None:
            mean_track_length = self.global_params['track_length_mean']
        #get the track length distribution
        if track_length_distribution is None:
            track_length_distribution = self.global_params["track_distribution"]

        #get the exposure time for the SMT experiment in ms
        if exposure_time is None:
            exposure_time = self.global_params['exposure_time'] #ms
        #get the number of frames to be simulated from simulation_cube
        movie_frames = simulation_cube.shape[0]
        #get the lengths of the tracks given a distribution
        track_lengths = self._get_lengths(track_distribution=track_length_distribution,track_length_mean=mean_track_length,total_tracks=num_tracks)
        #if track_lengths is larger than the number of frames then set that to the number of frames -1
        track_lengths = np.array([i if i < movie_frames else movie_frames-1 for i in track_lengths])
        #for each track_lengths find the starting frame
        starting_frames = np.array([random.randint(0,movie_frames-i) for i in track_lengths])


        if isinstance(initials,dict): 
            #do the _initial_checker()
            self._initial_checker(initials=initials)
            #the diffusion_coefficients will be in um^2/s so we need to convert to pix^2/(frame_time)
            diff_coef_condensate = self._update_units(unit=initials['diffusion_coefficient'],
                                                        orig_type='um^2/s',
                                                        update_type='pix^2/ms')
            initials['diffusion_coefficient'] = diff_coef_condensate
            #initialize the Condensates. TODO

            self.create_condensate_dict(**initials,units_time=np.array(["20ms"]*len(initials['diffusion_coefficient'])))
            #define the top_hat class that will be used to sample the condensates
            top_hat_func = pf.multiple_top_hat_probability(
                num_subspace = len(initials['diffusion_coefficient']),
                subspace_centers = initials['initial_centers'],
                subspace_radius = initials['initial_scale'],
                density_dif = self.global_params['density_dif'],
                space_size = self.cell_params['cell_space'],
            )

            #make a placeholder for the initial position array with all 0s
            initials = np.zeros((num_tracks,3))
            #lets use the starting frames to find the inital position based on the position of the condensates
            for i in range(num_tracks):
                #get the starting frame
                starting_frame = starting_frames[i]
                #condensate positions
                condensate_positions = np.zeros((len(self.condensates),2))
                #loop through the condensates
                for ID,cond in self.condensates.items():
                    condensate_positions[int(ID)] = cond(int(starting_frame),"20ms")["Position"]
                #update the top_hat_func with the new condensate positions
                top_hat_func.update_parameters(subspace_centers=condensate_positions)
                #sample the top hat to get the initial position
                initials[i][:2] = sf.generate_points_from_cls(top_hat_func,total_points=1,min_x=self.cell_params["cell_space"][0],max_x=self.cell_params["cell_space"][1],density_dif=self.global_params["density_dif"])[0]



            
        else:
            #make sure that the number of tracks is equal to the number of diffusion coefficients and initials with raise an error if not
            assert len(hursts) == len(diffusion_coefficients) == len(initials) == num_tracks, 'The number of tracks is not equal to the number of diffusion coefficients or initials or hurst exponents'
        
        #check to see if there is 2 or 3 values in the second dimension of initials
        if initials.shape[1] == 2:
            #add a third dimension of zeros so that the final shape is (num_tracks,3) with (:,3) being 0s
            initials = np.hstack((initials,np.zeros((num_tracks,1)))) 
            


        #create tracks
        tracks = {}
        points_per_frame = dict(zip([str(i) for i in range(movie_frames)],[[] for i in range(movie_frames)]))
        for i in range(num_tracks):
            if track_type=="fbm":
                start_frame = starting_frames[i]
                xyz,t = self._make_fbm_track(length=track_lengths[i],end_time=track_lengths[i],hurst=hursts[i],return_time=True,dim=3)
                #convery the time to ints
                t = np.array(t,dtype=int)

                #shift the frames to start at the start_frame
                frames = start_frame + t
                
                #shift the track with the initial position and diffusion coefficient
                xyz = (xyz * np.sqrt(2*diffusion_coefficients[i])) + initials[i]
                #add this to the dictionary of tracks
                tracks[i] = {'xy':xyz,'frames':frames,'diffusion_coefficient':diffusion_coefficients[i],'initial':initials[i],'hurst':hursts[i]}
                #add the number of points per frame to the dictionary
                for j in range(len(frames)):
                    points_per_frame[str(frames[j])].append(xyz[j])
            if track_type=="constant":
                #make a constant track
                xyz = np.array([initials[i] for ll in range(int(track_lengths[i]))])
                #make the time array
                t = np.arange(track_lengths[i],dtype=int)

                start_frame = starting_frames[i]
                #shift the frames to start at the start_frame
                frames = start_frame + t
                #add this to the dictionary of tracks
                tracks[i] = {'xy':xyz,'frames':frames,'diffusion_coefficient':0,'initial':initials[i],'hurst':0}
                #add the number of points per frame to the dictionary
                for j in range(len(frames)):
                    points_per_frame[str(frames[j])].append(xyz[j])
                    
        return tracks,points_per_frame
    
    @decorators.deprecated("This function is not useful, but is still here for a while in case I need it later")
    def _format_points_per_frame(self,points_per_frame):
        '''
        Docstring for _format_points_per_frame: format the points per frame dictionary so that for each key the set of tracks in it are 
        converted to a numpy array of N x 2 where N is the total amount of points in that frame. You don't need this function. 
        
        Parameters:
        -----------
        points_per_frame : dict
            keys = str(i) for i in range(movie_frames), values = list of tracks, which are collections of [x,y] coordinates
        
        Returns:
        --------
        points_per_frame : dict
            keys = str(i) for i in range(movie_frames), values = numpy array of N x 2 where N is the total amount of points in that frame
        
        '''
        for i in points_per_frame.keys():
            #each value is a list of K lists that are composed of M x 2 arrays where M can be different for each list
            #we want to convert this to a numpy array of N x 2 where N is the total amount of points in that frame
            point_holder = []
            for j in points_per_frame[i]:
                point_holder.append(j)
            points_per_frame[i] = np.array(point_holder)
        return points_per_frame

    def _update_map(self,map,points_per_frame,axial_func="ones"):
        ''' Docstring for _update_map: update the map with the points per frame

        Parameters:
        -----------
        map : array-like
            empty space for simulation, shape = (movie_frames,dims[0],dims[1])
        points_per_frame : dict
            keys = str(i) for i in range(movie_frames), values = list of points in each frame as a list of (x,y,z) tuples
        axial_func : str 
            function to use to update the axial intensity of the points, default is "ones" which means that the axial intensity factor is 1
            

        Returns:
        --------
        map : array-like
            map with points per frame added to it using the generate_map_from_points function from base class
    
        '''

        for i in range(map.shape[0]):
            # we want to update the point_intensity based on the axial position of the point (z)
            if len(points_per_frame[str(i)]) == 0:
                abs_axial_position = self.point_intensity
                points_per_frame_xy = np.array(points_per_frame[str(i)])
            else:
                abs_axial_position = self.point_intensity*sf.axial_intensity_factor(np.abs(np.array(points_per_frame[str(i)])[:,2]),func = axial_func)
                points_per_frame_xy = np.array(points_per_frame[str(i)])[:,:2]
            map[i],_ = self.generate_map_from_points(points_per_frame_xy,point_intensity=abs_axial_position,map=map[i],movie=True)
        return map
    
    def get_cell(self,dims=None,movie_frames=None,num_tracks=None,diffusion_coefficients=None,initials=None,
                hursts=None,track_type=None,mean_track_length=None,
                track_length_distribution=None,exposure_time=None):
        ''' Docstring for get_cell: get the cell for simulation

        Parameters:
        -----------
        dims : tuple, Default = None
            dimensions of the simulation cube
        movie_frames : int, Default = None
            number of frames to be simulated
        num_tracks : int, Default = None
            number of tracks to be simulated
        diffusion_coefficients : list, Default = None
            list of diffusion coefficients for each track
        initials : list, Default = None 
            list of initial positions for each track
        hursts : list, Default = None
            list of hurst exponents for each track
        track_type : str, Default = 'fbm'
            type of track to be simulated, currently only 'fbm' is supported
        mean_track_length : int, Default = None
            mean track length for the tracks
        track_length_distribution : str, Default = None
            distribution of track lengths, currently only 'poisson' is supported
        exposure_time : int, Default = None
            exposure time for the SMT experiment in ms

        Returns:
        --------
        cell : dict
            dictionary containing the tracks and points per frame for the cell
            Keys = 'map','tracks','points_per_frame'
            Values = simulation_cube,tracks,points_per_frame

        '''
        if dims is None:
            dims = self.cell_params['dims']
        if movie_frames is None:
            movie_frames = self.cell_params['movie_frames']
        if num_tracks is None:
            num_tracks = self.cell_params['num_tracks']
        if diffusion_coefficients is None:
            diffusion_coefficients = self.cell_params['diffusion_coefficients']
        if initials is None:
            initials = self.cell_params['initials']
        if hursts is None:
            hursts = self.cell_params['hursts']
        if mean_track_length is None:
            mean_track_length = self.global_params['track_length_mean']
        if track_length_distribution is None:
            track_length_distribution = self.global_params["track_distribution"]
        if exposure_time is None:
            exposure_time = self.global_params['exposure_time']
        if track_type is None:
            track_type = self.cell_params['track_type']
        

        #create the simulation cube
        simulation_cube = self._define_space(dims=dims,movie_frames=movie_frames)
        #create the tracks
        tracks,points_per_frame = self._create_cell_tracks(num_tracks=num_tracks,diffusion_coefficients=diffusion_coefficients,
                                                        initials=initials,simulation_cube=simulation_cube,hursts=hursts,
                                                        track_type=track_type,mean_track_length=mean_track_length,
                                                        track_length_distribution=track_length_distribution,exposure_time=exposure_time)
        #update the map
        map = self._update_map(map=simulation_cube,points_per_frame=points_per_frame,axial_func=self.global_params['axial_function'])
        return {"map":map,"tracks":tracks,"points_per_frame":points_per_frame}
    
    def _initial_checker(self,initials):
            ''' Docstring for _initial_checker: check the initials dictionary
            This only applies which initials is a dictionary; this occurs when we are defining condensate movement with the Condensate class

            Parameters:
            -----------
            initials : dict
                dictionary containing the initial centers, initial scale, diffusion coefficient and hurst exponent
            '''
            #make sure when the initials are a dictionary that it contains the keys: 1) "inital_centers", 2) "initial_scale"
            # 3) "diffusion_coefficient" and 4) "hurst_exponent" at the very least
            #keys to check for
            keys = ["initial_centers","initial_scale","diffusion_coefficient","hurst_exponent"]
            #check if the keys are in the dictionary
            if all(k in initials.keys() for k in keys):
                pass
            else:
                raise ValueError("The initials dictionary must contain the following keys at the very least: {}".format(keys))

            pass
        
    def sample_cells(self,dims=None,movie_frames=None,num_tracks=None,diffusion_coefficients=None,initials=None,
                hursts=None,track_type='fbm',mean_track_length=None,
                track_length_distribution=None,exposure_time=None,subsample_sizes=None,**kwargs):
        ''' Docstring for sample_cells: create a true set of tracks given the variables and then create subsamples of the tracks to be used for the cell simulation

        Parameters:
        -----------
        dims : tuple, Default = None
            dimensions of the simulation cube
        movie_frames : int, Default = None
            number of frames to be simulated
        num_tracks : int, Default = None
            number of tracks to be simulated
        diffusion_coefficients : list, Default = None
            list of diffusion coefficients for each track
        initials : list, Default = None 
            list of initial positions for each track
        hursts : list, Default = None
            list of hurst exponents for each track
        track_type : str, Default = 'fbm'
            type of track to be simulated, currently only 'fbm' is supported
        mean_track_length : int, Default = None
            mean track length for the tracks
        track_length_distribution : str, Default = None
            distribution of track lengths, currently only 'poisson' is supported
        exposure_time : int, Default = None
            exposure time for the SMT experiment in ms
        subsample_sizes : list or array-like dim = 1, Default = None
            list of subsample sizes to be used for the simulation

        KWARGS: (not implemented yet)
        -------
        subsample_weights : list must be same length as num_tracks
            list of weights for each track in the subsample based on weight_var
        weight_var : str (can be 'diffusion_coefficient','hurst')
            variable to be used for weighting the tracks
        
        Returns:
        --------
        output_subsample : list of dicts
            list of dictionaries containing the tracks and points per frame for each subsample
            Keys = 'map','tracks','points_per_frame'
            Values = simulation_cube,tracks,points_per_frame
        output_true : dict
            dictionary containing the tracks and points per frame for the true set of tracks (all of them)
            Keys = 'map','tracks','points_per_frame'
            Values = simulation_cube,tracks,points_per_frame

        '''
        if dims is None:
            dims = self.cell_params['dims']
        if movie_frames is None:
            movie_frames = self.cell_params['movie_frames']
        if num_tracks is None:
            num_tracks = self.cell_params['num_tracks']
        if diffusion_coefficients is None:
            diffusion_coefficients = self.cell_params['diffusion_coefficients']
        if initials is None:
            initials = self.cell_params['initials']
        if hursts is None:
            hursts = self.cell_params['hursts']
        if mean_track_length is None:
            mean_track_length = self.global_params['track_length_mean']
        if track_length_distribution is None:
            track_length_distribution = self.global_params["track_distribution"]
        if exposure_time is None:
            exposure_time = self.global_params['exposure_time']


        #if the subsample sizes are not defined, return the output of self.get_cell without subsampling
        if subsample_sizes is None:
            return [self.get_cell(dims=dims,movie_frames=movie_frames,num_tracks=num_tracks,diffusion_coefficients=diffusion_coefficients,
                                initials=initials,hursts=hursts,track_type=track_type,mean_track_length=mean_track_length,
                                track_length_distribution=track_length_distribution,exposure_time=exposure_time)]
        #make sure each subsample size is less than the number of tracks
        for i in subsample_sizes:
            if i > num_tracks:
                raise ValueError("Subsample size must be less than the number of tracks")

        #create the simulation cube
        simulation_cube = self._define_space(dims=dims,movie_frames=movie_frames)
        #create the tracks
        tracks,points_per_frame = self._create_cell_tracks(num_tracks=num_tracks,diffusion_coefficients=diffusion_coefficients,
                                                        initials=initials,simulation_cube=simulation_cube,hursts=hursts,
                                                        track_type=track_type,mean_track_length=mean_track_length,
                                                        track_length_distribution=track_length_distribution,exposure_time=exposure_time)
        full_map = self._update_map(map=simulation_cube,points_per_frame=points_per_frame,axial_func=self.global_params['axial_function'])
        #if kwargs is empty make a list of wieghts that are all 1
        if len(kwargs) == 0:
            weights = np.ones(num_tracks)/num_tracks

        #for each subsample size, create a subsample of the tracks and make a new map
        output = []
        for i in subsample_sizes:
            #create the simulation cube for this new sample 
            simulation_cube_sample = self._define_space(dims=dims,movie_frames=movie_frames)
            #create the subsample
            #from the full sample tracks, choose i random tracks without replacement
            subsample_tracks = {k:tracks[k] for k in np.random.choice(list(tracks.keys()),i,replace=False, p=weights)}
            #make a new points per frame dictionary
            subsample_points_per_frame = self._convert_track_dict_points_per_frame(subsample_tracks,movie_frames)
            #update the map
            subsample_map = self._update_map(map=simulation_cube_sample,points_per_frame=subsample_points_per_frame,axial_func=self.global_params['axial_function'])
            #append the output
            output.append({"map":subsample_map,"tracks":subsample_tracks,"points_per_frame":subsample_points_per_frame})
        output_subsample = output
        output_overall = {"map":full_map,"tracks":tracks,"points_per_frame":points_per_frame}
        return output_subsample,output_overall
    
    def _convert_track_dict_points_per_frame(self,tracks,movie_frames):
        '''Docstring for _convert_track_dict_points_per_frame: convert the track dictionary to a dictionary of points per frame

        Parameters:
        -----------
        tracks : dict
            dictionary of tracks, keys = track number, values = dictionary with keys = 'xy','frames','diffusion_coefficient','initial','hurst'
        movie_frames : int
            number of frames in the movie
        
        Returns:
        --------
        points_per_frame : dict
            dictionary of points per frame, keys = frame number, values = list of (x,y,z) tuples
        
        '''
        points_per_frame = dict(zip([str(i) for i in range(movie_frames)],[[] for i in range(movie_frames)]))
        for i,j in tracks.items():
            for k in range(len(j["frames"])):
                points_per_frame[str(j["frames"][k])].append(j["xy"][k])

        return points_per_frame

    def _convert_track_dict_msd(self,tracks):
        '''Docstring for _convert_track_dict_msd: convert the track dictionary to a dictionary of tracks with the format
        required for the msd function

        Parameters:
        -----------
        tracks : dict
            dictionary of tracks, keys = track number, values = dictionary with keys = 'xy','frames','diffusion_coefficient','initial','hurst'
        
        Returns:
        --------
        track_msd : dict
            dictionary of tracks with the format required for the msd function, keys = track number, values = list of (x,y,T) tuples
            
        
        '''
        track_msd = {}
        for i,j in tracks.items():
            #make a (x,y,T) tuple for each point
            track_msd[i] = []
            for k in range(len(j["xy"])):
                track_msd[i].append((j["xy"][k][0],j["xy"][k][1],j["frames"][k]))
            #add this to the dictionary
            track_msd[i] = np.array(track_msd[i])
        return track_msd

def save_tiff(image,path,img_name=None):
    ''' Docstring for save_tiff: save the image as a tiff file
    
    Parameters:
    -----------
    image : array-like
        image to be saved
    path : str
        path to save the image
    img_name : str, Default = None
        name of the image
    
    Returns:
    --------
    None
    '''
    if img_name is None:
        skimage.io.imsave(path,image)
    else:
        skimage.io.imsave(path+img_name+".tiff",image)
    return  

#function to perform the subsegmentation
def sub_segment(img,sub_frame_num,img_name=None,subsegment_type="mean"):
    ''' Docstring for sub_segment: perform subsegmentation on the image
    
    Parameters:
    -----------
    img : array-like
        image to be subsegmented
    sub_frame_num : int
        number of subsegments to be created
    img_name : str, Default = None
        name of the image
    subsegment_type : str, Default = "mean"
        type of subsegmentation to be performed, currently only "mean" is supported
    
    Returns:
    --------
    hold_img : list
        list of subsegments
    hold_name : list
        list of names of the subsegments
    
    
    '''
    #get the dimensions of the image
    dims = img.shape
    #get the number of frames
    num_frames = dims[0]
    #find the number of frames per subsegment
    frames_per_subsegment = int(num_frames/sub_frame_num)
    hold_img = []
    hold_name = []
    for j in np.arange(sub_frame_num):
        if subsegment_type == "mean":
            hold_img.append(np.mean(img[int(j*frames_per_subsegment):int((j+1)*frames_per_subsegment)],axis=0))
        elif subsegment_type == "max":
            hold_img.append(np.max(img[int(j*frames_per_subsegment):int((j+1)*frames_per_subsegment)],axis=0))
        elif subsegment_type == "std":
            hold_img.append(np.std(img[int(j*frames_per_subsegment):int((j+1)*frames_per_subsegment)],axis=0))
        
    return hold_img

def make_directory_structure(cd,img_name,img,subsegment_type,sub_frame_num,**kwargs):
    ''' Docstring for make_directory_structure: make the directory structure for the simulation and save the image + the data and parameters
    Also perform the subsegmentation and save the subsegments in the appropriate directory
    
    Parameters:
    -----------
    cd : str
        directory to save the simulation
    img_name : str
        name of the image
    img : array-like    
        image to be subsegmented
    subsegment_type : str
        type of subsegmentation to be performed, currently only "mean" is supported
    sub_frame_num : int
        number of subsegments to be created
    **kwargs : dict
        dictionary of keyword arguments
    
    KWARGS:
    -------
    data : dict, Default = None
        dictionary of data to be saved, Keys = "map","tracks","points_per_frame" Values = array-like. 
        See the return of the function simulate_cell_tracks for more details
    parameters : dict, Default = None (dict with the keys "cell_parameters","global_parameters")
        cell_parameters : dict, Default = None
            dictionary of cell parameters, Keys = "num_tracks",
                                                  "diffusion_coefficients",
                                                  "initials",
                                                  "simulation_cube",
                                                  "hursts",
                                                  "track_type",
                                                  "mean_track_length",
                                                  "track_length_distribution",
                                                  "exposure_time"
        global_parameters : dict, Default = None
            dictionary of global parameters, Keys = "num_cells",
                                                    "num_subsegments",
                                                    "subsegment_type",
                                                    "subsegment_num_frames",
                                                    "subsegment_num_points",
                                                    "subsegment_num_tracks",
                                                    "subsegment_num_diffusion_coefficients",
                                                    "subsegment_num_initials",
                                                    "subsegment_num_hursts",
                                                    "subsegment_num_track_type",
                                                    "subsegment_num_mean_track_length",
                                                    "subsegment_num_track_length_distribution",
                                                    "subsegment_num_exposure_time"
                                                    (names might be different, but should be self explanatory once you load the dictionary)
    
    Returns:
    --------
    array-like
        list of subsegment images
    '''
    #make the directory if it does not exist
    if not os.path.exists(cd):
        os.makedirs(cd)

    #saves the data if it is passed as a keyword argument (map,tracks,points_per_frame)
    with open(cd+'Track_dump.pkl', 'wb+') as f:
        pickle.dump(kwargs.get("data",{}), f)
    #saves the parameters used to generate the simulation
    with open(cd+'params_dump.pkl', 'wb+') as f:
        pickle.dump(kwargs.get("parameters",{}), f)

    #in this directory, dump the parameters into a json file
    if "parameters" in kwargs.keys():
        with open('{0}/parameters.json'.format(cd), 'w') as fp:
            #check if parameter values are dictionaries
            for i,j in kwargs["parameters"].items():
                if type(j) == dict:
                    for k,l in j.items():
                        if type(l) == np.ndarray and k !="initials":
                            #if true then find the unique values in the array with the number of times they occur and save it as a dictionary "unique_values":count
                            unique, counts = np.unique(l, return_counts=True)
                            #convert the arrays to lists
                            unique = unique.tolist()
                            counts = counts.tolist()
                            kwargs["parameters"][i][k] = dict(zip(unique, counts))
                        elif k=="initials":
                            if isinstance(l,dict):
                                temp_dict = {}
                                for m,n in l.items():
                                    temp_dict[m] = n.tolist()

                                kwargs["parameters"][i][k] = temp_dict
                            else:
                                #from the collection of [[x,y]...] find the unique values of [x,y] combinations and save it as a dictionary "unique_values":count
                                unique, counts = np.unique(l, axis=0,return_counts=True)
                                #convert the arrays to lists
                                unique = map(str,unique.tolist())
                                counts = counts.tolist()
                                kwargs["parameters"][i][k] = dict(zip(unique, counts))
                        
                            
            json.dump(kwargs["parameters"], fp,indent=4)


    #make a diretory inside cd called Analysis if it does not exist
    if not os.path.exists('{0}/Analysis'.format(cd)):
        os.makedirs('{0}/Analysis'.format(cd))

    #save the img file with its name in the cd directory
    save_tiff(img,cd,img_name=img_name)
    #make a directory inside cd called segmented if it does not exist
    if not os.path.exists('{0}/segmented'.format(cd)):
        os.makedirs('{0}/segmented'.format(cd))
    #perform subsegmentation on the image
    hold_img = sub_segment(img,sub_frame_num,img_name=img_name,subsegment_type=subsegment_type)
    #create the names for the subsegmented images
    hold_name = []
    for i in np.arange(sub_frame_num):
        hold_name.append( '{0}segmented/{1}.tif'.format(cd,str(int(i)+1)+"_"+img_name))
    #save the subsegmented images
    for i in np.arange(sub_frame_num):
    
        img = Image.fromarray(hold_img[i])
        img.save(hold_name[i])
    return hold_img