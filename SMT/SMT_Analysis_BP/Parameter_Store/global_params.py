'''
This file contains all the global parameters that are used in the code. 
These parameters are used in different modules of the code, and may not be useful for the user to change.

Author: Baljyot Singh Parmar
Date: 2023-04-20
'''

PIXELSIZES = {"olympus_pixel_size":130,
              "confocal_pixel_size":79,
              "olympus_pixel_size_1x1":65}

#AUD to photon conversion factor parameters
PHOTON_CONVERSION_FACTORS = {"olympus_sCMOS_old":0.49,
                             "olympus_sCMOS_new":0.23,
                             "confocal":None}

READOUT_NOISE = {"olympus_sCMOS_old":1.9,
                 "olympus_sCMOS_new":1.35,
                 "confocal":None} #electrons rms

DARK_OFFSET = {"olympus_sCMOS_old":100,
               "olympus_sCMOS_new":100,
               "confocal":None} #AUD (analog to digital units)

QUANTUM_EFFICIENCY_561 = {"olympus_sCMOS_old":0.71,
                          "olympus_sCMOS_new":0.95,
                          "confocal":None} #in percent at 561nm wavelength