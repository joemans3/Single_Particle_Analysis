import os
import sys
import numpy as np

#allowed types for the start values in Counter_start_stop
ALLOWED_START_TYPES = [int,float]


#define a class that acts as a counter which gives the next integer given a starting integer. 
#it needs a call method which returns the next integer and a reset method which resets the counter to the starting integer

class Counter_start_stop:
    def __init__(self,start,increment=1):
        #check the type of start
        if type(start) not in ALLOWED_START_TYPES:
            raise TypeError("start must be of type: {}".format(ALLOWED_START_TYPES))
        #check the type of increment
        if type(increment) not in ALLOWED_START_TYPES:
            raise TypeError("increment must be of types: {}".format(ALLOWED_START_TYPES))
        #cache the start and increment
        self.start = start
        self.increment = increment
        self.counter = start
    def __call__(self):
        self.counter += 1
        return self.counter - 1
    def reset(self):
        self.counter = self.start




