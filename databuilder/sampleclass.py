"""
Sample Class : Store data samples in dictionaries

"""

import numpy as np
import copy

class SampleDict(dict):
    def __init__(self, *arg, **kw):
        super(SampleDict, self).__init__(*arg, **kw)
    
        self.__setitem__("x", []) 
        self.__setitem__("y", [])


    def concat(self, f_dict):
        for key in self:
            if len(self[key]) == 0:
                self[key] = f_dict[key]
            elif len(f_dict[key]) == 0:
                pass
            else:
                self[key] = np.concatenate((self[key], f_dict[key]), axis = 0)
