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

