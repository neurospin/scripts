# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:54:54 2016

@author: ad247405
"""

import numpy as np
from sklearn import metrics 
import os

import os
import json
import time
import shutil

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from brainomics import plot_utilities
from parsimony.utils import plot_map2d



INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation"
INPUT_BASE_DATA_DIR = os.path.join(INPUT_BASE_DIR, "data_0.1")
INPUT_BASE_RESULTS_DIR = os.path.join(INPUT_BASE_DIR, "results_0.1_1e-8")
INPUT_MASK_DIR = os.path.join(INPUT_BASE_DATA_DIR, "masks")
INPUT_DATA_DIR_FORMAT = os.path.join(INPUT_BASE_DATA_DIR,
                                     "data_100_100_{set}")
INPUT_RESULTS_DIR_FORMAT = os.path.join(INPUT_BASE_RESULTS_DIR,
                                    "data_100_100_{set}")
                                    
                                     
for set in range(1,50):
    input_dir = INPUT_RESULTS_DIR_FORMAT.format(set=set)
    print input_dir
    os.chdir(input_dir)   
    map_cmd=" mapreduce.py --map config.json --ncore=5"
    os.system(map_cmd)
    reduce_cmd = " mapreduce.py --reduce config.json --ncore=2"  
    os.system(reduce_cmd)
    
    
