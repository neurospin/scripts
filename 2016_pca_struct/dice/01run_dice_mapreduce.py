# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:54:54 2016

@author: ad247405
"""

import os




INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice/"
#INPUT_BASE_RESULTS_DIR = os.path.join(INPUT_BASE_DIR, "results_10comp")
#INPUT_RESULTS_DIR_FORMAT = os.path.join(INPUT_BASE_RESULTS_DIR,"data_100_100_{set}")

INPUT_BASE_RESULTS_DIR = os.path.join(INPUT_BASE_DIR,"local_minima_experiment")
INPUT_RESULTS_DIR_FORMAT = os.path.join(INPUT_BASE_RESULTS_DIR,"data_100_100_{set}")
                                     
                                     
for set in range(0,50):
    input_dir = INPUT_RESULTS_DIR_FORMAT.format(set=set)
    print (input_dir)
    os.chdir(input_dir)   
    map_cmd=" mapreduce.py --map config.json --ncore=3"
    os.system(map_cmd)
#    print ("set: ", set)
#    reduce_cmd = " mapreduce.py --reduce config_dCV.json --ncore=1"  
#    os.system(reduce_cmd)
    
    

