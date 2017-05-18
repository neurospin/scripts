# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:54:54 2016

@author: ad247405
"""

import os




INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice/2017/results_Jenatton"
INPUT_RESULTS_DIR_FORMAT = os.path.join(INPUT_BASE_DIR,"data_100_100_{set}")
                                     
                                     
for set in range(0,50):
    input_dir = INPUT_RESULTS_DIR_FORMAT.format(set=set)
    print (input_dir)
    os.chdir(input_dir)   
    #map_cmd=" mapreduce.py --map config_alpha_dCV.json --ncore=6"
    #os.system(map_cmd)
    print ("set: ", set)
    reduce_cmd = " mapreduce.py --reduce config_dCV.json --ncore=1"  
    os.system(reduce_cmd)
#    
    



