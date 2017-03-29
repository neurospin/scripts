#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:18:08 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import glob


#############################################################################
#SCHIZCONNECT cohort
#############################################################################
os.chdir("/neurospin/abide/schizConnect/processed/freesurfer/screenshots_seg")

name_subjects = "/neurospin/abide/schizConnect/processed/list_subjects.txt"

list_subjects = []
with open(name_subjects) as f:
    subjects = f.readlines()
for i in subjects:
    list_subjects.append(i.rstrip('\n'))
    
for i in range(len(list_subjects)):
    cmd = "convert " + "sagital_1_"+list_subjects[i] + ".png sagital_2_" +list_subjects[i]\
                                             + ".png sagital_3_"  +list_subjects[i] + ".png "\
                                              + "QC_planche/"+list_subjects[i] + ".pdf" 
                                                                           
    os.system(cmd)
    print (i)                                                                       




#############################################################################
#VIP cohort
#############################################################################
os.chdir("/neurospin/brainomics/2016_schizConnect/analysis/VIP/data/original_data/screenshots_seg")

name_subjects = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/data/original_data/list.txt"

list_subjects = []
with open(name_subjects) as f:
    subjects = f.readlines()
for i in subjects:
    list_subjects.append(i.rstrip('\n'))
    
for i in range(len(list_subjects)):
    cmd = "convert " + "sagital_1_"+list_subjects[i] + ".png sagital_2_" +list_subjects[i]\
                                             + ".png sagital_3_"  +list_subjects[i] + ".png "\
                                              + "QC_planche/"+list_subjects[i] + ".pdf" 
                                                                           
    os.system(cmd)
    print (i)                                                                       
