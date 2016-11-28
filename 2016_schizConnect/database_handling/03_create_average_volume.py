#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:23:58 2016

@author: ad247405
"""


import glob
import os
import numpy as np
import os
import pandas as pd

BASE_PATH =  '/neurospin/abide/schizConnect'
DATA_PATH = os.path.join(BASE_PATH,"data_nifti_format")

in_file = os.path.join(BASE_PATH,"list_nifti_images.csv")
images = pd.read_csv(in_file,delimiter=',')



assert len(images.subjectid.unique()) == 763
assert len(images) == len(images.path.unique()) == 3268


list_subjects = images.subjectid.unique()
number_subjects = len(images.subjectid.unique())
for i in list_subjects:  
    current_subject = images[images.subjectid ==i].reset_index(drop=True)
    for j in  current_subject.img_date.unique():            
        current_visit = current_subject[current_subject.img_date == j].reset_index()
        current_visit.path        
        if len(current_visit.path) >1:
            cmd = 'fsl5.0-flirt_average ' + str(len(current_visit.path)) + ' '
            for k in range(len(current_visit.path)):
                cmd = cmd + current_visit.path[k] + ' '
            output = os.path.dirname(os.path.dirname(current_visit.path[k]))   +"/average_T1.nii"
            cmd = cmd + output                                                
            os.system(cmd)
            print("Subject %s : multiple acquisitions have been correctly registered and averaged"%(i))
        else:  
            print("Subject %s : Only one acquisition"%(i))
    

           