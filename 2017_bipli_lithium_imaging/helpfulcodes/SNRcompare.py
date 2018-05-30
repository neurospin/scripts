# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:11:16 2018

@author: js247994
"""

""

import os
import nibabel
import numpy as np

matsize=[8,8,5]

directory="V:/projects/BIPLi7/Tests/2018_04_23/Phantomcomparisons/2017_12_01"
#directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".nii"):   #or filename.endswith(".py"): 
        babel_image = nibabel.load(os.path.join(directory,filename))      
        np.argmax(image)      
    else:
        continue