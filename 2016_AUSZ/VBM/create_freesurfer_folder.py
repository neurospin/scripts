# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:33:49 2016

@author: ad247405
"""

import os
import numpy as np
from scipy import ndimage
import os, os.path, sys
import pylab
import pandas as pd 
import nibabel as nib
import glob

BASE_PATH =  '/neurospin/brainomics/2016_AUSZ'
in_file ='/neurospin/brainomics/2016_AUSZ/data/list_dicom_dir.txt'


FS_directory = '/neurospin/brainomics/2016_AUSZ/data/Freesurfer'

inf=open(in_file, "r")


for subject_dir in inf.readlines():
    subject_dir = subject_dir.replace("\n","")   
    os.chdir(subject_dir)
    print(subject_dir)
    name = os.path.basename(os.path.split(subject_dir)[0])
    cmd = "cp T1_VBM/T1.nii "+ FS_directory+'/'+name+'_T1.nii'
    os.system(cmd)








#
#
#for subject_dir in inf.readlines():
#    subject_dir = subject_dir.replace("\n","")   
#    os.chdir(subject_dir)
#    print subject_dir
#    name = os.path.basename(os.path.split(subject_dir)[0])
#    os.chdir(FS_directory)
#    cmd = "mkdir " + name 
#    os.system(cmd)
#    os.chdir(subject_dir)
#    cmd = "cp T1_VBM/T1.nii "+ FS_directory+'/'+name+'/T1.nii'
#    os.system(cmd)
#    
    

 