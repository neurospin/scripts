# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:08:31 2016

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
in_file ='/neurospin/brainomics/2016_AUSZ/subject_DICOM_dir.txt'




inf=open(in_file, "r")


for subject_dir in inf.readlines():
    subject_dir = subject_dir.replace("\n","")   
    os.chdir(subject_dir)
    list_acquisitions = glob.glob(subject_dir+'*')
    print subject_dir
    for i in range (len(list_acquisitions)):
        seq = os.path.basename(list_acquisitions[i])
        name = "sequence_" + seq
        cmd = "dinifti "+ list_acquisitions[i] + " " + name
        os.system(cmd)
 

#CAREFUL, differetn nomenclature for '/neurospin/brainomics/2016_AUSZ/data/AUSZ_2013/Lp130187/Ausz - 11446'
#dinifti '/neurospin/brainomics/2016_AUSZ/data/AUSZ_2013/Lp130187/Ausz - 11446/SAG_FSPGR_3D_IR_3' sequence_SAG_T1.nii

#For AUSZ 2012 and 2013, the sequence_3 correspond to the T1 acquisition
#For AUSZ 2014, 2015 and 2016, the sequence SE0 correspond to the T1 acquisition