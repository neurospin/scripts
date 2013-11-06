# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:11:50 2013

@author: jl237561
"""

import os, sys, fnmatch
import getpass
import numpy, pandas
import tables
import nibabel
sys.path.append('/home/vf140245/gits/nilearn')
from nilearn.image.resampling import resample_img
import numpy as np


def convert_path(path):
# this function can be removed after jinpeng get right to access data
    if getpass.getuser() == "jl237561":
        path = '~' + path
        path = os.path.expanduser(path)
    return path

# Input
BASE_DIR = '/neurospin/brainomics/2013_imagen_bmi/'
BASE_DIR = convert_path(BASE_DIR)
DATA_DIR=os.path.join(BASE_DIR, 'data')
CLINIC_DIR=os.path.join(DATA_DIR, 'clinic');

# Output files
OUT_DIR=os.path.join(DATA_DIR, 'dataset_pa_prace')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ======================================================================
# 1 Image subjects
df = pandas.io.parsers.read_csv(os.path.join(CLINIC_DIR, '1534bmi-vincent2.csv'))
subject_indices = df.Subjects


# ======================================================================
# 2 Cov subjects

cfn = os.path.join(CLINIC_DIR, '1534bmi-vincent2.csv')
# load this file to check that there are 1534 common subjects accros
#   - csv file
#   - genotyping file
#   - image file

covdata = open(cfn).read().split('\n')[:-1]
cov_header = covdata[0]
covdata = covdata[1:]
cov_subj = ["%012d"%int(i.split(',')[0]) for i in covdata]