# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 2013

@author: vf140245

This script perform :
    - the reading of the SNP data
    - the Nan fix
    - th writing in a hdf5 

It uses a recent version of igutils.
The git repository can be cloned at https://github.com:VincentFrouin/igutils.git

Add the following in the script supposing you cloned the repos in ~/gits
import sys
sys.path.append('~/gits/igutils')

"""
import sys
import getpass
sys.path.append('/home/vf140245/gits/igutils')
import os
import igutils as ig
import numpy as np


def convert_path(path):
    if getpass.getuser() == "jl237561":
        path = '~' + path
        path = os.path.expanduser(path)
    return path


def check_array_NaN(nparray):
    if np.isnan(nparray).any():
        raise ValueError("np.array contain NaN")

# Input
BASE_DIR='/neurospin/brainomics/2013_imagen_bmi/'
BASE_DIR = convert_path(BASE_DIR)
DATA_DIR=os.path.join(BASE_DIR, 'data')
CLINIC_DIR=os.path.join(DATA_DIR, 'clinic')


gfn = os.path.join(DATA_DIR, 'qc_sub_qc_gen_all_snps_common_autosome')
genotype = ig.Geno(gfn)
data = genotype.snpGenotypeAll()


#
data.shape
data.dtype
data.size
np.unique(data)
check_array_NaN(data)
data.shape
genotype.assayIID()
genotype.snpList()
len(genotype.snpList())

#Prepare the data
# - imput with the median 128=Nan
# - get the list of individuals
# - get the list of SNPs
#Push it in an hdf5 cache file
