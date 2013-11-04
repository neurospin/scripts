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
sys.path.append('/home/vf140245/gits/igutils')
import os
import igutils as ig
import numpy as np

# Input
BASE_DIR='/neurospin/brainomics/2013_imagen_bmi/'
DATA_DIR=os.path.join(BASE_DIR, 'data')
CLINIC_DIR=os.path.join(DATA_DIR, 'clinic')


gfn = os.path.join(DATA_DIR, 'bmi_snp')
genotype = ig.Geno(gfn)
data = genotype.snpGenotypeAll()

#
data.shape
data.dtype
data.size
np.unique(data)
data.shape
genotype.assayIID()
genotype.snpList()
len(genotype.snpList())

#Prepare the data
# - imput with the median 128=Nan
# - get the list of individuals
# - get the list of SNPs
#Push it in an hdf5 cache file