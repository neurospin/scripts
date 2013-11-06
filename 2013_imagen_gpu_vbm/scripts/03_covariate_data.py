# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5th 2013

@author: vf140245

This script perform :
    - the writing in a hdf5 

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


cfn = os.path.join(CLINIC_DIR, '1534bmi-vincent2.csv')
# load this file to check that there are 1534 common subjects accros
#   - csv file
#   - genotyping file
#   - image file

covdata = open(cfn).read().split('\n')[:-1]
cov_header = covdata[0]
covdata = covdata[1:]
cov_subj = ["%012d"%int(i.split(',')[0]) for i in covdata]

gfn = os.path.join(DATA_DIR, 'qc_sub_qc_gen_all_snps_common_autosome')
genotype = ig.Geno(gfn)
geno_subj = genotype.assayIID()

len(set(cov_subj).intersection(set(geno_subj)))



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
