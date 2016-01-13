# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:40:59 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import sys
import os
import subprocess

# get scrpipt path
scriptpath = os.environ['HOME']
scriptpath += '/gits/scripts/2015_imagen_height_study/'
sys.path.append(os.path.abspath(scriptpath))
#
from study_central_ancil import init_data, var_explained_pgs, univariate, multivariate

# Set pathes
IMAGEN_CENTRAL = '/neurospin/brainomics/imagen_central/reproducing/height/'

# read data
print '[------------------------ reading data -------------------------------]'
covar, height, hippo, studyPgS, snps = init_data(IMAGEN_CENTRAL)

### heritability
print '[---------------------- comput heritab gcta --------------------------]'
cmd = 'python heritability_gcta.py'
result = subprocess.check_output(cmd, shell=True)
print result

# Study of Height ################
# - Polygenic score effect size
print '[---------------------- polygenic score  -----------------------------]'
subjects, lm = var_explained_pgs(covar, studyPgS)


# - now perform univariate fit
# get the ordered subject list
print '[-------------------------- MULM score  -----------------------------]'
mask = [snps.subject_ids.tolist().index(i) for i in subjects]
X, Y = univariate(mask, snps, studyPgS)



# - now perform multivariate analysis
print '[----------------------- mutlivariate analysis -----------------------]'
multivariate(X, Y)
