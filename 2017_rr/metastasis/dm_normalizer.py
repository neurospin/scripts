# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:29:35 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import argparse
import os.path
from glob import glob
import json
import numpy as np
import nibabel as ni



doc = """
python $HOME/gits/scripts/2017_rr/metastasis/dm_normalizer.py \
   --subject  \
   --out 
"""


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out',  required=True,
                    help='outfn.')
parser.add_argument('-s', '--subjectdir',
                    required=True,
                    help='infn')
#
args = parser.parse_args()
subject = os.path.basename(args.subjectdir)

im = ni.load(subject)
imd = im.get_data()
oud = np.zeros(imd.shape, dtype='float')
oud[imd>0] = (imd[imd>0]*1. - np.mean(imd[imd>0]))/np.std(imd[imd>0])

ni.save(ni.Nifti1Image(oud, affine=im.affine), args.out)