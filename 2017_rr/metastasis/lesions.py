# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:50:13 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import nibabel as ni
import numpy as np
from glob import glob
import os.path
import argparse

tissues = ['edema', 'enh', 'necrosis']

doc = """
This command to help reorganize manually segmented data
"""
parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-p", "--process", dest="process", action='store_true',
    help="if activated execute.")


args = parser.parse_args()

masks = {}
for lesion in range(1, 3):
    for tissue in tissues:
        p = '*_model10_mask_{}_{}.nii.gz'.format(tissue, lesion)
        p = glob(p)
        print p
        if len(p) == 1:
            p = p[0]
            if not lesion in masks:
                masks[lesion] = {}
            masks[lesion][tissue] = ni.load(p)

for lesion in masks:
    for tissue in tissues:
        if tissue in masks[lesion]:
            print lesion, "-", tissue, ": ", np.unique(masks[lesion][tissue].get_data())

if args.process:
    for lesion in masks:
        for tissue in tissues:
            if tissue in masks[lesion]:
                ttype = masks[lesion][tissue].get_data_dtype()
                tshape = list(masks[lesion][tissue].header['dim'][1:4])+[len(masks[lesion])]
                taffine = masks[lesion][tissue].affine
            break;
        buf = np.zeros(tshape, dtype=ttype)
        # now create dyn image file to store all subtypes in one lesion file        
        i = 0
        for tissue in tissues:
            if tissue in masks[lesion]:
                buf[:,:,:,i] = masks[lesion][tissue].get_data()
                i += 1
        
        flesion = os.path.basename(masks[lesion][tissue].get_filename()).split('_')[0]
        mlesion = os.path.basename(masks[lesion][tissue].get_filename()).split('_')[1]
        flesion = flesion + '_' + mlesion + '_mask_lesion-{}'.format(lesion) + '.nii.gz'
#        print flesion
        ni.save(ni.Nifti1Image(buf, affine=taffine), flesion)
            