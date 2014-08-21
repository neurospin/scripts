#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:31:17 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import os, sys
import argparse
import nibabel, numpy as np
import json


parser = argparse.ArgumentParser()
# Config file
parser.add_argument('config', help='Configuration json file')
parser.add_argument('--keys', '-k', help='keys list separated by space')

options = parser.parse_args()

if not options.config:
    print 'Required config file'
    sys.exit(1)

if not options.keys:
    print 'keys list separated '
    sys.exit(1)

KEYS = options.keys.split()

if os.path.dirname(options.config) : 
    os.chdir(os.path.dirname(options.config))

config = json.load(open(options.config))
PENALTY_START = config["penalty_start"]
INPUT_MASK = config['mask_filename']


OUTPUT = os.path.join("results", "weigths_map")
if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

mask_image = nibabel.load(INPUT_MASK)
mask = mask_image.get_data() != 0
outfilenames = list()
for key in KEYS:
    #key = keys[0]
    beta = np.load(os.path.join("results/0", key, "beta.npz"))['arr_0']
    arr = np.zeros(mask.shape)
    arr[mask] = beta[PENALTY_START:].ravel()
    out_im = nibabel.Nifti1Image(arr,affine=mask_image.get_affine())
    if OUTPUT is None:
        outfilename = os.path.join("results/0", key, "beta_%s.nii.gz" % key)
    else:
        outfilename = os.path.join(OUTPUT, "beta_%s.nii.gz" % key)
    out_im.to_filename(outfilename)
    outfilenames.append(outfilename)


for nii in outfilenames:
    mesh_cmd = os.path.join(os.path.dirname(__file__), "weigths_map_mesh.py")
    cmd = "bv_env %s --input %s" % (mesh_cmd, os.path.abspath(nii))
    print cmd
