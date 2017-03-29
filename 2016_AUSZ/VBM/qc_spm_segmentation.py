#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:32:47 2017

@author: ad247405
"""

import os
import numpy as np
from scipy import ndimage
import os, os.path, sys
import pylab
from soma import aims

os.chdir('/neurospin/brainomics/2016_AUSZ/')
in_file = os.path.join('/neurospin/brainomics/2016_AUSZ',"preproc_VBM","list_subjects","list_T1.txt")
out_file = os.path.join('/neurospin/brainomics/2016_AUSZ',"preproc_VBM","tissues-volumes-globals_spm-segmentation.csv")



suffix_im = "mprage_noface.nii"
inf=open(in_file, "r")
outf=open(out_file, "w")
outf.write("subject"+" "+"dimX"+" "+"dimY"+" "+"dimZ"+" "+
        "voxsizeX"+" "+"voxsizeY"+" "+"voxsizeZ"+" "+
        "nvoxGM_nomask"+" "+"nvoxWM_nomask"+" "+"nvoxCSF_nomask"+" "+
        "nvoxGM"+" "+"nvoxWM"+" "+"nvoxCSF\n")
outf.flush()


r = aims.Reader()
for cur_anat_path in inf.readlines():
    cur_anat_path = cur_anat_path.replace("\n","")
    cur_dir = os.path.dirname(cur_anat_path)
    image = os.path.basename(cur_anat_path)
    image= image[:-3]
    

    # read segmeted images
    gm_im = r.read(os.path.join(cur_dir, "c1"+image+'nii'))
    wm_im = r.read(os.path.join(cur_dir, "c2"+image+'nii'))
    csf_im = r.read(os.path.join(cur_dir, "c3"+image+'nii'))
    skull_im = r.read(os.path.join(cur_dir, "c4"+image+'nii'))
    outside_im = r.read(os.path.join(cur_dir, "c5"+image+'nii'))

    gm_arr = gm_im.arraydata().squeeze()
    wm_arr = wm_im.arraydata().squeeze()
    csf_arr = csf_im.arraydata().squeeze()
    skull_arr = skull_im.arraydata().squeeze()
    outside_arr = outside_im.arraydata().squeeze()
    
    print "classify/",;sys.stdout.flush()
    # classify
    gm_bool_arr = (gm_arr > wm_arr) & (gm_arr > csf_arr) & \
                  (gm_arr > skull_arr) & (gm_arr > outside_arr)

    # pylab.matshow(gm_bool_arr[np.int(gm_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)

    wm_bool_arr = (wm_arr > gm_arr) & (wm_arr > csf_arr) & \
                  (wm_arr > skull_arr) & (wm_arr > outside_arr)

    # pylab.matshow(wm_bool_arr[np.int(wm_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)

    csf_bool_arr = (csf_arr > gm_arr) & (csf_arr > wm_arr) & \
                   (csf_arr > skull_arr) & (csf_arr > outside_arr)

    # pylab.matshow(csf_bool_arr[np.int(csf_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)
    
    # union of voxels segmented as qm, wm or csf (all tissues)
    alltissues_arr = gm_bool_arr | wm_bool_arr | csf_bool_arr
    #pylab.matshow(alltissues_arr[np.int(alltissues_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)
    gm_nomask = gm_bool_arr.sum()
    wm_nomask = wm_bool_arr.sum()
    csf_nomask = csf_bool_arr.sum()

    print "CCs/",;sys.stdout.flush()
    # label connected components (CCs) (26 connexity)
    alltissues_arr_labels, nlabels = ndimage.label(alltissues_arr, np.ones((3, 3, 3)))

    # get label (CCs) size
    lab_sizes = [(lab, np.sum(alltissues_arr_labels == lab)) for lab in
        np.unique(alltissues_arr_labels)]

    print "maxCC/",;sys.stdout.flush()
    # get brain mask: CC of maximal size wich is not the background
    max_size = 0
    brain_lab = None
    for lab, size in lab_sizes:
        #print lab, size
        if size>10 and\
            not (np.all((alltissues_arr_labels==lab) == (alltissues_arr==False))):
            if size > max_size:
                max_size = size
                brain_lab = lab

    brain_mask_arr = (alltissues_arr_labels == brain_lab)

    print "morpho/",;sys.stdout.flush()
    brain_mask_arr = ndimage.binary_closing(brain_mask_arr, structure=np.ones((3, 3, 3)))

    #pylab.matshow(brain_mask_arr[np.int(brain_mask_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)
    #pylab.matshow(brain_mask_arr[:,np.int(brain_mask_arr.shape[1]/2),:], cmap=pylab.cm.gray)
    #pylab.matshow(brain_mask_arr[:,:,np.int(brain_mask_arr.shape[2]/2)], cmap=pylab.cm.gray)

    print "mask/",;sys.stdout.flush()
    # apply mask
    gm_bool_arr[brain_mask_arr == False] = False
    wm_bool_arr[brain_mask_arr == False] = False
    csf_bool_arr[brain_mask_arr == False] = False

    print "to_csv";sys.stdout.flush()
    vox_vol_mm3 = np.prod(gm_im.header()['voxel_size'].arraydata())

    gm = gm_bool_arr.sum()
    wm = wm_bool_arr.sum()
    csf = csf_bool_arr.sum()
    vox_sizes = gm_im.header()['voxel_size'].arraydata()

    center = cur_dir.split("/")[-4]
    subject = cur_dir.split("/")[-3]

    outf.write(subject+" "+
          str(gm_im.getSizeX())+" "+str(gm_im.getSizeY())+" "+str(gm_im.getSizeZ())+" "+
          str(vox_sizes[0])+" "+str(vox_sizes[1])+" "+str(vox_sizes[2])+" "+
          str(gm_nomask)+" "+str(wm_nomask)+" "+str(csf_nomask)+" "+
          str(gm)+" "+str(wm)+" "+str(csf)+"\n")
    outf.flush()

outf.close()


#Make nice plot to vizualize possible bug in segmentation 
######


import csv
import pandas as pd
os.chdir('/neurospin/brainomics/2016_AUSZ/')
out_file = os.path.join('/neurospin/brainomics/2016_AUSZ',"preproc_VBM","tissues-volumes-globals_spm-segmentation.csv")


table = pd.read_csv(out_file,delim_whitespace=True)


table["propGM"] = (table['nvoxGM'])/(table['nvoxGM'] + table['nvoxWM']+table['nvoxCSF'])
table["propWM"] = (table['nvoxWM'])/(table['nvoxGM'] + table['nvoxWM'] + table['nvoxCSF'])
table["propCSF"] = (table['nvoxCSF'])/(table['nvoxGM'] + table['nvoxWM'] + table['nvoxCSF'])
table["propVoxOutMask"] = (table['nvoxGM_nomask'] + table['nvoxWM_nomask']+ table['nvoxCSF_nomask'] - (table['nvoxGM'] + table['nvoxWM']+table['nvoxCSF']))/ (table['nvoxGM_nomask'] + table['nvoxWM_nomask']+table['nvoxCSF_nomask'])

import matplotlib.pyplot as plt

plt.plot(table["propGM"],'o',label = r'$\frac{GM}{GM + WM +CSF}$')
plt.plot(table["propWM"],'o',label = r'$\frac{WM}{GM + WM +CSF}$')
plt.plot(table["propCSF"],'o',label = r'$\frac{CSF}{GM + WM +CSF}$')
plt.xlabel('subjects')
plt.title('Segmentation - Tissues proportions')
plt.grid()
plt.legend(bbox_to_anchor=(1.4,0.7))

plt.plot(table["propVoxOutMask"],'o')
plt.xlabel('subjects')
plt.title('Proportion of voxels outside of mask')
plt.grid()

