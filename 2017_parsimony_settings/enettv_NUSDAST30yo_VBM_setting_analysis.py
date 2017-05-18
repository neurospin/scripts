#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:27:44 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import array_utils
import nilearn
from nilearn import image
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting


INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/mask.nii'

babel_mask  = nibabel.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3

#Save Beta map for each conesta iteration in nifti format and display glass brain with iterations info

BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
VBM/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)
    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    os.makedirs(snap_path, exist_ok=True)
    beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
    os.makedirs(beta_path, exist_ok=True)
    conesta_ite = sorted(os.listdir(snap_path))
    nb_conesta = len(conesta_ite)
    i=1
    pdf_path = os.path.join(BASE_PATH,p,"weight_map_across_iterations.pdf")
    pdf = PdfPages(pdf_path)
    fig = plt.figure(figsize=(11.69, 8.27))
    for ite in conesta_ite:
        path = os.path.join(snap_path,ite)
        conesta_ite_number = ite[-11:-4]
        print (conesta_ite_number)
        ite = np.load(path)
        fista_ite_nb =ite['continuation_ite_nb'][-1]
        beta = ite["beta"][penalty_start:,:]
        arr = np.zeros(mask_bool.shape);
        arr[mask_bool] = beta.ravel()
        out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
        filename = os.path.join(beta_path,"beta_"+conesta_ite_number+".nii.gz")
        out_im.to_filename(filename)
        beta = nibabel.load(filename).get_data()
        beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
        fig.add_subplot(nb_conesta,1,i)
        title = "CONESTA iterations: " + str(i) + " -  FISTA iterations : " + str(fista_ite_nb)
        nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,title = title)
        pdf.savefig()
        plt.close(fig)
        i = i +1
    pdf.close()



