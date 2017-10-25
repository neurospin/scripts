#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:47:10 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import array_utils
import nilearn
from nilearn import plotting
from nilearn import image
import matplotlib.pyplot as plt


#https://neurovault.org/images/1700/
OUTPUT = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/figures_ROIs"


STRIATUM_MASK = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/data_centered_excluding_striatum/mask_striatum.nii"


nilearn.plotting.plot_glass_brain(STRIATUM_MASK,colorbar=False,plot_abs=False,vmax=1.5)
plt.savefig(os.path.join(OUTPUT, "striatum.png"))



ATLAS_PATH = "/neurospin/brainomics/2016_schizConnect/atlas"

sub_image = nibabel.load(os.path.join(ATLAS_PATH,"HarvardOxford-sub-maxprob-thr0-1.5mm.nii.gz"))
sub_arr = sub_image.get_data()




putamen = np.logical_or(sub_arr == 17,sub_arr == 6)
out_im = nibabel.Nifti1Image(putamen.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "putamen.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "putamen.nii"),colorbar=False,plot_abs=False,vmax=1.5)
plt.savefig(os.path.join(OUTPUT, "putamen.png"))


caudate = np.logical_or(sub_arr == 16,sub_arr == 5)
out_im = nibabel.Nifti1Image(caudate.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "caudate.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "caudate.nii"),colorbar=False,plot_abs=False,vmax=1.5)
plt.savefig(os.path.join(OUTPUT, "caudate.png"))



pallidum = np.logical_or(sub_arr == 18,sub_arr == 7)
out_im = nibabel.Nifti1Image(pallidum.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "pallidum.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "pallidum.nii"),colorbar=False,plot_abs=False,vmax=1.5)
plt.savefig(os.path.join(OUTPUT, "pallidum.png"))





hippocampus = np.logical_or(sub_arr == 19   ,sub_arr == 9)
out_im = nibabel.Nifti1Image(hippocampus.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "hippocampus.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "hippocampus.nii"),colorbar=False,plot_abs=False,vmax=1.5,cmap="Blues")
plt.savefig(os.path.join(OUTPUT, "hippocampus.png"))



accumbens = np.logical_or(sub_arr == 11    ,sub_arr == 21)
out_im = nibabel.Nifti1Image(accumbens.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "accumbens.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "accumbens.nii"),colorbar=False,plot_abs=False,vmax=1.5,cmap="Blues")
plt.savefig(os.path.join(OUTPUT, "accumbens.png"))




thalamus = np.logical_or(sub_arr == 4    ,sub_arr == 15)
out_im = nibabel.Nifti1Image(thalamus.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "thalamus.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "thalamus.nii"),colorbar=False,plot_abs=False,vmax=1.5,cmap="Blues")
plt.savefig(os.path.join(OUTPUT, "thalamus.png"))



amygdala = np.logical_or(sub_arr == 10    ,sub_arr == 20)
out_im = nibabel.Nifti1Image(amygdala.astype("int16"),
                             affine=sub_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "amygdala.nii"))
nilearn.plotting.plot_glass_brain(os.path.join(OUTPUT, "amygdala.nii"),colorbar=False,plot_abs=False,vmax=1.5,cmap="Blues")
plt.savefig(os.path.join(OUTPUT, "amygdala.png"))
