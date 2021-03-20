import sys
import os
import time

import numpy as np
import nibabel
import pandas as pd
import matplotlib.pylab as plt
import nilearn
from nilearn.image import resample_to_img

from nilearn import plotting
import argparse
import glob
from scipy import ndimage

from matplotlib.backends.backend_pdf import PdfPages
from nitk.atlases import fetch_atlas_harvard_oxford
from nitk.atlases import fetch_cortex
from nitk.image import make_sphere
from  nitk.image import rm_small_clusters

#from  nitk.image import compute_brain_mask
#from  nitk.image import rm_small_clusters
# sys.path.append('/home/ed203246/git/scripts/2021_wmh_memento+rundmc')

FS = "/home/ed203246/data"
# RADIUS = 5 # 5mm from cortex and ventriculus
RADIUS = 10 # 5mm from cortex and ventriculus

#%% MEMENTO_RUNDMC
MEMENTO_RUNDMC_PATH = "{FS}/2021_wmh_memento+rundmc".format(FS=FS)
MEMENTO_RUNDMC_DATA = os.path.join(MEMENTO_RUNDMC_PATH, "data")

#%% MEMENTO
MEMENTO_PATH = "{FS}/2017_memento/analysis/WMH".format(FS=FS)
MEMENTO_DATA = os.path.join(MEMENTO_PATH, "data")

#%% RUNDMC
RUNDMC_PATH = "{FS}/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca".format(FS=FS)
RUNDMC_DATA = os.path.join(RUNDMC_PATH, "data")

memento_mask_img_filename = os.path.join(MEMENTO_DATA, "mask.nii.gz")
rundmc_mask_img_filename = os.path.join(RUNDMC_DATA, "mask.nii.gz")

memento_mask_img = nibabel.load(memento_mask_img_filename)
rundmc_mask_img = nibabel.load(rundmc_mask_img_filename)

memento_mask_img.affine.round(3)
rundmc_mask_img.affine.round(3)

#mask_img = memento_mask_img

################################################################################
# Cortex and dilated cortex

fsl_home = "/usr/share/fsl"
MNI152_T1_1mm_brain_mask_img = nibabel.load(os.path.join(fsl_home, "data/standard/MNI152_T1_1mm_brain_mask.nii.gz"))

mask_cortex_1mm_img = fetch_cortex(fsl_home=fsl_home)
cort_arr = mask_cortex_1mm_img.get_fdata() != 0

struct = make_sphere(box_shape=(21, 21, 21), center=[10, 10, 10], radius=RADIUS)

# 1x1x1 mm3 image, <1CM from cortex
# binary_dilation: "i.e. are considered as neighbors of the central element.
# Elements up to a **squared distance** of connectivity from the center are considered neighbors."
cort_arr_dill = ndimage.binary_dilation(cort_arr, structure=struct, iterations=1)
cort_arr_dill = cort_arr_dill & (MNI152_T1_1mm_brain_mask_img.get_fdata() != 0)

mask_cortex_dill5mm_1mm_img = nilearn.image.new_img_like(mask_cortex_1mm_img, cort_arr_dill)

mask_cortex_1mm_img.to_filename("/tmp/mask_cortex_1mm.nii.gz")
mask_cortex_dill5mm_1mm_img.to_filename("/tmp/mask_cortex-dill5mm_1mm.nii.gz")

################################################################################
# Ventricles and dilated Ventricles

fsl_home = "/usr/share/fsl"
MNI152_T1_1mm_brain_mask_img = nibabel.load(os.path.join(fsl_home, "data/standard/MNI152_T1_1mm_brain_mask.nii.gz"))

cort_img, cort_labels, sub_img, sub_labels = fetch_atlas_harvard_oxford()
# cort_labels = {name:lab for lab, name in enumerate(cort_labels)}
sub_labels = {name:lab for lab, name in enumerate(sub_labels)}
mask_vent_arr = (sub_img.get_fdata() == sub_labels['Left Lateral Ventrical']) |\
    (sub_img.get_fdata() == sub_labels['Right Lateral Ventricle'])

struct = make_sphere(box_shape=(21, 21, 21), center=[10, 10, 10], radius=RADIUS)

# 1x1x1 mm3 image, <1CM from cortex
# binary_dilation: "i.e. are considered as neighbors of the central element.
# Elements up to a **squared distance** of connectivity from the center are considered neighbors."
mask_vent_arr_dill = ndimage.binary_dilation(mask_vent_arr, structure=struct, iterations=1)

mask_vent_1mm_img = nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, mask_vent_arr)
mask_vent_dill5mm_1mm_img = nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, mask_vent_arr_dill)

mask_vent_1mm_img.to_filename("/tmp/mask_ventricles_1mm.nii.gz")
mask_vent_dill5mm_1mm_img.to_filename("/tmp/mask_ventricles-dill5mm_1mm.nii.gz")

################################################################################
# Deep WM

# struct = make_sphere(box_shape=(41, 41, 41), center=[20, 20, 20], radius=20)
# cort_arr_closed = ndimage.binary_closing(cort_arr_dill, structure=struct, iterations=1)
# nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, cort_arr_closed).to_filename("/tmp/mask_cortex_closed_1mm.nii.gz")
# wmdeep_arr = cort_arr_closed.copy()

wmdeep_arr = (sub_img.get_fdata() == sub_labels['Left Cerebral White Matter']) |\
             (sub_img.get_fdata() == sub_labels['Right Cerebral White Matter'])
wmdeep_arr = rm_small_clusters(wmdeep_arr, clust_size_thres=100000)
nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, wmdeep_arr).to_filename("/tmp/test.nii.gz")

wmdeep_arr = ndimage.binary_dilation(wmdeep_arr, iterations=2)
wmdeep_arr[cort_arr_dill | mask_vent_arr_dill] = 0
mask_wmdeep_1mm_img = nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, wmdeep_arr)
mask_wmdeep_1mm_img.to_filename("/tmp/mask_wmdeep_1mm.nii.gz")


################################################################################
# Merge

mask_cortex_ventricles_dill_deepwm_mni_1mm = cort_arr_dill.copy().astype(int)
mask_cortex_ventricles_dill_deepwm_mni_1mm[mask_vent_arr_dill] = 2
mask_cortex_ventricles_dill_deepwm_mni_1mm[wmdeep_arr] = 3
mask_cortex_ventricles_dill_deepwm_mni_1mm_img = nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, mask_cortex_ventricles_dill_deepwm_mni_1mm)

tissue_filename = os.path.join(MEMENTO_RUNDMC_DATA, "mask_cortex_ventricles_dill%imm_deepwm_mni_1mm.nii.gz" % RADIUS)
mask_cortex_ventricles_dill_deepwm_mni_1mm_img.to_filename(tissue_filename)

################################################################################
# ICBM DTI-81 Atlas http://www.bmap.ucla.edu/portfolio/atlases/ICBM_DTI-81_Atlas/
# Same mask but use ICBM DTI-81 Atlas to determine deepwm
# mask_pericortex_ventricles_dill_ mask_cortex_ventricles_dill_deepwm_mni_1mm.copy()

jhu_icbm_labels_mni_1mm_img = nibabel.load(os.path.join(fsl_home, "data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz"))
assert np.all(jhu_icbm_labels_mni_1mm_img.affine == mask_cortex_ventricles_dill_deepwm_mni_1mm_img.affine)
mask_jhu_icbm = jhu_icbm_labels_mni_1mm_img.get_fdata() != 0

# Same mask but old deepwm becomes "other" ie 1
mask_other_ventricles_dill_icbmdti81 = mask_cortex_ventricles_dill_deepwm_mni_1mm_img.get_fdata().copy()
mask_other_ventricles_dill_icbmdti81[mask_other_ventricles_dill_icbmdti81 == 3] = 1

# deepwm stem from mask_jhu_icbm
mask_other_ventricles_dill_icbmdti81[mask_jhu_icbm] = 3

mask_other_ventricles_dill_icbmdti81_img = nilearn.image.new_img_like(MNI152_T1_1mm_brain_mask_img, mask_other_ventricles_dill_icbmdti81)

tissue_filename = os.path.join(MEMENTO_RUNDMC_DATA, "mask_other_ventricles_dill%imm_jhu-icbm-dti-81_mni_1mm.nii.gz" % RADIUS)
mask_other_ventricles_dill_icbmdti81_img.to_filename(tissue_filename)
# tissue_filename = '/home/ed203246/data/2021_wmh_memento+rundmc/data/mask_other_ventricles_dill10mm_jhu-icbm-dti-81_mni_1mm.nii.gz'