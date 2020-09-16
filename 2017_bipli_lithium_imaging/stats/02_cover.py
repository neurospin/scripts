import os
import numpy as np
import scipy
import pandas as pd
import nibabel
#import brainomics.image_atlas
import nilearn
from nilearn import plotting
from mulm import MUOLS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
#import array_utils
#import proj_classif_config
import re
import glob
import json

import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting


WD = "/home/ed203246/documents/publications/2020/7Li_MRI_Jacques/cover"
INPUT_DATA = os.path.join(WD, 'stats')


import scipy

map_img = nibabel.load(os.path.join(INPUT_DATA, "li~1+age+sex_tstat1.nii.gz"))
map_arr = map_img.get_data()
map_arr[map_arr < 0] = 0

logpval_img = nibabel.load(os.path.join(INPUT_DATA, "li~1+age+sex_vox_p_tstat-mulm_log10.nii.gz"))
mask_pval = logpval_img.get_data() > 2.8
map_arr[~mask_pval] = 0

########################################################################################################################
# Build a mask of stat > 2 with intersection with significant ROIs

atlas_sub_img = nibabel.load(os.path.join(INPUT_DATA, "atlas_harvard_oxford_sub.nii.gz"))
atlas_sub_arr = atlas_sub_img.get_data()
atlas_sub_msk = (atlas_sub_arr == 9) | (atlas_sub_arr == 19) | (atlas_sub_arr == 7) | (atlas_sub_arr == 18)
atlas_Hippocampus_msk_arr = (atlas_sub_arr == 9)# | (atlas_sub_arr == 19)
atlas_Hippocampus_msk_img = nibabel.Nifti1Image(atlas_Hippocampus_msk_arr.astype(int), map_img.affine)
atlas_Pallidum_msk_arr = (atlas_sub_arr == 18)# | (atlas_sub_arr == 19)
atlas_Pallidum_msk_img = nibabel.Nifti1Image(atlas_Pallidum_msk_arr.astype(int), map_img.affine)

# "Left Hippocampus": 9
# "Right Hippocampus": 19
# "Left Pallidum": 7
#  "Right Pallidum": 18

#threstval = 3
#
map_labels, n_clusts = scipy.ndimage.label(map_arr > 0)
print([[lab, np.sum(map_labels == lab)] for lab in  np.unique(map_labels)[1:]])
# [[1, 96], [2, 266], [3, 14], [4, 33], [5, 86], [6, 133], [7, 108], [8, 553], [9, 41], [10, 14], [11, 123]]
# intersection ROI stat > 3
map_labels_in_roi = np.unique(map_labels[atlas_sub_msk & (map_labels != 0)])
mask_map_in_roi = np.zeros(map_arr.shape, dtype=bool)
for lab in map_labels_in_roi:
    mask_map_in_roi[map_labels == lab] = True

map_arr[~mask_map_in_roi] = 0
map_img = nibabel.Nifti1Image(map_arr, map_img.affine)
map_img.to_filename( os.path.join(WD, "li~1+age+sex_tstat1_thresh.nii.gz"))

map_arr_mask = np.copy(map_arr)
map_arr_mask[map_arr != 0] = 1
map_mask_img = nibabel.Nifti1Image(map_arr_mask.astype(int), map_img.affine)
map_mask_img = nibabel.Nifti1Image(map_arr_mask, map_img.affine)
map_mask_img.to_filename(os.path.join(WD, "li~1+age+sex_tstat1_thresh_mask.nii"))
map_mask_img.header["data_type"]

"""
AimsFileConvert li~1+age+sex_tstat1_thresh_mask.nii li~1+age+sex_tstat1_thresh_mask.nii -t S16
AimsMeshBrain -i li~1+age+sex_tstat1_thresh_mask.nii -o li~1+age+sex_tstat1_thresh_mask.gii

AimsThreshold -i stats/li~1+age+sex_tstat1.nii.gz -t 2 -m ge -o stats/li~1+age+sex_tstat1_thres2.nii.gz

AimsMesh -i li~1+age+sex_tstat1_thresh_mask.nii -o li~1+age+sex_tstat1_thresh_mask.mesh
AimsMeshSmooth -i li~1+age+sex_tstat1_thresh_mask.mesh

# Open anatomist
anatomist /neurospin/brainomics/neuroimaging_ressources/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz stats/li~1+age+sex_tstat1.nii.gz li~1+age+sex_tstat1_thresh_mask_1_1.mesh /neurospin/brainomics/neuroimaging_ressources/bv_typical/mni_single_subject/single_subj_T1/t1mri/default_acquisition/single_subj_T1.nii.gz /neurospin/brainomics/neuroimaging_ressources/bv_typical/mni_single_subject/single_subj_T1/t1mri/default_acquisition/default_analysis/segmentation/mesh/single_subj_T1_*hemi.gii


cd /volatile/duchesnay/mega/documents/publications/2020/7Li_MRI_Jacques/cover/SNAPSHOTS

cp originals/*.png ./

ls *.png|while read input; do
convert  $input -trim /tmp/toto.png;
convert  /tmp/toto.png -transparent black $input;
convert $input -resize 20% -quality .1 "${input%.png}_small.png"
done


"""