#file_ll_path = "/media/mma/mescog/originals/Munich/CAD_bioclinica_nifti/1001/1001-M0-LL.nii.gz"
base_dir = "/neurospin/mescog/neuroimaging/cadasil"

import os, os.path
#from soma import aims
#import tempfile
#import scipy, scipy.ndimage
import glob
import numpy as np
import nibabel as nib

file_paths = glob.glob(os.path.join(base_dir, "CAD_bioclinica_nifti", "*", "*M0-WMH.nii.gz"))
print len(file_paths) # 882

# QC and warp into MNI
# ====================
file_path = file_paths[0]
# fsl5.0-applywarp -i invol -o outvol -r refvol -w warpvol
# --premat=flirted_old_brain.mat

# refvol
# /i2bm/local/fsl-5.0.6/data/standard/MNI152lin_T1_2mm.nii.gz
# warpvol
# /neurospin/mescog/neuroimaging/cadasil/Normalization/CAD/<id>/

#Compose 
#rFLAIR_to_rT1.mat : Rigid boby transfor (FSL flirt 4.1.6)
#With
#rT1_to_MNI_warp.nii.gz: non linear tranfo
#Using 

#run scripts/2013_mescog/proj_wmh_patterns/00_build_dataset.py
#X = np.vstack(arr_list)

# Compute mask
# ============

# Mask out
# ========

# Save X
# ======