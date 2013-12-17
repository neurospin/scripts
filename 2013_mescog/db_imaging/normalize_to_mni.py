import os, os.path
#from soma import aims
#import tempfile
#import scipy, scipy.ndimage
import glob
import numpy as np
import nibabel as nib

#file_ll_path = "/media/mma/mescog/originals/Munich/CAD_bioclinica_nifti/1001/1001-M0-LL.nii.gz"
base_dir = "/neurospin/mescog/neuroimaging/cadasil"
base_dir = "/home/edouard/data/mescog/neuroimaging/cadasil"
fsl_mni_filepath = "/usr/share/data/fsl-mni152-templates/MNI152lin_T1_2mm.nii.gz"
temp_dir = "/tmp"


subject_paths = glob.glob(os.path.join(base_dir, "CAD_bioclinica_nifti", "*"))

# QC and warp into MNI
# ====================
subject_path = subject_paths[0]
#if not os.path.isdir(dir_path): continue:
#, "*M0-WMH.nii.gz"))
subject_id = os.path.basename(subject_path)

wmh_filepath = glob.glob(os.path.join(subject_path, "*M0-WMH.nii.gz"))
rflair_filepath = glob.glob(os.path.join(subject_path, "*-M0-rFLAIR.nii.gz"))
rt1_filepath = glob.glob(os.path.join(subject_path, "*-M0-rT1.nii.gz"))

trm_basedir = os.path.join(base_dir, "Normalization", "CAD", patient_id)
trm_rFLAIR_to_rT1_filepath = os.path.join(trm_basedir, "rFLAIR_to_rT1.mat")
trm_rT1_to_MNI_filepath = os.path.join(trm_basedir, "rT1_to_MNI_warp.nii.gz")

#invol = "/home/edouard/data/mescog/neuroimaging/cadasil/CAD_bioclinica_nifti/1001/1001-M0-rT1.nii.gz"
invol = file_path
outvol = '/tmp/toto2.nii'
refvol = fsl_mni_filepath
premat = trm_rFLAIR_to_rT1_filepath
warpvol = trm_rT1_to_MNI_filepath
#'fsl5.0-applywarp -i %s -o %s -r %s --premat %s -w %s' % (invol, outvol, refvol, warpvol)

'fsl5.0-applywarp -i %s -o %s -r %s --premat=%s -w %s' % (invol, outvol, refvol, premat, warpvol)


/home/edouard/data/mescog/neuroimaging/cadasil/Normalization/CAD/1001/rT1_to_MNI_warp.nii.gz'

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