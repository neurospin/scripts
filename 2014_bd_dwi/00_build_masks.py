import os
import nibabel as nib
import numpy as np


BASE_PATH = "/volatile/share/2014_bd_dwi"

INPUT_IMAGE = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii.gz")
INPUT_IMAGE_SK = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA_skeletonised.nii.gz")

OUTPUT_DIR = os.path.join(BASE_PATH, "masks")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_MASK_FILENAME = "mask.nii.gz" #mask use to filtrate our images
OUTPUT_MASK_TRUNC_FILENAME = "mask_trunc.nii.gz" #mask use to filtrate our images
OUTPUT_MASK_SK_FILENAME = "mask_sk.nii.gz" #mask use to filtrate our images

FA_THRESHOLD = 0.05
ATLAS_FILENAME = "/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz" # Image use to improve our mask
ATLAS_LABELS_RM = [0, 13, 2, 8]  # cortex, trunc
# Slices to remove after the cingulate gyri
MASK_SLICE = range(71)

####################################
# Creation of mask for base images #
####################################

# Load base images

images4d = nib.load(INPUT_IMAGE)
image_arr = images4d.get_data()

shape = image_arr.shape
fa_mean = np.mean(image_arr, axis=3)
mask = fa_mean > FA_THRESHOLD

# Remove cortex, trunc (thanks to atlas)
atlas = nib.load(ATLAS_FILENAME)
assert np.all(images4d.get_affine() == atlas.get_affine())
for label_rm in ATLAS_LABELS_RM:
    mask[atlas.get_data() == label_rm] = False
n_voxel_in_mask = np.count_nonzero(mask)
print "Number of voxels in mask:", n_voxel_in_mask
assert(n_voxel_in_mask == 507383)

# Store mask
out_im = nib.Nifti1Image(mask.astype(np.uint8), affine=images4d.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DIR, OUTPUT_MASK_FILENAME))

# Truncate mask
trunc_mask = np.copy(mask)
trunc_mask[:, MASK_SLICE, :] = 0
n_voxel_in_trunc_mask = np.count_nonzero(trunc_mask)
print "Number of voxels in truncated mask:", trunc_mask.sum()
assert(n_voxel_in_trunc_mask == 448334)

# Store truncated mask
out_im = nib.Nifti1Image(trunc_mask.astype(np.uint8), affine=images4d.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DIR, OUTPUT_MASK_TRUNC_FILENAME))

del image_arr, images4d, mask, trunc_mask

#############################################
# Creation of mask with skeletonised images #
#############################################

# Load skeletonised images

images4d_sk = nib.load(INPUT_IMAGE_SK)
image_arr_sk = images4d_sk.get_data()

shape = image_arr_sk.shape
fa_mean = np.mean(image_arr_sk, axis=3)
mask_sk = fa_mean > FA_THRESHOLD

# Remove cortex, trunc (thanks to atlas)
atlas = nib.load(ATLAS_FILENAME)
assert np.all(images4d_sk.get_affine() == atlas.get_affine())
for label_rm in ATLAS_LABELS_RM:
    mask_sk[atlas.get_data() == label_rm] = False
n_voxel_in_mask_sk = np.count_nonzero(mask_sk)
print "Number of voxels in mask:", n_voxel_in_mask_sk
assert(n_voxel_in_mask_sk == 105799)

# Store truncated mask
out_im = nib.Nifti1Image(mask_sk.astype(np.uint8), affine=images4d_sk.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DIR, OUTPUT_MASK_SK_FILENAME))