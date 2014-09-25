# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:01 2014

@author: christophe

Create a dataset with non-skeletonised FA images and covariates:
 - intercept
 - sex
 - site
 - age (centered and scaled)
We use the standard mask.
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd

BASE_PATH = "/neurospin/brainomics/2014_bd_dwi"

INPUT_IMAGES = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii")
INPUT_SK_IMAGES = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA_skeletonised.nii")
INPUT_CSV = os.path.join(BASE_PATH, "population.csv")
# Atlas used to improve our mask
INPUT_ATLAS = "/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz"

OUTPUT_DIR = os.path.join(BASE_PATH, "datasets")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_Y = os.path.join(OUTPUT_DIR,
                        "Y.npy")
# Output masks
OUTPUT_STD_MASK = os.path.join(OUTPUT_DIR,
                               "mask.nii.gz")
OUTPUT_TRUNC_MASK = os.path.join(OUTPUT_DIR,
                                 "mask_trunc.nii.gz")
OUTPUT_SKEL_MASK = os.path.join(OUTPUT_DIR,
                                "mask_skel.nii.gz")

###############################################################################
# Parameters                                                                  #
###############################################################################

FA_THRESHOLD = 0.05
ATLAS_LABELS_RM = [0, 13, 2, 8]  # cortex, trunc
# Slices to remove everything after the cingulate gyri
TRUNC_MASK_SLICE = range(71)

###############################################################################
# Functions                                                                   #
###############################################################################


def create_mask(images, atlas, fa_threshold,
                output_file,
                slices=None):
    """Create mask by thresholding the FA and eventually slicing.
    Values in ATLAS_LABELS_RM
    """
    #shape = images.shape
    fa_mean = np.mean(images, axis=3)
    mask = fa_mean > fa_threshold

    # Remove cortex, trunc (thanks to atlas)
    for label_rm in ATLAS_LABELS_RM:
        mask[atlas.get_data() == label_rm] = False

    # Slice the mask
    if slices is not None:
        mask[:, slices, :] = 0

    n_voxels_in_mask = np.count_nonzero(mask)
    print "Number of voxels in mask:", n_voxels_in_mask

    # Store mask
    out_im = nib.Nifti1Image(mask.astype(np.uint8),
                             affine=atlas.get_affine())
    out_im.to_filename(output_file)


def create_dataset_from_mask(images, population, mask_file,
                             cat_covar_df, cont_covar_df,
                             output_x, output_x_desc,
                             add_intercept=True):
    """Create a dataset.
    Continuous variables are centered and scaled.
    If add_intercept is True, add an intercept.
    """
    #n_images = images.shape[-1]
    n = population.shape[0]
    # Open mask
    #print "Reading mask:", mask_file
    babel_mask = nib.load(mask_file)
    mask = babel_mask.get_data()
    bin_mask = mask != 0
    n_voxels_in_mask = np.count_nonzero(mask)
    #print "Number of voxels in mask:", n_voxels_in_mask

    # Apply mask on images & center
    #print "Application of mask on all the images & centering"
    masked_images = np.zeros((n, n_voxels_in_mask))
    for i, ID in enumerate(population.index):
        cur = population.iloc[i]
        slice_index = cur['SLICE']
        image = images[:, :, :, slice_index]
        masked_images[i, :] = image[bin_mask]
    masked_images -= masked_images.mean(axis=0)
    masked_images /= masked_images.std(axis=0)

    # Add categorical covariables
    cat_covar = cat_covar_df.as_matrix()
    # Center and add continuous covariables
    cont_covar = cont_covar_df.as_matrix()
    cont_covar -= cont_covar.mean(axis=0)
    cont_covar /= cont_covar.std(axis=0)

    # Concatenation
    covar = covar = np.hstack((cont_covar, cat_covar))
    if add_intercept:
        Xtot = np.hstack([np.ones((covar.shape[0], 1)), covar, masked_images])
        n, p = Xtot.shape
        assert Xtot.shape == (n, n_voxels_in_mask + covar.shape[1] + 1)
    else:
        Xtot = np.hstack([covar, masked_images])
        n, p = Xtot.shape
        assert Xtot.shape == (n, n_voxels_in_mask + covar.shape[1])

    # Store dataset and description
    np.save(output_x, Xtot)
    covar_names = cat_covar_df.columns.tolist() + \
                  cont_covar_df.columns.tolist()
    fh = open(output_x_desc, "w")
    fh.write('shape = (%i, %i): ' % (n, p))
    if add_intercept:
        fh.write('Intercept + ')
    covar_string = ' + '.join(covar_names)
    fh.write(covar_string)
    fh.write(' + %i voxels' % n_voxels_in_mask)
    fh.close()

###############################################################################
# Loading data                                                                #
###############################################################################

print "Reading data"
babel_images = nib.load(INPUT_IMAGES)
images = babel_images.get_data()
babel_skel_images = nib.load(INPUT_SK_IMAGES)
skel_images = babel_skel_images.get_data()

population = pd.read_csv(INPUT_CSV,
                         index_col=0)
n, _ = population.shape

# Load atlas
atlas = nib.load(INPUT_ATLAS)
assert np.all(babel_images.get_affine() == atlas.get_affine())
assert np.all(babel_skel_images.get_affine() == atlas.get_affine())


print '###################'
print '# Creating target #'
print '###################'
# Create Y (same for all the cases)
Ytot = np.asarray(population.BD_HC, dtype='float64').reshape(n, 1)
np.save(OUTPUT_Y, Ytot)

###############################################################################
# Create masks                                                                #
###############################################################################

print '##################'
print '# Creating masks #'
print '##################'

# First mask: simple threshold on pixel values
print "Creating:", OUTPUT_STD_MASK
create_mask(images, atlas, FA_THRESHOLD, OUTPUT_STD_MASK)

# Second mask: threshold + slice
print "Creating:", OUTPUT_TRUNC_MASK
create_mask(images, atlas, FA_THRESHOLD, OUTPUT_TRUNC_MASK,
            slices=TRUNC_MASK_SLICE)

# Third mask: threshold on skeletonized images
print "Creating:", OUTPUT_SKEL_MASK
create_mask(skel_images, atlas, FA_THRESHOLD, OUTPUT_SKEL_MASK)

###############################################################################
# Create datasets                                                             #
###############################################################################

print '#####################'
print '# Creating datasets #'
print '#####################'

## First case: no intercept, SEX and AGE, standard mask
## This case is no longer of interest. It's keep here just for record.
#output_x = os.path.join(OUTPUT_DIR,
#                        "X_nointercept.npy")
#output_x_desc = output_x.replace("npy", "txt")
#print "Creating", output_x
#create_dataset_from_mask(images, population, OUTPUT_STD_MASK,
#                         population[["SEX"]], population[["AGEATMRI"]],
#                         output_x, output_x_desc,
#                         add_intercept=False)
#
## Second case: intercept, SEX and AGE, standard mask
## This case is no longer of interest. It's keep here just for record.
#output_x = os.path.join(OUTPUT_DIR,
#                        "X.npy")
#output_x_desc = output_x.replace("npy", "txt")
#print "Creating", output_x
#create_dataset_from_mask(images, population, OUTPUT_STD_MASK,
#                         population[["SEX"]], population[["AGEATMRI"]],
#                         output_x, output_x_desc)

# Third case: intercept, SEX, SCANNER and AGE, standard mask
output_x = os.path.join(OUTPUT_DIR,
                        "X_site.npy")
output_x_desc = output_x.replace("npy", "txt")
print "Creating", output_x
create_dataset_from_mask(images, population, OUTPUT_STD_MASK,
                         population[["SEX",
                                     "SCANNER#1",
                                     "SCANNER#2",
                                     "SCANNER#3"]],
                         population[["AGEATMRI"]],
                         output_x, output_x_desc)

# Fourth case: intercept, SEX, SCANNER and AGE, truncated mask
output_x = os.path.join(OUTPUT_DIR,
                        "X_trunc.npy")
output_x_desc = output_x.replace("npy", "txt")
print "Creating", output_x
create_dataset_from_mask(images, population, OUTPUT_TRUNC_MASK,
                         population[["SEX",
                                     "SCANNER#1",
                                     "SCANNER#2",
                                     "SCANNER#3"]],
                         population[["AGEATMRI"]],
                         output_x, output_x_desc)

# Fifth case: skletonized images, SEX, SCANNER and AGE, skel mask
output_x = os.path.join(OUTPUT_DIR,
                        "X_skel.npy")
output_x_desc = output_x.replace("npy", "txt")
print "Creating", output_x
create_dataset_from_mask(skel_images, population, OUTPUT_SKEL_MASK,
                         population[["SEX",
                                     "SCANNER#1",
                                     "SCANNER#2",
                                     "SCANNER#3"]],
                         population[["AGEATMRI"]],
                         output_x, output_x_desc)

###############################################################################
