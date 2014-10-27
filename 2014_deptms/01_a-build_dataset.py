# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:43:03 2014

@author: cp243490

Read the data (clinic data, ROI, MRI AND PET images)
Construct the masks, Build and save the datasets given a MODALITY
(MRI/PET) for:
    - the whole brain (implicit mask)
    - each ROI (using harvard and oxford atlas to define the regions and
                the implicit mask and finally dilate the obtained mask
                to make sure all the region is contained in the mask )

INPUTs
- Clinic data: /neurospin/brainomics/2014_deptms/clinic/deprimPetInfo.csv
- ROIs : /neurospin/brainomics/2014_deptms/ROI_labels.csv
- resampled cortical and subcortical harvard oxford atlases:
    /neurospin/brainomics/2014_deptms/images/atalses/
        - HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz
        - HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz
- MRI images :
    /neurospin/brainomics/2014_deptms/images/MRI_images/smwc1*.img
- PET images :
    /neurospin/brainomics/2014_deptms/images/PET_j0Scaled_images/smw*.img

OUTPUTs
- implicit mask for the whole brain and mask for each ROI :
    /neurospin/brainomics/2014_deptms/*{MRI, PET}/mask_*.nii
- X for the whole brain and for each ROI associated to a modality :
    /neurospin/brainomics/2014_deptms/*{MRI, PET}/X*.npy
    (Intercept + Age + Gender + (MRI or PET) images
    for the whole brain and each ROI)
- y : /neurospin/brainomics/2014_deptms/*{MRI, PET}/y.npy
    (response to the treatment)

"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

REP_MAP = {"norep": 0, "rep": 1}

MODALITY = "PET"  # modalities: {"MRI", "PET"}

print "Modality: ", MODALITY

BASE_PATH = "/neurospin/brainomics/2014_deptms"

INPUT_CSV = os.path.join(BASE_PATH,          "clinic", "deprimPetInfo.csv")
INPUT_ROIS_CSV = os.path.join(BASE_PATH,     "ROI_labels.csv")

OUTPUT_CSI = os.path.join(BASE_PATH,          MODALITY)

if not os.path.exists(OUTPUT_CSI):
    os.makedirs(OUTPUT_CSI)

# Read pop csv
pop = pd.read_csv(INPUT_CSV, sep="\t")
pop['rep_norep.num'] = pop["rep_norep"].map(REP_MAP)

# Read ROIs csv
atlas = []
dict_rois = {}
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    label_ho = cur["label_ho"].values[0]
    atlas_ho = cur["atlas_ho"].values[0]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["label_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if not roi_name in dict_rois:
            print "ROI_name_deptms", roi_name
            labels = np.asarray(label_ho.split(), dtype="int")
            dict_rois[roi_name] = [labels]
            dict_rois[roi_name].append(atlas_ho)
            print dict_rois[roi_name]
            print "\n"

#############################################################################
# Read images
n = len(pop)
assert n == 34
Z = np.zeros((n, 3))  # Intercept + Age + Gender
Z[:, 0] = 1  # Intercept
y = np.zeros((n, 1))  # rep_norep
images = list()
fileName = ""
image_path = ""
if MODALITY == "MRI":
    fileName = "MRIm_G_file"
    image_path = "MRI_images"
elif MODALITY == "PET":
    fileName = "PET_file"
    image_path = "PET_j0Scaled_images"

for i, MRIm_G_file in enumerate(pop["MRIm_G_file"]):
    cur = pop[pop.MRIm_G_file == MRIm_G_file]
    # print cur
    f = cur[fileName].values[0]
    imagefile_name = os.path.join(BASE_PATH,
                                   "images",
                                   image_path,
                                   f)
    babel_image = nib.load(imagefile_name)
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Sex"]]).ravel()
    y[i, 0] = cur["rep_norep.num"]

shape = babel_image.get_data().shape
#############################################################################
# Compute implicit mask and X and y matrix for the whole brain
#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a given threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)

mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)

out_im = nib.Nifti1Image(mask.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI,
                                "mask_" + MODALITY + "_wb.nii"))
babel_mask = nib.load(os.path.join(OUTPUT_CSI,
                                   "mask_" + MODALITY + "_wb.nii"))
assert np.all(mask == (babel_mask.get_data() != 0))

#############################################################################
# Xcsi for the whole brain
X = Xtot[:, mask.ravel()]
X = np.hstack([Z, X])
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
np.save(os.path.join(OUTPUT_CSI, "X_" + MODALITY + "_wb.npy"), X)
fh = open(os.path.join(OUTPUT_CSI,
                       "X_" + MODALITY + "_wb.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. ' \
            'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

#############################################################################
# Do the same for each ROI
#############################################################################
# Compute binarized mask with atlas for the specific ROIs

# Resample HarvardOxford atlas (cortical and subcortical) into reference
# space and interpolation

#fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s --interp=nn"
#ref=imagefile_name[0]
#input_sub_filename = os.path.join("/usr/share/data",
#                        "harvard-oxford-atlases",
#                        "HarvardOxford",
#                        "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")
#input_cort_filename = os.path.join("/usr/share/data",
#                        "harvard-oxford-atlases",
#                        "HarvardOxford",
#                        "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
#output_sub_filename = os.path.join(BASE_PATH, "images", "atlases",
#                        "HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz")
#output_cort_filename = os.path.join(BASE_PATH,"images", "atlases",
#                        "HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz")
#os.system(fsl_cmd % (input_sub_filename, ref, output_sub_filename))
#os.system(fsl_cmd % (input_cort_filename, ref, output_cort_filename))

sub_image = nib.load(os.path.join(BASE_PATH,
                                   "images",
                                   "atlases",
                                   "HarvardOxford-sub-maxprob-thr0-1mm-nn." \
                                       "nii.gz"))
sub_arr = sub_image.get_data()
cort_image = nib.load(os.path.join(BASE_PATH,
                                   "images",
                                   "atlases",
                                   "HarvardOxford-cort-maxprob-thr0-1mm-nn." \
                                       "nii.gz"))
cort_arr = cort_image.get_data()

# Compute the mask associated to each ROI by
# applying the implicit mask to the atlas
# and by identifying the ROI in the atlas (with the labels)
# Then dilate the mask
for ROI, values in dict_rois.iteritems():
    print "ROI: ", ROI
    labels = values[0]
    print "labels", labels
    atlas = values[1]
    print "atlas: ", atlas

    # Compute the mask associated to each ROI by
    # applying the implicit mask to the atlas
    # and by identifying the ROI in the atlas (with the labels)
    # Then dilate the mask
    if atlas == "sub":
        mask_atlas_ROI = np.copy(sub_arr)
    elif atlas == "cort":
        mask_atlas_ROI = np.copy(cort_arr)

    if len(labels) == 1:
        mask_atlas_ROI[np.logical_or(np.logical_not(mask),
                                     mask_atlas_ROI != labels[0])] = 0
    elif len(labels) == 2:
        mask_atlas_ROI[np.logical_or(np.logical_not(mask),
                            np.logical_and(mask_atlas_ROI != labels[0],
                                           mask_atlas_ROI != labels[1]))] = 0
    # dilate
    # 3x3 structuring element with connectivity 1 and 2 iterations
    mask_bool_ROI = ndimage.morphology.binary_dilation(
                                mask_atlas_ROI,
                                iterations=2).astype(mask_atlas_ROI.dtype)
    mask_bool_ROI = mask_bool_ROI.astype("bool")

    out_im = nib.Nifti1Image(mask_bool_ROI.astype("int16"),
                             affine=babel_image.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_CSI, "mask_" + MODALITY + "_" +
                                            ROI + ".nii"))
    im = nib.load(os.path.join(OUTPUT_CSI, "mask_" + MODALITY + "_" +
                                        ROI + ".nii"))
    assert np.all(mask_bool_ROI == im.get_data())

    # Xcsi for the specific ROIs
    X_ROI = Xtot[:, mask_bool_ROI.ravel()]
    X_ROI = np.hstack([Z, X_ROI])
    X_ROI -= X_ROI.mean(axis=0)
    X_ROI /= X_ROI.std(axis=0)
    X_ROI[:, 0] = 1.
    n, p = X_ROI.shape
    np.save(os.path.join(OUTPUT_CSI,
                         "X_" + MODALITY + "_" + ROI + ".npy"), X_ROI)
    fh = open(os.path.join(OUTPUT_CSI,
                           "X_" + MODALITY + "_" +
                               ROI + ".npy").replace("npy", "txt"), "w")
    fh.write('Centered and scaled data. ' \
                'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
        (n, p, mask_bool_ROI.sum()))
    fh.close()
    print '\n'


np.save(os.path.join(OUTPUT_CSI, "y.npy"), y)