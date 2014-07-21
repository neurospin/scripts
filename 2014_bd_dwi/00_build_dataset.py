# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:01 2014

@author: christophe
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd

#loading files
BASE_PATH =  "/volatile/share/2014_bd_dwi"

#BASE_PATH =  "/volatile/share/2014_bd_dwi/all_FA/nii/stats"


INPUT_IMAGE_INDEX = os.path.join(BASE_PATH, "clinic", "image_list.txt")
INPUT_IMAGE = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii.gz")
INPUT_CSV_BASE = os.path.join(BASE_PATH, "clinic", "BD_clinic.csv")
OUTPUT = os.path.join(BASE_PATH, "bd_dwi_enettv")
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
OUTPUT_CSV = os.path.join(OUTPUT, "population.csv")
OUTPUT_X = os.path.join(OUTPUT, "X.npy")
OUTPUT_Y = os.path.join(OUTPUT, "Y.npy")

MASK_FILENAME =  os.path.join(BASE_PATH, "all_FA/nii/stats/mask.nii.gz") #mask use to filtrate our images
FA_THRESHOLD = 0.05
ATLAS_FILENAME = "/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz" # Image use to improve our mask
ATLAS_LABELS_RM = [0, 13, 2, 8]  # cortex, trunc
#OUTPUT = os.path.join(BASE_PATH, "DATA")

#############################################################################
## Build population file

# Read image order and convert to match the ID in clinic file
image_id = list()
for l in open(INPUT_IMAGE_INDEX).readlines():
    l = l.replace("\n", "")
    l = l.replace("P_S_", "")
    if l.find("C_") == 0:
        l = l[:-4]
    if l.find("M_H_C6HB") == 0:  # M_H_C6HB02 => HB_002
        l = l.replace("M_H_C6HB", "HB_0")
    if l.find("M_H_C6HK") == 0:  # M_H_C6HK01 => HK_001
        l = l.replace("M_H_C6HK", "HK_0")
    if l.find("M_Z_") == 0:  # M_Z_103843 => MZ103843
        l = l.replace("M_Z_", "MZ")
    image_id.append(l)

# Create a dataframe of image position
images = pd.DataFrame(range(len(image_id)), columns=['SLICE'],
                      index=image_id)

# Read some columns clinic data
clinic = pd.read_csv(INPUT_CSV_BASE,
                     index_col = 0)
clinic = clinic[["BD_HC", "SCANNER", "AGEATMRI", "SEX"]]

# Merge dataframes
population = pd.merge(clinic, images,
                      right_index = True,
                      left_index = True)
assert(population.shape[0] == 194)

# Print ID of subjects for which clinic or image is missing
in_clinic_not_in_image = clinic.index[~clinic.index.isin(images.index)]
for ID in in_clinic_not_in_image:
    print "ID", ID, "matched 0 images"
assert(len(in_clinic_not_in_image)==10)
#ID 204776 matched 0 images
#ID 209861 matched 0 images
#ID 211297 matched 0 images
#ID 212485 matched 0 images
#ID 213141 matched 0 images
#ID 23012 matched 0 images
#ID 27308 matched 0 images
#ID 31108 matched 0 images
#ID C_cn00018 matched 0 images
#ID MZ150743 matched 0 images

in_image_not_in_clinic = images.index[~images.index.isin(clinic.index)]
for ID in in_image_not_in_clinic:
    print "ID", ID, "matched 0 clinic"
assert(len(in_image_not_in_clinic)==6)
#ID 023012 matched 0 clinic
#ID C_pb090397 matched 0 clinic
#ID 031108 matched 0 clinic
#ID C_gb100352 matched 0 clinic
#ID 027308 matched 0 clinic
#ID 214469 matched 0 clinic

population.to_csv(OUTPUT_CSV)

#
#############################################################################
## Build dataset

## Threshold FA mean map
images4d = nib.load(INPUT_IMAGE)
image_arr = images4d.get_data()

#test if mask already exist

babel_mask = nib.load(MASK_FILENAME)

# to do if mask has not been registered yet
#creation du mask
shape = image_arr.shape
fa_mean = np.mean(image_arr, axis=3)
mask = fa_mean > FA_THRESHOLD

# ATLAS, remove: cortex, trunc
atlas = nib.load(ATLAS_FILENAME)
assert np.all(images4d.get_affine() == atlas.get_affine())
for label_rm in ATLAS_LABELS_RM:
    mask[atlas.get_data() == label_rm] = False

#registration of mask
out_im = nib.Nifti1Image(mask.astype(int), affine=images4d.get_affine())
out_im.to_filename(MASK_FILENAME)
# Check that the stored image is the same than mask
babel_mask = nib.load(MASK_FILENAME)
assert np.all(mask == (babel_mask.get_data() != 0))
print mask.sum()

n_voxel_in_mask = np.count_nonzero(mask)
assert(n_voxel_in_mask == 507383)

#application of mask on all the images

n, _ = population.shape

Ytot = np.asarray(population.BD_HC, dtype='float64').reshape(n, 1)
Ztot = population[["AGEATMRI", "SEX"]].as_matrix()

Xtot = np.zeros((n, n_voxel_in_mask))
for i, ID in enumerate(population.index):
    cur = population.iloc[i]
    slice_index = cur.SLICE
    image = image_arr[:, :, :, slice_index]
    Xtot[i, :] = image[mask]

## sans le cs
Xtot = np.hstack([Ztot, Xtot])
assert Xtot.shape == (n, n_voxel_in_mask+2)

Xtot -= Xtot.mean(axis = 0)
Xtot /= Xtot.std(axis = 0)


n, p = Xtot.shape
np.save(OUTPUT_X, Xtot)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()



np.save(OUTPUT_Y, Ytot)

## avec le cs





#enettv = LinearRegressionL1L2TV(l1, l2, tv, A, algorithm=StaticCONESTA(max_iter=500))
#yte_pred_enettv = enettv.fit(Xtrain, y_train).predict(Xtest)

## TODO : shape, apprentissage, calcul des y estimés
## TODO : vérification croisée des résultats
#############################################################################

