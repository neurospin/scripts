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
 
 
INPUT_TBSS = os.path.join(BASE_PATH, "clinic", "filelist_tbss.txt")
INPUT_IMAGE = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii.gz")
INPUT_CSV_BASE = os.path.join(BASE_PATH, "clinic", "BD_clinic.csv")
INPUT_CSV_NEW = os.path.join(BASE_PATH, "bd_dwi_enettv", "population.csv")
OUTPUT = os.path.join(BASE_PATH, "bd_dwi_enettv")
OUTPUT_CSV = os.path.join(OUTPUT, "population.csv")

MASK_FILENAME =  os.path.join(BASE_PATH, "all_FA/nii/stats/mask.nii.gz") #mask use to filtrate our images
FA_THRESHOLD = 0.05
ATLAS_FILENAME = "/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz" # Image use to improve our mask
ATLAS_LABELS_RM = [0, 13, 2, 8]  # cortex, trunc
#OUTPUT = os.path.join(BASE_PATH, "DATA")

#############################################################################
## Build population file
tbss = open(INPUT_TBSS).readlines()
tmp = list()
for l in tbss:
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
    tmp.append(l)

tbss = tmp
clinic = pd.read_csv(INPUT_CSV_BASE)

tbss2 = None
empty = pd.DataFrame([[0]*6], columns=["ID","BD_HC", "SCANNER", "AGEATMRI", "SEX", "with_clinic"])
for ID in tbss: 
    match = clinic.ID == ID
    if np.sum(match) == 1:
        t = clinic[match][["ID","BD_HC", "SCANNER", "AGEATMRI", "SEX"]]
        t["with_clinic"] = 1
    else:
        t = empty.copy()
        print "ID",ID, "matched ", np.sum(match), "clinic"
    if tbss2 is None:
        tbss2 = t
    else:
        tbss2 = tbss2.append(t, ignore_index=True)


# ID which didnt match that we have to add to tbss2

#ID C_gb100352 matched  0 clinic
#ID C_pb090397 matched  0 clinic
#ID 023012 matched  0 clinic
#ID 027308 matched  0 clinic
#ID 031108 matched  0 clinic
#ID 214469 matched  0 clinic


n_in_clinic_without_image = np.sum(np.logical_not(clinic.ID.isin(tbss)))

print "n_tbss", len(tbss), "n_clinic", clinic.shape[0], \
"n_in_clinic_without_image", n_in_clinic_without_image

assert len(tbss2) == 200

tbss2.to_csv(OUTPUT_CSV)

# 
#############################################################################
## Build dataset

os.makedirs(OUTPUT)

image = list()


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
babel_mask = nib.load(MASK_FILENAME)
assert np.all(mask == (babel_mask.get_data() != 0))
print mask.sum()

#application of mask on all the images

i = 0
subject = pd.read_csv(INPUT_CSV_NEW, header=0)

ntot = 0

for i, SID in enumerate(subject["ID"]):
    cur = subject.iloc[i]
    if cur.with_clinic == 0:
        i = i + 1
    else:
        ntot = ntot + 1
        i = i + 1

Ytot = np.zeros((ntot, 1))
Xtot = np.zeros((ntot, 507383))

i = 0
k = 0

Ztot = np.zeros((ntot, 2))

while k < len(Ytot) and i < 194:
    cur = subject.iloc[k]
    print("k = ", k)
    if cur.with_clinic == 0:
        k = k + 1
    else:
        print("BD_HC = %i",cur.BD_HC)
        image.append(image_arr[:, :, :, k].ravel())
        Xtot[i, :] = np.vstack(image)[:, mask.ravel()]
        del image[0]
        Ytot[i] = cur.BD_HC
        Ztot[:, :] = np.asarray(cur[["AGEATMRI", "SEX"]]).ravel()
        i = i + 1
        k = k + 1

## sans le cs
Xtot = np.hstack([Ztot, Xtot])
assert Xtot.shape == (194, 507385)

Xtot -= Xtot.mean(axis = 0)
Xtot /= Xtot.std(axis = 0)


n, p = Xtot.shape
np.save(os.path.join(OUTPUT, "Xtot.npy"), Xtot)
fh = open(os.path.join(OUTPUT, "Xtot.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()



np.save(os.path.join(OUTPUT, "Ytot.npy"), Ytot)

## avec le cs





#enettv = LinearRegressionL1L2TV(l1, l2, tv, A, algorithm=StaticCONESTA(max_iter=500))
#yte_pred_enettv = enettv.fit(Xtrain, y_train).predict(Xtest)

## TODO : shape, apprentissage, calcul des y estimés
## TODO : vérification croisée des résultats
#############################################################################

