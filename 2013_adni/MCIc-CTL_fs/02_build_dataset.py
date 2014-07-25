# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:46:15 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb

#import proj_classif_config
GENDER_MAP = {'Female': 0, 'Male': 1}

BASE_PATH = "/neurospin/brainomics/2013_adni"
INPUT_IMAGE_DIR = os.path.join(BASE_PATH, "MCIc-CTL_fs" ,"fs_assembled_data")


INPUT_CSV = os.path.join(BASE_PATH,          "MCIc-CTL_fs", "population.csv")

OUTPUT = os.path.join(BASE_PATH,             "MCIc-CTL_fs")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['PTGENDER.num'] = pop["PTGENDER"].map(GENDER_MAP)

#############################################################################
# Read images
n = len(pop)
assert n == 201
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, PTID in enumerate(pop['PTID']):
    cur = pop[pop.PTID == PTID]
    print cur
    #cur = pop.iloc[0]
    left = nb.load(cur["mri_path_lh"].values[0]).get_data().ravel()
    right = nb.load(cur["mri_path_rh"].values[0]).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    Z[i, :] = np.asarray(cur[["AGE", "PTGENDER.num"]]).ravel()
    y[i, 0] = cur["DX.num"]


Xtot = np.vstack(surfaces)
assert Xtot.shape == (201, 327684)


means = np.mean(Xtot, axis=0)
stds = np.std(Xtot, axis=0)
mins = np.min(Xtot, axis=0)
maxs = np.max(Xtot, axis=0)
print "Means:", means.min(), means.max(), means.mean()
print "Std:", stds.min(), stds.max(), stds.mean()
print "Mins:", mins.min(), mins.max(), mins.mean()
print "Maxs:", maxs.min(), maxs.max(), maxs.mean(), (maxs == 0).sum()
#In [26]: Means: 0.0 4.70434 2.12934374275
#In [27]: Std: 0.0 2.18101 0.573928598131
#In [28]: Mins: 0.0 3.02653 0.906595988818
#In [29]: Maxs: 0.0 5.0 3.81341894935 10595
print (stds > 1e-8).sum()
#317089


mask = ((maxs > 0) & (stds > 1e-2))
assert mask.sum() == 317089

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

# Xcs
X = Xtot[:, mask]
X = np.hstack([Z, X])
assert X.shape == (201, 317091)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i surface points' % \
    (n, p, mask.sum()))
fh.close()

