#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:59:22 2017

@author: edouard.duchesnay@cea.fr

"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil

BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST"
INPUT_FS = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/freesurfer_assembled_data_fsaverage"
TEMPLATE_PATH = os.path.join(BASE_PATH,"Freesurfer","freesurfer_template")

INPUT_CSV = os.path.join(BASE_PATH,"Freesurfer","population_30yo.csv")
OUTPUT = "/home/ed203246/mega/studies/parsimony/logistic_enettv_NUSDAST30yo_FS"
penalty_start = 3


#############################################################################
# utils

def save_texture(filename, data):
    from nibabel import gifti
    import codecs
    darray = gifti.GiftiDataArray(data)
    gii = gifti.GiftiImage(darrays=[darray])
    f = codecs.open(filename, 'wb')
    f.write(gii.to_xml(enc='utf-8'))
    f.close()


#############################################################################
## Build mesh template
import brainomics.mesh_processing as mesh_utils
cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "rh.pial.gii"))
cor = np.vstack([cor_l, cor_r])
assert cor.shape == (327684, 3)
tri_r += cor_l.shape[0]
tri = np.vstack([tri_l, tri_r])
assert tri.shape == (655360, 3)
mesh_utils.mesh_from_arrays(cor, tri, path=os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"), os.path.join(OUTPUT , "lrh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT , "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT , "rh.pial.gii"))


#############################################################################
# Read images
# Read pop csv
pop = pd.read_csv(INPUT_CSV)

n = len(pop)
assert n == 165
Z = np.zeros((n, 3)) # Intercept + Age + Gender #only one site, no need for a site covariate
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, ID in enumerate(pop["id"]):
    cur = pop[pop["id"] == ID]
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    y[i, 0] = cur["dx_num"]
    Z[i, 1:] = np.asarray(cur[["age", "sex_num"]]).ravel()
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (165, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() == 299731

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape ==  (165, 299731)


#############################################################################

X = np.hstack([Z, X])
assert X.shape == (165, 299734)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (165, 299734)



np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)

#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov
mask = np.load(os.path.join(OUTPUT, "mask.npy"))

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (299731, 299731) for a in Atv])
assert np.all(np.array([np.max(np.abs(Atv[i] - Atv_[i])) for i in range(len(Atv))]) == 0)

#############################################################################
# fit models
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
from parsimony.utils.linalgs import LinearOperatorNesterov

"""
l1, l2, tv = np.round(0.1 * np.array([0.02, 0.18, 0.8]), 3)
#l1, l2, tv = np.round(0.1 * np.array([0.1, 0.1, 0.8]), 3)
param = "0.01_0.72_0.08_0.2"
param = "0.01_0.02_0.18_0.8" # en cours
param = "0.1_0.02_0.18_0.8"
"""

Atv = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))

X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))

params = ["0.01_0.72_0.08_0.2",
"0.01_0.08_0.72_0.2",
"0.01_0.18_0.02_0.8",
"0.1_0.18_0.02_0.8",
"0.1_0.02_0.18_0.8",
"0.01_0.02_0.18_0.8",
"0.1_0.08_0.72_0.2",
"0.1_0.72_0.08_0.2"]

"""
alphas = [.01, 0.1]
tv_ratio = [0.2, 0.8]
l1l2_ratio = [0.1, 0.9]
algos = ["enettv", "enetgn"]
import itertools
params_enet_tvgn = [list(param) for param in itertools.product(algos, alphas, l1l2_ratio, tv_ratio)]
assert len(params_enet_tvgn) == 16
"""

for param in params:
    # param = "0.01_0.08_0.72_0.2"
    key = param.split("_")
    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    #param = "_".join([str(x) for x in [l1, l2, tv]])
    #Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask)
    for max_iter in [500, 1000, 10000]:
        # max_iter = 10000
        print(param, max_iter)
        conesta = algorithms.proximal.CONESTA(max_iter=max_iter)
        mod = estimators.LogisticRegressionL1L2TV(l1, l2,tv, Atv, algorithm=conesta, class_weight="auto", penalty_start=penalty_start)
        mod.fit(X, y.ravel())
        y_pred = mod.predict(X)
        proba_pred = mod.predict_probability(X)
        #beta_path = os.path.join(OUTPUT, "%s_beta.npz" % param)
        beta_path = os.path.join(OUTPUT, "%s_beta_%iite.npz" % (param, max_iter))
        np.savez_compressed(beta_path, mod.beta)


#############################################################################
# Reload models and evaluate stability
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd

#from parsimony.utils.linalgs import LinearOperatorNesterov
#import parsimony.estimators as estimators

mask_mesh = np.load(os.path.join(OUTPUT, "mask.npy"))
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
#Atv = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))

# Hemispheres masks of mesh and beta
half = int(mask_mesh.shape[0] / 2)
assert mask_mesh.shape[0] / 2 % 2 == 0
mask_mesh_l = np.copy(mask_mesh)
mask_mesh_l[half:] = False
mask_mesh_r = np.copy(mask_mesh)
mask_mesh_r[:half] = False
assert mask_mesh_l.sum() + mask_mesh_r.sum() + penalty_start == X.shape[1]

mask_beta_l = np.repeat(False, X.shape[1])
mask_beta_l[penalty_start:(penalty_start + mask_mesh_l.sum())] = True
mask_beta_r = np.repeat(False, X.shape[1])
mask_beta_r[(penalty_start + mask_mesh_l.sum()):] = True
assert np.all(np.logical_xor(mask_beta_l, mask_beta_r)[penalty_start:])
assert np.all(np.logical_xor(mask_beta_l, mask_beta_r)[:penalty_start] == False)
assert mask_beta_l.sum() + mask_beta_r.sum() + penalty_start == X.shape[1]

decision_funcs = dict()
betas = dict()

beta_paths = glob.glob(os.path.join(OUTPUT, "*beta*.npz"))

for beta_path in beta_paths:
    #beta_path = '/home/ed203246/mega/studies/parsimony/logistic_enettv_NUSDAST30yo_FS/0.01_0.02_0.18_0.8_beta_500ite.npz'
    output_path, _ = os.path.splitext(beta_path)
    # Stability of prediction
    beta = np.load(beta_path)['arr_0'].ravel()
    key, _ = os.path.splitext(os.path.basename(beta_path))
    key = tuple(key.split("_beta_"))
    betas[key] = beta
    decision_funcs[key] = np.dot(X, beta)

    # small thersholding for texture
    from brainomics import array_utils
    beta_t, t = array_utils.arr_threshold_from_norm2_ratio(beta[penalty_start:], 0.99)
    beta_t = np.hstack([beta[:penalty_start], beta_t])
    beta_t = beta
    assert beta_t.shape == mask_beta_l.shape
    # left
    arr = np.zeros(mask_mesh_l.shape)
    arr[mask_mesh_l] = beta_t[mask_beta_l]
    save_texture(output_path + "lh.gii", arr)


decision_funcs.keys()
zip(*decision_funcs.keys())
params, ites = [l for l in zip(*decision_funcs.keys())]
params, ites = set(params), set(ites)

sns.set_style(None)
pdf = PdfPages(os.path.join(OUTPUT, "decision_functions_pairplots.pdf"))

params_l = list(params)
params_l.sort()
for param in params_l:
    print("################################################################")
    print(param)
    decfuncs = pd.DataFrame(np.array([decision_funcs[(param, ite)] for ite in ites]).T, columns=ites)
    variables = decfuncs.columns.sort_values()
    decfuncs["y"] = y
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_title("TOTO")
    sns.pairplot(decfuncs, hue="y", vars=variables)
    #sns.set_style('whitegrid')
    #plt.title('This is the title')
    plt.suptitle(param)
    betas_map = np.array([betas[(param, ite)] for ite in ites])
    print("\nDecision function correlations\n")
    print(decfuncs.corr())
    print("\nDecision function correlations\n")
    print(np.corrcoef(betas_map))
    pdf.savefig()  # saves the current figure into a pdf page
    plt.clf()

pdf.close()
