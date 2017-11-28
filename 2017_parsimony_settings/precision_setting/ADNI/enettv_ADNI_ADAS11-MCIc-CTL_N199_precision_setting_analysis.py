#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:27:44 2017

@author: ad247405
"""

import os
import pandas as pd
import numpy as np
import nibabel
import array_utils
import nilearn
from nilearn import image
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting
import glob
import seaborn as sns


INPUT_MASK_PATH = '/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL/mask.nii.gz'
babel_mask  = nibabel.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 0

WD = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL"
os.chdir(WD)
arxiv = np.load("ADNI_ADAS11-MCIc-CTL_N199.npz")
X = arxiv["X"]
y = arxiv["y"]



#1) x:iteration y:precision+gap
###########################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL/run"
snap_path = os.path.join(BASE_PATH,"conesta_ite_snapshots")
conesta_ite = sorted(os.listdir(snap_path))
nb_conesta = len(conesta_ite)
ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))

pdf_path = os.path.join(BASE_PATH,"precision_iterations.pdf")
pdf = PdfPages(pdf_path)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(ite_final["gap"],label = r"$gap$")
plt.plot(ite_final["func_val"] - ite_final["func_val"][-1],\
         label = r"$f(\beta^{k})$ - $f(\beta^{*})$" )
plt.yscale('log')
plt.xscale('log')

plt.xlabel("iterations")
plt.ylabel(r"precision")
plt.legend(prop={'size':15})
plt.title("ADNI - MCI-CTL - 286214 features")
pdf.savefig()
plt.close(fig)
pdf.close()


#2) x:precision, y: cor(Xbeta*, Xbeta_hat) + cor(beta*, beta_hat)
###########################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL/run"
snap_path = os.path.join(BASE_PATH,"conesta_ite_snapshots")
beta_path = os.path.join(BASE_PATH,"conesta_ite_beta")
conesta_ite = sorted(os.listdir(snap_path))
nb_conesta = len(conesta_ite)
ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
beta_star =  ite_final["beta"]

corr = np.zeros((nb_conesta))
decfunc =  np.zeros((nb_conesta))

i=0
for ite in conesta_ite:
    path = os.path.join(snap_path,ite)
    ite = np.load(path)
    corr[i] = np.corrcoef(ite["beta"][:,0],beta_star[:,0])[0][1]
    decfunc[i] = np.corrcoef(np.dot(X, ite["beta"][:,0]),np.dot(X, beta_star[:,0]))[0][1]
    i = i + 1
gap = np.zeros((nb_conesta))
func = np.zeros((nb_conesta))
for i in range(len(conesta_ite)):
     fista_number = ite['continuation_ite_nb'][i]
     gap[i] = ite["gap"][fista_number -1]
     func[i] = ite["func_val"][fista_number -1]


#cor(beta*, beta_hat)
pdf_path = os.path.join(BASE_PATH,"precision_vs_corr_beta.pdf")
pdf = PdfPages(pdf_path)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(gap,corr,label = r"$gap$")
plt.plot(func - func[-1],corr,label = r"$f(\beta^{k})$ - $f(\beta^{*})$")
plt.xscale('log')
plt.ylabel(r"$corr(\beta^{k}$, $\beta^{*})$ ")
plt.xlabel(r"precision")
plt.legend(prop={'size':15})
plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()
pdf.savefig()
plt.close(fig)
pdf.close()

#Plot precision vs cor(Xbeta*, Xbeta_hat)
pdf_path = os.path.join(BASE_PATH,"precision_vs_corr_decision_function.pdf")
pdf = PdfPages(pdf_path)
fig = plt.figure(figsize=(11.69, 8.27))
plt.plot(gap,decfunc,label = r"$gap$")
plt.plot(func - func[-1],decfunc,label = r"$f(\beta^{k})$ - $f(\beta^{*})$")
plt.xscale('log')
plt.ylabel(r"$corr(X\beta^{k}$, $X\beta^{*})$ ")
plt.xlabel(r"precision")
plt.legend(prop={'size':15})
plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()
pdf.savefig()
plt.close(fig)
pdf.close()




# 3) Save Beta map for each conesta iteration in nifti format and display glass brain with iterations info
##############################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL/run"


snap_path = os.path.join(BASE_PATH,"conesta_ite_snapshots")
os.makedirs(snap_path, exist_ok=True)
beta_path = os.path.join(BASE_PATH,"conesta_ite_beta")
os.makedirs(beta_path, exist_ok=True)
conesta_ite = sorted(os.listdir(snap_path))
nb_conesta = len(conesta_ite)
i=1
pdf_path = os.path.join(BASE_PATH,"weight_map_across_iterations.pdf")
pdf = PdfPages(pdf_path)
fig = plt.figure(figsize=(11.69, 8.27))
for ite in conesta_ite:
    path = os.path.join(snap_path,ite)
    conesta_ite_number = ite[-11:-4]
    print ("........Iterations: " + str(conesta_ite_number)+"........")
    ite = np.load(path)
    fista_ite_nb =ite['continuation_ite_nb'][-1]
    beta = ite["beta"][penalty_start:,:]
    beta_t, t = array_utils.arr_threshold_from_norm2_ratio(beta[penalty_start:], 0.99)
    prop_non_zero = float(np.count_nonzero(beta_t)) / float(np.prod(beta.shape))
    print ("Proportion of non-zeros voxels " + str(prop_non_zero))
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = beta.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(beta_path,"beta_"+conesta_ite_number+".nii.gz")
    out_im.to_filename(filename)
    beta = nibabel.load(filename).get_data()
    beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
    fig.add_subplot(nb_conesta,1,i)
    title = "CONESTA iterations: " + str(i) + " -  FISTA iterations : " + str(fista_ite_nb)
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,\
                                      title = title,cmap=plt.cm.bwr)
    plt.text(-43,0.023,"proportion of non-zero voxels:%.4f" % round(prop_non_zero,4))
    pdf.savefig()
    plt.close(fig)
    i = i +1
pdf.close()
#############################################################################
