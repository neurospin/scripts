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
import sys
sys.path.insert(0, os.path.join(os.getenv("HOME"), 'git/scripts/brainomics'))
import array_utils
import nilearn
from nilearn import image
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting
import glob
#import seaborn as sns
#sns.set_style("whitegrid")


WD = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL"
os.chdir(WD)

arxiv = np.load("ADNI_ADAS11-MCIc-CTL_N199.npz")
X = arxiv["X"]
y = arxiv["y"]

babel_mask  = nibabel.load('mask.nii.gz')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

penalty_start = 0
# Palette
# sns.palplot(sns.xkcd_palette(["denim blue", "pale red", "medium green"]).as_hex())
colors = ['#3b5b92', '#d9544d', '#39ad48']

###############################################################################
# 2) gM, since eps ~ gap_mu + mu * gM
###############################################################################
ALPHA = 0.01 #
_, _, g = ALPHA * np.array([0.3335, 0.3335, 0.333])

# gM = function.eps_max(1.0)
# gM = self.tv.l * self.tv.M()
M = X.shape[1] / 2.0# self._A[0].shape[0] / 2.0
gM = g * M
# stoping criterium:
# gap_mu + mu * gM < self.eps:



###############################################################################
# 1) x:iteration y:precision+gap
###############################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL/run"
snap_path = os.path.join(BASE_PATH,"conesta_ite_snapshots")
conesta_ite = sorted(os.listdir(snap_path))
nb_conesta = len(conesta_ite)
ite_final = np.load(os.path.join(snap_path, conesta_ite[-1]))

fstar = ite_final["func_val"][-1]
gap = ite_final["gap"]
func_val = ite_final["func_val"]
mu = ite_final["mu"]
ite = np.arange(1, 1+len(mu))

# pick continuation points
mask = np.concatenate([np.diff(mu) != 0, [0]], 0).astype(bool)
gap = gap[mask]
func_val = func_val[mask]
mu = mu[mask]
ite = ite[mask]

# tune
plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
y_max = max(ite_final["gap"].max(), ite_final["func_val"].max())
y_min = 10 ** (-9)

# plot
pdf_path = os.path.join(BASE_PATH,"fig_mri3d_adni_adas_l1l2tv_regression_conesta_precision_iterations.pdf")
#pdf_path = os.path.join(BASE_PATH,"precision_iterations.pdf")

pdf = PdfPages(pdf_path)

# -
#fig = plt.figure(figsize=(6, 4.5))# Nice size
fig = plt.figure(figsize=(6, 2.3))# reduce height
#plt.plot(ite_final["gap"],label = r"$\textsc{Gap}_{\mu}$")
plt.plot(ite, gap + mu * gM, color=colors[0],
         label = r"$\varepsilon\equiv \textsc{Gap}_{\mu^k}(\beta^{k}) + \mu^k\gamma M$")

plt.plot(ite, func_val - fstar, color=colors[1],
         label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$" )
plt.yscale('log')
plt.xscale('log')
#plt.ylim([y_min, y_max])
plt.ylim(1e-5, 1e5)
plt.xlim(1e1, 1e5)
plt.yticks([1e4, 1e1, 1e-1, 1e-2, 1e-3, 1e-5])
plt.xlabel("Iterations [k]")
plt.ylabel(r"Precision [$\varepsilon$]")
plt.grid()
plt.legend()#prop={'size':15})
#plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()
# -

pdf.savefig()
plt.close(fig)
pdf.close()

###############################################################################
# 2) x:precision, y: cor(Xbeta*, Xbeta_hat) and cor(beta*, beta_hat)
###############################################################################

BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL/run"
snap_path = os.path.join(BASE_PATH,"conesta_ite_snapshots")
beta_path = os.path.join(BASE_PATH,"conesta_ite_beta")
conesta_ite = sorted(os.listdir(snap_path))
nb_conesta = len(conesta_ite)
ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
beta_star =  ite_final["beta"]

corr = np.zeros((nb_conesta))
decfunc =  np.zeros((nb_conesta))
decfunc_mse =  np.zeros((nb_conesta))
decfunc_mse_norm =  np.zeros((nb_conesta))
decfunc_mae =  np.zeros((nb_conesta)) # mean absolute error
decfunc_mae_norm =  np.zeros((nb_conesta)) # mean absolute error

i=0
Xbetastar = np.dot(X, beta_star[:,0])
Xbetastar_nrm  = np.sqrt(np.sum(Xbetastar ** 2))
Xbetastar_nrm1  = np.sum(np.abs(Xbetastar))

for ite in conesta_ite:
    path = os.path.join(snap_path,ite)
    ite = np.load(path)
    corr[i] = np.corrcoef(ite["beta"][:,0], beta_star[:,0])[0][1]
    Xbetak = np.dot(X, ite["beta"][:,0])
    decfunc[i] = np.corrcoef(Xbetak, Xbetastar)[0][1]
    decfunc_mse[i] = np.sqrt(np.sum((Xbetak - Xbetastar) ** 2)) / len(Xbetastar)
    decfunc_mse_norm[i] = np.sqrt(np.sum((Xbetak - Xbetastar) ** 2)) / Xbetastar_nrm
    decfunc_mae[i] = np.mean(np.abs(Xbetak - Xbetastar))
    decfunc_mae_norm[i] = np.sum(np.abs(Xbetak - Xbetastar)) / Xbetastar_nrm1
    i = i + 1

plt.plot(decfunc_mse, decfunc_mae)

gap = np.zeros((nb_conesta))
func = np.zeros((nb_conesta))
for i in range(len(conesta_ite)):
     fista_number = ite['continuation_ite_nb'][i]
     gap[i] = ite["gap"][fista_number -1]
     func[i] = ite["func_val"][fista_number -1]

eps_true = func - func[-1]

eps_thrsld = 1e-3
gap_thrsld_idx = np.where(gap <= eps_thrsld)[0][0]
eps_thrsld_idx = np.where(eps_true <= eps_thrsld)[0][0]

eps_thrsld_loose = 0.01009564#1e-2
gap_thrsld_idx_loose = np.where(gap <= eps_thrsld_loose)[0][0]
eps_thrsld_idx_loose = np.where(eps_true <= eps_thrsld_loose)[0][0]
#eps_true[ (eps_true >= 1e-3) & (eps_true <= 1e-1)]

print("Stopping at gap <= 1e-3, lead to a corr(betak, beta*) of %.2f" % corr[gap_thrsld_idx])
print("Stopping at eps <= 1e-3, lead to a corr(betak, beta*) of %.2f" % corr[eps_thrsld_idx])



#x_min = min(gap.min(), (func - func[-1]).min())

###############################################################################
# Plot cor(beta*, beta_hat)
cor_up = max(corr[gap_thrsld_idx], corr[eps_thrsld_idx])


# tune
plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})

pdf_path = os.path.join(BASE_PATH,"precision_vs_corr_beta.pdf")
pdf = PdfPages(pdf_path)

# -
y_axis = corr
#fig = plt.figure(figsize=(6, 4.5))# Nice size
fig = plt.figure(figsize=(6, 2.3))# reduce height
plt.plot(gap, corr, label = r"$\varepsilon\equiv \textsc{Gap}_{\mu^k}(\beta^{k}) + \mu^k\gamma M$", color=colors[0])
plt.plot(func - func[-1], corr, label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$", color=colors[1])

# vline @ 1e-2 on gap
plt.plot([eps_thrsld_loose, eps_thrsld_loose], [0, y_axis[gap_thrsld_idx_loose]], "k--")
plt.text(eps_thrsld_loose + eps_thrsld_loose/10, y_axis[gap_thrsld_idx_loose]- 0.05,
         "%.3f" % y_axis[gap_thrsld_idx_loose], verticalalignment='top')
# vline @ 1e-2 on true eps
(func - func[-1])[y_axis <= 2e-2]

plt.plot([eps_thrsld_loose, eps_thrsld_loose], [0, y_axis[eps_thrsld_idx_loose]], "k--")
plt.text(eps_thrsld_loose + eps_thrsld_loose/10, y_axis[eps_thrsld_idx_loose],
         "%.3f" % y_axis[eps_thrsld_idx_loose], verticalalignment='top')

# vline @ 1e-3 on gap
plt.plot([eps_thrsld, eps_thrsld], [0, y_axis[gap_thrsld_idx]], "k--")
plt.text(eps_thrsld + eps_thrsld/10, y_axis[gap_thrsld_idx] + 0.1,
         "%.3f" % y_axis[gap_thrsld_idx], verticalalignment='top')
# vline @ 1e-3 on true eps
plt.plot([eps_thrsld, eps_thrsld], [0, y_axis[eps_thrsld_idx]], "k--")
plt.text(eps_thrsld + eps_thrsld/10, y_axis[eps_thrsld_idx],
         "%.3f" % y_axis[eps_thrsld_idx], verticalalignment='top')

# vline @ 1e-3
#plt.plot([eps_thrsld, eps_thrsld], [0, cor_up], "k--")
#plt.text(eps_thrsld, cor_up + cor_up/100, "%.2f" % cor_up)

plt.xlim(1e-5, 1e3)
plt.ylim(0, 1.1)
plt.xscale('log')
plt.ylabel(r"corr$(\beta^{k}$, $\beta^{*})$ ")
plt.xlabel(r"Precision [$\varepsilon$]")
plt.grid()
plt.legend()#prop={'size':15})
#plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()
# -

pdf.savefig()
plt.close(fig)
pdf.close()


"""
Stopping at gap <= 1e-3, lead to a corr(betak, beta*) of 0.97
Stopping at eps <= 1e-3, lead to a corr(betak, beta*) of 0.92
"""
###############################################################################
# Plot precision vs cor(Xbeta*, Xbeta_hat)

# tune
plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

#pdf_path = os.path.join(BASE_PATH,"precision_vs_err_decision_function.pdf")
pdf_path = os.path.join(BASE_PATH,"precision_vs_mae-norm_decision_function.pdf")
pdf = PdfPages(pdf_path)

# -
fig = plt.figure(figsize=(6, 4.5))#(11.69, 8.27))

#y_axis = decfunc_mse
#y_axis = decfunc_mse_norm
#y_axis = decfunc_mae
y_axis = decfunc_mae_norm

y_axis_topvline = y_axis[gap_thrsld_idx]
plt.plot(gap, y_axis,label = r"$\varepsilon\equiv\textsc{Gap}_{\mu^k}(\beta^k)$", color=colors[0])
plt.plot(func - func[-1], y_axis,label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$", color=colors[1])

# vline @ 1e-2 on gap
plt.plot([eps_thrsld_loose, eps_thrsld_loose], [0, y_axis[gap_thrsld_idx_loose]], "k--")
plt.text(eps_thrsld_loose + eps_thrsld_loose/10, y_axis[gap_thrsld_idx_loose],
         "%.3f" % y_axis[gap_thrsld_idx_loose], verticalalignment='top')
# vline @ 1e-2 on true eps
(func - func[-1])[y_axis <= 2e-2]

plt.plot([eps_thrsld_loose, eps_thrsld_loose], [0, y_axis[eps_thrsld_idx_loose]], "k--")
plt.text(eps_thrsld_loose + eps_thrsld_loose/10, y_axis[eps_thrsld_idx_loose]/1.2,
         "%.3f" % y_axis[eps_thrsld_idx_loose], verticalalignment='top')

# vline @ 1e-3 on gap
plt.plot([eps_thrsld, eps_thrsld], [0, y_axis[gap_thrsld_idx]], "k--")
plt.text(eps_thrsld + eps_thrsld/10, y_axis[gap_thrsld_idx],
         "%.3f" % y_axis[gap_thrsld_idx], verticalalignment='top')
# vline @ 1e-3 on true eps
plt.plot([eps_thrsld, eps_thrsld], [0, y_axis[eps_thrsld_idx]], "k--")
plt.text(eps_thrsld + eps_thrsld/10, y_axis[eps_thrsld_idx],
         "%.3f" % y_axis[eps_thrsld_idx], verticalalignment='top')

plt.xlim(1e-6, 1e0)
plt.ylim(1e-4, 1e-1)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"${\Vert X \beta^{k} - X\beta^{*}\Vert_1 / \Vert X\beta^{*}\Vert_1$ ")
#plt.ylabel(r"${\Vert X \beta^{k} - X\beta^{*}\Vert_2 / \Vert X\beta^{*}\Vert_2$ ")
plt.xlabel(r"Precision [$\varepsilon$]")
plt.grid()
plt.legend(loc="lower right")#prop={'size':15})
#plt.title("ADNI - MCI-CTL - 286214 features")

fig.tight_layout()
# -

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
