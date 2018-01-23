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
pdf = PdfPages(pdf_path)
#fig = plt.figure(figsize=(6, 4.5))# Nice size
fig = plt.figure(figsize=(6, 2.3))# reduce height
#plt.plot(ite_final["gap"],label = r"$\textsc{Gap}_{\mu}$")
plt.plot(ite, gap + mu * gM, color=colors[0],
         label = r"$\varepsilon\equiv \textsc{Gap}_{\mu^k}(\beta^{k}) + \mu^k\gamma M$")

plt.plot(ite, func_val - fstar, color=colors[1],
         label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$" )
plt.yscale('log')
plt.xscale('log')
plt.ylim([y_min, y_max])

plt.xlabel("Iterations [k]")
plt.ylabel(r"Precision [$\varepsilon$]")
plt.grid()
plt.legend()#prop={'size':15})
#plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()

pdf.savefig()
plt.close(fig)
pdf.close()

###############################################################################
# 2) x:precision, y: cor(Xbeta*, Xbeta_hat) + cor(beta*, beta_hat)
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

i=0
Xbetastar = np.dot(X, beta_star[:,0])
Xbetastar_nrm  = np.sqrt(np.sum(Xbetastar ** 2))

for ite in conesta_ite:
    path = os.path.join(snap_path,ite)
    ite = np.load(path)
    corr[i] = np.corrcoef(ite["beta"][:,0], beta_star[:,0])[0][1]
    Xbetak = np.dot(X, ite["beta"][:,0])
    decfunc[i] = np.corrcoef(Xbetak, Xbetastar)[0][1]
    decfunc_mse[i] = np.sqrt(np.sum((Xbetak - Xbetastar) ** 2))
    decfunc_mse_norm[i] = np.sqrt(np.sum((Xbetak - Xbetastar) ** 2)) / Xbetastar_nrm
    i = i + 1

gap = np.zeros((nb_conesta))
func = np.zeros((nb_conesta))
for i in range(len(conesta_ite)):
     fista_number = ite['continuation_ite_nb'][i]
     gap[i] = ite["gap"][fista_number -1]
     func[i] = ite["func_val"][fista_number -1]

eps_thrsld = 1e-3
gap_thrsld_idx = np.where(gap <= eps_thrsld)[0][0]
eps_thrsld_idx = np.where(func - func[-1] <= eps_thrsld)[0][0]

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
#fig = plt.figure(figsize=(6, 4.5))# Nice size
fig = plt.figure(figsize=(6, 2.3))# reduce height
plt.plot(gap, corr, label = r"$\varepsilon\equiv \textsc{Gap}_{\mu^k}(\beta^{k}) + \mu^k\gamma M$", color=colors[0])
plt.plot(func - func[-1], corr, label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$", color=colors[1])
# vline @ 1e-3
plt.plot([eps_thrsld, eps_thrsld], [0, cor_up], "k--")
plt.text(eps_thrsld, cor_up + cor_up/100, "%.2f" % cor_up)

plt.xscale('log')
plt.ylabel(r"corr$(\beta^{k}$, $\beta^{*})$ ")
plt.xlabel(r"Precision [$\varepsilon$]")
plt.grid()
plt.legend()#prop={'size':15})
#plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()
pdf.savefig()
plt.close(fig)
pdf.close()


"""
Stopping at gap <= 1e-3, lead to a corr(betak, beta*) of 0.97
Stopping at eps <= 1e-3, lead to a corr(betak, beta*) of 0.92
"""
###############################################################################
# Plot precision vs cor(Xbeta*, Xbeta_hat)

decfunc_mse_norm_up = decfunc_mse_norm[gap_thrsld_idx]
decfunc_mse_up = decfunc_mse[gap_thrsld_idx]

# tune
plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

pdf_path = os.path.join(BASE_PATH,"precision_vs_err_decision_function.pdf")
pdf = PdfPages(pdf_path)
fig = plt.figure(figsize=(6, 4.5))#(11.69, 8.27))
#plt.plot(gap,decfunc,label = r"$gap$")
#plt.plot(func - func[-1],decfunc,label = r"$f(\beta^{k})$ - $f(\beta^{*})$")
plt.plot(gap, decfunc_mse,label = r"$\varepsilon\equiv\textsc{Gap}_{\mu^k}(\beta^k)$", color=colors[0])
plt.plot(func - func[-1], decfunc_mse,label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$", color=colors[1])
#plt.plot(gap, decfunc_mse_norm,label = r"$\varepsilon\equiv\textsc{Gap}_{\mu^k}(\beta^k)$")
#plt.plot(func - func[-1],decfunc_mse_norm,label = r"$\varepsilon\equiv f(\beta^{k})$ - $f(\beta^{*})$")

# vline @ 1e-3
plt.plot([eps_thrsld, eps_thrsld], [0, decfunc_mse_norm_up], "k--")
plt.text(eps_thrsld + eps_thrsld/10, decfunc_mse_norm_up,
         "%.3f" % decfunc_mse_norm_up, verticalalignment='top')

plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"${\Vert X \beta^{k} - X\beta^{*}\Vert_2 / \Vert X\beta^{*}\Vert_2$ ")
plt.xlabel(r"Precision [$\varepsilon$]")
plt.legend(loc="lower right")#prop={'size':15})
#plt.title("ADNI - MCI-CTL - 286214 features")
fig.tight_layout()

pdf.savefig()
plt.close(fig)
pdf.close()
