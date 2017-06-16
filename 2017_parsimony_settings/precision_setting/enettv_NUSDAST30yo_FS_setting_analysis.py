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


INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/mask.npy'

mask  = np.load(INPUT_MASK_PATH)
penalty_start = 3

# 4 OUTPUTS ARE EXPECTED:
    # 1) Weight maps display across conesta iterations
    # 2) Decision function display across iterations
    # 3) Decrease of gap
    # 4) correlation between current iteration and final *  of several metrics



# 1
#Save Beta map for each conesta iteration in nifti format and display glass brain with iterations info
# Anatomist for meshes..
#############################################################################

# 2
#Plot correlation between decision function and y at all conesta iterations
##############################################################################
STATUS_MAP = {0.0 : 'controls', 1.0 : 'patients'}

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
Freesurfer/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)
    i=1
    fig = plt.figure(figsize=(11.69, 8.27))
    decision_funcs = dict()
    betas = dict()
    pdf_path = os.path.join(BASE_PATH,p,"decision_functions_pairplots.pdf")
    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    conesta_ite = sorted(os.listdir(snap_path))
    sns.set_style(None)
    pdf = PdfPages(pdf_path)
    for ite in conesta_ite:
        path = os.path.join(snap_path,ite)
        ite = np.load(path)
        fista_ite_nb =ite['continuation_ite_nb'][-1]
        beta = ite["beta"]
        key = path[-11:-4]
        betas[key] = beta
        decision_funcs[key] = np.dot(X, beta)
        print("################################################################")
        print(key)
        decfuncs = pd.DataFrame(np.array(decision_funcs[key]))
        variables = decfuncs.columns.sort_values()
        decfuncs["y"] = y
        decfuncs['status'] = decfuncs["y"].map(STATUS_MAP)
        print(decfuncs.corr())
        corr = decfuncs.corr()[0][1]

        beta = ite["beta"][penalty_start:,:]
        beta_t, t = array_utils.arr_threshold_from_norm2_ratio(beta[penalty_start:], 0.99)
        prop_non_zero = float(np.count_nonzero(beta_t)) / float(np.prod(beta.shape))
        print ("Proportion of non-zeros voxels " + str(prop_non_zero))

        sns.set()
        ax = sns.pairplot(decfuncs, hue="status", vars=variables)
        title = "CONESTA ite: " + str(i) + " -  FISTA ite : " + str(fista_ite_nb)
        ax.set(xlabel='', ylabel='Decision function')
        plt.title(title, fontsize=7)
        plt.suptitle('Corr with y: %.3f - prop non-zeros voxels: %.3f' % (round(corr,3),round(prop_non_zero,3)), fontsize=8, fontweight='bold')
        plt.legend()
        ax.fig.tight_layout()
        handles = ax._legend_data.values()
        labels = ax._legend_data.keys()
        ax.fig.legend(handles=handles, labels=labels, loc='upper left', ncol=3)

        print("\nDecision function correlations\n")

        pdf.savefig()  # saves the current figure into a pdf page
        plt.clf()
        i = i +1
    pdf.close()


#3 Plot decrease of Gap, parameter of smoothing mu, and value of function
###########################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
Freesurfer/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)

    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
    conesta_ite = sorted(os.listdir(snap_path))
    nb_conesta = len(conesta_ite)
    ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))

    pdf_path = os.path.join(BASE_PATH,p,"metrics_vs_iterations.pdf")
    pdf = PdfPages(pdf_path)

    fig = plt.figure(figsize=(11.69, 8.27))
    plt.plot(np.log10(ite_final["gap"]))
    plt.xlabel("iterations")
    plt.ylabel(r"$\log (gap)$")
    plt.title("2D cortical thickness - 299731 features")
    pdf.savefig()
    plt.close(fig)

    fig = plt.figure(figsize=(11.69, 8.27))
    plt.plot(np.log10(ite_final["func_val"]))
    plt.xlabel("iterations")
    plt.ylabel(r"$\log (func_val)$")
    plt.title("2D cortical thickness - 299731 features")
    pdf.savefig()
    plt.close(fig)

    fig = plt.figure(figsize=(11.69, 8.27))
    plt.plot(np.log10(ite_final["mu"]))
    plt.xlabel("iterations")
    plt.ylabel(r"$\log (mu)$")
    plt.title("2D cortical thickness - 299731 features")
    pdf.savefig()
    plt.close(fig)

    pdf.close()


#4 Plot convergence toward beta *
###########################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
Freesurfer/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)
    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
    conesta_ite = sorted(os.listdir(snap_path))
    nb_conesta = len(conesta_ite)
    ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
    beta_star =  ite_final["beta"]

    pdf_path = os.path.join(BASE_PATH,p,"metrics_convergence.pdf")
    pdf = PdfPages(pdf_path)

    fig = plt.figure(figsize=(11.69, 8.27))
    plt.plot(np.log10(ite_final["func_val"] - ite_final["func_val"][-1]))
    plt.xlabel("iterations")
    plt.ylabel(r"$f(\beta^{k})$ - $f(\beta^{*})$ ")
    plt.title("2D cortical thickness - 299731 features")
    pdf.savefig()
    plt.close(fig)

    frob_norm = np.zeros((nb_conesta))
    i=0
    for ite in conesta_ite:
        path = os.path.join(snap_path,ite)
        ite = np.load(path)
        frob_norm[i]  = np.linalg.norm(ite["beta"] -beta_star,ord = 'fro')
        i = i +1

    fig = plt.figure(figsize=(11.69, 8.27))
    plt.plot(frob_norm)
    plt.xlabel("CONESTA iterations")
    plt.ylabel(r"$\|\beta^{k}$ - $\beta^{*}\|$ ")
    plt.title("2D cortical thickness - 299731 features")
    pdf.savefig()
    plt.close(fig)

    pdf.close()
###########################################################################

#Boxplot of precision at 99% correlation for the 6 parameters under studies at 5 differetn starting points
precision = np.zeros((5,8))
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer/model_selectionCV%s/0" %cv
    params =glob.glob(os.path.join(BASE_PATH,"0*"))
    p_index = 0
    for p in params:
        p =  os.path.basename(p)
        print (p)
        snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
        beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
        conesta_ite = sorted(os.listdir(snap_path))
        nb_conesta = len(conesta_ite)
        ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
        beta_star =  ite_final["beta"]
        corr = np.zeros((nb_conesta))
        i=0
        for ite in conesta_ite:
            path = os.path.join(snap_path,ite)
            ite = np.load(path)
            corr[i] = np.corrcoef(ite["beta"][:,0],beta_star[:,0])[0][1]

            i = i + 1
        gap = np.zeros((nb_conesta))
        for i in range(len(conesta_ite)):
             fista_number = ite['continuation_ite_nb'][i]
             gap[i] = ite["gap"][fista_number -1]


        #plt.plot(corr,np.log10(gap))
        from scipy.interpolate import interp1d
        f = interp1d(corr,gap)
        xnew = np.linspace(0, 1, num=100, endpoint=True)
        #plt.plot(corr, gap, 'o', xnew, f(xnew), '-',)
        #plt.plot(corr,np.log10(gap),'o',xnew,np.log10(f(xnew)))
        #plt.legend(['data', 'linear'], loc='best')
        #plt.show()
        precision[cv,p_index] = f(0.99)
        p_index = p_index + 1

    data_to_plot = [np.log10(precision[0,:]),np.log10(precision[1,:]),np.log10(precision[2,:]),\
                   np.log10(precision[3,:]),np.log10(precision[4,:])]
    df = pd.DataFrame(data=data_to_plot)
    for i in range(8):
      df =  df.rename(columns = {i:os.path.basename(params[i])})
    ax = sns.boxplot(df)
    ax.set_xticklabels([os.path.basename(params[0]), os.path.basename(params[1]),\
                       os.path.basename(params[2]), os.path.basename(params[3]),\
                       os.path.basename(params[4]), os.path.basename(params[5]),\
                       os.path.basename(params[6]), os.path.basename(params[7])])
    plt.xticks(rotation=70)
    plt.title(r" Gap required to obtain $corr(\beta^{k}$ - $\beta^{*})= 0.99$ at different starting points")
    plt.tight_layout()
   #plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/gap_required.png")
    plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer/gap_required_99%.pdf")


###########################################################################

#Boxplot of number of iteration at 99% correlation for the 6 parameters under studies at 5 differetn starting points
precision = np.zeros((5,8))
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer/model_selectionCV%s/0" %cv
    params =glob.glob(os.path.join(BASE_PATH,"0*"))
    p_index = 0
    for p in params:
        p =  os.path.basename(p)
        print (p)
        snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
        beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
        conesta_ite = sorted(os.listdir(snap_path))
        nb_conesta = len(conesta_ite)
        ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
        beta_star =  ite_final["beta"]
        corr = np.zeros((nb_conesta))
        i=0
        for ite in conesta_ite:
            path = os.path.join(snap_path,ite)
            ite = np.load(path)
            corr[i] = np.corrcoef(ite["beta"][:,0],beta_star[:,0])[0][1]

            i = i + 1
        fista_ite = ite_final['continuation_ite_nb']
        #plt.plot(corr,np.log10(gap))
        from scipy.interpolate import interp1d
        f = interp1d(corr,fista_ite)
        xnew = np.linspace(0, 1, num=100, endpoint=True)
        #plt.plot(corr, gap, 'o', xnew, f(xnew), '-',)
        #plt.plot(corr,np.log10(gap),'o',xnew,np.log10(f(xnew)))
        #plt.legend(['data', 'linear'], loc='best')
        #plt.show()
        precision[cv,p_index] = f(0.90)
        p_index = p_index + 1

    data_to_plot = [(precision[0,:]),(precision[1,:]),(precision[2,:]),(precision[3,:]),(precision[4,:])]
    df = pd.DataFrame(data=data_to_plot)
    for i in range(8):
       df =  df.rename(columns = {i:os.path.basename(params[i])})
    ax = sns.boxplot(df)
    ax.set_xticklabels([os.path.basename(params[0]), os.path.basename(params[1]),\
                       os.path.basename(params[2]), os.path.basename(params[3]),\
                       os.path.basename(params[4]), os.path.basename(params[5]),\
                       os.path.basename(params[6]), os.path.basename(params[7])])
    plt.xticks(rotation=70)
    plt.title(r" Number of FISTA iterations required to obtain $corr(\beta^{k}$ - $\beta^{*})= 0.90$ at different starting points")
    plt.tight_layout()
   #plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/number_FISTA_required.png")
    plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer/number_FISTA_required_90%.pdf")




