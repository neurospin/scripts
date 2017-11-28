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


INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/mask.nii'

babel_mask  = nibabel.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3




# 4 OUTPUTS ARE EXPECTED:
    # 1) Weight maps display across conesta iterations
    # 2) Decision function display across iterations
    # 3) Decrease of gap
    # 4) correlation between current iteration and final *  of several metrics



# 1
#Save Beta map for each conesta iteration in nifti format and display glass brain with iterations info
##############################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
VBM/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)
    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    os.makedirs(snap_path, exist_ok=True)
    beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
    os.makedirs(beta_path, exist_ok=True)
    conesta_ite = sorted(os.listdir(snap_path))
    nb_conesta = len(conesta_ite)
    i=1
    pdf_path = os.path.join(BASE_PATH,p,"weight_map_across_iterations.pdf")
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
        nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,title = title)
        plt.text(-43,0.023,"proportion of non-zero voxels:%.4f" % round(prop_non_zero,4))
        pdf.savefig()
        plt.close(fig)
        i = i +1
    pdf.close()
#############################################################################

#some test with plotting.plot.stat

BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
VBM/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)
    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    os.makedirs(snap_path, exist_ok=True)
    beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
    os.makedirs(beta_path, exist_ok=True)
    conesta_ite = sorted(os.listdir(snap_path))
    nb_conesta = len(conesta_ite)
    i=1
    pdf_path = os.path.join(BASE_PATH,p,"weight_map_across_iterations_statmap.pdf")
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
        nilearn.plotting.plot_stat_map(filename,display_mode='ortho',threshold = t,title = title)
        plt.text(-43,0.023,"proportion of non-zero voxels:%.4f" % round(prop_non_zero,4))
        pdf.savefig()
        plt.close(fig)
        i = i +1
    pdf.close()


#some test with plotting.plot.stat same CUT all the time

BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
VBM/model_selectionCV/0"

params = os.listdir(BASE_PATH)
for p in params:
    print (p)
    snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
    os.makedirs(snap_path, exist_ok=True)
    beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
    os.makedirs(beta_path, exist_ok=True)
    conesta_ite = sorted(os.listdir(snap_path))
    nb_conesta = len(conesta_ite)
    i=1
    pdf_path = os.path.join(BASE_PATH,p,"weight_map_across_iterations_statmap_0_-15_4.pdf")
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
        nilearn.plotting.plot_stat_map(filename,display_mode='ortho',threshold = t,title = title,cut_coords = [0,-15,4])
        plt.text(-43,0.023,"proportion of non-zero voxels:%.4f" % round(prop_non_zero,4))
        pdf.savefig()
        plt.close(fig)
        i = i +1
    pdf.close()


#save each weight map separatly in png format to create a gif afterward
## convert -delay 100 *.png CONESTA_iterations.gif
# commadn to create a GIF from serie of png images
for cv in range(1,5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
    params =glob.glob(os.path.join(BASE_PATH,"0*"))
    for p in params:
        p =  os.path.basename(p)
        print (p)
        snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
        os.makedirs(snap_path, exist_ok=True)
        beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
        os.makedirs(beta_path, exist_ok=True)
        conesta_ite = sorted(os.listdir(snap_path))
        nb_conesta = len(conesta_ite)
        i=1
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
            title = "CONESTA iterations: " + str(i) + " -  FISTA iterations : " + str(fista_ite_nb)
            nilearn.plotting.plot_stat_map(filename,display_mode='ortho',\
                                           threshold = t,title = title,cut_coords = [0,-15,4],draw_cross = False)
            plt.text(-43,0.023,"proportion of non-zero voxels:%.4f" % round(prop_non_zero,4))
            plt.savefig(os.path.join(BASE_PATH,p,"images",conesta_ite_number+ ".png"))
            i = i +1
        os.chdir(os.path.join(BASE_PATH,p,"images"))
        cmd_gif = "convert -delay 100 *.png %s.gif" %p
        os.system(cmd_gif )


# 2
#Plot correlation between decision function and y at all conesta iterations
##############################################################################
STATUS_MAP = {0.0 : 'controls', 1.0 : 'patients'}

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
VBM/model_selectionCV/0"
params =glob.glob(os.path.join(BASE_PATH,"0*"))
for p in params:
    p =  os.path.basename(p)
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
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
    params =glob.glob(os.path.join(BASE_PATH,"0*"))
    for p in params:
        p =  os.path.basename(p)
        print (p)
        snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
        beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
        conesta_ite = sorted(os.listdir(snap_path))
        nb_conesta = len(conesta_ite)
        ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))

        pdf_path = os.path.join(BASE_PATH,p,"precision.pdf")
        pdf = PdfPages(pdf_path)
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.plot(np.log10(ite_final["gap"]),label = r"$gap$")
        plt.plot(np.log10(ite_final["func_val"] - ite_final["func_val"][-1]),\
                 label = r"$f(\beta^{k})$ - $f(\beta^{*})$" )
        plt.xlabel("iterations")
        plt.ylabel(r"precision")
        plt.legend(prop={'size':15})
        plt.title("3D voxel-based GM maps - 257595 features")
        pdf.savefig()
        plt.close(fig)
        pdf.close()


        pdf_path_reduced_ite = os.path.join(BASE_PATH,p,"precision_first_ites.pdf")
        pdf_reduced_ite = PdfPages(pdf_path_reduced_ite)
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.plot(np.log10(ite_final["gap"][:10000]),label = r"$gap$")
        plt.plot(np.log10(ite_final["func_val"][:10000] - ite_final["func_val"][-1]),\
                 label = r"$f(\beta^{k})$ - $f(\beta^{*})$" )
        plt.xlabel("iterations")
        plt.ylabel(r"precision")
        plt.legend(prop={'size':15})
        plt.title("3D voxel-based GM maps - 257595 features")
        pdf_reduced_ite.savefig()
        plt.close(fig)
        pdf_reduced_ite.close()

#4 Plot convergence toward beta *
###########################################################################
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/\
VBM/model_selectionCV/0"

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)

import sklearn
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
    params =glob.glob(os.path.join(BASE_PATH,"0*"))
    for p in params:
        p =  os.path.basename(p)
        print (p)
        snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
        beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
        conesta_ite = sorted(os.listdir(snap_path))
        nb_conesta = len(conesta_ite)
        ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
        beta_star =  ite_final["beta"]

        frob_norm = np.zeros((nb_conesta))
        corr = np.zeros((nb_conesta))
        decfunc =  np.zeros((nb_conesta))
        i=0
        for ite in conesta_ite:
            path = os.path.join(snap_path,ite)
            ite = np.load(path)
            corr[i] = np.corrcoef(ite["beta"][:,0],beta_star[:,0])[0][1]
            decfunc[i] = np.corrcoef(np.dot(X, ite["beta"][:,0]),np.dot(X, beta_star[:,0]))[0][1]
            #frob_norm[i]  = np.linalg.norm(ite["beta"] -beta_star,ord = 'fro')
            i = i +1

        pdf_path = os.path.join(BASE_PATH,p,"correlation.pdf")
        pdf = PdfPages(pdf_path)
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.plot(ite['continuation_ite_nb'],corr,label = r"$corr(\beta^{k}$ - $\beta^{*})$ ")
        plt.plot(ite['continuation_ite_nb'],decfunc,label = r"$corr(X\beta^{k}$ - $X\beta^{*})$ ")
        plt.xlabel("iterations")
        plt.ylabel("Correlation coefficient")
        plt.legend(prop={'size':15})
        plt.title("3D voxel-based GM maps - 257595 features")
        pdf.savefig()
        plt.close(fig)
        pdf.close()

        pdf_path_10000 = os.path.join(BASE_PATH,p,"correlation_first_ites.pdf")
        pdf_10000 = PdfPages(pdf_path_10000)
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.plot(ite['continuation_ite_nb'][:20],corr[:20],label = r"$corr(\beta^{k}$ - $\beta^{*})$ ")
        plt.plot(ite['continuation_ite_nb'][:20],decfunc[:20],label = r"$corr(X\beta^{k}$ - $X\beta^{*})$ ")
        plt.xlabel("iterations")
        plt.ylabel("Correlation coefficient")
        plt.legend(prop={'size':15})
        plt.title("3D voxel-based GM maps - 257595 features")
        pdf_10000.savefig()
        plt.close(fig)
        pdf_10000.close()


###########################################################################
X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)


###PLot correlationbetween beta stability and precision
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
    params =glob.glob(os.path.join(BASE_PATH,"0*"))
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


        pdf_path = os.path.join(BASE_PATH,p,"corr_vs_gap.pdf")
        pdf = PdfPages(pdf_path)
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.plot(corr,np.log10(gap),label = r"$gap$")
        plt.plot(corr,np.log10(func - func[-1]),label = r"$f(\beta^{k})$ - $f(\beta^{*})$")
        plt.xlabel(r"$corr(\beta^{k}$ - $\beta^{*})$ ")
        plt.ylabel(r"precision")
        plt.legend(prop={'size':15})
        plt.title("3D voxel-based GM maps - 257595 features")
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)
        pdf.close()


###########################################################################
###PLot correlationbetween beta stability and precision
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
    pdf_path = os.path.join(BASE_PATH,"corr_vs_gap.pdf")
    pdf = PdfPages(pdf_path)
    paired_pal = sns.color_palette("Paired",10)
    colors = {("0.01_0.08_0.72_0.2"):paired_pal[0],
                 ("0.1_0.08_0.72_0.2"):paired_pal[1],
                 ("0.01_0.18_0.02_0.8"):paired_pal[2],
                 ("0.1_0.18_0.02_0.8"):paired_pal[3],
                 ("0.01_0.72_0.08_0.2"):paired_pal[4],
                 ("0.1_0.72_0.08_0.2"):paired_pal[5],
                  ("0.01_0.02_0.18_0.8"):paired_pal[6],
                 ("0.1_0.02_0.18_0.8"):paired_pal[7]}

    fig = plt.figure(figsize=(11.69, 8.27))

    params =glob.glob(os.path.join(BASE_PATH,"0*"))
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


        plt.plot(corr,np.log10(gap),color = colors[(p)],label = p)

    plt.xlabel(r"$corr(\beta^{k}$ - $\beta^{*})$ ")
    plt.ylabel(r"precision")
    plt.legend(prop={'size':15})
    plt.title("3D voxel-based GM maps - 257595 features")
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()
###########################################################################
#Boxplot of precision at 99% correlation for the 6 parameters under studies at 5 differetn starting points
precision = np.zeros((5,8))
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
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
        precision[cv,p_index] = f(0.90)
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
    plt.title(r" Gap required to obtain $corr(\beta^{k}$ - $\beta^{*})= 0.90$ at different starting points")
    plt.tight_layout()
   #plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/gap_required.png")
    plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/gap_required_90%.pdf")


###########################################################################

#Boxplot of number of iteration at 99% correlation for the 6 parameters under studies at 5 differetn starting points
precision = np.zeros((5,8))
for cv in range(5):
    print ("CV number : %s" %cv)
    BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
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
    plt.savefig("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/number_FISTA_required_90%.pdf")

#################################################################

 # compare the correlation between beta when different starting point   are used
OUTPUT = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/start_vectors"
precision = np.zeros((5,8))
for p in params:
        p =  os.path.basename(p)
        print (p)
        betas = np.zeros((257595,5,30))
        for cv in range(5):
            print ("CV number : %s" %cv)
            BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/VBM/model_selectionCV%s/0" %cv
            params =glob.glob(os.path.join(BASE_PATH,"0*"))
            p_index = 0
            snap_path = os.path.join(BASE_PATH,p,"conesta_ite_snapshots")
            beta_path = os.path.join(BASE_PATH,p,"conesta_ite_beta")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)
            i = 0
            for ite in conesta_ite:
                path = os.path.join(snap_path,ite)
                ite = np.load(path)
                beta =  ite["beta"][:,0]
                betas[:,cv,i] = beta_star
                corr = np.corrcoef(betas[:,:,i].T)
                mean_pairwise_corr = corr[corr!=1.].mean()
                print(mean_pairwise_corr)
                i = i + 1

    pdf_path = os.path.join(OUTPUT,"starting_vector_" + str(p) +"_.pdf")
    pdf = PdfPages(pdf_path)
    i=0
    frob_norm = np.zeros((nb_conesta))
    for ite in conesta_ite:
        path_0 = os.path.join(snap_path_0,ite)
        ite_0 = np.load(path_0)
        path_1 = os.path.join(snap_path_1,ite)
        ite_1 = np.load(path_1)
        frob_norm[i]  = np.linalg.norm(ite_0["beta"] -ite_1["beta"],ord = 'fro')
        i = i +1

    fig = plt.figure(figsize=(11.69, 8.27))
    plt.plot(frob_norm)
    plt.xlabel("CONESTA iterations")
    plt.ylabel(r"$\|\beta^{k}_{v0}$ - $\beta^{k}_{v1}\|$ ")
    plt.title("3D voxel-based GM maps - 257595 features")
    pdf.savefig()
    plt.close(fig)

    pdf.close()
