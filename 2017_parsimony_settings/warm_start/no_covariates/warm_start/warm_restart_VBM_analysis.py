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
import sklearn.metrics



#no covariates =
INPUT_MASK_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/\
VBM/no_covariates/warm_restart/all_all_as_start_vector/mask.nii"
INPUT_DATA_X = '/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/y.npy'
BASE_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/\
VBM/no_covariates/warm_restart/cv00_all_as_start_vector/model_selectionCV/*/*"
babel_mask  = nibabel.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
penalty_start = 0

def launch_analysis(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
    #create_gif(BASE_PATH)
    plot_precision(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y)
    plot_correlation(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y)
    plot_stabiility_vs_gap(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y)
    plot_stabiility_vs_gap_all_params(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y)



#save each weight map separatly in png format to create a gif afterward
## convert -delay 100 *.png CONESTA_iterations.gif
# commadn to create a GIF from serie of png images
def create_gif(BASE_PATH):

    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            os.makedirs(snap_path, exist_ok=True)
            beta_path = os.path.join(PATH,p,"conesta_ite_beta")
            os.makedirs(beta_path, exist_ok=True)
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)
            os.makedirs(os.path.join(PATH,p,"images"), exist_ok=True)
            i=1
            for ite in conesta_ite:
                conesta_ite_number = ite[-10:-4]
                print ("........Iterations: " + str(conesta_ite_number)+"........")
                if not os.path.exists(os.path.join(PATH,p,"images",conesta_ite_number+ ".png")):
                    path = os.path.join(snap_path,ite)
                    ite = np.load(path)
                    fista_ite_nb =ite['continuation_ite_nb'][-1]
                    beta = ite["beta"][penalty_start:,:]
                    beta_t, t = array_utils.arr_threshold_from_norm2_ratio(beta[penalty_start:], 0.99)
                    prop_non_zero = float(np.count_nonzero(beta_t)) / float(np.prod(beta.shape))
                    print ("Proportion of non-zeros voxels " + str(prop_non_zero))
                    arr = np.zeros(mask_bool.shape);
                    arr[mask_bool] = beta.ravel()
                    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.affine)
                    filename = os.path.join(beta_path,"beta_"+conesta_ite_number+".nii.gz")
                    out_im.to_filename(filename)
                    beta = nibabel.load(filename).get_data()
                    beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
                    title = "CONESTA iterations: " + str(i) + " -  FISTA iterations : " + str(fista_ite_nb)
                    nilearn.plotting.plot_stat_map(filename,display_mode='ortho',\
                                                   threshold = t,title = title,cut_coords = [0,-15,4],draw_cross = False)
                    plt.text(-43,0.023,"proportion of non-zero voxels:%.4f" % round(prop_non_zero,4))

                    plt.savefig(os.path.join(PATH,p,"images",conesta_ite_number+ ".png"))
                i = i +1
            os.chdir(os.path.join(PATH,p,"images"))
            cmd_gif = "convert -delay 100 *.png %s.gif" %p
            os.system(cmd_gif )



#3 Plot decrease of Gap, parameter of smoothing mu, and value of function
###########################################################################
def plot_precision(BASE_PATH):
    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)
            print ("Number of CONESTA : %s" %nb_conesta)
            if nb_conesta != 0:
                ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))

                pdf_path = os.path.join(PATH,p,"precision.pdf")
                pdf = PdfPages(pdf_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.plot((ite_final["gap"]),label = r"$gap$")
                plt.plot((ite_final["func_val"] - ite_final["func_val"][-1]),\
                         label = r"$f(\beta^{k})$ - $f(\beta^{*})$" )
                plt.xlabel("iterations")
                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel(r"precision")
                plt.legend(prop={'size':15})
                plt.title("3D voxel-based GM maps - 257595 features")
                pdf.savefig()
                plt.close(fig)
                pdf.close()

#4 Plot convergence toward beta *
###########################################################################

def plot_correlation(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)

    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)

            if nb_conesta != 0:
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
                    #frob_norm[i]  = np.linalg.norm(ite["beta"] -beta_star,ord = 'fro')
                    i = i +1

                pdf_path = os.path.join(PATH,p,"correlation.pdf")
                pdf = PdfPages(pdf_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.plot((ite['continuation_ite_nb']),corr,label = r"$corr(\beta^{k}$ - $\beta^{*})$ ")
                plt.plot((ite['continuation_ite_nb']),decfunc,label = r"$corr(X\beta^{k}$ - $X\beta^{*})$ ")
                plt.xscale('log')
                plt.xlabel("iterations")
                plt.ylabel("Correlation coefficient")
                plt.legend(prop={'size':15})
                plt.title("3D voxel-based GM maps - 257595 features")
                pdf.savefig()
                plt.close(fig)
                pdf.close()



def plot_mse_beta(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)

    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)

            if nb_conesta != 0:
                ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
                beta_star =  ite_final["beta"]

                mse = np.zeros((nb_conesta))

                i=0
                for ite in conesta_ite:
                    path = os.path.join(snap_path,ite)
                    ite = np.load(path)
                    mse[i] = sklearn.metrics.mean_squared_error(ite["beta"][:,0],beta_star[:,0])
                    i = i +1

                pdf_path = os.path.join(PATH,p,"mse_beta.pdf")
                pdf = PdfPages(pdf_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.plot((ite['continuation_ite_nb']),mse,label = r"$mse(\beta^{k}$ - $\beta^{*})$ ")
                plt.xscale('log')
                plt.xlabel("iterations")
                plt.ylabel("Mean squared Error")
                plt.legend(prop={'size':15})
                plt.title("3D voxel-based GM maps - 257595 features")
                pdf.savefig()
                plt.close(fig)
                pdf.close()

def plot_mse_decfunc(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)

    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)

            if nb_conesta != 0:
                ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
                beta_star =  ite_final["beta"]

                decfunc =  np.zeros((nb_conesta))
                i=0
                for ite in conesta_ite:
                    path = os.path.join(snap_path,ite)
                    ite = np.load(path)
                    decfunc[i] = sklearn.metrics.mean_squared_error(np.dot(X, ite["beta"][:,0]),np.dot(X, beta_star[:,0]))
                    i = i +1

                pdf_path = os.path.join(PATH,p,"mse_decfunc.pdf")
                pdf = PdfPages(pdf_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.plot((ite['continuation_ite_nb']),decfunc,label = r"$mse(X\beta^{k}$ - $X\beta^{*})$ ")
                plt.xscale('log')
                plt.xlabel("iterations")
                plt.ylabel("Mean squared Error")
                plt.legend(prop={'size':15})
                plt.title("3D voxel-based GM maps - 257595 features")
                pdf.savefig()
                plt.close(fig)
                pdf.close()
###########################################################################
def plot_stabiility_vs_gap(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
###PLot correlationbetween beta stability and precision
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)
    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)

            if nb_conesta != 0:
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


                pdf_path = os.path.join(PATH,p,"corr_vs_gap.pdf")
                pdf = PdfPages(pdf_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.plot(corr,(gap),label = r"$gap$")
                plt.plot(corr,(func - func[-1]),label = r"$f(\beta^{k})$ - $f(\beta^{*})$")
                plt.axvline(x=0.99)
                plt.yscale('log')
                plt.xlabel(r"$corr(\beta^{k}$ - $\beta^{*})$ ")
                plt.ylabel(r"precision")
                plt.legend(prop={'size':15})
                plt.title("3D voxel-based GM maps - 257595 features")
                fig.tight_layout()
                pdf.savefig()
                plt.close(fig)
                pdf.close()

def plot_mse_vs_gap(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
###PLot correlationbetween beta stability and precision
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)
    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
            conesta_ite = sorted(os.listdir(snap_path))
            nb_conesta = len(conesta_ite)

            if nb_conesta != 0:
                ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
                beta_star =  ite_final["beta"]


                mse = np.zeros((nb_conesta))

                i=0
                for ite in conesta_ite:
                    path = os.path.join(snap_path,ite)
                    ite = np.load(path)
                    mse[i] = sklearn.metrics.mean_squared_error(ite["beta"][:,0],beta_star[:,0])

                    i = i + 1
                gap = np.zeros((nb_conesta))
                func = np.zeros((nb_conesta))
                for i in range(len(conesta_ite)):
                     fista_number = ite['continuation_ite_nb'][i]
                     gap[i] = ite["gap"][fista_number -1]
                     func[i] = ite["func_val"][fista_number -1]


                pdf_path = os.path.join(PATH,p,"mse_vs_gap.pdf")
                pdf = PdfPages(pdf_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                plt.plot(mse,(gap),label = r"$gap$")
                plt.plot(mse,(func - func[-1]),label = r"$f(\beta^{k})$ - $f(\beta^{*})$")
                #plt.axvline(x=0.99)
                plt.yscale('log')
                plt.xlabel(r"$mse(\beta^{k}$ - $\beta^{*})$ ")
                plt.ylabel(r"precision")
                plt.legend(prop={'size':15})
                plt.title("3D voxel-based GM maps - 257595 features")
                fig.tight_layout()
                pdf.savefig()
                plt.close(fig)
                pdf.close()
###########################################################################
###PLot correlationbetween beta stability and precision
def plot_stabiility_vs_gap_all_params(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
###PLot correlationbetween beta stability and precision
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)
    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            print ("CV number : %s" %cv)
            pdf_path = os.path.join(PATH,"corr_vs_gap.pdf")
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
                snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
                conesta_ite = sorted(os.listdir(snap_path))
                nb_conesta = len(conesta_ite)
                if nb_conesta != 0:
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
            plt.axvline(x=0.99)
            plt.yscale('log')
            plt.legend(prop={'size':15})
            plt.title("3D voxel-based GM maps - 257595 features")
            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)
            pdf.close()
###########################################################################


def plot_mse_vs_gap_all_params(BASE_PATH,INPUT_DATA_X,INPUT_DATA_y):
###PLot correlationbetween beta stability and precision
    X = np.load(INPUT_DATA_X)
    y = np.load(INPUT_DATA_y)
    cv = glob.glob(BASE_PATH)
    for i in range(len(cv)):
        print ("CV : %s" %cv[i])
        PATH = cv[i]
        params = glob.glob(os.path.join(PATH,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            print ("CV number : %s" %cv)
            pdf_path = os.path.join(PATH,"mse_vs_gap.pdf")
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
                snap_path = os.path.join(PATH,p,"conesta_ite_snapshots")
                conesta_ite = sorted(os.listdir(snap_path))
                nb_conesta = len(conesta_ite)
                if nb_conesta != 0:
                    ite_final = np.load(os.path.join(snap_path,conesta_ite[-1]))
                    beta_star =  ite_final["beta"]
                    mse = np.zeros((nb_conesta))
                    i=0
                    for ite in conesta_ite:
                        path = os.path.join(snap_path,ite)
                        ite = np.load(path)
                        mse[i] = sklearn.metrics.mean_squared_error(ite["beta"][:,0],beta_star[:,0])

                        i = i + 1
                    gap = np.zeros((nb_conesta))
                    for i in range(len(conesta_ite)):
                         fista_number = ite['continuation_ite_nb'][i]
                         gap[i] = ite["gap"][fista_number -1]


                    plt.plot(mse,np.log10(gap),color = colors[(p)],label = p)

            plt.xlabel(r"$mse(\beta^{k}$ - $\beta^{*})$ ")
            plt.ylabel(r"precision")
            plt.yscale('log')
            plt.legend(prop={'size':15})
            plt.title("3D voxel-based GM maps - 257595 features")
            fig.tight_layout()
            pdf.savefig()
            plt.close(fig)
            pdf.close()

# Plot mse vs gap for two runs : one with random start, one with warm restart
#Need to computed on the same dataset
###############################################################################
BASE_PATH_WARM = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/warm_restart/all_all_as_start_vector/model_selectionCV"
BASE_PATH_RANDOM = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_warm_restart/model_selectionCV/"


BASE_PATH_WARM = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/warm_restart/cv04_all_as_start_vector/model_selectionCV/"
BASE_PATH_RANDOM = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/"


def plot_mse_beta_compare_start_vectors(BASE_PATH_RANDOM,BASE_PATH_WARM,INPUT_DATA_X,INPUT_DATA_y):
    cv = os.listdir(BASE_PATH_WARM)
    for cv_index in range(len(cv)):
        print ("CV : %s" %cv[cv_index])
        PATH_WARM = os.path.join(BASE_PATH_WARM,cv[cv_index],"all")
        PATH_RANDOM = os.path.join(BASE_PATH_RANDOM,cv[cv_index],"all")
        params = glob.glob(os.path.join(PATH_WARM,"0*"))
        for p in params:
            p =  os.path.basename(p)
            print (p)
            snap_path_warm = os.path.join(PATH_WARM,p,"conesta_ite_snapshots")
            conesta_ite_warm = sorted(os.listdir(snap_path_warm))
            nb_conesta_warm = len(conesta_ite_warm)

            snap_path_random = os.path.join(PATH_RANDOM,p,"conesta_ite_snapshots")
            conesta_ite_random = sorted(os.listdir(snap_path_random))
            nb_conesta_random = len(conesta_ite_random)


            ite_final_warm = np.load(os.path.join(snap_path_warm,conesta_ite_warm[-1]))
            beta_star_warm =  ite_final_warm["beta"]
            gap_warm = ite_final_warm["gap"]
            mse_warm = np.zeros((nb_conesta_warm))
            i=0
            for ite in conesta_ite_warm:
                path = os.path.join(snap_path_warm,ite)
                ite_warm = np.load(path)
                mse_warm[i] = sklearn.metrics.mean_squared_error(ite_warm["beta"][:,0],beta_star_warm[:,0])
                i = i +1

            ite_final_random = np.load(os.path.join(snap_path_random,conesta_ite_random[-1]))
            beta_star_random =  ite_final_random["beta"]
            gap_random = ite_final_random["gap"]
            mse_random = np.zeros((nb_conesta_random))

            i=0
            for ite in conesta_ite_random:
                path = os.path.join(snap_path_random,ite)
                ite_random = np.load(path)
                mse_random[i] = sklearn.metrics.mean_squared_error(ite_random["beta"][:,0],beta_star_random[:,0])
                i = i +1


            pdf_path = os.path.join(BASE_PATH_WARM,cv[cv_index],"all",p,"start_vector_effect_on_gap.pdf")
            pdf = PdfPages(pdf_path)
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.plot(gap_random,label = "Random start")
            plt.plot(gap_warm,label = "Warm start: Beta from all/all")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("iterations")
            plt.ylabel(r"$gap$")
            plt.legend(prop={'size':15})
            plt.title(p)
            pdf.savefig()
            plt.close(fig)
            pdf.close()



# Plot precision with different starting points
###############################################################################
BASE_PATH_WARM = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/warm_restart/"
BASE_PATH_RANDOM = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV"

def compare_start_vectors(BASE_PATH_RANDOM,BASE_PATH_WARM,INPUT_DATA_X,INPUT_DATA_y):
        for cv in range(5):
            params = glob.glob(os.path.join(BASE_PATH_RANDOM,"all/all/*"))
            for p in params:
                p =  os.path.basename(p)
                print (p)
                pdf_path = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/influence_of_start_vector/cv0%s"%cv
                os.makedirs(pdf_path, exist_ok=True)
                pdf = PdfPages(os.path.join(pdf_path,"%s.pdf" %p))
                fig = plt.figure(figsize=(11.69, 8.27))


                snap_path_random = os.path.join(BASE_PATH_RANDOM,"cv0%s/all"%cv,p,"conesta_ite_snapshots/")
                conesta_ite_random = sorted(os.listdir(snap_path_random))
                nb_conesta_random = len(conesta_ite_random)
                ite_final_random = np.load(os.path.join(snap_path_random,conesta_ite_random[-1]))
                beta_star_random =  ite_final_random["beta"]
                gap_random = ite_final_random["gap"]
                mse_random = np.zeros((nb_conesta_random))

                i=0
                for ite in conesta_ite_random:
                    path = os.path.join(snap_path_random,ite)
                    ite_random = np.load(path)
                    mse_random[i] = sklearn.metrics.mean_squared_error(ite_random["beta"][:,0],beta_star_random[:,0])
                    i = i +1

                plt.plot(gap_random,label = "Random start")

                for warm_num in range(5):
                    PATH_WARM = BASE_PATH_WARM + "cv0%s_all_as_start_vector/model_selectionCV/cv0%s/all" %(warm_num,cv)
                    snap_path_warm = os.path.join(PATH_WARM,p,"conesta_ite_snapshots")
                    conesta_ite_warm = sorted(os.listdir(snap_path_warm))
                    nb_conesta_warm = len(conesta_ite_warm)


                    ite_final_warm = np.load(os.path.join(snap_path_warm,conesta_ite_warm[-1]))
                    beta_star_warm =  ite_final_warm["beta"]
                    gap_warm = ite_final_warm["gap"]
                    mse_warm = np.zeros((nb_conesta_warm))
                    i=0
                    for ite in conesta_ite_warm:
                        path = os.path.join(snap_path_warm,ite)
                        ite_warm = np.load(path)
                        mse_warm[i] = sklearn.metrics.mean_squared_error(ite_warm["beta"][:,0],beta_star_warm[:,0])
                        i = i +1

                    plt.plot(gap_warm,label = "Warm start: Beta from cv0%s/all" %warm_num)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel("iterations")
                    plt.ylabel(r"$gap$")
                    plt.legend(prop={'size':15})
                    plt.title(p)
                    plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                pdf.close()









