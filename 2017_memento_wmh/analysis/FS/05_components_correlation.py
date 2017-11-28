#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:46:40 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt

BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all"
MASK_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/data/mask.npy"
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/population.csv"



pop = pd.read_csv(INPUT_CSV)
assert  pop.shape == (2164, 27)


# Standard PCA
COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")
PROJ_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","X_test_transform.npz")

components = np.load(COMP_PATH)["arr_0"]
projections = np.load(PROJ_PATH)["arr_0"]
assert components.shape == (299879, 10)
assert projections.shape == (2164, 10)


#Correlation with age
for i in range(10):
    x = pop["age_cons"][np.isnan(pop["age_cons"])==False]
    y = projections[:,i][np.array(np.isnan(pop["age_cons"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("age_cons score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))




#Correlation with age
for i in range(10):
    x = pop["age_cons"][np.isnan(pop["age_cons"])==False]
    y = projections[:,i][np.array(np.isnan(pop["age_cons"])==False)]
    plt.figure()
    plt.plot(x[pop["mci"]==0],y[np.array(pop["mci"]==0)],'o',label = "Non MCI")
    plt.plot(x[pop["mci"]==1],y[np.array(pop["mci"]==1)],'o',label = "aMCI pur")
    plt.plot(x[pop["mci"]==2],y[np.array(pop["mci"]==2)],'o',label = "aMCI multidomain")
    plt.plot(x[pop["mci"]==3],y[np.array(pop["mci"]==3)],'o',label = "naMCI pur")
    plt.plot(x[pop["mci"]==4],y[np.array(pop["mci"]==4)],'o',label = "naMCI multidomain")
    plt.xlabel("age_cons score")
    plt.ylabel("Score on component %r"%i)
    plt.legend(loc = "bottom left")



for i in range(10):
    x = pop["mmssctot"][np.isnan(pop["mmssctot"])==False]
    y = projections[:,i][np.array(np.isnan(pop["mmssctot"])==False)]
    mci_status = pop["mci"][np.isnan(pop["mmssctot"])==False]
    plt.figure()
    plt.plot(x[mci_status==0],y[np.array(mci_status==0)],'o',label = "Non MCI")
    plt.plot(x[mci_status==1],y[np.array(mci_status==1)],'o',label = "aMCI pur")
    plt.plot(x[mci_status==2],y[np.array(mci_status==2)],'o',label = "aMCI multidomain")
    plt.plot(x[mci_status==3],y[np.array(mci_status==3)],'o',label = "naMCI pur")
    plt.plot(x[mci_status==4],y[np.array(mci_status==4)],'o',label = "naMCI multidomain")
    plt.xlabel("mmssctot score")
    plt.ylabel("Score on component %r"%i)
    plt.legend(loc = "bottom left")



#Correlation with MMSE
for i in range(10):
    x = pop["mmssctot"][np.isnan(pop["mmssctot"])==False]
    y = projections[:,i][np.array(np.isnan(pop["mmssctot"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("MMSE score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))




#Correlation with cdrscr
for i in range(10):
    x = pop["cdrscr"][np.isnan(pop["cdrscr"])==False]
    y = projections[:,i][np.array(np.isnan(pop["cdrscr"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("cdrscr score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))

#Correlation with flu_p
for i in range(10):
    x = pop["flu_p"][np.isnan(pop["flu_p"])==False]
    y = projections[:,i][np.array(np.isnan(pop["flu_p"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("Flup_p score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))

#Correlation with flu_anim
for i in range(10):
    x = pop["flu_anim"][np.isnan(pop["flu_anim"])==False]
    y = projections[:,i][np.array(np.isnan(pop["flu_anim"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("flu_anim score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


#Correlation with risctotim
for i in range(10):
    x = pop["risctotim"][np.isnan(pop["risctotim"])==False]
    y = projections[:,i][np.array(np.isnan(pop["risctotim"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("risctotim score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))

#Correlation with risctotrl
for i in range(10):
    x = pop["risctotrl"][np.isnan(pop["risctotrl"])==False]
    y = projections[:,i][np.array(np.isnan(pop["risctotrl"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("risctotrl score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


#Correlation with EAVMEM
for i in range(10):
    x = pop["eavmem"][np.isnan(pop["eavmem"])==False]
    y = projections[:,i][np.array(np.isnan(pop["eavmem"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("eavmem score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))

#Correlation with eavatt
for i in range(10):
    x = pop["eavatt"][np.isnan(pop["eavatt"])==False]
    y = projections[:,i][np.array(np.isnan(pop["eavatt"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("eavatt score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


#Correlation with eavlang
for i in range(10):
    x = pop["eavlang"][np.isnan(pop["eavlang"])==False]
    y = projections[:,i][np.array(np.isnan(pop["eavlang"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("eeavlang score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))

#Correlation with tmta_taux
for i in range(10):
    x = pop["tmta_taux"][np.isnan(pop["tmta_taux"])==False]
    y = projections[:,i][np.array(np.isnan(pop["tmta_taux"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("tmta_taux score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


#Correlation with tmta_taux
for i in range(10):
    x = pop["tmtb_taux"][np.isnan(pop["tmtb_taux"])==False]
    y = projections[:,i][np.array(np.isnan(pop["tmtb_taux"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("tmtb_taux score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))

#Correlation with apoe_eps4
for i in range(10):
    x = pop["apoe_eps4"][np.isnan(pop["apoe_eps4"])==False]
    y = projections[:,i][np.array(np.isnan(pop["apoe_eps4"])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    plt.figure()
    plt.plot(x,y,'o')
    plt.xlabel("apoe_eps4 score")
    plt.ylabel("Score on component %r"%i)
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))