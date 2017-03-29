#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:46:47 2017

@author: ad247405
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:06:35 2017

@author: ad247405
"""
import os
import numpy as np
import pandas as pd
import nibabel
from mulm import MUOLS
import brainomics
from brainomics import create_texture


BASE_PATH = '/neurospin/brainomics/2016_AUSZ'
INPUT_CSV= os.path.join(BASE_PATH,"results","Freesurfer","population.csv")
MASK_PATH = os.path.join(BASE_PATH,"results","Freesurfer","data","mask.npy")
TEMPLATE_PATH = "/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template"
mask_arr = np.load(MASK_PATH)

#asd vs controls
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","Freesurfer","asd_vs_controls")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0,0], pval=True)
np.savez(os.path.join(OUTPUT,"pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(OUTPUT,"tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals","pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"tvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"tvals","tvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)                                   

create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"log10pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH =MASK_PATH,
                                   beta_path = os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)  


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0,0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])


create_texture.create_texture_file(OUTPUT = os.path.join(OUTPUT,"pvals_corr"),
                                   TEMPLATE_PATH = TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
#########################


#scz-asd vs asd
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","Freesurfer","scz_asd_vs_asd")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0,0], pval=True)
np.savez(os.path.join(OUTPUT,"pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(OUTPUT,"tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals","pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"tvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"tvals","tvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)                                   

create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"log10pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH =MASK_PATH,
                                   beta_path = os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)  


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0,0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])


create_texture.create_texture_file(OUTPUT = os.path.join(OUTPUT,"pvals_corr"),
                                   TEMPLATE_PATH = TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),
                                   penalty_start = 3,
                                   threshold = False)
#################################################################################


#scz_asd_vs_controls
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","Freesurfer","scz_asd_vs_controls")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0,0], pval=True)
np.savez(os.path.join(OUTPUT,"pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(OUTPUT,"tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals","pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"tvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"tvals","tvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)                                   

create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"log10pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH =MASK_PATH,
                                   beta_path = os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)  


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0,0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])


create_texture.create_texture_file(OUTPUT = os.path.join(OUTPUT,"pvals_corr"),
                                   TEMPLATE_PATH = TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),
                                   penalty_start = 3,
                                   threshold = False)
#################################################################################



#scz_vs_asd
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","Freesurfer","scz_vs_asd")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0,0], pval=True)
np.savez(os.path.join(OUTPUT,"pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(OUTPUT,"tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals","pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"tvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"tvals","tvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)                                   

create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"log10pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH =MASK_PATH,
                                   beta_path = os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)  


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0,0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])


create_texture.create_texture_file(OUTPUT = os.path.join(OUTPUT,"pvals_corr"),
                                   TEMPLATE_PATH = TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),
                                   penalty_start = 3,
                                   threshold = False)#################################################################################




#scz_vs_controls
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","Freesurfer","scz_vs_controls")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0,0], pval=True)
np.savez(os.path.join(OUTPUT,"pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(OUTPUT,"tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals","pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"tvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"tvals","tvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)                                   

create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"log10pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH =MASK_PATH,
                                   beta_path = os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)  


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0,0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])


create_texture.create_texture_file(OUTPUT = os.path.join(OUTPUT,"pvals_corr"),
                                   TEMPLATE_PATH = TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),
                                   penalty_start = 3,
                                   threshold = False)
#################################################################################


#scz_vs_scz-asd
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","Freesurfer","scz_vs_scz-asd")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0,0], pval=True)
np.savez(os.path.join(OUTPUT,"pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(OUTPUT,"tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals","pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)
                                   
create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"tvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"tvals","tvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)                                   

create_texture.create_texture_file(OUTPUT=os.path.join(OUTPUT,"log10pvals"),
                                   TEMPLATE_PATH =TEMPLATE_PATH,
                                   MASK_PATH =MASK_PATH,
                                   beta_path = os.path.join(OUTPUT,"log10pvals","log10pvals.npz"),
                                   penalty_start = 3,
                                   threshold = False)  


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0,0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])


create_texture.create_texture_file(OUTPUT = os.path.join(OUTPUT,"pvals_corr"),
                                   TEMPLATE_PATH = TEMPLATE_PATH,
                                   MASK_PATH = MASK_PATH ,
                                   beta_path = os.path.join(OUTPUT,"pvals_corr","pvals_corrected_perm.npz"),
                                   penalty_start = 3,
                                   threshold = False)
#################################################################################
