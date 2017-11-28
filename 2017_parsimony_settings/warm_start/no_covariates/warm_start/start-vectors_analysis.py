#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:25:11 2017

@author: ad247405
"""


beta_start = np.load("/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv00/all/0.1_0.08_0.72_0.2/beta.npz")

ite_final = np.load("/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv00/all/0.1_0.08_0.72_0.2/\
conesta_ite_snapshots/conesta_ite_000029.npz")
beta_final = ite_final['beta']

ite_conesta = np.load("/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/warm_restart/cv00_all_as_start_vector/model_selectionCV/cv02/all/0.1_0.08_0.72_0.2/conesta_ite_snapshots/conesta_ite_00001.npz")
beta_start_conesta =  ite_conesta["beta"]

ite_fista = np.load("/neurospin/brainomics/2017_parsimony_settings/warm_restart/\
NUSDAST_30yo/VBM/no_covariates/warm_restart/FISTA_snapshots/cv00_all_as_start_vector/model_selectionCV/cv02/all/0.1_0.08_0.72_0.2/fista_ite_snapshots/fista_ite_00001.npz")
beta_start_fista =  ite_fista["beta"]


beta_start['arr_0'] ==  beta_final

beta_start['arr_0'] ==  beta_start_conesta

beta_start['arr_0'] ==  beta_start_fista

beta_start_conesta ==  beta_start_fista


start = np.load("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/\
VBM/no_covariates/warm_restart/cv00_all_as_start_vector/model_selectionCV/cv02/all/0.1_0.08_0.72_0.2/beta_start.npz")["arr_0"]
start == beta_final


ite = np.load("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/\
VBM/no_covariates/warm_restart/FISTA_snapshots/cv00_all_as_start_vector/\
model_selectionCV/cv03/all/0.1_0.08_0.72_0.2/fista_ite_snapshots/fista_ite_00001.npz")
beta_new = ite["beta_new"]
beta_old = ite["beta_old"]

beta_start['arr_0'] == beta_old
beta_start['arr_0'] == beta_new