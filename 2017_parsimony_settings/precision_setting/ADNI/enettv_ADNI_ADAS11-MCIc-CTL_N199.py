#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:13:00 2017

@author: ad247405
"""

###############################################################################
# import

import os
import numpy as np
import parsimony.functions.nesterov.tv as tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import nibabel
from statsmodels.stats.inter_rater import fleiss_kappa
from parsimony.algorithms.utils import Info
from parsimony.algorithms.utils import AlgorithmSnapshot
import parsimony.utils.weights as weights

WD = "/neurospin/brainomics/2017_parsimony_settings/precision_setting/ADNI_ADAS11-MCIc-CTL"
os.chdir(WD)
mask_filename = "mask.nii.gz"
dataset_filename = "ADNI_ADAS11-MCIc-CTL_N199.npz"

###############################################################################
# load dataset set some parameters

arxiv = np.load("ADNI_ADAS11-MCIc-CTL_N199.npz")
X = arxiv["X"]
y = arxiv["y"]
beta_start = arxiv["beta_start"]
assert X.shape == (199, 286214)
TAU = 0.2
EPS = 1e-6  # PRECISION FOR THE PAPER

###############################################################################
# Fit model

ALPHA = 0.01 #
l, k, g = ALPHA * np.array([0.3335, 0.3335, 0.333])

mask_ima = nibabel.load(os.path.join(WD,  mask_filename))
Atv = tv.linear_operator_from_mask(mask_ima.get_data())

out = os.path.join(WD,"run","conesta_ite_snapshots/")
snapshot = AlgorithmSnapshot(out, saving_period=1).save_conesta


info = [Info.converged,Info.num_iter,Info.time,Info.func_val,Info.mu,Info.gap,Info.converged,Info.fvalue]
conesta = algorithms.proximal.CONESTA(callback_conesta = snapshot)
algorithm_params = dict(max_iter=1000000, info=info)
os.makedirs(out, exist_ok=True)

algorithm_params["callback"] = snapshot


mod = estimators.LinearRegressionL1L2TV(
                 l, k, g, A=Atv, algorithm=conesta,algorithm_params=algorithm_params,penalty_start=0, mean=True)

mod.fit(X, y, beta_start)

