#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:05:24 2017

@author: ad247405
"""

#Validation of the models on a totally independent dataset



import os
import numpy as np
import nibabel
import nilearn  
from nilearn import plotting
import matplotlib.pylab as plt
import pandas as pd
import brainomics.image_atlas
import brainomics.array_utils
import subprocess
import json
from scipy.stats.stats import pearsonr 

BASE_PATH="/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST"


svm_weight_map = os.path.join(BASE_PATH,"results/svm/svm_model_selection_5folds_NUDAST/model_selectionCV/all/all/1e-05/beta.npz")

