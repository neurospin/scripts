# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:12:03 2016

@author: ad247405
"""
import os, sys
import json
import time
import numpy as np
import pandas as pd
import nibabel
import argparse
import parsimony.utils.consts as consts
import pca_tv

import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.functions.nesterov.l1tv as l1tv

import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils.start_vectors as start_vectors
import brainomics.mesh_processing as mesh_utils

from parsimony.algorithms.utils import AlgorithmSnapshot
#import sklearn.decomposition
from parsimony.utils import plot_map2d
import os, sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import itertools
import scipy.ndimage as ndimage
import glob



INPUT = '/neurospin/brainomics/2014_pca_struct/adni/fs/adni_5folds/results/0'


list = glob.glob(os.path.join(INPUT,'*'))

for l in list:
    frob = np.load(os.path.join(l,"frobenius_test.pkl"))
    components = np.load(os.path.join(l,"components.npz"))['arr_0']
    non_zero_voxels_ratio_comp2 = float( sum(components[:,1] == 0)) / components[:,1].shape[0]
    non_zero_voxels_ratio_comp3 =  float( sum(components[:,2] == 0)) / components[:,2].shape[0]
    print os.path.basename(l)
    print frob
    print non_zero_voxels_ratio_comp2
    print non_zero_voxels_ratio_comp3
