# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 10:31:25 2014

@author: md238665

Check that we can correctly re-project individuals (i.e. components) and
create a CSV file of components aligned with population file (for later use to
correlate with clinic scores).

"""

import os
import glob

import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib

from parsimony.functions.nesterov.tv import A_from_mask

import pca_tv

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/mescog/mescog_5folds"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "results")
INPUT_MESCOG_DIR = "/neurospin/mescog/proj_wmh_patterns"

INPUT_POPULATION_FILE = os.path.join(INPUT_MESCOG_DIR,
                                     "population.csv")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,
                             "X.npy")
INPUT_MASK = os.path.join(INPUT_BASE_DIR,
                          "mask_bin.nii")

OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/mescog/mescog_5folds"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,
                                 "components.csv")

##############
# Parameters #
##############

GLOB_PATTERN = os.path.join(INPUT_DIR,
                            "0",
                            "struct_pca_*")
N_COMP = 3
# We don'really need those parameters here
ll1 = ll2 = ltv = 0

##########
# Script #
##########

# Read population (we just need the ID)
population = pd.io.parsers.read_csv(INPUT_POPULATION_FILE)
subjects_id = population['Subject ID']

# Read data
X = np.load(INPUT_DATASET)
n, p = X.shape
assert(n == len(subjects_id))

# Read mask
babel_mask = nib.load(INPUT_MASK)

# Compute A matrices
Atv, n_compacts = A_from_mask(babel_mask.get_data())
Al1 = sp.sparse.eye(p, p)

# For all struct PCA models in fold 0, check that we can reproduce components
# and store components in a file
total_df = None
input_dir_list = glob.glob(GLOB_PATTERN)
for input_dir in input_dir_list:
    components_file = os.path.join(input_dir, "components.npz")
    components = np.load(components_file)['arr_0']
    # Create model with the same loadings and recompute projections
    model = pca_tv.PCA_SmoothedL1_L2_TV(n_components=N_COMP,
                                        l1=ll1, l2=ll2, ltv=ltv,
                                        Atv=Atv,
                                        Al1=Al1,
                                        criterion="frobenius",
                                        eps=1e-6,
                                        max_iter=100,
                                        inner_max_iter=int(1e4),
                                        output=False)
    model.V = components
    computed_projections, _ = model.transform(X)
    # Projections
    projections_file = os.path.join(input_dir, "X_train_transform.npz")
    projections = np.load(projections_file)['arr_0']
    # Compare both
    assert (np.allclose(projections, computed_projections))
    # Create df for this model
    model_name = os.path.split(input_dir)[-1]
    index = pd.MultiIndex.from_arrays([subjects_id, np.asarray([model_name]*n)],
                                      names=['Subject ID', 'model'])
    df = pd.DataFrame(data=projections,
                      index=index,
                      columns=['PC{i}'.format(i=i+1) for i in range(N_COMP)])
    # Append to large df
    if total_df is None:
        total_df = df
    else:
        total_df = total_df.append(df)
total_df.to_csv(OUTPUT_COMPONENTS)
