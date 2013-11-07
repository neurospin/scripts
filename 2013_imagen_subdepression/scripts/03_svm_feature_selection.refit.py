# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:41:36 2013

@author: md238665

Refit the best pipeline on the whole dataset (used to plot weight maps).

"""

# Standard library modules
import os, sys
# Numpy and friends
import numpy
import sklearn, sklearn.svm, sklearn.feature_selection
# For reading HDF5 files
import tables
# Nipy
import nibabel

import epac

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_imagen_subdepression", "lib"))
import data_api, utils

TEST_MODE = False

if TEST_MODE:
    DB_PATH='/volatile/DB/micro_subdepression/'
    LOCAL_PATH='/volatile/DB/cache/micro_subdepression.hdf5'
else:
    DB_PATH='/neurospin/brainomics/2013_imagen_subdepression'
    LOCAL_PATH='/volatile/DB/cache/imagen_subdepression.hdf5'

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm_feature_selection/')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

##############
# Parameters #
##############
IMAGES = "masked_images_Gender_Age_VSF_Scanner_Type"
K = 256
REGULARIZATION = 'l1'
C = 0.5

#############
# Load data #
#############

babel_mask = nibabel.load(data_api.get_mask_file_path(DB_PATH))
mask = babel_mask.get_data()
binary_mask = mask != 0

csv_file_name = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(csv_file_name)
# Class 0 is ctl, 1 is subdep
y = numpy.asarray(utils.numerical_coding(df, variables=['group_sub_ctl']).group_sub_ctl)

h5file = tables.openFile(LOCAL_PATH)
X = numpy.asarray(data_api.get_images(h5file, IMAGES))

##################
# Run classifier #
##################

wf = epac.Pipe(sklearn.feature_selection.SelectKBest(k=K),
               sklearn.preprocessing.StandardScaler(),
               sklearn.svm.LinearSVC(class_weight='auto',
                               C=C, penalty=REGULARIZATION,
                               dual=False))

wf.run(X=X, y=y)
results = wf.reduce()

##############
# Make a map #
##############

selected_features = wf.wrapped_node.get_support()
svm = wf.children[0].children[0].wrapped_node
selected_betas = svm.coef_
betas = numpy.zeros(selected_features.shape)
betas[selected_features] = selected_betas
img = utils.make_image_from_array(betas, babel_mask)
nibabel.save(img, os.path.join(OUT_DIR, 'betas.nii'))

h5file.close()