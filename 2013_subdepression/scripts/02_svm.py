# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

This script uses SVM for classification:
 1) fit various SVM without feature selection and inspect the best parameters

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

import epac, epac.map_reduce.engine

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_subdepression", "lib"))
import data_api, utils

TEST_MODE = False

if TEST_MODE:
    DB_PATH='/home/md238665/DB/micro_subdep/'
    LOCAL_PATH='/volatile/micro_subdepression.hdf5'
else:
    DB_PATH='/neurospin/brainomics/2012_imagen_subdepression'
    LOCAL_PATH='/volatile/imagen_subdepression.hdf5'

#########################
# Oth step: access data #
#########################

csv_file_name = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(csv_file_name)

babel_mask = nibabel.load(data_api.get_mask_file_path(DB_PATH))
mask = babel_mask.get_data()
binary_mask = mask != 0

h5file = tables.openFile(LOCAL_PATH)
masked_images = data_api.get_images(h5file)

###########################################
# 1st step: SVM without feature selection #
#           We cross-validate the results #
###########################################

if TEST_MODE:
    C_VALUES = [0.1, 1, 10]
else:
    C_VALUES = [0.1, 0.5, 1, 5, 10]
REGULARIZATION_METHODS = ['l1', 'l2']

if TEST_MODE:
    N_FOLDS_NESTED  = 3
    N_FOLDS_EVAL    = 3
else:
  N_FOLDS_NESTED  = 5
  N_FOLDS_EVAL    = 10  


OUT_DIR='results/svm/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
OUT_IMAGE_FORMAT = 'C={C}_penalty={penalty}.CV.nii'

# Create all the classifiers
svms = epac.Pipe(sklearn.preprocessing.StandardScaler(),
                 epac.Methods(*[sklearn.svm.LinearSVC(class_weight='auto',
                           C=C, penalty=penalty,
                           dual=False)
                           for C in C_VALUES
                           for penalty in REGULARIZATION_METHODS]))
# Select the best with CV
svms_auto = epac.CVBestSearchRefit(svms, n_folds=N_FOLDS_NESTED)

# Evaluate it
svms_auto_cv = epac.CV(svms_auto, n_folds=N_FOLDS_EVAL)

# Run model selection
X = numpy.asarray(masked_images)
y = numpy.asarray(utils.numerical_coding(df, variables=['group_sub_ctl']).group_sub_ctl)

engine = epac.map_reduce.engine.LocalEngine(svms_auto_cv,
                                            num_processes=5)
svms_auto_cv = engine.run(X=X, y=y)
svms_auto_cv_results = svms_auto_cv.reduce()
# Voir résultats (reduce) et les paramètres séléctionnés

# Re-fit the best classifier on the whole datatset. Warning: biased !!!
svms_auto.run(X=X, y=y)
thetas = svms_auto.best_params
print "Best SVM parameters:", thetas[1]
betas = svms_auto.refited.children[0].estimator.coef_
print betas

# Store in an image
betas_img = numpy.zeros(binary_mask.shape)
betas_img[binary_mask] = betas[0, :]
outimg = nibabel.Nifti1Image(betas_img, babel_mask.get_affine())
nibabel.save(outimg, os.path.join(OUT_DIR, OUT_IMAGE_FORMAT.format(**thetas[1])))

h5file.close()
