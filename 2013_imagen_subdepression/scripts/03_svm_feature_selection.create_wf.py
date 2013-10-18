# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

This script creates a workflow for classification with SVM and feature selection.
We create a large number of SVM and put them in a CV loop.
There is also a parameter selection loop (also in CV).

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
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_imagen_subdepression", "lib"))
import data_api, utils

TEST_MODE = False
MAX_N_FEATURES = 336188 # All features
IMAGES = ["masked_images", "masked_images_Gender_Age_VSF_ImagingCentreCity",
          "masked_images_Gender_Age_VSF_Scanner_Type"]

if TEST_MODE:
    DB_PATH='/volatile/DB/micro_subdepression/'
    LOCAL_PATH='/volatile/DB/cache/micro_subdepression.hdf5'
else:
    DB_PATH='/neurospin/brainomics/2013_imagen_subdepression'
    LOCAL_PATH='/volatile/DB/cache/imagen_subdepression.hdf5'

if TEST_MODE:
    C_VALUES = [0.1, 1, 10]
else:
    C_VALUES = [0.1, 0.5, 1, 5, 10]

REGULARIZATION_METHODS = ['l1', 'l2']

if TEST_MODE:
    K_VALUES = [32, MAX_N_FEATURES]
else:
    K_VALUES = utils.range_log2(32, MAX_N_FEATURES)

if TEST_MODE:
    N_FOLDS_NESTED  = 3
    N_FOLDS_EVAL    = 3
else:
    N_FOLDS_NESTED  = 5
    N_FOLDS_EVAL    = 10

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm_feature_selection/')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
WF_NAME_PATTERN="svm_feature_selection_{images}"

#########################
# Oth step: access data #
#########################

csv_file_name = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(csv_file_name)

babel_mask = nibabel.load(data_api.get_mask_file_path(DB_PATH))
mask = babel_mask.get_data()
binary_mask = mask != 0

h5file = tables.openFile(LOCAL_PATH)

####################
# Create workflows #
####################

# Base workflow: SVM + feature selection
svms = pipelines = epac.Methods(*[
      epac.Pipe(sklearn.feature_selection.SelectKBest(k=k),
                sklearn.preprocessing.StandardScaler(),
                epac.Methods(*[sklearn.svm.LinearSVC(class_weight='auto',
                               C=C, penalty=penalty,
                               dual=False)
                               for C in C_VALUES
                               for penalty in REGULARIZATION_METHODS]))
                               for k in K_VALUES])

# Feature selection workflow
svms_auto_cv = epac.CVBestSearchRefit(svms, n_folds=N_FOLDS_NESTED)

cv = epac.CV(epac.Methods(svms, svms_auto_cv), n_folds=N_FOLDS_EVAL)

#################################
# Store workflow per image type #
#################################

y = numpy.asarray(utils.numerical_coding(df, variables=['group_sub_ctl']).group_sub_ctl)
for images_name in IMAGES:
    images = data_api.get_images(h5file, images_name)
    X = numpy.asarray(images)

    wf_name = WF_NAME_PATTERN.format(images=images_name)
    path = os.path.join(OUT_DIR, wf_name)
    print "Creating workflow at {path}".format(path=path)

    engine = epac.map_reduce.engine.SomaWorkflowEngine(cv,
                                                       resource_id="md238665@gabriel",
                                                       login="md238665",
                                                       num_processes=10,
                                                       remove_finished_wf=False)
    engine.export_to_gui(path, X=X, y=y)
#    print "Running for {images} images".format(images=images_name)
#    cv.run(X=X, y=y)

h5file.close()
