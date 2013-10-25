# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:19:17 2013

@author: md238665

This script reload a workflow (SVM+feature selection), reduce it and write results in a CSV file.

"""

# Standard library modules
import os, sys

import epac, epac.map_reduce.engine

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_imagen_subdepression", "lib"))
import data_api, utils

TEST_MODE     = False
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

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm_feature_selection/')
WF_NAME_PATTERN="svm_feature_selection_{images}"

####################
# Reduce workflows #
####################
for images_name in IMAGES:
    wf_name = WF_NAME_PATTERN.format(images=images_name)
    wf_path = os.path.join(OUT_DIR, wf_name)
    print "Processing", wf_path
    if not os.path.exists(wf_path):
        raise Exception('{path} not found'.format(path=wf_path))

    cv = epac.map_reduce.engine.SomaWorkflowEngine.load_from_gui(wf_path)
    print "Workflow loaded"
    cv_results = cv.reduce()
    print "Reduce done"
    # Write CV results in a CSV file
    epac.export_csv(cv, cv_results, os.path.join(wf_path, "cv_results.csv"))
