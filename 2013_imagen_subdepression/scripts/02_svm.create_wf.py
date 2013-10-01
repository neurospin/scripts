# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

This script creates a workflow for classification with SVM.

"""

# Standard library modules
import os, sys, argparse
# Numpy and friends
import numpy
import sklearn, sklearn.svm, sklearn.feature_selection
# For reading HDF5 files
import tables
# Nipy
import nibabel

import epac, epac.map_reduce.engine

TEST_MODE = False
DEFAULT_WF_NAME    = "svm_wf"
DEFAULT_IMAGES_NAME = "masked_images"

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_subdepression", "lib"))
import data_api, utils

parser = argparse.ArgumentParser(description='''Create a workflow for SVM classification.''')

parser.add_argument('--wf_name',
      type=str, default=DEFAULT_WF_NAME,
      help='Name of the workflow (default: %s)'% (DEFAULT_WF_NAME))

parser.add_argument('--images_name',
      type=str, default=DEFAULT_IMAGES_NAME,
      help='Name of the image dataset (default: %s)'% (DEFAULT_IMAGES_NAME))

args = parser.parse_args()

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
    N_FOLDS_NESTED  = 3
    N_FOLDS_EVAL    = 3
else:
  N_FOLDS_NESTED  = 5
  N_FOLDS_EVAL    = 10

OUT_DIR=os.path.join(DB_PATH, 'results', 'svm')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
WORKFLOW_PATH= os.path.join(OUT_DIR, args.wf_name)
OUT_IMAGE_FORMAT = 'C={C}_penalty={penalty}.CV.nii'

###############
# access data #
###############

csv_file_name = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(csv_file_name)

babel_mask = nibabel.load(data_api.get_mask_file_path(DB_PATH))
mask = babel_mask.get_data()
binary_mask = mask != 0

h5file = tables.openFile(LOCAL_PATH)
masked_images = data_api.get_images(h5file, name=args.images_name)

X = numpy.asarray(masked_images)
y = numpy.asarray(utils.numerical_coding(df, variables=['group_sub_ctl']).group_sub_ctl)

########################################
# Create workflow:                     #
#       SVM without feature selection  #
#       & cross-validation             #
########################################

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
engine = epac.map_reduce.engine.SomaWorkflowEngine(svms_auto_cv,
                                                   resource_id="md238665@gabriel",
                                                   login="md238665",
                                                   num_processes=10,
                                                   remove_finished_wf=False)
print "Creating workflow at {path}".format(path=WORKFLOW_PATH)
engine.export_to_gui(WORKFLOW_PATH, X=X, y=y)
print "Execute the workflow at {path} with SOMA Workflow GUI and re-run this script in analysis mode".format(path=WORKFLOW_PATH)

h5file.close()
