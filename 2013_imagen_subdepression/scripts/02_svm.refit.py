# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:29:11 2013

@author: md238665

Create the same pipeline than in 02_svm.create_wf.py and fit it on the whole dataset.
Warning: this is biased and used only for illustration.

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
DEFAULT_IMAGES_NAME = "masked_images"
OUT_IMAGE_FORMAT = 'C={C}_penalty={penalty}.CV.nii'

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_imagen_subdepression", "lib"))
import data_api, utils

parser = argparse.ArgumentParser(description='''Create a simple workflow for SVM classification and evaluate it.''')

parser.add_argument('--out_dir', required=True,
      type=str,
      help='Output directory')

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
FULL_OUT_DIR= os.path.join(OUT_DIR, args.out_dir)
if not os.path.exists(FULL_OUT_DIR):
    os.makedirs(FULL_OUT_DIR)

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

##########
# Re-fit #
##########

print "Refitting on the whole data set"
# Create all the classifiers
svms = epac.Pipe(sklearn.preprocessing.StandardScaler(),
                 epac.Methods(*[sklearn.svm.LinearSVC(class_weight='auto',
                           C=C, penalty=penalty,
                           dual=False)
                           for C in C_VALUES
                           for penalty in REGULARIZATION_METHODS]))
# Select the best with CV
svms_auto = epac.CVBestSearchRefit(svms, n_folds=N_FOLDS_NESTED)

svms_auto.run(X=X, y=y)
thetas = svms_auto.best_params
print "Best SVM parameters:", thetas[1]
betas = svms_auto.refited.children[0].wrapped_node.coef_
print betas

# Store in an image
betas_img = numpy.zeros(binary_mask.shape)
betas_img[binary_mask] = betas[0, :]
outimg = nibabel.Nifti1Image(betas_img, babel_mask.get_affine())
nibabel.save(outimg, os.path.join(FULL_OUT_DIR, OUT_IMAGE_FORMAT.format(**thetas[1])))