# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:02:18 2013

@author: md238665
"""

# Standard library modules
import os, sys, argparse
# Numpy and friends
import numpy
import tables
# Nipy
import nibabel

from mulm import LinearRegression

# Local import
try:
    # When executed as a script
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
except NameError:
    # When executed from spyder
    sys.path.append(os.path.join(os.environ["HOME"] , "Code", "scripts", "2013_subdepression", "lib"))
import data_api, utils

parser = argparse.ArgumentParser(description='''Residualize data.''')

parser.add_argument('--test_mode',
      action='store_true',
      help='Use test mode')

args = parser.parse_args()

TEST_MODE = args.test_mode

if TEST_MODE:
    DB_PATH='/volatile/DB/micro_subdepression/'
    LOCAL_PATH='/volatile/DB/cache/micro_subdepression.hdf5'
else:
    DB_PATH='/neurospin/brainomics/2012_imagen_subdepression'
    LOCAL_PATH='/volatile/DB/cache/imagen_subdepression.hdf5'


###############
# access data #
###############

csv_file_name = data_api.get_clinic_file_path(DB_PATH)
df = data_api.read_clinic_file(csv_file_name)

babel_mask = nibabel.load(data_api.get_mask_file_path(DB_PATH))
mask = babel_mask.get_data()
binary_mask = mask != 0

h5file = tables.openFile(LOCAL_PATH, 'r+')
images = data_api.get_images(h5file)

#####################
# Fit GLM & project #
#####################

# 1st model

MODEL = ["Gender", "Age", "VSF", "Scanner_Type"]

design_mat = utils.make_design_matrix(df, regressors=MODEL).as_matrix()

isnan = numpy.isnan(design_mat)
if isnan.any():
    bad_subject_ind = numpy.where(isnan)[0]
    print "Removing subject", bad_subject_ind
    design_mat = numpy.delete(design_mat, bad_subject_ind, axis=0)
    images = numpy.delete(images, bad_subject_ind, axis=0)

# Fit LM & compute residuals
lm = LinearRegression()
lm.fit(X=design_mat, Y=images)
images_pred = lm.predict(X=design_mat)
res = images - images_pred

# Write to file
residual_name = 'masked_images_' + '_'.join(MODEL)
data_api.write_images(h5file, res, residual_name)

# 2nd model

MODEL = ["Gender", "Age", "VSF", "ImagingCentreCity"]

design_mat = utils.make_design_matrix(df, regressors=MODEL).as_matrix()

isnan = numpy.isnan(design_mat)
if isnan.any():
    bad_subject_ind = numpy.where(isnan)[0]
    print "Removing subject", bad_subject_ind
    design_mat = numpy.delete(design_mat, bad_subject_ind, axis=0)
    images = numpy.delete(images, bad_subject_ind, axis=0)

# Fit LM & compute residuals
lm = LinearRegression()
lm.fit(X=design_mat, Y=images)
images_pred = lm.predict(X=design_mat)
res = images - images_pred

# Write to file
residual_name = 'masked_images_' + '_'.join(MODEL)
data_api.write_images(h5file, res, residual_name)

h5file.close()