# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:53:51 2014

@author: hl237680
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.environ["HOME"], "gits", "scripts", "2013_imagen_subdepression", "lib"))
import utils


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')

# For more details, see: /neurospin/brainomics/2013_imagen_subdepression/lib: utils.py and data_api.py
# List the values of the categorical variables and the associated numeric values.
# The first value is the reference level in dummy coding
# Using OrderedDict allow deterministic order which is important for dummy coding.

## Values are arbitrary
#GenderMap = collections.OrderedDict((('Male',0), ('Female',1)))
#
#CityMap = collections.OrderedDict((
#                ('LONDON',1),
#                ('NOTTINGHAM',2),
#                ('DUBLIN',3),
#                ('BERLIN',4),
#                ('HAMBURG',5),
#                ('MANNHEIM',6),
#                ('PARIS',7),
#                ('DRESDEN',8)))
#                
## Categorical variables mappings
#REGRESSOR_MAPPINGS = {
#"Gender de Feuil2": GenderMap,
#"ImagingCentreCity": CityMap
#}


# Dataframe
COFOUND = ["Subject", "Gender de Feuil2", "ImagingCentreCity", "tiv_gaser", "mean_pds"]
df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH, "1534bmi-vincent2.csv"), index_col=0)
df = df[COFOUND]

# Conversion dummy coding
design_mat = utils.make_design_matrix(df, regressors=COFOUND).as_matrix()

# Keep only subjects for which we have all data
subjects_id = np.genfromtxt(os.path.join(DATA_PATH, "subjects_id.csv"), dtype=None, delimiter=',', skip_header=1)

design_mat = np.delete(design_mat, np.where(np.in1d(design_mat[:,0], np.delete(design_mat, np.where(np.in1d(design_mat[:,0], subjects_id)), 0))), 0)

