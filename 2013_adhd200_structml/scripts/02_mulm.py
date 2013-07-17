# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:59:39 2013

@author: ed203246


edit "01_fill-missing.py" modify "OUTPUT_PATH"
Then set ADHD200_DATA_BASE_PATH to previously OUTPUT_PATH
"""

import numpy as np
import mulm
import nibabel as nib

# Scripts PATH
try:
    ADHD200_SCRIPTS_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
  ADHD200_SCRIPTS_BASE_PATH = os.path.join(os.environ["HOME"] , "git", "scripts", "2013_adhd200_structml")

#ADHD200_DATA_BASE_PATH = "/neurospin/adhd200"
ADHD200_DATA_BASE_PATH = "/volatile/duchesnay/data/2013_adhd_structml"
INPUT_PATH = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")

execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "data_api.py"))
execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "utils_image.py"))

## ==========================================================================
feature = "mw"


# DATA PATH
#INPUT_PATH2 = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")
## !! X and Y a exchanged
Y_train , x_train = get_data(INPUT_PATH, feature, test=False)
Y_test , y_test = get_data(INPUT_PATH, feature, test=True)
mask_im = nib.load(get_mask_path(INPUT_PATH))
mask = mask_im.get_data() == 1

# Build design matrix
X_train = np.zeros((x_train.shape[0], 2))
X_train[x_train == 0, 0] = 1 # CTL regressor
X_train[x_train == 1, 1] = 1 # ADHD regressor

linreg = mulm.LinearRegression()

linreg.fit(X_train, Y_train)
tval, pvalt = linreg.stats(X_train, Y_train, contrast=[1, -1], pval=True)



# Bonferroni correction
print np.sum(pvalt <= 0.05 / Y_train.shape[1])
pvalt_fwer = pvalt.copy()
pvalt_fwer = pvalt_fwer * Y_train.shape[1]
pvalt_fwer[pvalt_fwer > 1] = 1


image_to_file(values=tval, file_path="/tmp/t_ctl-sup-hdhd", mask_image=mask_im)
image_to_file(values=pvalt, file_path="/tmp/pt_ctl-sup-hdhd", mask_image=mask_im, background=1)
image_to_file(values=pvalt_fwer, file_path="/tmp/pt_fwer_ctl-sup-hdhd", mask_image=mask_im, background=1)


# anatomist /tmp/*ctl*hdhd*