# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:11:27 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause


ABIDE 1109 Subjects. keep only MP-RAGE: 1099 subjects
"""

import os, glob
import pandas as pd
import numpy as np

BASE_PATH = "/neurospin/abide"

nii_filenames = glob.glob(os.path.join(BASE_PATH, "loni/ABIDE/*/MP-RAGE/*/*/*.nii"))
clinic_loni_filename =  os.path.join(BASE_PATH, "clinic", "ABIDE_6_18_2014.csv")
clinic_nitrc_filename = os.path.join(BASE_PATH, "clinic", "Phenotypic_V1_0b.csv")
# OUTPUT:
clinic_filename = os.path.join(BASE_PATH, "clinic", "clinic.csv")

DX_LONI_MAP = {'Control': 0, 'Autism': 1}
SEX_LONI_MAP = {'F': 0, 'M': 1,}
DX_NITRC_MAP = {1: 1, 2: 0}
SEX_NITRC_MAP = {2: 0, 1: 1}

########################################################################
## Check NI from loni verify that we have the MP-RAGE IN ABIDE_6_18_2014.csv
clinic_loni = pd.read_csv(clinic_loni_filename)
clinic_loni = clinic_loni[clinic_loni.Description == 'MP-RAGE']
subj_nii = set([int(f.split("/")[5]) for f in nii_filenames])

subj_csv = set(clinic_loni.Subject)
assert len(subj_nii) == len(subj_csv) == 1099

assert len(subj_csv.difference(subj_nii)) == 0

########################################################################
## Check NI from loni verify that we have the MP-RAGE IN ABIDE_6_18_2014.csv
clinic_nitrc = pd.read_csv(clinic_nitrc_filename)
clinic_nitrc['DX'] = clinic_nitrc["DX_GROUP"].map(DX_NITRC_MAP)
clinic_nitrc['Sex'] = clinic_nitrc["SEX"].map(SEX_NITRC_MAP)
clinic_nitrc = clinic_nitrc[["SUB_ID", "DX", "Sex", "AGE_AT_SCAN", "SITE_ID"]]
clinic_nitrc.columns = ["Subject", "DX", "Sex", "Age", "Site"]

clinic_loni = pd.read_csv(clinic_loni_filename)
clinic_loni = clinic_loni[clinic_loni.Description == 'MP-RAGE']
clinic_loni['DX'] = clinic_loni["Group"].map(DX_LONI_MAP)
clinic_loni['Sex'] = clinic_loni["Sex"].map(SEX_LONI_MAP)
clinic_loni = clinic_loni[["Subject", "DX", "Sex", "Age"]]


m = pd.merge(clinic_loni, clinic_nitrc, on=["Subject", "DX", "Sex"])

assert len(m) == 1099
assert np.max(np.abs(m.Age_x - m.Age_y)) < 1
print m[1:10]
m = m[["Subject", "DX", "Sex", "Age_x", "Site"]]
m.columns = ["Subject", "DX", "Sex", "Age", "Site"]
m.to_csv(clinic_filename, index=False)
assert m.shape == (1099, 5)