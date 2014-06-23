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
import shutil

BASE_PATH = "/neurospin/abide"

nii_filenames = glob.glob(os.path.join(BASE_PATH, "loni/ABIDE/*/MP-RAGE/*/*/*.nii"))
clinic_loni_filename =  os.path.join(BASE_PATH, "clinic", "ABIDE_6_18_2014.csv")
clinic_nitrc_filename = os.path.join(BASE_PATH, "clinic", "Phenotypic_V1_0b.csv")
# OUTPUT:
OUTPUT_CSV = os.path.join(BASE_PATH, "population.csv")
OUTPUT_SAMPLE_CSV = os.path.join(BASE_PATH, "population_sample.csv")

OUTPUT_IMAGES = os.path.join(BASE_PATH, "ni/{Site}/{Subject}/t1mri/default_acquisition/default_analysis/{Subject}_mprage.nii")

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


pop = pd.merge(clinic_loni, clinic_nitrc, on=["Subject", "DX", "Sex"])

assert len(pop) == 1099
assert np.max(np.abs(pop.Age_x - pop.Age_y)) < 1
print pop[1:10]
pop = pop[["Subject", "DX", "Sex", "Age_x", "Site"]]
pop.columns = ["Subject", "DX", "Sex", "Age", "Site"]

#pop["mri_path"] = [os.path.join(OUTPUT_IMAGES, str(s), str(s)+"_mprage.nii.gz") for s in pop.Subject]
pop["mri_path"] = [OUTPUT_IMAGES.format(Site=pop.iloc[i]["Site"], Subject=pop.iloc[i]["Subject"]) for i in xrange(len(pop))]

pop.to_csv(OUTPUT_CSV, index=False)
assert pop.shape == (1099, 6)

########################################################################
## copy images into ni/<SITE>/

for src_nii_filename in nii_filenames:
    subj_nii = int(src_nii_filename.split("/")[5])
    t = pop[pop.Subject == subj_nii]
    dst_nii_filename = t.mri_path.values[0]
    if not os.path.exists(os.path.dirname(dst_nii_filename)):
        os.makedirs(os.path.dirname(dst_nii_filename))
    print src_nii_filename, dst_nii_filename
    shutil.copyfile(src_nii_filename, dst_nii_filename)
    ret = os.system("diff %s %s" % (src_nii_filename, dst_nii_filename))
    if ret != 0:
        raise ValueError()
    #os.system("gzip %s" % dst_nii_filename)

sample = list()
# build sample
for site in np.unique(pop.Site.values):
    #site = np.unique(m.Site.values)[0]
    s=pop[(pop.Site == site) & (pop.DX == 1)].iloc[0]
    subject = str(s["Subject"])
    sample.append(s)
    s=pop[(pop.Site == site) & (pop.DX == 0)].iloc[0]
    sample.append(s)

sample = pd.DataFrame(sample)
sample.to_csv(OUTPUT_SAMPLE_CSV, index=False)

## Compress images:
## find ni_t1 -name "*mprage.nii"|while read f ; do gzip $f ; done