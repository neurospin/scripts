# -*- coding: utf-8 -*-
"""

@author: md238665

Creates a CSV file for the population.
=> intersection of subject_list.txt and adnimerge_simplified.csv of in CTL or AD

"""
import os
import numpy as np
import pandas as pd

#import proj_classif_config
GROUP_MAP = {'CTL': 0, 'MCIc': 1}

BASE_PATH = "/neurospin/brainomics/2013_adni"
INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "adnimerge_simplified.csv")
INPUT_SUBJECTS_LIST_FILENAME = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC",
                                   "subject_list.txt")

OUTPUT_CSV = os.path.join(BASE_PATH,
                          "MCIc-CTL",
                          "population.csv")

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read subjects with image
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
input_subjects = [x[:10] for x in input_subjects[1]]

# intersect with subject with image
clinic = clinic[clinic["PTID"].isin(input_subjects)]
assert  clinic.shape == (456, 92)

# Extract sub-population 
# MCIc = MCI at bl converion to AD within 800 days
TIME_TO_CONV = 800#365 * 2
mcic_m = (clinic["DX.bl"] == "MCI") &\
       (clinic.CONV_TO_AD < TIME_TO_CONV) &\
       (clinic["DX.last"] == "AD")
assert np.sum(mcic_m) == 82

mcic = clinic[mcic_m][['PTID', 'AGE', 'PTGENDER', "DX.bl"]]
mcic["DX"] = "MCIc"

# CTL: CTL at bl no converion to AD
ctl_m = (clinic["DX.bl"] == "CTL") &\
      (clinic["DX.last"] == "CTL")
assert np.sum(ctl_m) == 120

ctl = clinic[ctl_m][['PTID', 'AGE', 'PTGENDER', "DX.bl"]]
ctl["DX"] = "CTL"

pop = pd.concat([mcic, ctl])
assert len(pop) == 202

# Map group
pop['DX.num'] = pop["DX"].map(GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
