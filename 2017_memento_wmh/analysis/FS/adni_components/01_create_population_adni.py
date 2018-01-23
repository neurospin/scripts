
"""

Creates a CSV file for the population.
=> intersection of subject_list.txt and adnimerge_simplified.csv of in CTL or AD

"""
import os
import numpy as np
import pandas as pd
import re
import glob

#import proj_classif_config
GROUP_MAP = {'CTL': 0, 'MCInc': 1, 'MCIc': 2,"AD":3}

BASE_PATH = "/neurospin/brainomics/2016_pca_struct/adni"
INPUT_CLINIC_FILENAME = '/neurospin/cati/ADNI/push_ida_adni1/2015_ADNI/csv/population.csv'
INPUT_FS = os.path.join(BASE_PATH,"freesurfer_assembled_data_fsaverage")

OUTPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/data/adni/population.csv"

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = re.search("^([0-9]{3}_S_[0-9]{4})+_lh", os.path.basename(path)).group(1)
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = re.search("^([0-9]{3}_S_[0-9]{4})+_rh", os.path.basename(path)).group(1)
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["Subject ID", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (817, 3)


# intersect with subject with image
clinic = clinic.merge(input_subjects_fs, on="Subject ID")
assert  clinic.shape == (815, 125)


#Find Controls and converters according to DX at different time points (Conversion time < 2 years is considered)


controls_bool = clinic["status.sc"] == 'NL'
ad_bool = clinic["status.sc"] == 'AD'

converters_bool = (((clinic["status.sc"] == 'MCI')  & (clinic["status.m06"] =='AD')) |
    ((clinic["status.sc"] == 'MCI')  & (clinic["status.m12"] =='AD')) |
     ((clinic["status.sc"] == 'MCI')  & (clinic["status.m18"] =='AD')) |
      ((clinic["status.sc"] == 'MCI')  & (clinic["status.m24"] =='AD')) )


non_converters_bool = ((clinic["status.sc"] == 'MCI')  & (clinic["status.m24"] =='MCI'))

assert sum(ad_bool) == (191,)
assert sum(controls_bool) == (227,)
assert sum(converters_bool) == (133,)
assert sum(non_converters_bool) == (157,)



mcic = clinic[converters_bool][['Subject ID',"Center Code", 'Age at inclusion', 'Sex', "status.sc",\
"MMSE Total Score.sc","ADAS11.sc","ADAS13.sc", "mri_path_lh",  "mri_path_rh"]]
mcic["DX"] = "MCIc"

mcinc = clinic[non_converters_bool][['Subject ID',"Center Code", 'Age at inclusion', 'Sex', "status.sc",\
"MMSE Total Score.sc","ADAS11.sc","ADAS13.sc",  "mri_path_lh",  "mri_path_rh"]]
mcinc["DX"] = "MCInc"

ctl = clinic[controls_bool][['Subject ID',"Center Code", 'Age at inclusion', 'Sex', "status.sc",\
"MMSE Total Score.sc","ADAS11.sc","ADAS13.sc",  "mri_path_lh",  "mri_path_rh"]]
ctl["DX"] = "CTL"

ad = clinic[ad_bool][['Subject ID',"Center Code", 'Age at inclusion', 'Sex', "status.sc",\
"MMSE Total Score.sc","ADAS11.sc","ADAS13.sc",  "mri_path_lh",  "mri_path_rh"]]
ad["DX"] = "AD"


pop = pd.concat([ctl,mcinc,mcic,ad])
assert pop.shape == (708, 11)

assert len(pop["Subject ID"].unique())  == 707
#136_S_0429 is count as copnverter and non converter. Remove it

pop = pop[pop["Subject ID"] != "136_S_0429"]
assert pop.shape == (706, 11)
assert len(pop["Subject ID"].unique())  == 706

# Map group
pop['DX.num'] = pop["DX"].map(GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
