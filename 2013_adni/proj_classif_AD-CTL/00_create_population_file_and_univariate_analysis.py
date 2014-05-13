# -*- coding: utf-8 -*-
"""

@author: md238665

Creates a CSV file for the population.
=> intersection of subject_list.txt and adnimerge_baseline.csv of in CTL or AD

Create files for two-sample t-test with SPM (CTL vs AD).
This analysis will create the mask.

"""
import os
import glob
import pandas as pd

#import proj_classif_config
GROUP_MAP = {'CTL': 0, 'AD': 1}

BASE_PATH = "/neurospin/brainomics/2013_adni"
INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "adnimerge_baseline.csv")
INPUT_SUBJECTS_LIST_FILENAME = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC",
                                   "subject_list.txt")

OUTPUT_CSV = os.path.join(BASE_PATH,
                          "proj_classif_AD-CTL",
                          "population.csv")

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read input subjects
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
input_subjects = [x[:10] for x in input_subjects[1]]


# Extract sub-population
is_input_groups = clinic["DX.bl"].isin(GROUP_MAP)
is_in_input = clinic["PTID"].isin(input_subjects)
pop = clinic[is_input_groups & is_in_input]
pop = pop[['PTID', 'AGE', 'PTGENDER', "DX.bl"]]
n = len(pop)
print "Found", n
#Found 270

# Map group
pop['DX.bl.num'] = pop["DX.bl"].map(GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)



#############################################################################
## THIS PART HAS NOT BEEN RE-TESTED
if False:
# Create file
    OUTPUT_SPM_FILENAME = os.path.join(OUTPUT_BASE_PATH,
                               "SPM",
                               "template_FinalQC_CTL_AD",
                               "spm_file.txt")
    
    if not os.path.exists(os.path.dirname(OUTPUT_SPM_FILENAME)):
        os.makedirs(os.path.dirname(OUTPUT_SPM_FILENAME))
    OUTPUT_SPM_FILENAME = open(OUTPUT_SPM_FILENAME, "w")
    for group in INPUT_GROUPS:
        group_num = GROUP_MAP[group]
        print >> OUTPUT_SPM_FILENAME, "[{group_num} ({group})]".format(group_num=group_num,
                                                               group=group)
        is_sub_pop = pop['Group.num'] == group_num
        sub_pop = pop[is_sub_pop]
        n_sub = len(sub_pop)
        print "Found", n_sub, "in", group
        for ptid in sub_pop.index:
            print ptid
            imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=ptid)
            imagefile_name = glob.glob(imagefile_pattern)[0]
            print >> OUTPUT_SPM_FILENAME, imagefile_name
        print >> OUTPUT_SPM_FILENAME
    
    OUTPUT_SPM_FILENAME.close()
    
    # Display some statistics
    import itertools
    levels = list(itertools.product(INPUT_GROUPS, ['training', 'testing']))
    index = pd.MultiIndex.from_tuples(levels,
                                      names=['Class', 'Set'])
    count = pd.DataFrame(columns=['Count'], index=index)
    for i, (group, sample) in enumerate(levels):
        is_in = (pop['Group.ADNI'] == group) & (pop['Sample'] == sample)
        count['Count'].loc[group, sample] = is_in.nonzero()[0].shape[0]
