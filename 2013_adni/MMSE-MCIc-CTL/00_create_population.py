# -*- coding: utf-8 -*-
"""

@author: edouard.duchesnay@cea.fr

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
                          "MMSE-MCIc-CTL",
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
assert  clinic.shape == (456, 244)

# Extract sub-population 
# MCIc = MCI at bl converion to AD within 800 days
TIME_TO_CONV = 800#365 * 2
mcic_m = (clinic["DX.bl"] == "MCI") &\
       (clinic.CONV_TO_AD < TIME_TO_CONV) &\
       (clinic["DX.last"] == "AD") &\
       (np.logical_not(clinic["MMSE.d800"].isnull()))
assert np.sum(mcic_m) == 81

mcic = clinic[mcic_m][['PTID', 'AGE', 'PTGENDER', "DX.bl", "MMSE.bl", "MMSE.d800", "ADAS11.d800", "ADAS13.d800"]]
mcic["DX"] = "MCIc"
print "Mean MMMSE decrease in MCIc=", np.mean(mcic["MMSE.bl"] - mcic["MMSE.d800"])
# Mean MMMSE decrease in MCIc= 4.02469135802

# CTL: CTL at bl no converion to AD
ctl_m = (clinic["DX.bl"] == "CTL") &\
        (clinic["DX.last"] == "CTL") &\
        (np.logical_not(clinic["MMSE.d800"].isnull()))
assert np.sum(ctl_m) == 120

ctl = clinic[ctl_m][['PTID', 'AGE', 'PTGENDER', "DX.bl", "MMSE.bl", "MMSE.d800", "ADAS11.d800", "ADAS13.d800"]]
ctl["DX"] = "CTL"
print "Mean MMMSE decrease in CTL=", np.mean(ctl["MMSE.bl"] - ctl["MMSE.d800"])
# Mean MMMSE decrease in CTL= 0.0416666666667

pop = pd.concat([mcic, ctl])
assert len(pop) == 201

# Map group
pop['DX.num'] = pop["DX"].map(GROUP_MAP)


# Save population information
pop.to_csv(OUTPUT_CSV, index=False)

#
import pylab as plt
def rand_jitter(arr, jit=.1):
    range_ = np.max(arr) - np.min(arr)
    range_ = 1 if range_ == 0 else range_
    stdev = jit * range_
    return arr + np.random.randn(len(arr)) * stdev


#plt.scatter(pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==0]["DX.num"])
plt.plot(rand_jitter(pop[pop["DX.num"]==0]["DX.num"]+1, .01), pop[pop["DX.num"]==0]["MMSE.d800"], "ob",
         rand_jitter(pop[pop["DX.num"]==1]["DX.num"]+1, .01), pop[pop["DX.num"]==1]["MMSE.d800"], "or",
alpha=.3)
plt.boxplot([pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==1]["MMSE.d800"]])
plt.show()

#plt.scatter(pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==0]["DX.num"])
plt.plot(rand_jitter(pop[pop["DX.num"]==0]["DX.num"]+1, .01), pop[pop["DX.num"]==0]["ADAS11.d800"], "ob",
         rand_jitter(pop[pop["DX.num"]==1]["DX.num"]+1, .01), pop[pop["DX.num"]==1]["ADAS11.d800"], "or",
alpha=.3)
plt.boxplot([pop[pop["DX.num"]==0]["ADAS11.d800"], pop[pop["DX.num"]==1]["ADAS11.d800"]])
plt.show()

#plt.scatter(pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==0]["DX.num"])
plt.plot(rand_jitter(pop[pop["DX.num"]==0]["DX.num"]+1, .01), pop[pop["DX.num"]==0]["ADAS13.d800"], "ob",
         rand_jitter(pop[pop["DX.num"]==1]["DX.num"]+1, .01), pop[pop["DX.num"]==1]["ADAS13.d800"], "or",
alpha=.3)
plt.boxplot([pop[pop["DX.num"]==0]["ADAS13.d800"], pop[pop["DX.num"]==1]["ADAS13.d800"]])
plt.show()

#plt.scatter(pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==0]["DX.num"])
plt.plot(pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==0]["ADAS11.d800"], "ob",
         pop[pop["DX.num"]==1]["MMSE.d800"], pop[pop["DX.num"]==1]["ADAS11.d800"], "or", alpha=.3)
plt.show()

#plt.scatter(pop[pop["DX.num"]==0]["MMSE.d800"], pop[pop["DX.num"]==0]["DX.num"])
plt.plot(pop[pop["DX.num"]==0]["ADAS13.d800"], pop[pop["DX.num"]==0]["ADAS11.d800"], "ob",
         pop[pop["DX.num"]==1]["ADAS13.d800"], pop[pop["DX.num"]==1]["ADAS11.d800"], "or", alpha=.3)
plt.show()


"""
d = read.csv("/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/population.csv")
summary(aov(ADAS11.d800 ~ DX, data=d))
summary(aov(MMSE.d800 ~ DX, data=d))

"""