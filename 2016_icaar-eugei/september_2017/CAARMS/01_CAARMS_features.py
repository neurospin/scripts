#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:38:49 2017

@author: ad247405
"""


import os
import numpy as np
import pandas as pd
import nibabel
import brainomics.image_atlas
import mulm
import nilearn
from nilearn import plotting
from mulm import MUOLS
import sklearn
from sklearn import preprocessing



INPUT_CSV_ICAAR = "/neurospin/brainomics/2016_icaar-eugei/september_2017/VBM/ICAAR/population+scores.csv"
OUTPUT_DATA = "/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS/data"

pop = pd.read_csv(INPUT_CSV_ICAAR)
pop['M0_CA7_7B'][22]= 0
pop['M0_CA2_1B'][33]= 0
pop['M0_CA7_7B'] = pop['M0_CA7_7B'].astype(np.float)
pop['M0_CA2_1B'] = pop['M0_CA7_7B'].astype(np.float)

assert pop.shape == (53,120)
#Keep only subjects for which we have MRI data
pop = pop[np.isnan(pop["group_outcom.num"])==False]
assert pop.shape == (41,120)
pop = pop[pop["sex.num"]==1]
assert pop.shape == (28,120)
np.save(os.path.join(OUTPUT_DATA,"y.npy"),pop["group_outcom.num"].values)


#BPSR score
###################################################################################
BPRS24_score = pop["BPRS24 TOTAL"].values
BPRS24_score[np.isnan(BPRS24_score)] = np.nanmean(BPRS24_score)
np.save(os.path.join(OUTPUT_DATA,"X_BPRS.npy"),BPRS24_score)
###################################################################################


features = pop.keys()

features_CAARMS_all_names = ['M0_CA1_1A',
       'M0_CA1_1B', 'M0_CA1_2A', 'M0_CA1_2B', 'M0_CA1_3A', 'M0_CA1_3B',
       'M0_CA1_4A', 'M0_CA1_4B', 'M0_CA2_1A', 'M0_CA2_1B', 'M0_CA2_2A',
       'M0_CA3_1A', 'M0_CA3_1B', 'M0_CA3_2A', 'M0_CA3_2B', 'M0_CA3_3A',
       'M0_CA3_3B', 'M0_CA4_1A', 'M0_CA4_1B', 'M0_CA4_2A', 'M0_CA4_2B',
       'M0_CA4_3A', 'M0_CA4_3B', 'M0_CA5_1A', 'M0_CA5_1B', 'M0_CA5_2A',
       'M0_CA5_2B', 'M0_CA5_3A', 'M0_CA5_3B', 'M0_CA5_4A', 'M0_CA5_4B',
       'M0_CA6_1A', 'M0_CA6_1B', 'M0_CA6_2A', 'M0_CA6_3A', 'M0_CA6_3B',
       'M0_CA6_4A', 'M0_CA6_4B', 'M0_CA7_1A', 'M0_CA7_1B', 'M0_CA7_2A',
       'M0_CA7_2B', 'M0_CA7_3A', 'M0_CA7_3B', 'M0_CA7_4A', 'M0_CA7_4B',
       'M0_CA7_5A', 'M0_CA7_5B', 'M0_CA7_6A', 'M0_CA7_6B', 'M0_CA7_7A',
       'M0_CA7_7B', 'M0_CA7_8A', 'M0_CA7_8B']

features_CAARMS_severity_names = ['M0_CA1_1A','M0_CA1_2A','M0_CA1_3A',
                                  'M0_CA1_4A','M0_CA2_1A','M0_CA2_2A',
                                  'M0_CA3_1A', 'M0_CA3_2A', 'M0_CA3_3A',
                                  'M0_CA4_1A', 'M0_CA4_2A','M0_CA4_3A',
                                  'M0_CA5_1A','M0_CA5_2A','M0_CA5_3A',
                                  'M0_CA5_4A','M0_CA6_1A','M0_CA6_2A',
                                  'M0_CA6_3A', 'M0_CA6_4A','M0_CA7_1A',
                                  'M0_CA7_2A', 'M0_CA7_3A','M0_CA7_4A',
                                  'M0_CA7_5A','M0_CA7_6A','M0_CA7_7A','M0_CA7_8A']

features_CAARMS_frequence_names = ['M0_CA1_1B','M0_CA1_2B','M0_CA1_3B',
                                  'M0_CA1_4B','M0_CA2_1B',
                                  'M0_CA3_1B', 'M0_CA3_2B', 'M0_CA3_3B',
                                  'M0_CA4_1B', 'M0_CA4_2B','M0_CA4_3B',
                                  'M0_CA5_1B','M0_CA5_2B','M0_CA5_3B',
                                  'M0_CA5_4B','M0_CA6_1B',
                                  'M0_CA6_3B', 'M0_CA6_4B','M0_CA7_1B',
                                  'M0_CA7_2B', 'M0_CA7_3B','M0_CA7_4B',
                                  'M0_CA7_5B','M0_CA7_6B','M0_CA7_7B','M0_CA7_8B']




features_CAARMS_all_scores = pop[features_CAARMS_all_names].fillna(0.0).values
features_CAARMS_severity_scores = pop[features_CAARMS_severity_names].fillna(0).values
features_CAARMS_frequence_scores = pop[features_CAARMS_frequence_names].fillna(0).values

assert features_CAARMS_all_scores.shape == (28,54)
assert features_CAARMS_severity_scores.shape == (28,28)
assert features_CAARMS_frequence_scores.shape == (28,26)



np.save(os.path.join(OUTPUT_DATA,"X_CAARMS_all.npy"),features_CAARMS_all_scores)
np.save(os.path.join(OUTPUT_DATA,"X_CAARMS_severity.npy"),features_CAARMS_severity_scores)
np.save(os.path.join(OUTPUT_DATA,"X_CAARMS_frequence.npy"),features_CAARMS_frequence_scores)


np.save(os.path.join(OUTPUT_DATA,"features_CAARMS_all.npy"),features_CAARMS_all_names)
np.save(os.path.join(OUTPUT_DATA,"features_CAARMS_severity.npy"),features_CAARMS_severity_names)
np.save(os.path.join(OUTPUT_DATA,"features_CAARMS_frequence.npy"),features_CAARMS_frequence_names)