#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:04:30 2018

@author: ed203246
"""

###############################################################################
# 1) Merge participants with demographics

import os
import pandas as pd

WD = "/neurospin/psy/canbind"

demo = pd.read_csv(os.path.join(WD, "data/sourcedata/clinic_demo/CBN-database_20180618.csv")).iloc[:, 1:]
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')


participants.head()
demo.head()
#  SUBJLABEL    Group_x  AGE_x  SEX_x
#0  CAM_0002    Control   26.0    2.0
#1  CAM_0004    Control   29.0    2.0
#2  CAM_0005  Treatment   24.0    1.0
#3  CAM_0006    Control   23.0    2.0
#4  CAM_0007  Treatment   23.0    1.0

demo.columns = ["participant_id", "group", "age", "sex_x"]

# append sub- and change _ to -
demo["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in demo["participant_id"]]

participants.head()
#  participant_id  ses-01  ses-01-alt03  ses-01-alt16  ses-01SE02  ses-02  ses-03  ses-03SE02 site
#0       CAM-0002     1.0           NaN           NaN         NaN     1.0     1.0         NaN  CAM
#1       CAM-0004     1.0           NaN           NaN         NaN     1.0     1.0         NaN  CAM
#2       CAM-0005     NaN           NaN           NaN         NaN     1.0     1.0         NaN  CAM
#3       CAM-0006     1.0           NaN           NaN         NaN     1.0     1.0         NaN  CAM
#4       CAM-0007     NaN           NaN           NaN         NaN     1.0     1.0         NaN  CAM

assert participants.shape == (310, 9)
assert demo.shape == (332, 4)


df = pd.merge(demo, participants, on="participant_id", how='outer')
df.head()


with_ima = df[[s for s in df.columns if s.count("ses")]].sum(axis=1) > 0
assert with_ima.sum() == participants.shape[0] == 310


# append "sub-"
df.to_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t', index=False)
