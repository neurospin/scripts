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

# demo_clinic
#demo_clinic = pd.read_csv(os.path.join(WD, "data/sourcedata/clinic_demo/CBN-database_20180618.csv")).iloc[:, 1:]
#demo_clinic = pd.read_csv(os.path.join(WD, "data/sourcedata/clinic_demo/CBN-database_20180703.csv"))#.iloc[:, 1:]
demo_clinic = pd.read_csv(os.path.join(WD, "raw/clinic_demo/CBN-database_20180706.csv"))#.iloc[:, 1:]
# rename columns
cols = {"SUBJLABEL":"participant_id",
        "EVENTNAME":"time_point",
        "AGE_x":"age",
        "SEX_x":"sex",
        "EDUC":"educ",
        "Group_x":"treatment",
        "RESPOND_WK16":"respond_wk16",
        "PSYHIS_MDD_AGE":"age_onset",
        "MADRS_TOT_PRO_RATED":"madrs",
        "PSYHIS_MDE_NUM":"mde_num"}

demo_clinic = demo_clinic[list(cols.keys())]
demo_clinic.columns = [cols[name] for name in demo_clinic.columns]
demo_clinic.head()
demo_clinic["participant_id"] = [s.replace("_", "-") for s in demo_clinic["participant_id"]]
# get site
demo_clinic["site"] = [s.split("-")[0] for s in demo_clinic["participant_id"]]


# Get One line per subject: pivot madrs measures
madrs = demo_clinic[['participant_id', 'time_point', 'madrs']]
madrs = madrs[madrs.time_point.notnull()]
madrs = madrs.pivot(index='participant_id', columns='time_point', values='madrs')
madrs.columns = ['madrs_%s' % c.replace('Week ', "wk") for c in madrs.columns]
madrs = madrs.reset_index()
assert madrs.shape == (323, 11)

# Drop duplicates in the other measures
demo_clinic = demo_clinic[['participant_id',
 'age',
 'sex',
 'educ',
 'treatment',
 'respond_wk16',
 'age_onset',
 'mde_num']].drop_duplicates()
assert demo_clinic.shape == (332, 8)

# merge clinic with pivoted madrs
demo_clinic = pd.merge(demo_clinic, madrs, on='participant_id', how='left')
assert demo_clinic.shape == (332, 18) and (len(set(demo_clinic.participant_id)) == 332)

# Participants with MRI
participants_mri = pd.read_csv(os.path.join(WD, "data", "participants_mri.tsv"), sep='\t')
participants_mri.head()
participants_mri["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in participants_mri["participant_id"]]
assert participants_mri.shape == (310, 9) and (len(set(participants_mri.participant_id)) == 310)

# demo_clinic[["participant_id", "age", "educ", "respond_wk16",,,,]]
# assert demo_clinic.shape == (332, 7) and (len(set(demo_clinic.participant_id)) == 332)
#assert demo_clinic.shape == (2353, 149) and (len(set(demo_clinic.participant_id)) == 332)
#assert demo_clinic.shape == (309, 149) and (len(set(demo_clinic.participant_id)) == 309)


df = pd.merge(demo_clinic, participants_mri, how='outer')
assert df.shape == (349, 26) and (len(set(df.participant_id)) == 349)

with_ima = df[[s for s in df.columns if s.count("ses")]].sum(axis=1) > 0
assert with_ima.sum() == participants_mri.shape[0] == 310


participants = df
participants.to_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t', index=False)


###############################################################################
# Merge FS-ROIs
rois = pd.read_csv(os.path.join(WD, "data/sourcedata/canbind_FS/ROIs/thickness_subcortical_features.csv"))#.iloc[:, 1:]
rois.columns = ["fs-roi_%s"% s for s in rois.columns]
rois.columns = pd.Series(rois.columns).replace({"fs-roi_SUBJLABEL":"participant_id"})
rois["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in rois["participant_id"]]
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')
df = pd.merge(participants, rois, on="participant_id", how='right')
assert df.shape == (278, 82)
df.to_csv(os.path.join(WD, "data/derivatives/FS/ROIs/thickness_subcortical_features.csv"), index=False)

###############################################################################
# 2) How to Merge volume with participants
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')
vols = pd.read_csv(os.path.join(WD, "data/derivatives/spmsegment/spmsegment_volumes.csv"))
df = pd.merge(participants, vols, on="participant_id", how='right')
df.head()
