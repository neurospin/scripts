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
demo_clinic = pd.read_csv(os.path.join(WD, "data/sourcedata/clinic_demo/CBN-database_20180703.csv"))#.iloc[:, 1:]

demo_clinic.head()
demo_clinic.columns = ["participant_id", "age", "sex", "respond_wk16", "group", "psyhis_mdd_age"]
# append sub- and change _ to -
demo_clinic["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in demo_clinic["participant_id"]]
# get site
demo_clinic["site"] = [s.split("-")[1] for s in demo_clinic["participant_id"]]

# Participants
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')
participants.head()
participants["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in participants["participant_id"]]


assert participants.shape == (310, 9) and (len(set(participants.participant_id)) == 310)
assert demo_clinic.shape == (332, 7) and (len(set(demo_clinic.participant_id)) == 332)

df = pd.merge(demo_clinic, participants, how='outer')
assert df.shape == (349, 14) and (len(set(df.participant_id)) == 349)

with_ima = df[[s for s in df.columns if s.count("ses")]].sum(axis=1) > 0
assert with_ima.sum() == participants.shape[0] == 310


df.to_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t', index=False)
