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

# demo
demo = pd.read_csv(os.path.join(WD, "data/sourcedata/clinic_demo/CBN-database_20180618.csv")).iloc[:, 1:]
demo.head()
demo.columns = ["participant_id", "group", "age", "sex_x"]
# append sub- and change _ to -
demo["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in demo["participant_id"]]

# Response
resp = pd.read_csv(os.path.join(WD, "data/sourcedata/clinic_demo//response_20180627.csv"))
resp.columns = ["participant_id", "sex_x", "age", "group", "Respond_WK16"]
# append sub- and change _ to -
resp["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in resp["participant_id"]]

# Participants
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')
participants.head()
participants["participant_id"] = ["sub-%s" % s.replace("_", "-") for s in participants["participant_id"]]


assert participants.shape == (310, 9) and (len(set(participants.participant_id)) == 310)
assert demo.shape == (332, 4) and (len(set(demo.participant_id)) == 332)
assert resp.shape == (332, 5) and (len(set(resp.participant_id)) == 332)



df = pd.merge(demo, participants, how='outer')
assert df.shape == (349, 12) and (len(set(df.participant_id)) == 349)

df = pd.merge(df, resp, how='outer')
assert df.shape == (349, 13) and (len(set(df.participant_id)) == 349)

df.head()


with_ima = df[[s for s in df.columns if s.count("ses")]].sum(axis=1) > 0
assert with_ima.sum() == participants.shape[0] == 310


# append "sub-"
df.to_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t', index=False)
