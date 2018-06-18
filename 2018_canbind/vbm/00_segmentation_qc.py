#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:04:30 2018

@author: ed203246
"""

###############################################################################
#

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

WD = "/neurospin/psy/canbind"

vols = pd.read_csv(os.path.join(WD, "data/derivatives/spmsegment/QC_segment/spmsegment_volumes.csv"))
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')

vols.head()
participants.head()

vols["scan_id"]

import re

#vols["scan_id"]

regex = re.compile("^.+(sub-.+)_(ses-.+)_T1w")
df = pd.DataFrame([regex.findall(s)[0] for s in vols["scan_id"]], columns=["participant_id", "session"])


df = pd.concat([df, vols], axis=1, sort=False)

df = pd.merge(participants, df, on="participant_id", how='right')
tissues_vol = df[['participant_id', 'site', 'group', 'age', 'sex_x', 'session', 'GMratio', 'WMratio', 'CSFratio']]

# Plot of Volume ratios
#pdf = PdfPages(os.path.join(output_dir, "spmsegment_volumes.pdf"))

df = tissues_vol.melt(id_vars=["participant_id", 'site', "session", "group", "age", "sex_x"])

fig = plt.figure(figsize=(26.6, 15))
#set(df.variable)
#Out[101]: {'CSFratio', 'GMratio', 'WMratio'}

fig.add_subplot(311)
sns.violinplot("variable", "value", hue="site", data=df[df.variable=="GMratio"], inner="quartile")
fig.add_subplot(312)
sns.violinplot("variable", "value", hue="site", data=df[df.variable=="WMratio"], inner="quartile")
fig.add_subplot(313)
sns.violinplot("variable", "value", hue="site", data=df[df.variable=="CSFratio"], inner="quartile")

#sns.swarmplot("variable", "value", hue="site", color="white", data=df, size=2, alpha=0.5)
#pdf.savefig()
plt.close(fig)



set(df.site)

df.participant_id.map({'CAM':'001', 'MCM':'001', 'MCU':'001',
                       'QNS':'001', 'TGH':'001', 'TWH':'001', 'UBC':'001', 'UCA':'001'})



def change(col):
    """
    Out[51]: {'CAM', 'MCM', 'MCU', 'QNS', 'TGH', 'TWH', 'UBC', 'UCA'}
    """
    col = col.str.replace('CAM', '001')
    col = col.str.replace('MCM', '002')
    col = col.str.replace('MCU', '003')
    col = col.str.replace('QNS', '004')
    col = col.str.replace('TGH', '005')
    col = col.str.replace('TWH', '006')
    col = col.str.replace('UBC', '007')
    col = col.str.replace('UCA', '008')
    return col


