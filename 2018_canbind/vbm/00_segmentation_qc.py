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
import pandas as pd

WD = "/neurospin/psy/canbind"

###############################################################################
# Merge volume with participants
vols = pd.read_csv(os.path.join(WD, "data/derivatives/spmsegment/spmsegment_volumes.csv"))
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')


df = pd.merge(participants, vols, on="participant_id", how='right')
df.head()

tissues_vol = df#[['participant_id', 'site', 'group', 'age', 'sex_x', 'session', 'GMratio', 'WMratio', 'CSFratio']]
tissues_vol.sex_x = tissues_vol.sex_x.map({1:'M', 2:'F'})
tissues_vol = tissues_vol[['participant_id', 'group', 'age', 'sex_x', 'site',
       'session', 'scan_id', 'GMvol_l', 'WMvol_l', 'CSFvol_l', 'TIV_l',
       'GMratio', 'WMratio', 'CSFratio']]

###############################################################################
# Descriptive statistics

tissues_vol.groupby("site").mean()
#            age   GMvol_l   WMvol_l  CSFvol_l     TIV_l   GMratio   WMratio  CSFratio
#site
#CAM   31.261905  0.730541  0.468349  0.322158  1.521049  0.480597  0.307876  0.211527
#MCM         NaN  0.705211  0.396029  0.274779  1.376019  0.513849  0.289095  0.197056
#MCU   37.263736  0.676801  0.443452  0.330423  1.450676  0.467173  0.305122  0.227705
#QNS   37.459459  0.707953  0.464334  0.299720  1.472007  0.481748  0.315855  0.202396
#TGH   35.111702  0.704542  0.432962  0.346712  1.484216  0.475781  0.291539  0.232680
#TWH   43.600000  0.687026  0.422127  0.312070  1.421222  0.483119  0.297371  0.219509
#UBC   34.901042  0.737337  0.417496  0.271659  1.426492  0.519285  0.292699  0.188016
#UCA   31.821839  0.711214  0.452408  0.305720  1.469342  0.485513  0.308093  0.206393

tissues_vol.groupby("site").std()
#            age   GMvol_l   WMvol_l  CSFvol_l     TIV_l   GMratio   WMratio  CSFratio
#site
#CAM   11.225101  0.082429  0.055590  0.074135  0.129720  0.038173  0.024048  0.043873
#MCM         NaN  0.090110  0.051789  0.091770  0.188708  0.033927  0.025229  0.041836
#MCU   13.364825  0.081335  0.071879  0.065456  0.151268  0.040962  0.045896  0.040116
#QNS   13.633349  0.096462  0.057634  0.086284  0.164979  0.044968  0.023848  0.049007
#TGH   12.300744  0.077332  0.056058  0.081745  0.152383  0.036587  0.021292  0.041747
#TWH   10.406729  0.108790  0.061236  0.048966  0.218653  0.003141  0.003410  0.003420
#UBC   11.572019  0.081231  0.060854  0.087152  0.179978  0.038719  0.020997  0.045868
#UCA   10.537430  0.071745  0.051693  0.076410  0.152071  0.036403  0.018407  0.039075

desc = tissues_vol.groupby("site").describe(include="all")
desc.to_excel(os.path.join(WD, 'data/derivatives/spmsegment/QC_segment/spmsegment_volumes_statsdesc.xlsx'))


pdf = PdfPages(os.path.join(WD, 'data/derivatives/spmsegment/QC_segment/spmsegment_volumes_statsdesc.pdf'))

sns.set(style="whitegrid")

fig = plt.figure(figsize=(13.3, 15))
fig.add_subplot(411)
sns.violinplot("site", "GMratio", hue="group", data=tissues_vol.ix[tissues_vol.site != 'TWH', ["site", "GMratio", "group"]], palette="Set3")#, inner="quartile")
fig.add_subplot(412)
sns.violinplot("site", "WMratio", hue="group", data=tissues_vol.ix[tissues_vol.site != 'TWH', ["site", "WMratio", "group"]], palette="Set3")#, inner="quartile")
fig.add_subplot(413)
sns.violinplot("site", "CSFratio", hue="group", data=tissues_vol.ix[tissues_vol.site != 'TWH', ["site", "CSFratio", "group"]], palette="Set3")#, inner="quartile")
fig.add_subplot(414)
sns.violinplot("site", "age", hue="group", data=tissues_vol.ix[tissues_vol.site != 'TWH', ["site", "age", "group"]], palette="Set3")#, inner="quartile")


pdf.savefig()
plt.close(fig)
pdf.close()

###############################################################################
# Outliers
tissues_vol[((tissues_vol.WMratio < .2) | (tissues_vol.WMratio > .5))]
#   participant_id site    group   age sex_x session   GMratio   WMratio  CSFratio
#75   sub-MCU-0044  MCU  Control  51.0     F  ses-01  0.578449  0.057570  0.363982
#85   sub-MCU-0047  MCU  Control  25.0     M  ses-03  0.280478  0.598257  0.121265

###############################################################################
# Recode

recoded = tissues_vol.copy()
import re
dictmap = {'CAM':'S1', 'MCM':'S2', 'MCU':'S3','QNS':'S4', 'TGH':'S5', 'TWH':'S6', 'UBC':'S7', 'UCA':'S8'}
regex = re.compile('|'.join(sorted(re.escape(k) for k in dictmap)))
f = lambda m, s: s.replace(m, dictmap[m])

recoded.participant_id = [f(regex.findall(s)[0], s) for s in tissues_vol.participant_id]

recoded.site = [f(regex.findall(s)[0], s) for s in tissues_vol.site]

recoded.columns = ['participant_id', 'group', 'age', 'sex', 'site',
       'session', 'scan_id', 'gm_vol', 'wm_vol', 'csf_vol', 'tiv_vol',
       'gm_ratio', 'wm_ratio', 'csf_ratio']

recoded = recoded[['participant_id', 'group', 'age', 'sex', 'site','session', 'gm_vol', 'wm_vol', 'csf_vol']]

recoded.to_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/brain_volumes.csv", index=False)



demo = recoded[['participant_id', 'site', 'group', 'age', 'sex']]
demo = demo.drop_duplicates()

gm = recoded[['participant_id',  'session', 'gm_vol']]
wm = recoded[['participant_id',  'session', 'wm_vol']]
csf = recoded[['participant_id',  'session', 'csf_vol']]

demo.to_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/demo.csv", index=False)
gm.to_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/gm.csv", index=False)
wm.to_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/wm.csv", index=False)
csf.to_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/csf.csv", index=False)

import pandas as pd
demo = pd.read_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/demo.csv")
gm = pd.read_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/gm.csv")
wm = pd.read_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/wm.csv")
csf = pd.read_csv("/home/ed203246/git/pystatsml/datasets/brain_volumes/csf.csv")

brain_vol = pd.merge(pd.merge(pd.merge(demo, gm), wm), csf)
assert brain_vol.shape == (808, 9)
print(brain_vol.describe(include='all'))
brain_vol["tiv_vol"] = brain_vol["gm_vol"] + brain_vol["wm_vol"] + brain_vol["csf_vol"]

brain_vol["gm_ratio"] = brain_vol["gm_vol"] / brain_vol["tiv_vol"]
brain_vol["wm_ratio"] = brain_vol["wm_vol"] / brain_vol["tiv_vol"]

sns.lmplot("age", "gm_ratio", hue="group", data=brain_vol[brain_vol.group.notnull()])
