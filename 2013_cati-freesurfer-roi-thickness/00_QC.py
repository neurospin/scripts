# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:54:59 2013

@author: ed203246
"""

import os
import pandas as pd
import numpy as np

WD = "/home/ed203246/data/2013_cati-freesurfer-roi-thickness"
csvfilename = os.path.join(WD,"data/freesurfer.csv")
datasets_filepath = os.path.join(WD,"data/AD_CTL.npz")

## Read and clean the table
df = pd.read_table(csvfilename, header=0)
# import utils
run /home/ed203246/git/scripts/2013_cati-freesurfer-roi-thickness/utils.py
d_num, d_cat, d_nas, stats = dataframe_quality_control(df)

print d_cat.to_string()
print stats.to_string()

X = np.asarray(d_num)
print set(d_cat.DIAGNOSTIC)
# subset
idx = np.array(d_cat.DIAGNOSTIC == "AD") | np.array(d_cat.DIAGNOSTIC == "CONTROL")
X = d_num[idx]
y = np.array(d_cat.DIAGNOSTIC[idx] == "AD", dtype=int)

np.savez(datasets_filepath, X=X, y=y)