# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:44:13 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

from glob import glob
import os
from shutil import rmtree
from numpy import unique

ROOT='/neurospin/radiomics/studies/metastasis'
pold = os.path.join(ROOT, 'baseOld')
pnew = os.path.join(ROOT, 'base')
logs_a = os.path.join(pnew, 'anat_logs')
logs_1 = os.path.join(pnew, 'model01_logs')
logs_2 = os.path.join(pnew, 'model02_logs')

fenh = glob(os.path.join(old, 'raw', '*', 'AxT1enhanced', '*'))
sid = [i.split('/')[7] for i in fenh]
sid = unique(sid).tolist()

for s in sid:
    e = os.path.join(pnew, s)
    _ = [os.makedirs(os.path.join(e, i)) for i in ['anat', 'model01', 'model02'] 
                    if not os.path.exists(os.path.join(e, i))]
    os.makedirs(os.path.join(e, 'anat'))
    os.makedirs(os.path.join(e, 'model01'))
    os.makedirs(os.path.join(e, 'model02'))

for e in [logs_a, logs_1, logs_2]:
    if not os.path.exists(e):
        os.makedirs(e)
