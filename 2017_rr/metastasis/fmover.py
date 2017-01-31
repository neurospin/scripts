# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:44:13 2017

@author: vf140245, cp251292
Copyrignt : CEA NeuroSpin - 2014
"""

from glob import glob
import os
from shutil import copy2
from numpy import unique

ROOT='/neurospin/radiomics/studies/metastasis'
pold = os.path.join(ROOT, 'baseOld')
pnew = os.path.join(ROOT, 'baseTest')

fenh = glob(os.path.join(pold, 'raw', '*', 'AxT1enhanced', '*'))
sid = [i.split('/')[7] for i in fenh]
sid = unique(sid).tolist()

for s in sid:
    print(s)
    ind_path = os.path.join(pnew, s)
    anat_dir = os.path.join(ind_path, 'anat')
    if not os.path.exists(anat_dir):
        os.makedirs(anat_dir)
    T1 = os.path.join(pold, 'raw', s, 'AxT1enhanced', 'AxT1enhanced.nii.gz')
    T1_new = "{path}/{id}_{seq_type}.{ext}".format(path=os.path.join(pnew, s,
 'anat'), id=s, seq_type='enh-gado_T1w', ext='nii.gz')
    #print(T1)
    #print(T1_new)
    copy2(T1, T1_new)
    T1_json = os.path.join(pold, 'raw', s, 'AxT1enhanced', 'logs', '*.json')
    T1_json_new = "{path}/{id}_{seq_type}.{ext}".format(path=os.path.join(pnew, 
s, 'anat'), id=s, seq_type='enh-gado_T1w', ext='json')
    copy2(T1, T1_json_new)
    T2 = os.path.join(pold, 'raw', s, 'AxT2', 'AxT2.nii.gz')
    T2_new = "{path}/{id}_{seq_type}.{ext}".format(path=os.path.join(pnew, 
s, 'anat'), id=s, seq_type='FLAIR', ext='nii.gz')
    copy2(T2, T2_new)
    T2_json = os.path.join(pold, 'raw', s, 'AxT2', 'logs', '*.json')
    T2_json_new = "{path}/{id}_{seq_type}.{ext}".format(path=os.path.join(pnew, 
s, 'anat'), id=s, seq_type='FLAIR', ext='json')
    copy2(T2, T2_json_new)
    logs_dir = os.path.join(anat_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logs_T1_dir = os.path.join(logs_dir, s+'_enh-gado_T1w')
    if not os.path.exists(logs_T1_dir):
        os.makedirs(logs_T1_dir)
    for j in ['inputs', 'outputs', 'runtime']:
        log1 = "{path}/{id}/{seq_type}/logs/{file}".format(path=os.path.join(pold, 'raw'), 
id=s, seq_type='AxT1enhanced', file=j+'.json')
        log1_new = '/'.join([logs_T1_dir, j+'.json'])
        copy2(log1, log1_new)
    logs_T2_dir = os.path.join(logs_dir, s+'_FLAIR')
    if not os.path.exists(logs_T2_dir):
        os.makedirs(logs_T2_dir)
    for k in ['inputs', 'outputs', 'runtime']:
        log2 = "{path}/{id}/{seq_type}/logs/{file}".format(path=os.path.join(pold, 'raw'), 
id=s, seq_type='AxT2', file=k+'.json')
        log2_new = '/'.join([logs_T2_dir, k+'.json'])
        copy2(log2, log2_new)
    qc_dir = os.path.join(anat_dir, 'qc')
    if not os.path.exists(qc_dir):
        os.makedirs(qc_dir)
    qc1 = "{path}/{id}/{seq_type}/{file}".format(path=os.path.join(pold, 'raw'), 
id=s, seq_type='AxT1enhanced', file='AxT1enhanced.pdf')
    qc1_new = '/'.join([qc_dir, s+'_enh-gado_T1w.pdf'])
    copy2(qc1, qc1_new)
    qc2 = "{path}/{id}/{seq_type}/{file}".format(path=os.path.join(pold, 'raw'), 
id=s, seq_type='AxT2', file='AxT2.pdf')
    qc2_new = '/'.join([qc_dir, s+'_FLAIR.pdf'])
    copy2(qc2, qc2_new)
    if not os.path.exists(os.path.join(ind_path, 'model01')):
        os.makedirs(os.path.join(ind_path, 'model01'))
    if not os.path.exists(os.path.join(ind_path, 'model02')):
        os.makedirs(os.path.join(ind_path, 'model02'))
