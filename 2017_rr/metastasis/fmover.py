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
import re

ROOT='/neurospin/radiomics/studies/metastasis/'
pold = os.path.join(ROOT, 'baseOld')
pnew = os.path.join(ROOT, 'base')

#fenh = glob(os.path.join(pold, 'raw', '*', 'AxT1enhanced', '*'))
fenh = glob(os.path.join(ROOT, 'base', '*'))
sid = [i.split('/')[6] for i in fenh]
sid = unique(sid).tolist()[0:-5]

#anat
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

#registry
for s in sid:
    print(s)
    ind_path = os.path.join(pnew, s)
    reg_dir = os.path.join(ind_path, 'model01')
    if not os.path.exists(reg_dir):
        os.makedirs(reg_dir)
    reg = os.path.join(pold, 'preprocess', s, 'rAxT2.nii.gz')
    reg_new = "{path}/{id}_{seq_type}.{ext}".format(path=reg_dir, id=s, seq_type='rAxT2', ext='nii.gz')
    copy2(reg, reg_new)
    reg_txt = os.path.join(pold, 'preprocess', s, 'rAxT2.nii.txt')
    reg_txt_new = "{path}/{id}_{seq_type}.{ext}".format(path=reg_dir, id=s, seq_type='rAxT2', ext='nii.txt')
    copy2(reg_txt, reg_txt_new)
    #print(T1)
    #print(T1_new)
    rlogs_dir = os.path.join(reg_dir, 'logs')
    if not os.path.exists(rlogs_dir):
        os.makedirs(rlogs_dir)
    rlog = "{path}/{file}".format(path=os.path.join(pold, 'preprocess'),file='log.txt')
    rlog_new = "{path}/{file}".format(path=rlogs_dir, file='log.txt')
    copy2(rlog, rlog_new)
    rqc_dir = os.path.join(reg_dir, 'qc')
    if not os.path.exists(rqc_dir):
        os.makedirs(rqc_dir)
    rqc_ax = "{path}/{id}/{file}".format(path=os.path.join(pold, 'preprocess'), 
id=s, file='qc_axi.pdf')
    rqc_ax_new = "{path}/{id}_{file}".format(path=rqc_dir, id=s, file='axi.pdf')
    copy2(rqc_ax, rqc_ax_new)
    rqc_sag = "{path}/{id}/{file}".format(path=os.path.join(pold, 'preprocess'), 
id=s, file='qc_sag.pdf')
    rqc_sag_new = "{path}/{id}_{file}".format(path=rqc_dir, id=s, file='sag.pdf')
    copy2(rqc_sag, rqc_sag_new)

#segmentation MJ -> model10
seg_old = os.path.join(ROOT, 'resource', 'poumon', 'metas_poumon_MJ')
mod = 'model10'
for s in sid:
    print(s)
    edema = glob(os.path.join(seg_old, s, 'mask_edema*'))
    lesion = glob(os.path.join(seg_old, s, 'mask_lesion*'))
    necrosis = glob(os.path.join(seg_old, s, 'mask_necrosis*'))
    enh = glob(os.path.join(seg_old, s, 'mask_enh*'))
    ind_path = os.path.join(pnew, s)
    seg_dir = os.path.join(ind_path, mod)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    n=0
    for e in edema : 
        n+=1
        new_ed = "{path}/{id}_{model}_{type}_{num}.{ext}".format(path=seg_dir, 
id=s, model=mod, type='mask_edema', num=n, ext='nii.gz')
        copy2(e, new_ed)
    n=0
    for l in lesion : 
        n+=1
        new_le = "{path}/{id}_{model}_{type}_{num}.{ext}".format(path=seg_dir, 
id=s, model='model10', type='mask_lesion', num=n, ext='nii.gz')
        copy2(l, new_le)
    n=0
    for ne in necrosis : 
        n+=1
        new_ne = "{path}/{id}_{model}_{type}_{num}.{ext}".format(path=seg_dir, 
id=s, model='model10', type='mask_necrosis', num=n, ext='nii.gz')
        copy2(ne, new_ne)
    n=0
    for en in enh : 
        n+=1
        new_en = "{path}/{id}_{model}_{type}_{num}.{ext}".format(path=seg_dir, 
id=s, model='model10', type='mask_enh', num=n, ext='nii.gz')
        copy2(en, new_en)
    

#segmentation ADP -> model11
seg_old = os.path.join(ROOT, 'resource', 'poumon', 'Alberto.ADP')
mod = 'model11'
edema=[]
enh=[]
necrosis=[]
for s in sid :
    print(s)
    edema = glob(os.path.join(seg_old, s, 'AxT1enhanced', 'mask_edema*'))
    print(edema)
    lesion = glob(os.path.join(seg_old, s, 'AxT1enhanced', 'mask_lesion*'))
    print(lesion)
    necrosis = glob(os.path.join(seg_old, s, 'AxT1enhanced', 'mask_necrosis*'))
    print(necrosis)
    enh = glob(os.path.join(seg_old, s, 'AxT1enhanced', 'mask_enh*'))
    print(enh)
    ind_path = os.path.join(pnew, s)
    seg_dir = os.path.join(ind_path, mod)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    if len(edema) != 0 : 
        if len(edema) > 1 :
            n=0
            for e in edema : 
                n = re.search(r"[0-9]+", e.split('/')[10])
                new_ed = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_edema', num=n.group(0), ext='nii.gz')
                print(e)
                print(new_ed)
                copy2(e, new_ed)
                #print(new_ed)
        else : 
            new_ed = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_edema', num=1, ext='nii.gz')
            print(edema[0])
            print(new_ed)
            copy2(edema[0], new_ed)
            #print(new_ed)
    if len(lesion) != 0 :
        if len(lesion) > 1 :
            n=0
            for l in lesion : 
                n = re.search(r"[0-9]+", l.split('/')[10])
                new_le = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_lesion', num=n.group(0), ext='nii.gz')
                copy2(l, new_le)
                print(l)
                print(new_le)
        else :
            new_le = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_lesion', num=1, ext='nii.gz')
            copy2(lesion[0], new_le)
            print(lesion[0])
            print(new_le)
    if len(necrosis) != 0 :
        if len(necrosis) > 1 :
            n=0
            for ne in necrosis : 
                n = re.search(r"[0-9]+", ne.split('/')[10])
                new_ne = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_necrosis', num=n.group(0), ext='nii.gz')
                copy2(ne, new_ne)
                print(ne)
                print(new_ne)
        else :
            new_ne = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_necrosis', num=1, ext='nii.gz')
            copy2(necrosis[0], new_ne)
            print(necrosis[0])
            print(new_ne)
    if len(enh) != 0 :
        if len(enh) > 1 :
            n=0
            for en in enh : 
                n = re.search(r"[0-9]", en.split('/')[10])
                new_en = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_enh', num=n.group(0), ext='nii.gz')
                copy2(en, new_en)
                print(en)
                print(new_en)
        else :
            new_en = "{path}/{pt}_{model}_{tp}_{num}.{ext}".format(path=seg_dir, 
        pt=s, model=mod, tp='mask_enh', num=1, ext='nii.gz')
            copy2(enh[0], new_en)
            print(enh[0])
            print(new_en)
    