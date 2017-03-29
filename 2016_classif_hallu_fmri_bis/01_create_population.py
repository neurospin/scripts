# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:59:07 2016

@author: ad247405
"""


import os
import numpy as np
from scipy import ndimage
import os, os.path, sys
import pylab
import pandas as pd 
import nibabel as nib
import glob

BASE_PATH =  '/neurospin/brainomics/2016_classif_hallu_fmri_bis'
#in_file = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/preproc/lists_subjects/list_subjects_path.txt'
in_file = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/preproc/preproc_26_oct/list_subjects_path.txt'
out_file = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/population26oct.txt'

demographic_file = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/data/Epidemio_46sujets_MULTIMODHAL_012017.xlsx"
demo = pd.read_excel(demographic_file)

inf=open(in_file, "r")
outf=open(out_file, "w")
outf.write("subject_name"+" "+"subject_folder"+" "+"data_path"+" "+"state_path"+" "+"age"+" "+"sex"+"\n")
outf.flush()

list_age = []
list_sex = []

for subject_folder in inf.readlines():
    #Name
    subject_folder = subject_folder.replace("\n","")
    subject_name = os.path.basename(subject_folder)
    print(subject_name)
    age = demo[demo.INCL==subject_name]["Age"].tolist()[0]
    sex = demo[demo.INCL==subject_name]["sexe_num"].tolist()[0]
    list_age.append(age)
    list_sex.append(sex)
    
    #State
    labels_file = glob.glob(subject_folder+'/labels/'+"*.csv")[0]
    labels_table = pd.read_csv(labels_file)
    labels = labels_table.Period
    labels = labels.as_matrix()
    labels_path = os.path.join(subject_folder+'/labels','labels.npy')
    np.save(labels_path,labels)

    #data

    list_files = glob.glob(subject_folder+'/scans/T1_VBM'+"/w*presto*")
    if not list_files:
        list_files = glob.glob(subject_folder+'/scans/T1_VBM'+"/w*PRESTO*")
    data_path = list_files[0]    
    outf.write(subject_name+" "+
          subject_folder+" "+data_path+" "+labels_path+" "+str(age)+" "+str(sex)+"\n")
    outf.flush()

outf.close()