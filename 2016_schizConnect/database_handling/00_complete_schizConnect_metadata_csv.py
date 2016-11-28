# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:37:23 2016

@author: ad247405
"""

import re
import glob
import os
import nibabel as nibabel
import numpy as np
import os
import pandas as pd

BASE_PATH =  '/neurospin/abide/schizConnect'

in_file = '/neurospin/abide/schizConnect/schizconnect_metaData_1829.csv'
out_file = '/neurospin/abide/schizConnect/completed_schizconnect_metaData_1829.csv'
outf = open(out_file, "w")
outf.write("study"+","+"subjectid"+","+"age"+","+"sex"+","+"dx"+","+"field_strength"+","+"img_date"+","+"datauri"+","+"maker"+","+"model"+","+"szc_protocol_hier"+","+"notes"+","+"imaging_protocol_site\n")
outf.flush()

images = pd.read_csv(in_file,delimiter=',')
n=0
id = 0
for i in range(len(images)):
        if images.study[i] == 'fBIRNPhaseII__0010':
            fBIRN_data_path = '/neurospin/abide/schizConnect/data/schizconnect_fBIRNPhaseII__0010_images_1829.7z.001_FILES'
            new_path = os.path.join(fBIRN_data_path,images.datauri[i]) 
            
        if images.study[i] == 'NUSDAST':
            NUSDAST_data_path = '/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_images_1829.7z.001_FILES'
            subject_directory = os.path.split(os.path.dirname(images.datauri[i]))[0]
            path = os.path.join(NUSDAST_data_path,subject_directory) 
            path = glob.glob(path+'/*')[0]
            new_path = os.path.join(path,"scans",os.path.split(images.datauri[i])[1]) +'-T1/resources/ANALYZE/files/'
            
        if images.study[i] == 'NMorphCH':
            NMorphCH_data_path = '/neurospin/abide/schizConnect/data/schizconnect_NMorphCH_images_1829.7z.001_FILES'
            subject_directory = os.path.split(os.path.dirname(images.datauri[i]))[0]            
            path = os.path.join(NMorphCH_data_path,subject_directory) 
            path = glob.glob(path+'/*')[0]
            new_path = os.path.join(path,"scans",os.path.split(images.datauri[i])[1]) +'-T1_MPR/resources/DICOM/files/'
            
        if images.study[i] == 'MCICShare':
            MCICShare_data_path = '/neurospin/abide/schizConnect/data/schizconnect_MCICShare_images_1829.7z.001_FILES'
            new_path = os.path.join(MCICShare_data_path,images.datauri[i])    
        
        if images.study[i] == 'COBRE':
            COBRE_data_path = '/neurospin/abide/schizConnect/data/schizconnect_COBRE_images_1829.7z.001_FILES'
            new_path = os.path.join(COBRE_data_path,images.datauri[i])
            
        if len(glob.glob(new_path+'/*img')) or len(glob.glob(new_path+'/*dcm'))  > 0:                
            outf.write(images.study[i]+","+images.subjectid[i]+","+str(images.age[i])+","+images.sex[i]+","+images.dx[i]+","+ str(images.field_strength[i])+","+images.img_date[i]+","+ new_path +","+ images.maker[i]+","+images.model[i]+","+ images.szc_protocol_hier[i]+","+ str(images.notes[i]) +","+ images.imaging_protocol_site[i]+"\n")
            outf.flush()
            n = n+1    
        else :
            print new_path    
outf.close()
assert n==3669


print "number of subject in fBIRN: ", len(images[images.study == 'fBIRNPhaseII__0010'].subjectid.unique())
print "number of subject in NUSDAST: ",len(images[images.study == 'NUSDAST'].subjectid.unique())
print "number of subject in NMorphCH: ",len(images[images.study == 'NMorphCH'].subjectid.unique())
print "number of subject in MCICShare: ",len(images[images.study == 'MCICShare'].subjectid.unique())
print "number of subject in COBRE: ",len(images[images.study == 'COBRE'].subjectid.unique())


#images[images.study == 'MCICShare'].imaging_protocol_site.unique()