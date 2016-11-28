# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:55:52 2016

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

population_file = '/neurospin/abide/schizConnect/schizconnect_metaData_1829.csv'

images = pd.read_csv(population_file,delimiter=',')
n=0
id = 0
for i in range(len(images)):
    if id != images.subjectid[i]:
         id = images.subjectid[i]
         n = n+1        
assert n==889        

########FBIRN, ONly 171 subject have a T1
##############################################################################

out_population_fBIRN_file = '/neurospin/abide/schizConnect/population_fBIRN.csv'
outf=open(out_population_fBIRN_file, "w")
outf.write("subejctid"+","+"age"+","+"sex"+","+"dx"+","+"path"+","+"scanner\n")
outf.flush()

fBIRN_data_path = '/neurospin/abide/schizConnect/data/schizconnect_fBIRNPhaseII__0010_images_1829.7z.001_FILES'
fBIRN_images = images[images.study == 'fBIRNPhaseII__0010']
n=0
id = 0
for i in (fBIRN_images.index):
    if id != fBIRN_images.subjectid[i]:
         id = fBIRN_images.subjectid[i]
         path = os.path.join(fBIRN_data_path,fBIRN_images.datauri[i])       
         if len(glob.glob(path+'/*.img')) >0:
             path = glob.glob(path+'/*.img')[0]
             outf.write(fBIRN_images.subjectid[i]+","+str(fBIRN_images.age[i])+","+fBIRN_images.sex[i]+","+fBIRN_images.dx[i]+","+path+","+ fBIRN_images.maker[i]+"\n")
             outf.flush()
             n = n+1  
assert n==171
outf.close()
pop = pd.read_csv(out_population_fBIRN_file,delimiter=',')
sum(pop.dx=='No_Known_Disorder')
sum(pop.dx=='Schizophrenia_Strict')
sum(pop.sex=='female')
sum(pop.sex=='male')
pop.age.mean()
pop.age.std()

##############################################################################


###########NUSDAST : 334 subjects : Some have only FLASH acq some only MPR
##############################################################################
out_population_NUSDAST_file = '/neurospin/abide/schizConnect/population_NUSDAST.csv'
outf=open(out_population_fBIRN_file, "w")
outf.write("subejctid"+","+"age"+","+"sex"+","+"dx"+","+"path_flash"+","+"path_mpr"+","+ "scanner\n")
outf.flush()
NUSDAST_data_path = '/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_images_1829.7z.001_FILES'
NUSDAST_images = images[images.study == 'NUSDAST']
NUSDAST_images.set_index
n=0
id = 0
c1=0
c2=0
for i in (NUSDAST_images.index):
    if id != NUSDAST_images.subjectid[i]:
         id = NUSDAST_images.subjectid[i]
         subject_directory = os.path.split(os.path.split(NUSDAST_images.datauri[i])[0])[0]
         path = os.path.join(NUSDAST_data_path,subject_directory) 
         if len(glob.glob(path+'/*/scans/FLASH1-T1/resources/ANALYZE/files/*.img')) >0:
             path_flash = glob.glob(path+'/*/scans/FLASH1-T1/resources/ANALYZE/files/*.img')[0]
             c1=c1+1
         if len(glob.glob(path+'/*/scans/MPR1-T1/resources/ANALYZE/files/*.img')) > 0:   
             path_mpr = glob.glob(path+'/*/scans/MPR1-T1/resources/ANALYZE/files/*.img')[0]
             c2=c2+1
         outf.write(NUSDAST_images.subjectid[i]+","+str(NUSDAST_images.age[i])+","+NUSDAST_images.sex[i]+","+NUSDAST_images.dx[i]+","+path_flash+","+path_mpr+","+ NUSDAST_images.maker[i]+"\n")
         outf.flush()
         n = n+1        
assert n==334
outf.close()
pop = pd.read_csv(out_population_NUSDAST_file,delimiter=',')
sum(pop.dx=='No_Known_Disorder')
sum(pop.dx=='Schizophrenia_Strict')
sum(pop.sex=='female')
sum(pop.sex=='male')
pop.age.mean()
pop.age.std()
##############################################################################




#####NMORPHCH, 88 images, DICOM FORMAT: Maybe should convert first
##############################################################################

out_population_NMorphCH_file = '/neurospin/abide/schizConnect/population_NMorphCH.csv'
outf=open(out_population_NMorphCH_file, "w")
outf.write("subejctid"+","+"age"+","+"sex"+","+"dx"+","+"path_flash"+","+"path_mpr"+","+ "scanner\n")
outf.flush()
NMorphCH_data_path = '/neurospin/abide/schizConnect/data/schizconnect_NMorphCH_images_1829.7z.001_FILES'
NMorphCH_images = images[images.study == 'NMorphCH']
NMorphCH_images.set_index
n=0
id = 0
for i in (NMorphCH_images.index):
     if id != NMorphCH_images.subjectid[i]:
         id = NMorphCH_images.subjectid[i]
         subject_directory = os.path.split(os.path.split(NMorphCH_images.datauri[i])[0])[0]
         path = os.path.join(NMorphCH_data_path,subject_directory)       
         if len(glob.glob(path+'/*/scans/3-T1_MPR/resources/DICOM/files/*.dcm')) >0:
             path_t1 = glob.glob(path+'/*/scans/3-T1_MPR/resources/DICOM/files/*.dcm')[0]
         if len(glob.glob(path+'/*/scans/4-T1_MPR/resources/DICOM/files/*.dcm')) >0:
             path_t1 = glob.glob(path+'/*/scans/4-T1_MPR/resources/DICOM/files/*.dcm')[0]    
         outf.write(NMorphCH_images.subjectid[i]+","+str(NMorphCH_images.age[i])+","+NMorphCH_images.sex[i]+","+NMorphCH_images.dx[i]+","+path_t1+","+NMorphCH_images.maker[i]+"\n")
         outf.flush()
         n = n+1  
assert n==88
outf.close()
pop = pd.read_csv(out_population_NMorphCH_file,delimiter=',')
sum(pop.dx=='No_Known_Disorder')
sum(pop.dx=='Schizophrenia_Strict')
sum(pop.sex=='female')
sum(pop.sex=='male')
pop.age.mean()
pop.age.std()

##############################################################################



##############################################################################

out_population_MCICShare_file = '/neurospin/abide/schizConnect/population_MCICShare.csv'
outf=open(out_population_MCICShare_file, "w")
outf.write("subejctid"+","+"age"+","+"sex"+","+"dx"+","+"path"+","+"path"+","+ "scanner\n")
outf.flush()
MCICShare_data_path = '/neurospin/abide/schizConnect/data/schizconnect_MCICShare_images_1829.7z.001_FILES'
MCICShare_images = images[images.study == 'MCICShare']
MCICShare_images.set_index
n=0
id = 0
for i in (MCICShare_images.index):
    if id != MCICShare_images.subjectid[i]:
         id = MCICShare_images.subjectid[i]
         path = os.path.join(MCICShare_data_path,MCICShare_images.datauri[i])       
         if len(glob.glob(path+'/*.dcm')) >0:
             path = glob.glob(path+'/*.dcm')[0]
             outf.write(MCICShare_images.subjectid[i]+","+str(MCICShare_images.age[i])+","+MCICShare_images.sex[i]+","+MCICShare_images.dx[i]+","+path+","+ MCICShare_images.maker[i]+"\n")
             outf.flush()
             n = n+1  
assert n== 95
pop = pd.read_csv(out_population_MCICShare_file,delimiter=',')
sum(pop.dx=='No_Known_Disorder')
sum(pop.dx=='Schizophrenia_Strict')
sum(pop.sex=='female')
sum(pop.sex=='male')
pop.age.mean()
pop.age.std()
##############################################################################


##############################################################################


###COBRE, DICOM Format
out_population_COBRE_file = '/neurospin/abide/schizConnect/population_COBRE.csv'
outf=open(out_population_COBRE_file, "w")
outf.write("subejctid"+","+"age"+","+"sex"+","+"dx"+","+"path"+","+"path"+","+ "scanner\n")
outf.flush()
COBRE_data_path = '/neurospin/abide/schizConnect/data/schizconnect_COBRE_images_1829.7z.001_FILES'
COBRE_images = images[images.study == 'COBRE']
COBRE_images.set_index
n=0
id = 0
for i in (COBRE_images.index):
    if id != COBRE_images.subjectid[i]:
         id = COBRE_images.subjectid[i]
         path = os.path.join(COBRE_data_path,COBRE_images.datauri[i])       
         if len(glob.glob(path+'/*.dcm')) >0:
             path = glob.glob(path+'/*.dcm')[0]
             outf.write(COBRE_images.subjectid[i]+","+str(COBRE_images.age[i])+","+COBRE_images.sex[i]+","+COBRE_images.dx[i]+","+path+","+COBRE_images.maker[i]+"\n")
             outf.flush()
             n = n+1  
assert n== 172
pop = pd.read_csv(out_population_COBRE_file,delimiter=',')
sum(pop.dx=='No_Known_Disorder')
sum(pop.dx=='Schizophrenia_Strict')
sum(pop.sex=='female')
sum(pop.sex=='male')
pop.age.mean()
pop.age.std()
##############################################################################
