#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:31:53 2016

@author: ad247405
"""


import glob
import os
import numpy as np
import os
import pandas as pd

BASE_PATH =  '/neurospin/abide/schizConnect'
DATA_PATH = os.path.join(BASE_PATH,"data_nifti_format")

in_file = os.path.join(BASE_PATH,"completed_schizconnect_metaData_1829.csv")
images = pd.read_csv(in_file,delimiter=',')
#Remove images from MCICShare study.
images = images[(images.study != "MCICShare")]

out_file = os.path.join(BASE_PATH,"list_nifti_images.csv")
outf = open(out_file, "w")
outf.write("study"+","+"subjectid"+","+"age"+","+"sex"+","+"dx"+","+"acquisition_number"+ \
","+"field_strength"+","+"img_date"+","+"path"+","+"maker"+","+"model"+","+"imaging_protocol_site\n")
outf.flush()

                                          
#
list_subjects = images.subjectid.unique()
number_subjects = len(images.subjectid.unique())
for i in range(number_subjects): 
    current = images[images.subjectid == list_subjects[i]].reset_index()
    print (i)
    if current.study[0] == 'fBIRNPhaseII__0010':
        visit_dates = current.img_date.unique()
        visit_path = glob.glob(os.path.join(DATA_PATH,"fBIRN",'*',current.subjectid[0],'*'))
        number_visits = len(visit_path)
        
        for j in range(number_visits):
            current_visit = current[current.img_date == visit_dates[j]].reset_index()
            paths = glob.glob(os.path.join(visit_path[j],'NIFTI','*.nii'))
            number_acquisitions = len(paths)

            for n in range(number_acquisitions):
                outf.write(current_visit.study[n]+","+current_visit.subjectid[n]+","\
                +str(current_visit.age[n])+","+current_visit.sex[n]+","+current_visit.dx[n]\
                +","+str(n+1)+","+ str(current_visit.field_strength[n])+","+current_visit.img_date[n]
                +","+ paths[n] +","+ current_visit.maker[n]+","+current_visit.model[n]+","+ \
                current_visit.imaging_protocol_site[n]+"\n")
                outf.flush()

          

    if current.study[0] == 'NUSDAST':
        visit_dates = current.img_date.unique()
        visit_path = glob.glob(os.path.join(DATA_PATH,'NUSDAST','*',current.subjectid[0],'*'))
        number_visits = len(visit_path)
        
        for j in range(number_visits):
            current_visit = current[current.img_date == visit_dates[j]].reset_index()
            paths = glob.glob(os.path.join(visit_path[j],'MPR*','*.nii'))
            if len(paths)==0:
                paths = glob.glob(os.path.join(visit_path[j],'FLASH*','*.nii'))
            number_acquisitions = len(paths)

            for n in range(number_acquisitions):
                outf.write(current_visit.study[n]+","+current_visit.subjectid[n]+","\
                +str(current_visit.age[n])+","+current_visit.sex[n]+","+current_visit.dx[n]\
                +","+str(n+1)+","+ str(current_visit.field_strength[n])+","+current_visit.img_date[n]
                +","+ paths[n] +","+ current_visit.maker[n]+","+current_visit.model[n]+","+ \
                current_visit.imaging_protocol_site[n]+"\n")
                outf.flush()

                  
    if current.study[0] == 'NMorphCH':
        visit_dates = current.img_date.unique()
        visit_path = glob.glob(os.path.join(DATA_PATH,'NMorphCH','*',current.subjectid[0],'*'))
        number_visits = len(visit_path)
        
        for j in range(number_visits):
            current_visit = current[current.img_date == visit_dates[j]].reset_index()
            paths = glob.glob(os.path.join(visit_path[j],'MPR*','*.nii'))
            number_acquisitions = len(paths)

            for n in range(number_acquisitions):
                outf.write(current_visit.study[n]+","+current_visit.subjectid[n]+","\
                +str(current_visit.age[n])+","+current_visit.sex[n]+","+current_visit.dx[n]\
                +","+str(n+1)+","+ str(current_visit.field_strength[n])+","+current_visit.img_date[n]
                +","+ paths[n] +","+ current_visit.maker[n]+","+current_visit.model[n]+","+ \
                current_visit.imaging_protocol_site[n]+"\n")
                outf.flush()

          

#CObre is more complicated since one original actually contained 5 T1. We need to take into account this.
    if current.study[0] == 'COBRE':
        print ("COBRE")
        visit_dates = current.img_date.unique()
        visit_path = glob.glob(os.path.join(DATA_PATH,'COBRE','*',current.subjectid[0],'*'))
        number_visits = len(visit_path)
        
        for j in range(number_visits):
            current_visit = current[current.img_date == visit_dates[j]].reset_index()
            paths = glob.glob(os.path.join(visit_path[j],'*MR*','*.nii*'))
            number_acquisitions = len(paths)

            for n in range(number_acquisitions):              
                outf.write(current_visit.study[0]+","+current_visit.subjectid[0]+","\
                +str(current_visit.age[0])+","+current_visit.sex[0]+","+current_visit.dx[0]\
                +","+str(n+1)+","+ str(current_visit.field_strength[0])+","+current_visit.img_date[0]
                +","+ paths[n] +","+ current_visit.maker[0]+","+current_visit.model[0]+","+ \
                current_visit.imaging_protocol_site[0]+"\n")
                outf.flush()    

          
        
         
    
   