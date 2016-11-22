# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:08:44 2016

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

in_file = '/neurospin/abide/schizConnect/completed_schizconnect_metaData_1829.csv'
images = pd.read_csv(in_file,delimiter=',')

new_directory = '/neurospin/abide/schizConnect/data_nifti_format'

for i in range(len(images)):
        if images.study[i] == 'fBIRNPhaseII__0010':
            study_path = os.path.join(new_directory,'fBIRN')
        if images.study[i] == 'NUSDAST':
            study_path = os.path.join(new_directory,'NUSDAST')   
        if images.study[i] == 'NMorphCH':
            study_path = os.path.join(new_directory,'NMorphCH')  
        if images.study[i] == 'MCICShare':
            study_path = os.path.join(new_directory,'MCICShare')    
        if images.study[i] == 'COBRE':
            study_path = os.path.join(new_directory,'COBRE')                
        try:
            os.stat(study_path)
        except:
            os.mkdir(study_path)
            
        site_path = os.path.join(study_path,images.imaging_protocol_site[i])    
        try:
            os.stat(site_path)
        except:
            os.mkdir(site_path)
            
        subject_new_path = os.path.join(site_path,images.subjectid[i])
        try:
            os.stat(subject_new_path)
        except:
            os.mkdir(subject_new_path) 
            
        subject_visit_path = os.path.join(subject_new_path,images.img_date[i][:10])
        try:
            os.stat(subject_visit_path)
        except:
            os.mkdir(subject_visit_path) 
            
        #case of multiple acquisitiion on the same day. Createsubfolders            
        if images.study[i] == 'NUSDAST':
            sequence_acq_path = os.path.join(subject_visit_path,images.notes[i])
            try:
                os.stat(sequence_acq_path)
            except:
                os.mkdir(sequence_acq_path) 
            #copy analyze files (both img and hdr)             
            copy_cmd_img = "cp "+ images.datauri[i] +'/*.img ' + sequence_acq_path
            os.system(copy_cmd_img)
            copy_cmd_hdr = "cp "+ images.datauri[i] +'/*.hdr ' + sequence_acq_path
            os.system(copy_cmd_hdr)
            #convert in nifti
            convert_cmd = "fsl5.0-fslchfiletype NIFTI " + glob.glob(sequence_acq_path +'/*.img')[0]
            os.system(convert_cmd)
   
            
                
        if images.study[i] == 'NMorphCH':
            sequence_acq_path = os.path.join(subject_visit_path,'MPRAGE_'+images.notes[i])
            try:
                os.stat(sequence_acq_path)
            except:
                os.mkdir(sequence_acq_path) 
                
            #copy dicom files (.dcm)             
            #copy_cmd_dcm = "cp "+ images.datauri[i] +'*.dcm ' + sequence_acq_path
            #convert dicom  in nifti    
            convert_cmd = "dinifti " + images.datauri[i] + '*.dcm '+ sequence_acq_path+'/'+ images.img_date[i][:10]+'_'+images.subjectid[i]+'.nii'
            os.system(convert_cmd)
            
            
        if images.study[i] == 'COBRE':
            sequence_acq_path = os.path.join(subject_visit_path,os.path.basename(glob.glob(images.datauri[i]+'/*dcm')[0])[0:12])
            try:
                os.stat(sequence_acq_path)
            except:
                os.mkdir(sequence_acq_path)  
            #copy dicom files (.dcm)             
            copy_cmd_dcm = "cp "+ images.datauri[i] +'/*.dcm ' + sequence_acq_path
            os.system(copy_cmd_dcm)
            #convert dicom  in nifti       
            convert_cmd = "dcm2nii " +  sequence_acq_path + '/*.dcm'
            os.system(convert_cmd)
            os.chdir(sequence_acq_path)
            os.system("rm -rf co*nii.gz")
            os.system("rm -rf o*nii.gz")
            os.system("rm *.dcm -rf")

 
        if images.study[i] == 'fBIRNPhaseII__0010':
            sequence_acq_path = os.path.join(subject_visit_path,os.path.basename(images.datauri[i]))
            try:
                os.stat(sequence_acq_path)
            except:
                os.mkdir(sequence_acq_path)  
            #copy analyze files (both img and hdr)             
            copy_cmd_img = "cp "+ images.datauri[i] +'/*.img ' + sequence_acq_path
            os.system(copy_cmd_img)
            copy_cmd_hdr = "cp "+ images.datauri[i] +'/*.hdr ' + sequence_acq_path
            os.system(copy_cmd_hdr)    
            
            convert_cmd = "fsl5.0-fslchfiletype NIFTI " + glob.glob(sequence_acq_path +'/*.img')[0]
            os.system(convert_cmd)
            os.chdir(sequence_acq_path)
            os.system("rm *.hdr -rf")
            os.system("rm *.img -rf")
            
        print i     
                
                

    