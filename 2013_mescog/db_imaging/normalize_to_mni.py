#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Normalize rT1, LL, rFLAIR and WMH images into MNI.

See INPUT_DIR, OUTPUT_DIR
INPUT_DIR = "/neurospin/mescog/neuroimaging/original/munich"
OUTPUT_DIR = "/neurospin/mescog/neuroimaging/processed"
"""
import os, os.path
import glob
import subprocess
import shutil

INPUT_DIR = "/neurospin/mescog/neuroimaging/original/munich"
OUTPUT_DIR = "/neurospin/mescog/neuroimaging/processed"
TIMEPOINT = "M0"
fsl_mni_filepath = "/neurospin/mescog/neuroimaging/ressources/MNI152_T1_2mm.nii.gz"
fsl_warp_cmd = 'fsl5.0-applywarp'

subject_paths = glob.glob(os.path.join(INPUT_DIR, "CAD_bioclinica_nifti", "*"))
#subject_path = "/neurospin/mescog/neuroimaging/original/munich/CAD_bioclinica_nifti/1026"
# QC and warp into MNI
# ====================
#subject_path = subject_paths[0]
errors = list()
for subject_path in subject_paths:
    print subject_path
    #if not os.path.isdir(dir_path): continue:
    #, "*M0-WMH.nii.gz"))
    subject_id = os.path.basename(subject_path)
    subject_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "CAD_bioclinica_nifti", subject_id)
    if not os.path.exists(subject_OUTPUT_DIR):
        os.makedirs(subject_OUTPUT_DIR)
    # images input / output
    rt1_intput_filepath    = os.path.join(subject_path,       subject_id+"-"+TIMEPOINT+"-rT1.nii.gz")
    rt1_output_filepath    = os.path.join(subject_OUTPUT_DIR, subject_id+"-"+TIMEPOINT+"-rT1-MNI.nii.gz")
    ll_intput_filepath     = os.path.join(subject_path,       subject_id+"-"+TIMEPOINT+"-LL.nii.gz")
    ll_output_filepath     = os.path.join(subject_OUTPUT_DIR, subject_id+"-"+TIMEPOINT+"-LL-MNI.nii.gz")
    rflair_intput_filepath = os.path.join(subject_path,       subject_id+"-"+TIMEPOINT+"-rFLAIR.nii.gz")
    rflair_output_filepath = os.path.join(subject_OUTPUT_DIR, subject_id+"-"+TIMEPOINT+"-rFLAIR-MNI.nii.gz")
    wmh_intput_filepath    = os.path.join(subject_path,       subject_id+"-"+TIMEPOINT+"-WMH.nii.gz")
    wmh_output_filepath    = os.path.join(subject_OUTPUT_DIR, subject_id+"-"+TIMEPOINT+"-WMH-MNI.nii.gz")
    
    # Load transfo
    trm_basedir = os.path.join(INPUT_DIR, "Normalization", "CAD", subject_id)
    trm_rFLAIR_to_rT1_filepath = os.path.join(trm_basedir, "rFLAIR_to_rT1.mat")
    trm_rT1_to_MNI_filepath = os.path.join(trm_basedir, "rT1_to_MNI_warp.nii.gz")
    
    # rT1 to MNI
    cmd_rt1_to_mni = (fsl_warp_cmd, "-i", rt1_intput_filepath, "-o", rt1_output_filepath,
                      "-r", fsl_mni_filepath, "-w", trm_rT1_to_MNI_filepath)
    ret1 = subprocess.call(cmd_rt1_to_mni)
    
    # ll to MNI
    cmd_ll_to_mni = (fsl_warp_cmd, "-i", ll_intput_filepath, "-o", ll_output_filepath,
                     "-r", fsl_mni_filepath, "-w", trm_rT1_to_MNI_filepath)
    ret2 = subprocess.call(cmd_ll_to_mni)
    
    # rFLAIR to MNI
    cmd_rflair_to_mni = (fsl_warp_cmd, "-i", rflair_intput_filepath, "-o", rflair_output_filepath,
                         "-r", fsl_mni_filepath, "--premat="+trm_rFLAIR_to_rT1_filepath, "-w", trm_rT1_to_MNI_filepath)
    ret3 = subprocess.call(cmd_rflair_to_mni)
    
    # WMH to MNI
    cmd_wmh_to_mni = (fsl_warp_cmd, "-i", wmh_intput_filepath, "-o", wmh_output_filepath,
                      "-r", fsl_mni_filepath, "--premat="+trm_rFLAIR_to_rT1_filepath, "-w", trm_rT1_to_MNI_filepath)
    ret4 = subprocess.call(cmd_wmh_to_mni)
    if ret1 or ret2 or ret3 or ret4:
        shutil.rmtree(subject_OUTPUT_DIR)
        errors.append(errors)

print errors

#run scripts/2013_mescog/proj_wmh_patterns/00_build_dataset.py
#X = np.vstack(arr_list)
