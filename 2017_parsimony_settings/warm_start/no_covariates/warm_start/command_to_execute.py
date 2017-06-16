#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:49:48 2017

@author: ad247405
"""

import os
import glob
import subprocess
while True:

    BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/all/all/*/beta.npz"
    p = glob.glob(BETA_START_PATH)
    if( len(p) == 8):
        print("Start of warm restart on all/all")
        os.chdir("/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start")
        cmd = "python NUDAST_30yo_VBM_all_all_as_start_vector.py"
        if not os.path.exists("os.path.join(/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/all_all_as_start_vector"):
            os.system(cmd)
            print("Config file of all/all correctly created")
        os.chdir("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/all_all_as_start_vector")
        cmd_map = "mapreduce.py config_dCV.json  --map --ncore 6"
        os.system(cmd_map)

    BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv00/all/*/beta.npz"
    p = glob.glob(BETA_START_PATH)
    if(len(p)==8):
        print("Start of warm restart on cv00/all")
        os.chdir("/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start")
        cmd = "python NUDAST_30yo_VBM_cv00_all_as_start_vector.py"
        if not os.path.exists("os.path.join(/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/cv00_all_as_start_vector"):
            os.system(cmd)
        print("Config file of cv00/all correctly created")
        os.chdir("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/cv00_all_as_start_vector")
        cmd_map = "mapreduce.py config_dCV.json  --map --ncore 6"
        subprocess.call(cmd_map)

    BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv01/all/*/beta.npz"
    p = glob.glob(BETA_START_PATH)
    if(len(p)==8):
        print("Start of warm restart on cv01/all")
        os.chdir("/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start")
        cmd = "python NUDAST_30yo_VBM_cv01_all_as_start_vector.py"
        if not os.path.exists("os.path.join(/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/cv01_all_as_start_vector"):
            os.system(cmd)
        print("Config file of cv01/all correctly created")
        os.chdir("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/cv01_all_as_start_vector")
        cmd_map = "mapreduce.py config_dCV.json  --map --ncore 6"
        subprocess.call(cmd_map)

    BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv02/all/*/beta.npz"
    p = glob.glob(BETA_START_PATH)
    if(len(p)==8):
        print("Start of warm restart on cv02/all")
        os.chdir("/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start")
        cmd = "python NUDAST_30yo_VBM_cv02_all_as_start_vector.py"
        if not os.path.exists("os.path.join(/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/cv02_all_as_start_vector"):
            os.system(cmd)
        print("Config file of cv02/all correctly created")
        os.chdir("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/cv02_all_as_start_vector")
        cmd_map = "mapreduce.py config_dCV.json  --map --ncore 6"
        subprocess.call(cmd_map)

    BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv03/all/*/beta.npz"
    p = glob.glob(BETA_START_PATH)
    if(len(p)==8):
        print("Start of warm restart on cv03/all")
        os.chdir("/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start")
        cmd = "python NUDAST_30yo_VBM_cv03_all_as_start_vector.py"
        if not os.path.exists("os.path.join(/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/cv03_all_as_start_vector"):
            os.system(cmd)
        print("Config file of cv03/all correctly created")
        os.chdir("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/cv03_all_as_start_vector")
        cmd_map = "mapreduce.py config_dCV.json  --map --ncore 6"
        subprocess.call(cmd_map)

    BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/no_warm_restart/model_selectionCV/cv04/all/*/beta.npz"
    p = glob.glob(BETA_START_PATH)
    if(len(p)==8):
        print("Start of warm restart on cv04/all")
        os.chdir("/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start")
        cmd = "python NUDAST_30yo_VBM_cv04_all_as_start_vector.py"
        if not os.path.exists("os.path.join(/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/cv04_all_as_start_vector"):
            os.system(cmd)
        print("Config file of cv04/all correctly created")
        os.chdir("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_covariates/warm_restart/cv04_all_as_start_vector")
        cmd_map = "mapreduce.py config_dCV.json  --map --ncore 6"
        subprocess.call(cmd_map)


    print ("end")

