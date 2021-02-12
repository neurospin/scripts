#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:16:46 2021

@author: ed203246

Data:
cd /home/ed203246/data/2021_wmh_memento+rundmc

libreoffice documents/MEMENTO-RUNDMC_WMH_PCATV_2021/PROJECT\ PROPOSAL\ 20201215.docx

2) Links between component values and clinical status (systematic adjustment for age and level of education) -> if difficulties with age, this should be extremely clearly explained+++
    • C1 vs MMSE
    • C1 vs processing speed (TMTA MEMENTO, compound score RUNDMC)
    • C1 vs executive functions (fluencies MEMENTO Eric check, OK RUNDMC)
    • C1 vs memory (FCSRT MEMENTO, compound RUNDMC)

    • C2 vs MMSE
    • C2 vs processing speed (TMTA MEMENTO, compound score RUNDMC)
    • C2 vs executive functions (fluencies MEMENTO, ??? RUNDMC)
    • C2 vs memory (FCSRT MEMENTO, ??? RUNDMC)


Model:
MMSE ~ PC1 + Age + sex + education + Brain atrophy + Lacune NB + MB
"""

import os.path
import pandas as pd

FS_NS = "/neurospin/brainomics"
FS_LOCAL = "/home/ed203246/data"

FS = FS_LOCAL

"""
Sync Clinical_data_RUNDMC.* to NS:

print("rsync -azvun {src} {dst}".format(
    src="{FS}/2019_rundmc_wmh/sourcedata/{fil}".format(FS=FS_LOCAL, fil="Clinical_data_RUNDMC.*"),
    dst="is234606.intra.cea.fr:{FS}/2019_rundmc_wmh/sourcedata/".format(FS=FS_NS))
    )

rsync -azvun /home/ed203246/data/2019_rundmc_wmh/sourcedata/Clinical_data_RUNDMC.* is234606.intra.cea.fr:/neurospin/brainomics/2019_rundmc_wmh/sourcedata/


"""

#%% MEMENTO
MEMENTO_PATH = "{FS}/2017_memento/analysis/WMH".format(FS=FS)
MEMENTO_DATA = os.path.join(MEMENTO_PATH, "data")
MEMENTO_MODEL = os.path.join(MEMENTO_PATH, "models/pca_enettv_0.000010_1.000_0.001")
os.listdir(MEMENTO_MODEL)


#%% RUNDMC
RUNDMC_PATH = "{FS}/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca".format(FS=FS)
RUNDMC_DATA = os.path.join(RUNDMC_PATH, "data")
RUNDMC_MODEL = os.path.join(RUNDMC_PATH, "models/pca_enettv_0.000035_1.000_0.005")
os.listdir(RUNDMC_MODEL)

#%% MEMENTO

phenotypes_filename = os.path.join(MEMENTO_PATH , "population.csv")
phenotypes = pd.read_csv(phenotypes_filename)
assert phenotypes.shape == (1755, 26)
phenotypes.columns

"""
phenotypes.columns
Out[15]:
Index(['participant_id', 'nii_path', 'siteid', 'sex', 'age_cons', 'dmdip',
       'bac', 'apoe_eps4', 'mci', 'mmssctot', 'cdrscr', 'risctotrl',
       'risctotim', 'reymem3sc', 'flu_p', 'flu_anim', 'tmta_taux', 'tmtb_taux',
       'eavmem', 'eavatt', 'eavlang', 'npi_score_clin_d', 'statut', 'py',
       'etiol', 'etiol_pres'],
      dtype='object')
"""

memento_mapping = dict("participant_id=participant_id",
                       mmssctot="MMSE",
                       tmta_taux="processing_speed",
                       flu_p="executive_functions",
                       memory="???"
                       )

    • C1 vs memory (FCSRT MEMENTO, compound RUNDMC)

#%% RUNDMC

participants_filename = os.path.join("{FS}/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca".format(FS=FS) , "participants.csv")
phenotypes_filename = os.path.join("{FS}/2019_rundmc_wmh/sourcedata/".format(FS=FS) , "Clinical_data_RUNDMC.xlsx")

participants = pd.read_csv(participants_filename)
participants.participant_id = participants.participant_id.astype(str)

phenotypes = pd.read_excel(phenotypes_filename)
assert phenotypes.shape == (503, 13)

"""
phenotypes.columns
RUNDMC_ID;Age;Sex(1=male);Education;Education_7levels;MMSE;Processing_speed;Executive_function;Verbal_memory;Lacune_nr;MB_nr;Brain_volume;ICV

Index(['RUNDMC_ID', 'Age', 'Sex(1=male)', 'Education', 'Education_7levels',
       'MMSE', 'Processing_speed', 'Executive_function', 'Verbal_memory',
       'Lacune_nr', 'MB_nr', 'Brain_volume', 'ICV'],
      dtype='object')
"""

rundmc_mapping = dict(RUNDMC_ID="participant_id",
                      MMSE="MMSE",
                      Processing_speed="processing_speed",
                      Executive_function="executive_functions",
                      memory="Verbal_memory"
)