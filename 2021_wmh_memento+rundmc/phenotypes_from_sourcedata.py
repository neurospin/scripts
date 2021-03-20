#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:16:46 2021

@author: ed203246

Data:
cd /home/ed203246/data/2021_wmh_memento+rundmc
"""

import os.path
import numpy as np
import pandas as pd
import seaborn as sns

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
memento_tivo_filename = os.path.join(MEMENTO_PATH, "../../sourcedata/tissues_volumes_spm12Segment_M000.csv")
os.listdir(MEMENTO_MODEL)

#%% RUNDMC
RUNDMC_PATH = "{FS}/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca".format(FS=FS)
RUNDMC_DATA = os.path.join(RUNDMC_PATH, "data")
RUNDMC_MODEL = os.path.join(RUNDMC_PATH, "models/pca_enettv_0.000035_1.000_0.005")
os.listdir(RUNDMC_MODEL)

########################################################################################################################
# MEMENTO

phenotypes_filename = os.path.join(MEMENTO_PATH , "population.csv")
phenotypes = pd.read_csv(phenotypes_filename)
assert phenotypes.shape == (1755, 26)

tivo = pd.read_csv(memento_tivo_filename)
tivo = tivo.rename(columns={"subject":'participant_id'})
assert tivo.shape == (2185, 5)
phenotypes = pd.merge(phenotypes, tivo)
assert phenotypes.shape == (1755, 30)

phenotypes["brain_volume"] = phenotypes["grey"] + phenotypes["white"]

# Sex mapping is [1, 2]
phenotypes[["sex", "tiv"]].groupby("sex").mean()
#             tiv
# sex
# 1    1509.414611
# 2    1313.277454
# => 1:M 2:F
phenotypes["sex"] = phenotypes["sex"].map({1:0, 2:1})

"""
phenotypes.columns

Index(['participant_id', 'nii_path', 'siteid', 'sex', 'age_cons', 'dmdip',
       'bac', 'apoe_eps4', 'mci', 'mmssctot', 'cdrscr', 'risctotrl',
       'risctotim', 'reymem3sc', 'flu_p', 'flu_anim', 'tmta_taux', 'tmtb_taux',
       'eavmem', 'eavatt', 'eavlang', 'npi_score_clin_d', 'statut', 'py',
       'etiol', 'etiol_pres', 'grey', 'white', 'csf', 'tiv'],
      dtype='object')


Je ne connais pas la nomenclature de Memento.
Est-ce que le tmta est en secondes ? de l’ordre de 30 à 180 ?

Pour les fluences oui c’est bien ça.

Pour la mémoire je ne sais pas. Tu n’as aucun moyen de vérifier ? Si tu fais ça dans les règles, tu peux contacter Vincent Bouteloup qui bosse avec Carole et qui pourra t’aider. Carole est au courant de l’étude que nous faisons.

"""
# Education
# bac = 5
sns.violinplot(x="bac", y="dmdip", data=phenotypes)
# education => dmdip
d = phenotypes.describe()

# Memory ?
sns.regplot(x='mmssctot', y='eavmem', data=phenotypes)
sns.violinplot(x="mmssctot", y="eavmem", data=phenotypes)

MEMENTO_MAPPING = dict(participant_id="participant_id",
                       sex='sex',
                       age_cons='age',
                       mmssctot="MMSE", # OK
                       tmta_taux="processing_speed",
                       flu_p="executive_functions", # OK Eric
                       # ?? = "memory",
                       dmdip = "education", # To verify
                       brain_volume = "brain_volume",
                       tiv = "tiv",
                       # ??:'lacune_nb',
                       # ??: 'mb_nb'
                       )
phenotypes = phenotypes.rename(columns=MEMENTO_MAPPING)

participants_with_phenotypes_filename = os.path.join(MEMENTO_PATH , "population_with_phenotypes.csv")
phenotypes.to_csv(participants_with_phenotypes_filename, index=False)
#    • C1 vs memory (FCSRT MEMENTO, compound RUNDMC)

########################################################################################################################
# RUNDMC

participants_filename = os.path.join(RUNDMC_DATA, "WMH_2006_participants.csv")
phenotypes_filename = os.path.join("{FS}/2019_rundmc_wmh/sourcedata/".format(FS=FS) , "Clinical_data_RUNDMC.xlsx")

participants = pd.read_csv(participants_filename)
participants.participant_id = participants.participant_id.astype(str)
assert participants.shape[0] == 267

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

RUNDMC_MAPPING = {'RUNDMC_ID':"participant_id",
                  'Age':'age',
                  'Sex(1=male)':'sex', # in [0. 1]
                  'MMSE':"MMSE",
                  'Processing_speed':"processing_speed",
                  'Executive_function':"executive_functions",
                  'Verbal_memory':"memory",
                  'Education':"education",
                  'Brain_volume':"brain_volume",
                  'ICV':"tiv",
                  'Lacune_nr':'lacune_nb',
                  'MB_nr':'mb_nb'
                  }
phenotypes = phenotypes.rename(columns=RUNDMC_MAPPING)
# Sex mapping is {1: "M", 0: "F"} => {1: "M", 0: "F"}
phenotypes["sex"] = phenotypes["sex"].map({1: "M", 0: "F"}).map({'M': 0, 'F': 1})
phenotypes.participant_id = phenotypes.participant_id.astype(str)

participants_with_phenotypes = pd.merge(participants, phenotypes, on='participant_id', how='inner')
assert participants_with_phenotypes.shape[0] == 267
assert np.all(participants.participant_id == participants_with_phenotypes.participant_id)

participants_with_phenotypes_filename = os.path.join(RUNDMC_DATA, "WMH_2006_participants_with_phenotypes.csv")
participants_with_phenotypes.to_csv(participants_with_phenotypes_filename, index=False)