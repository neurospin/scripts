# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 18:29:33 2014

@author: hl237680

Generation of various phenotype files from the dataframe on sulci maximal
depth obtained after quality control.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_depthMax_df.csv:
    sulci depthMax after quality control

OUTPUT:
As many .phe files as there are sulci who passed the QC (i.e. 85)
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_depthMax_beta_values_df.csv:
    Beta values from the General Linear Model run on sulci maximal depth.

"""


import os
import csv
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')

# Output results
OUTPUT_DIR = os.path.join(QC_PATH, 'depthMax_phenotype')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Sulci depthMax
sulci_depthMax_df = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                                    'sulci_depthMax_df.csv'),
                                            sep=',',
                                             index_col=0)

# Write .phe file corresponding to each sulcus
for sulci_column in sulci_depthMax_df.columns:
    # Pathname
    SULCI_PHENOTYPE = os.path.join(OUTPUT_DIR, '%s.phe' % (sulci_column))
    fp = open(SULCI_PHENOTYPE, 'wb')
    cw = csv.writer(fp, delimiter=' ')
    cw.writerow(['FID', 'IID', 'depthMax'])
    for i, s in enumerate(sulci_depthMax_df.index.tolist()):
        tmp = []
        # Family ID (FID)
        tmp.append('%012d' % (int(s)))
        # Individual ID (IID)
        tmp.append('%012d' % (int(s)))
        # BMI
        tmp.append('%.2f' % (sulci_depthMax_df[sulci_column].loc[s]))
        cw.writerow(tmp)
    fp.close()