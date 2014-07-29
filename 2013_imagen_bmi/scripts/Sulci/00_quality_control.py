# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:30:10 2014

@author: hl237680

Quality control on sulci data from the IMAGEN study.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry:
    one .csv file for each sulcus containing relevant features

OUTPUT:
- ~/gits/scripts/2013_imagen_bmi/scripts/Sulci/quality_control:
    sulci_df.csv: dataframe containing data for all sulci according to the
                  selected feature of interest.
                  We select subjects for who we have all genetic and
                  neuroimaging data.
                  We removed NaN rows.
    GM_thickness_qc.csv: statistics for each sulcus
    sulci_df_qc.csv: sulci_df.csv after quality control

The quality control first consists in removing sulci that are not recognized
in more than 25% of subjects.
Then, we get rid of outliers, that is we drop subjects for which more than
25% of the remaining robust sulci have not been detected.

"""

import os
import numpy as np
import pandas as pd
from glob import glob


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
SULCI_FILENAMES = os.listdir(SULCI_PATH)

OUTPUT_DIR = 'quality_control'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Sulci features of interest
#features = ['depthMean']
features = ['GM_thickness']
#features = ['opening']
#features = ['surface']
#features = ['depthMax']
#features = ['length']

# List all files containing information on sulci
sulci_file_list = []
for file in glob(os.path.join(SULCI_PATH, 'mainmorpho_*.csv')):
    sulci_file_list.append(file)

# Initialize dataframe that will contain data from all .csv sulci files
all_sulci_df = None
# Iterate along sulci files
for i, s in enumerate(sulci_file_list):
    sulc_name = s[83:-4]
    # Read each .csv file
    sulci_df = pd.io.parsers.read_csv(os.path.join(SULCI_PATH, s),
                                      sep=';',
                                      index_col=0)

    # If the sulcus has not been in 25% of the subjects, we don't take it
    # into account.
    # In this case the first quartile of the surface columns will be 0.
    surface_first_quart = sulci_df['surface'].describe()['25%']
    if (surface_first_quart == 0):
        print "Sulcus", sulc_name, "is not recognized in more than 25% of subjects. Ignore it."
    else:
        # Select column corresponding to features of interest
        recognized_sulci_df = sulci_df[features]

        # Rename columns according to the sulcus considered
        colname = ['.'.join((sulc_name, feature)) for feature in features]
        recognized_sulci_df.columns = colname
        if all_sulci_df is None:
            all_sulci_df = recognized_sulci_df
        else:
            all_sulci_df = all_sulci_df.join(recognized_sulci_df)

print "Loaded", all_sulci_df.shape[1], "sulci"

# Consider subjects for who we have neuroimaging and genetic data
subjects_id = np.genfromtxt(os.path.join(DATA_PATH,
                                         'subjects_id.csv'),
                            dtype=None,
                            delimiter=',',
                            skip_header=1)

sulci_data_df = all_sulci_df.loc[subjects_id]

# Drop rows that have any NaN values
sulci_data_df = sulci_data_df.dropna()

# Save this dataframe as a .csv file
sulci_data_df.to_csv((os.path.join(OUTPUT_DIR, 'sulci_df.csv')))

# Build a dataframe containing quality control parameters
QC_df = sulci_data_df.describe()
QC_df.to_csv(os.path.join(OUTPUT_DIR, 'GM_thickness_qc.csv'))
print "CSV file showing statistics for each sulcus has been saved."

# Get rid of outliers
# criterium: eliminate subjects for which more than 25% of sulci have not
# been detected
QC_subject_df = sulci_data_df.T.describe().T
subject_mask = (QC_subject_df['25%'] == 0.0)
print "Removing subjects:", sulci_data_df.loc[subject_mask].index.values

# Write results corresponding to this specific feature in a single csv file
# for all sulci
sulci_data_qc_df = sulci_data_df.loc[~subject_mask]
sulci_data_qc_df.to_csv((os.path.join(OUTPUT_DIR, 'sulci_df_qc.csv')))

# Control plot
pd.DataFrame.boxplot(sulci_data_qc_df)
from scipy.spatial.distance import squareform, pdist
import matplotlib.pylab as plt
D = squareform(pdist(sulci_data_qc_df.values))
plt.matshow(D)
plt.colorbar()
plt.show()