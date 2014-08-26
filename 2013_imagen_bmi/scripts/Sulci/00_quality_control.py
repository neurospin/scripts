# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:30:10 2014

@author: hl237680

Quality control on sulci data from the IMAGEN study.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/full_sulci/
    one .csv file containing relevant features for each reconstructed sulcus:
 'mainmorpho_S.T.s._right.csv',
 'mainmorpho_S.T.s._left.csv',
 'mainmorpho_S.T.pol._right.csv',
 'mainmorpho_S.T.pol._left.csv',
 'mainmorpho_S.T.i._right.csv',
 'mainmorpho_S.T.i._left.csv',
 'mainmorpho_S.s.P._right.csv',
 'mainmorpho_S.s.P._left.csv',
 'mainmorpho_S.Rh._right.csv',
 'mainmorpho_S.Rh._left.csv',
 'mainmorpho_S.R.inf._right.csv',
 'mainmorpho_S.T.s.ter.asc._right.csv',
 'mainmorpho_S.T.s.ter.asc._left.csv',
 'mainmorpho_S.R.inf._left.csv',
 'mainmorpho_S.Po.C.sup._right.csv',
 'mainmorpho_S.Po.C.sup._left.csv',
 'mainmorpho_S.Pe.C._right.csv',
 'mainmorpho_S.Pe.C._left.csv',
 'mainmorpho_S.Pa.t._right.csv',
 'mainmorpho_S.Pa.t._left.csv',
 'mainmorpho_S.Pa.sup._right.csv',
 'mainmorpho_S.Pa.sup._left.csv',
 'mainmorpho_S.Pa.int._right.csv',
 'mainmorpho_S.Pa.int._left.csv',
 'mainmorpho_S.p.C._right.csv',
 'mainmorpho_S.p.C._left.csv',
 'mainmorpho_S.Or._right.csv',
 'mainmorpho_S.Or._left.csv',
 'mainmorpho_S.Or.l._right.csv',
 'mainmorpho_S.Or.l._left.csv',
 'mainmorpho_S.Olf._right.csv',
 'mainmorpho_S.Olf._left.csv',
 'mainmorpho_S.O.T.lat._right.csv',
 'mainmorpho_S.O.T.lat._left.csv',
 'mainmorpho_S.O.p._right.csv',
 'mainmorpho_S.O.p._left.csv',
 'mainmorpho_S.Li._right.csv',
 'mainmorpho_S.Li._left.csv',
 'mainmorpho_S.F.sup._right.csv',
 'mainmorpho_S.F.sup._left.csv',
 'mainmorpho_S.F.polaire.tr._right.csv',
 'mainmorpho_S.F.polaire.tr._left.csv',
 'mainmorpho_S.F.orbitaire._right.csv',
 'mainmorpho_S.F.orbitaire._left.csv',
 'mainmorpho_S.F.median._right.csv',
 'mainmorpho_S.F.median._left.csv',
 'mainmorpho_S.F.marginal._right.csv',
 'mainmorpho_S.F.marginal._left.csv',
 'mainmorpho_S.F.inter._right.csv',
 'mainmorpho_S.F.inter._left.csv',
 'mainmorpho_S.F.int._right.csv',
 'mainmorpho_S.F.int._left.csv',
 'mainmorpho_S.F.inf._right.csv',
 'mainmorpho_S.F.inf._left.csv',
 'mainmorpho_S.Cu._right.csv',
 'mainmorpho_S.Cu._left.csv',
 'mainmorpho_S.Call._right.csv',
 'mainmorpho_S.Call._left.csv',
 'mainmorpho_S.C._right.csv',
 'mainmorpho_S.C.LPC._right.csv',
 'mainmorpho_S.C.LPC._left.csv',
 'mainmorpho_S.C._left.csv',
 'mainmorpho_OCCIPITAL_right.csv',
 'mainmorpho_OCCIPITAL_left.csv',
 'mainmorpho_INSULA_right.csv',
 'mainmorpho_INSULA_left.csv',
 'mainmorpho_F.P.O._right.csv',
 'mainmorpho_F.P.O._left.csv',
 'mainmorpho_F.I.P._right.csv',
 'mainmorpho_F.I.P.r.int.2_right.csv',
 'mainmorpho_F.I.P.r.int.2_left.csv',
 'mainmorpho_F.I.P.r.int.1_right.csv',
 'mainmorpho_F.I.P.r.int.1_left.csv',
 'mainmorpho_F.I.P.Po.C.inf._right.csv',
 'mainmorpho_F.I.P.Po.C.inf._left.csv',
 'mainmorpho_F.I.P._left.csv',
 'mainmorpho_F.Coll._right.csv',
 'mainmorpho_F.Coll._left.csv',
 'mainmorpho_F.Cal.ant.-Sc.Cal._right.csv',
 'mainmorpho_F.Cal.ant.-Sc.Cal._left.csv',
 'mainmorpho_F.C.M._right.csv',
 'mainmorpho_F.C.M._left.csv',
 'mainmorpho_F.C.L._right.csv',
 'mainmorpho_F.C.L.r.sc._right.csv',
 'mainmorpho_F.C.L.r.sc._left.csv',
 'mainmorpho_F.C.L.r._right.csv',
 'mainmorpho_F.C.L.r.retroC.tr._right.csv',
 'mainmorpho_F.C.L.r.retroC.tr._left.csv',
 'mainmorpho_F.C.L.r._left.csv',
 'mainmorpho_F.C.L._left.csv'

OUTPUT:
- ~/gits/scripts/2013_imagen_bmi/scripts/Sulci/quality_control:
    sulci_df.csv: dataframe containing data for all sulci according to the
                  selected feature of interest.
                  We select subjects for who we have all genetic and
                  neuroimaging data.
                  We removed NaN rows.
    sulci_df_qc.csv: sulci_df.csv after quality control

The quality control first consists in removing sulci that are not recognized
in more than 25% of subjects.
Then, we get rid of outliers, that is we drop subjects for which more than
25% of the remaining robust sulci have not been detected.
Finally, we eliminate subjects for whom at least one measure is aberrant,
that is we filter subjects whose features lie outside the interval
' mean +/- 3 * sigma '.

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
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')

QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')
if not os.path.exists(QC_PATH):
    os.makedirs(QC_PATH)

# Sulci features of interest
features = ['surface',
            'depthMax',
            'depthMean',
            'length',
            'GM_thickness',
            'opening']

# List all files containing information on sulci
sulci_file_list = []
for file in glob(os.path.join(FULL_SULCI_PATH, 'mainmorpho_*.csv')):
    sulci_file_list.append(file)

print "1) If the sulcus has not been recognized in 25% of the subjects, we do not take it into account."

# Initialize dataframe that will contain data from all .csv sulci files
all_sulci_df = None
# Iterate along sulci files
for i, s in enumerate(sulci_file_list):
    sulc_name = s[83:-4]
    # Read each .csv file
    sulci_df = pd.io.parsers.read_csv(os.path.join(SULCI_PATH, s),
                                      sep=';',
                                      index_col=0,
                                      usecols=np.hstack(('subject', features)))

    # If the sulcus has not been recognized in 25% of the subjects, we do
    # not take it into account.
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

print "Loaded", all_sulci_df.shape[1] / len(features), "sulci"


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
sulci_data_df.to_csv((os.path.join(QC_PATH, 'sulci_df.csv')))


# Get rid of outliers
# criterium 1: eliminate subjects for whom more than 25% sulci have not
# been detected
print "2) Eliminate subjects for whom more than 25% sulci have not been detected"
QC_subject_df = sulci_data_df.T.describe().T
subject_mask = (QC_subject_df['25%'] == 0.0)
sulci_data_df1 = sulci_data_df.loc[~subject_mask]
print "Removing subjects:", sulci_data_df.loc[subject_mask].index.values

# criterium 2: eliminate subjects for whom at least one measure is aberrant
# Filter subjects whose features lie outside the interval mean +/- 3 * sigma
print "3) Eliminate subjects for whom at least one measure is aberrant"
colnames = sulci_data_df1.columns.tolist()

hb_index = []
lb_index = []
for k in sulci_data_df.index:
    for c in colnames:
        opening_mean = sulci_data_df1[c].describe()['mean']
        opening_std = sulci_data_df1[c].describe()['std']
        higher_bound = opening_mean + 3 * opening_std
        lower_bound = opening_mean - 3 * opening_std
        if (sulci_data_df1[c][k] > higher_bound):
            hb_index.append(k)
        elif (sulci_data_df1[c][k] < lower_bound):
            lb_index.append(k)
hb_index_set = set(hb_index)
lb_index_set = set(lb_index)
# Union without common elements
index_set = hb_index_set ^ lb_index_set
masked_index_list = list(index_set)

# Keep subjects whose sulci features have been well recognized
sulci_data_qc_df = sulci_data_df1.drop(sulci_data_df1.loc[masked_index_list].
                                        index.values)
print "Removing subjects:", sulci_data_df1.loc[masked_index_list].index.values

# Write quality control results in a single csv file for all features of all
# sulci
sulci_data_qc_df.to_csv((os.path.join(QC_PATH, 'sulci_df_qc.csv')))

# Control plot
pd.DataFrame.boxplot(sulci_data_qc_df)
from scipy.spatial.distance import squareform, pdist
import matplotlib.pylab as plt
D = squareform(pdist(sulci_data_qc_df.values))
plt.matshow(D)
plt.colorbar()
plt.show()