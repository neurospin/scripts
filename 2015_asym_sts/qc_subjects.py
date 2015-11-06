# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:50:33 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
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
Then, we get rid of outliers, that is we drop subjects for whom more than
25% of the remaining robust sulci have not been detected.
Finally, we eliminate subjects for whom at least one measure is aberrant,
that is we filter subjects whose features lie outside the interval
' mean +/- 3 * sigma '.

"""

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from genibabel import imagen_subject_ids
import pheno

def _get_qc_sulci(sulci_path, out_path):
    """ Function to get the information on sulci
    
    """
    FULL_SULCI_PATH = sulci_path
    OUT_PATH =out_path
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
    
    print ('1) If the sulcus has not been recognized in 25% of the subjects, '
           'we do not take it into account.')
    
    # Initialize dataframe that will contain data from all .csv sulci files
    all_sulci_df = None
    # Iterate along sulci files
    sulcus_discarded = []
    for i, s in enumerate(sulci_file_list):
        sulc_name = s[83:-4]
    
        # Read each .csv file
        sulci_df = pd.io.parsers.read_csv(os.path.join(FULL_SULCI_PATH, s),
                                          sep=';',
                                          index_col=0,
                                          usecols=np.hstack(('subject',
                                                             features)))
    
        # If the sulcus has not been recognized in 25% of the subjects, we do
        # not take it into account.
        # In this case the first quartile of the surface columns will be 0.
        surface_first_quart = sulci_df['surface'].describe()['25%']
        if (surface_first_quart == 0):
            print "Sulcus", sulc_name, "is not recognized in more than 25% of subjects. Ignore it."
            sulcus_discarded.append(sulc_name)
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
                                          
    # Keep name of index column
    all_sulci_df.index = ['%012d' % int(i) for i in all_sulci_df.index]
    all_sulci_df.index.name = 'subjects_id'
    
    # Drop rows that have any NaN values
    all_sulci_df = all_sulci_df.dropna()

    # Save this dataframe as a .csv file
    all_sulci_df.to_csv(os.path.join(OUT_PATH, 'Q1_sulci_df.csv'))
    print "Original dataframe has been saved."

    return all_sulci_df, sulcus_discarded

def _get_sulci_for_subject_with_genetics(sulci_df) :
    """Function that keep only information for subject with genetic avail.
    this information is requested against the imagen2 server
    
    """
    # Consider subjects for who we have neuroimaging and genetic data
    # To fix genibabel should offer a iid function -direct request to server
    login = json.load(open(os.environ['KEYPASS']))['login']
    passwd = json.load(open(os.environ['KEYPASS']))['passwd']
    #Set the data set of interest ("QC_Genetics", "QC_Methylation" or "QC_Expression")
    data_set = "QC_Genetics"
    # Since login and password are not passed, they will be requested interactily
    subject_ids = imagen_subject_ids(data_of_interest=data_set, login=login,
                                     password=passwd)
#    subjects_id = imagen_genotype_measure(login, passwd,
#                                       gene_names=['NRXN3']).iid.tolist()
    subject_ids = set(subject_ids).intersection(set(sulci_df.index.tolist()))
    print 'DBG> ', 'difference between subject with genetics and sulci', \
            set(subject_ids).issubset(set(sulci_df.index.tolist()))

    return sulci_df.loc[list(subject_ids)]

def _get_sulci_qc_subject(sulci_data_df, out_path, percent_tol=0.03):
    """Perform QC on the subjects of the dataframe
    """
    # Get rid of outliers
    # criterium 1: eliminate subjects for whom more than 25% sulci have not
    # been detected
    OUTPATH =out_path
    print "2) Eliminate subjects for whom more than 25% sulci have not been detected"
    QC_subject_df = sulci_data_df.T.describe().T
    subject_mask = (QC_subject_df['25%'] == 0.0)
    sulci_data_df1 = sulci_data_df.loc[~subject_mask]
    print "Removing subjects:", sulci_data_df.loc[subject_mask].index.values

    # criterium 2: eliminate subjects for whom at least one measure is aberrant
    # Filter subjects whose features lie outside the interv mean +/- 3 * sigma
    print "3) Eliminate subjects for whom at least one measure is aberrant"
    colnames = sulci_data_df1.columns.tolist()
    
    #        opening_mean = sulci_data_df1[c].describe()['mean']
    #        opening_std = sulci_data_df1[c].describe()['std']
    num_features = len(colnames)
    print "Nb features:" + str(num_features)
    opening_mean = np.mean(sulci_data_df, axis=0)
    opening_std = np.std(sulci_data_df, axis=0)
    h = np.asarray(sulci_data_df) > \
        (opening_mean + 3 * opening_std).reshape(-1, num_features)
    b = np.asarray(sulci_data_df) < \
        (opening_mean - 3 * opening_std).reshape(-1, num_features)
    h = np.sum(h, axis=1) > (percent_tol * num_features)
    b = np.sum(b, axis=1) > (percent_tol * num_features)
    to_drop_index = h | b
#    print "DBG> ", sum(to_drop_index)

    # Keep subjects whose sulci features have been well recognized
    sulci_data_qc_df = sulci_data_df[~to_drop_index]
    print "Removing subjects:", sulci_data_df.loc[to_drop_index].index.values
    
    # Write quality control results in a single csv file for all features of all
    # sulci
    subject_with_sulci_qc = pd.DataFrame(dict(IID=['%012d' % int(i) for i in sulci_data_qc_df.index.tolist()]))
    subject_with_sulci_qc.to_csv(os.path.join(OUTPATH, 'Q2_sulci_df_qc.csv'),
                            index=False)
    print "Dataframe containing sulci data after quality control has been saved."
    return sulci_data_qc_df


def qc_sulci_qc_subject(percent_tol=0.03):
    """ Function to get information on the sulci with
    sulci   must be recognized at least 25 % among other sulci.
    subject must have at least 75 percent of sulci recognized
            with feature val that not oversized mean+3sd more that percent_tol
    """
    # Pathnames
    sulci_path = ('/neurospin/brainomics/2013_imagen_bmi/data/'
                  'Imagen_mainSulcalMorphometry/full_sulci/')
    out_path = '/neurospin/brainomics/2015_asym_sts/data'
    #
    sulci_dataframe, sulcus_discarded = _get_qc_sulci(sulci_path, out_path)
    #
    sulci_dataframe = _get_sulci_for_subject_with_genetics(sulci_dataframe)
    #
    sulci_dataframe = _get_sulci_qc_subject(sulci_dataframe, out_path, percent_tol)

    sulci_dataframe = pheno.fix_fid_iid_from_index(sulci_dataframe)
    return sulci_dataframe, sulcus_discarded

# Run
#########################################
if __name__ == "__main__":
    # Pathnames
    sulci_path = ('/neurospin/brainomics/2013_imagen_bmi/data/'
                  'Imagen_mainSulcalMorphometry/full_sulci/')
    out_path = '/neurospin/brainomics/2015_asym_sts/data2/'

    #
    sulci_dataframe = _get_qc_sulci(sulci_path, out_path)
    print sulci_dataframe.shape

    #
    sulci_dataframe = _get_sulci_for_subject_with_genetics(sulci_dataframe)
    print sulci_dataframe.shape

    #
    sulci_dataframe = _get_sulci_qc_subject(sulci_dataframe, out_path, percent_tol=0.03)    
    print sulci_dataframe.shape
    
    sulci_dataframe = pheno.fix_fid_iid_from_index(sulci_dataframe)

## Control plot
#pd.DataFrame.boxplot(sulci_data_qc_df)
#from scipy.spatial.distance import squareform, pdist
#import matplotlib.pylab as plt
#D = squareform(pdist(sulci_data_qc_df.values))
#plt.matshow(D)
#plt.colorbar()
#plt.show()
