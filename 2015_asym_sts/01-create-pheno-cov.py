"""
Created  01 12 2015

@author vf140245
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


# Pathnames
BASE_PATH = '/neurospin/brainomics/2015_asym_sts/'
DATA_PATH = os.path.join(BASE_PATH, 'data')

OUT_PATH = DATA_PATH

features = ['depthMax', ]
laterality = ['left', 'right']
all_sulci_df = None


# Load info from morphologist output.
for l in laterality:
    fpath = os.path.join(DATA_PATH, 'mainmorpho_S.T.s._%s.csv' % l)
    sulci_df = pd.io.parsers.read_csv(fpath,
                                      sep=';',
                                      index_col=0,
                                      usecols=np.hstack(('subject',
                                                         features)))
    colname = ['.'.join(("STs_%s" % l, feature)) for feature in features]
    sulci_df.columns = colname
    # a priori this sulci is always correctly recognized
    if all_sulci_df is None:
        all_sulci_df = sulci_df
    else:
        all_sulci_df = all_sulci_df.join(sulci_df)
print "Loaded", all_sulci_df.shape[1] / len(features), "sulci"

left = all_sulci_df[u'STs_left.depthMax']
right = all_sulci_df[u'STs_right.depthMax']
asym_index = 2 * (left - right) / (left + right)
fid_iid = ["%012d" % i for i in all_sulci_df.index]
asym_index_df = pd.DataFrame(asym_index, columns=['STs_asym'])
asym_index_df = asym_index_df.join(pd.DataFrame(dict(FID=fid_iid,
                                                     IID=fid_iid),
                                                index=all_sulci_df.index))


# Load info from demographics to get  cov
fpath = os.path.join(DATA_PATH, 'demographics.csv')
usecols = ['Subject', 'Gender from RecruitmentInfos',
           'Handedness from QualityReport',
           'ImagingCentreCity', ]
usecols = ['Subject', 'Gender from QualityReport',
           'Handedness from QualityReport',
           'ImagingCentreCity', ]
demog = pd.io.parsers.read_csv(fpath, sep=',', index_col=0,
                               usecols=np.hstack(usecols))
demog = demog.loc[demog.index <= 9999999999]  # filter out phantom and PSC1
#gender
gender = pd.get_dummies(demog['Gender from QualityReport'],
                        dummy_na=True)
gender.columns = gender.columns[:-1].tolist() + ['gender_nan']
gender['Female'] = gender['Female'] + gender['gender_nan'] * (-9)
gender = pd.DataFrame(gender['Female'], columns=['Female'])
#center
centre = pd.get_dummies(demog['ImagingCentreCity'],
                        dummy_na=True)
centre.columns = centre.columns[:-1].tolist() + ['centre_nan']
print '# of nan for centre: ', int(np.sum(centre.as_matrix()[:, -1]))
centre = centre.loc[:, centre.columns[:-2]]

#laterality
laterality = pd.get_dummies(demog['Handedness from QualityReport'],
                        dummy_na=True)
laterality = laterality.loc[:, laterality.columns[:-2]]
laterality.columns = laterality.columns[:-1].tolist() + ['laterality_nan']

fid_iid = ["%012d" % i for i in demog.index]
covar = pd.DataFrame(dict(FID=fid_iid, IID=fid_iid), index=demog.index)
covar = covar.join(gender).join(centre)
#covar = covar[demog['Handedness from QualityReport'] == 'Right']
covar.to_csv(os.path.join(OUT_PATH, 'sts_gender_centre.cov'),
                    sep='\t', index=False)


#Select
asym_index_df.to_csv(os.path.join(OUT_PATH, 'sts_asym.phe'),
                    sep='\t', index=False, cols=['FID', 'IID', 'STs_asym'])
asym_sub = \
   asym_index_df.loc[demog['Handedness from QualityReport'] == 'Right']
asym_sub.to_csv(os.path.join(OUT_PATH, 'sts_asym_rightonly.phe'),
                    sep='\t', index=False, cols=['FID','IID','STs_asym'])

plt.figure()
asym_index_df['STs_asym'].hist(bins=50)
plt.title('STs_asym all subjects: %d' % asym_index_df.shape[0])
plt.savefig(os.path.join(OUT_PATH, 'hist_sts.png'))

plt.figure()
asym_sub['STs_asym'].hist(bins=50)
plt.title('STs_asym righthanded subjects: %d' % asym_sub.shape[0])
plt.savefig(os.path.join(OUT_PATH, 'hist_sts_right.png'))

## Control plot
#pd.DataFrame.boxplot(asym_index_df)
#from scipy.spatial.distance import squareform, pdist
#import matplotlib.pylab as plt
#D = squareform(pdist(all_sulci_df.values))
#plt.matshow(D)
#plt.colorbar()
#plt.show()
#pval = pd.io.parsers.read_csv('/neurospin/brainomics/2015_asym_sts/data/distil.pval',
#                                      sep=' ',
#                                      index_col=0)