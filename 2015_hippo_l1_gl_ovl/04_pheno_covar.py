#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np
import pandas
from sklearn.linear_model import LinearRegression

GCTAOUT = '/volatile/frouin/baby_imagen/reacta/dataLinks'
#######################
# get Enigma2 dataset
#######################
fin = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
       'imagen_subcortCov_NP.csv')
df = pandas.DataFrame.from_csv(fin, sep=' ', index_col=False)
iid_fid = ["%012d" % int(i) for i in df['IID']]
iid_fid = pandas.DataFrame(np.asarray([iid_fid, iid_fid]).T,
                           columns=['FID', 'IID'])
df['IID'] = iid_fid['IID']
df['FID'] = iid_fid['FID']
#######################
# get phenotype Lhippo
#######################
sel_col = [u'ICV', u'Mhippo', u'Mthal',u'Mcaud',u'Mpal',u'Mput',u'Mamyg', u'Maccumb']
Lhippo = df[sel_col].join(iid_fid)
Lhippo = Lhippo.set_index(iid_fid['IID'])
for c in sel_col:
    Lhippo[c] = np.log10(Lhippo[c])
fname = GCTAOUT + '/LhippoLog.phe'
Lhippo.to_csv(fname, sep='\t', index = False, header=False, columns=[u'FID', u'IID'] + sel_col)

#fname = GCTAOUT + '/LhippoCor.phe'
#coni  = np.asarray(df[u'ICV']).reshape(-1,1)
#for s in [u'Lhippo', u'Rhippo', u'Lamyg', u'Laccumb', u'Rthal']:
#    y = np.asarray(df[s]).reshape(-1,1)
#    y_res = y - LinearRegression().fit(coni, y).predict(coni)
#    Lhippo[s] = y_res
#Lhippo.to_csv(fname, sep='\t', index = False,
#              columns=[u'FID', u'IID', u'Lhippo', u'Rhippo', u'Lamyg', u'Laccumb', u'Rthal'])

#######################
# get covariet info
#######################
covariate = iid_fid
covariate = covariate.join(pandas.get_dummies(df['ScanningCentre'],
                                              prefix='Centre')[range(7)])
covariate = covariate.join(df[['Age', 'Sex', 'AgeSq']])
covariate = covariate.set_index(iid_fid['IID'])
fname = GCTAOUT + '/SexCity.covar'
df[u'ScanningCentre']=['Centre_%d'%i for i in df[u'ScanningCentre']]
df.loc[:,u'Sex'].replace({0:'Male', 1:'Female'}, inplace=True)
df.to_csv(fname, sep='\t', index=False, header=False,
                 columns=[u'FID', u'IID', u'Sex', u'ScanningCentre'])
fname = GCTAOUT + '/SexScanner.covar'
df[u'ScanningCentre'].replace({'Centre_1':'GE', 'Centre_2':'Philips', 'Centre_3':'Philips', 'Centre_4':'Bruker',
     'Centre_5':'Siemens', 'Centre_6':'Siemens', 'Centre_7':'Siemens', 'Centre_8':'Siemens'}, inplace=True)
df.to_csv(fname, sep='\t', index=False, header=False,
                 columns=[u'FID', u'IID', u'Sex', u'ScanningCentre'])
fname = GCTAOUT + '/Sex.covar'
df.to_csv(fname, sep='\t', index=False, header=False,
                 columns=[u'FID', u'IID', u'Sex'])
fname = GCTAOUT + '/Age.qcovar'
df.to_csv(fname, sep='\t', index=False, header=False,
                 columns=[u'FID', u'IID', u'Age'])
fname = GCTAOUT + '/AgeIBS.qcovar'
df.to_csv(fname, sep='\t', index=False, header=False,
                 columns=[u'FID', u'IID', u'Age', u'C1', u'C2', u'C3', u'C4'])
                 

fin= ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
       '/recruitInfos_5_11_2015_17_53_0.csv')
ri = pandas.DataFrame.from_csv(fin, sep=',', index_col=False)
ri = ri.applymap(lambda x: np.nan if x==-1  else x)
ri[u'ni_height'] = ri[u'ni_height'].map(lambda x: np.nan if float(x)< 100.  else x)
ri = ri.dropna(subset=[u'ni_height'])
iid_fid = ["%012d" % int(i) for i in ri[u'Subject']]
ri = pandas.DataFrame(np.asarray([iid_fid, iid_fid, np.asarray(ri[u'ni_height']).tolist()]).T,
                           columns=['FID', 'IID', 'height'])
ri = ri.set_index(ri['IID'])
df = df.set_index(df['IID'])
result = pandas.merge(ri, df, left_index=True, right_index=True, how='inner');
fname = GCTAOUT + '/height.phe'
result.to_csv(fname, sep='\t', index = False, header=False,
              columns=[u'FID_x', u'IID_x', u'height'])

#                 
##
#from sklearn.linear_model import LinearRegression
#y = np.asarray(df['Lhippo']).reshape(-1,1)
#coni  = np.asarray(df[u'ICV']).reshape(-1,1)
#import matplotlib.pyplot as plt
#plt.plot(y, coni, '.')
#y_res = y - LinearRegression().fit(coni, y).predict(coni)
#plt.plot(y_res, coni, '.'); plt.show()
#Lhippo['Lhippo'] = y_res
