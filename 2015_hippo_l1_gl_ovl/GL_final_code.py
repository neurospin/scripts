# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:33:51 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas
import pickle
GCTAOUT = '/volatile/frouin/baby_imagen/reacta/dataLinks'

# read imputed imagen data
#############################################
isnps = pickle.load(open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/height_imputed_snps.pickle'))
maf = np.sum(isnps.data,axis = 0)/(2.*isnps.data.shape[0])
datas = {'snps':isnps.get_meta()[0].tolist(),
         'maf': maf}
imagenImputed = pandas.DataFrame(datas, columns=['snps', 'maf'],index=isnps.get_meta()[0].tolist())


fin = ('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
       'imagen_subcortCov_NP.csv')
covar = pandas.read_csv(fin, sep=' ', dtype={0:str, 1:str})
covar.index = covar['IID']
covar = covar[['FID','IID','Age','ScanningCentre', 'Sex']]

# get the information from the plosOne paper.
# comfront with IMAGEN imputed data
#############################################
plosList = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/SNPheight.csv'
orig = pandas.DataFrame.from_csv(plosList, sep=';')
orig['Beta'] = [float(i.replace('?','-')) for i in orig['Beta']]
orig['Freq'] = [float(i) for i in orig['Freq']]

plosOne = pandas.merge(orig[['A1', 'A2','Freq', 'Beta']], imagenImputed,
                       left_index=True, right_index=True, how='inner')
plosOne = plosOne.join(pandas.Series(plosOne['Freq']- plosOne['maf'],name='Diff'))
# reorder
plosOne = plosOne.loc[ isnps.get_meta()[0].tolist()]

# read height
#############################################
fname = GCTAOUT + '/height.phe'
height = pandas.read_csv(fname, sep='\t', dtype={1:str,0:str},header=None)
height.columns = ['FID','IID','height']
height.index = height['IID']



# create the PgS : 
beta  = np.asarray(plosOne['Beta']).reshape(173,-1)
PgS = np.dot(isnps.data, beta).reshape(-1)
studyPgS  = pandas.DataFrame({'PgS':PgS, 'IID':isnps.iid.reshape(-1)})
studyPgS = pandas.merge(pandas.merge(covar, height, how='inner',on='IID'), 
                        studyPgS, on='IID')
studyPgS = studyPgS[[u'FID_x', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS']]
studyPgS.columns = [u'FID', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS']
studyPgS[u'Sex'].replace({0:'Male', 1:'Female'}, inplace=True)
studyPgS[u'ScanningCentre'].replace({1:'GE', 2:'Centre_2', 3:'Centre_3', 4:'Centre_4',
     5:'Centre_5', 6:'Centre_6', 7:'Centre_7', 8:'Centre_8'}, inplace=True)


#save.
#############################################
fname = GCTAOUT + '/studyPgS.csv'
studyPgS.to_csv(fname, sep='\t', index=False, header=True,
                 columns=[u'FID', u'IID', u'Age', u'ScanningCentre', u'Sex', u'height', u'PgS'])
