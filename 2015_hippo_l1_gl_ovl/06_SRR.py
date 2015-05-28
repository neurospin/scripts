# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:33:51 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas

ssrFname = ('/neurospin/brainomics/bioinformatics_resources/data/deCODE/'
            'female.rmap')
df = pandas.DataFrame.from_csv(ssrFname, sep='\t')
# a priori les rmap refletent le contenue du SSR de Kong 2010 et Fontenla 2014
#annule les seqbin=°
df.loc[df.query('seqbin==0').index, 'stdrate'] = 0.


#seuil
m = df['stdrate'] < 10.
mbar = df['stdrate'] >= 10.
df.loc[m, 'stdrate'] = 0.
df.loc[mbar, 'stdrate'] = 1.

df.query('chr=="chr22"').plot(x='pos', y='stdrate')


import pickle

isnps = pickle.load(open('/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/height_imputed_snps.pickle'))

maf = 1 - np.sum(isnps.data,axis = 0)/(2.*isnps.data.shape[0])
datas = {'snps':isnps.get_meta()[0].tolist(),
         'maf': maf}
df = pandas.DataFrame(datas, columns=['snps', 'maf'],index=isnps.get_meta()[0].tolist())

plosList = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/SNPheight.csv'
orig = pandas.DataFrame.from_csv(plosList, sep=';')
orig['Beta'] = [float(i.replace('?','-')) for i in orig['Beta']]
orig['Freq'] = [float(i) for i in orig['Freq']]


tout = pandas.merge(orig[['A1', 'A2','Freq', 'Beta']], df, left_index=True, right_index=True, how='inner')
tout = tout.join(pandas.Series(tout['Freq']- tout['maf'],name='Diff'))
