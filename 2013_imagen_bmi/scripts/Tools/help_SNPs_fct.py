# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:03:01 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import numpy as np
import os 
import pandas as pd
import plinkio as ig



def extract(genotype, snps_dict):
    """ from a genotype instance provide various helpers
    """
    void_gene = [i for i in snps_dict if len(snps_dict[i])==0]
    _ = [snps_dict.pop(i) for i in void_gene]
    col =  []
    _ = [col.extend(snps_dict[i]) for i in snps_dict]
    col = [str(i) for i in col]
    data = genotype.snpGenotypeByName(col)
    data = impute_data_by_med(data, verbose=True, nan_symbol=128)
    row = genotype.assayIID()

    return data, col, row, snps_dict, void_gene



def impute_data_by_med(data, verbose=0, nan_symbol=128):
   """ This function cut/pasted from B DaMota (genim-stat)
   """
   eps2 = 1e-17
   asNan = (data == nan_symbol)
   if verbose:
      print ('impute %d data for a total of %d' %
         (asNan[asNan].size, data.shape[0] * data.shape[1]))
   med = np.array([int(np.median(data[:, i]))
               for i in range(0, data.shape[1])])
   if med[med > 2].size > 0:
      print 'med == %s :'% str(nan_symbol), med[med > 2].size
      print 'med shape :', med.shape
      print 'shape of repetition', data.shape[0]
   else:
      med = np.array([np.median(data[:, i])
               for i in range(0, data.shape[1])])
      med[med == 0] = eps2
   med_all = np.repeat(med, data.shape[0]).reshape((-1, data.shape[0])).T
   data[asNan] = med_all[asNan]

   return data

def get_snp_dict():
    a = open("/neurospin/brainomics/2013_imagen_bmi/data/genetics/blabla.annot").read().split('\n')[:-1]
    a = [i.split() for i in a]
    a = [(i[3], i[7].split('|')[1]) for i in a]
    snp_dict = dict()
    gene = list(np.unique([i[1] for i in a]))
    for i in gene:
        snp_dict[i] = []
    
    for i in a:
        snp_dict[i[1]].append(i[0])
    
    gfn = os.path.join('/neurospin/brainomics',
                       '2012_imagen_shfj',
                       'genetics',
                       'qc_sub_qc_gen_all_snps_common_autosome.bim')
    tmp = [i.split('\t')[1] for i in open(gfn).read().split('\n')[:-1]]
    universe = set(tmp)
    
    for i in snp_dict:
        snp_dict[i] = set(snp_dict[i]).intersection(universe)

    return snp_dict
    
    
#gfn = os.path.join('/neurospin/brainomics',
#               '2012_imagen_shfj',
#               'genetics',
#               'qc_sub_qc_gen_all_snps_common_autosome')
#genotype = ig.Genotype(gfn)
#snp_dict = get_snp_dict()
#snp_data, snp_data_columns, snp_data_rows, snp_dict, void_gene = \
#                                              extract(genotype, snp_dict)
#df = pd.DataFrame(snp_data,index=snp_data_rows,
#                          columns=snp_data_columns)

