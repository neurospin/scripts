# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 13:43:10 2014

@author: hl237680
"""
from genim import genibabel
#from glob import glob
#from genomics_bioresrql import BioresourcesDB
#import plinkio as ig


# List from graff paper Nature 2012
bmi_gene_list = ['SEC16B', 'TNNI3K', 'PTBP2', 'NEGR1', 'LYPLAL1', 'LZTR2', 'LRP1B', 'TMEM18',
 'POMC', 'FANCL', 'CADM2', 'SLC39A8', 'FLJ3577', 'HMGCR', 'NCR3', 'AIF1',
 'BAT2', 'NUDT3', 'TFAP2B', 'MSRA', 'LRRN6C', 'LMX1B', 'BDNF', 'MTCH2',
 'RPL27A', 'TUB', 'FAIM2', 'MTIF3', 'NRXN3', 'PRKD1', 'MAP2K5', 'GPRC5B',
 'ADCY9', 'SH2B1', 'APOB48', 'FTO', 'MC4R', 'QPCTL', 'KCTD15', 'TMEM160',
 'PRL', 'PTER', 'MAF', 'NPC1']

if __name__ == "__main__":
    snps_dict, void_gene, df = genibabel.load_from_genes(bmi_gene_list,
                                                         study='IMAGEN')
    #examples                         
    a = df.loc[[u'000037509984', u'000044836688', u'000063400084'],:].values                         
    b = df.loc[[u'000037509984', u'000063400084', u'000044836688'],:].values
