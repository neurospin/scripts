# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:01:58 2018

@author: am254847
"""
import os
import pandas

ROOT_DIR = "/neurospin/tmp/Angeline"
PHENO_DIR = "HBN_Phenotypic"

fn1 = os.path.join(ROOT_DIR,PHENO_DIR,"HBN_Demo_1.xlsx")
data1 = pandas.read_excel(fn1)
print data1.columns
print data1.head()
data1.plot.scatter("Sex","Age")

fn2 = os.path.join(ROOT_DIR,PHENO_DIR,"HBN_Demo_2.xlsx")
data2 = pandas.read_excel(fn2)

fn3 = os.path.join(ROOT_DIR,PHENO_DIR,"HBN_Demo_3.xlsx")
data3 = pandas.read_excel(fn3)

merged_pheno_dupl = pandas.concat([data1,data2,data3])
merged_pheno = merged_pheno_dupl.drop_duplicates("EID")

 merged_pheno.to_csv(outfile,index=False)
