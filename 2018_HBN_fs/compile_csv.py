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

# study the duplicted data
###############################################################################
print merged_pheno_dupl.head()
# sort the data inplace : ie data sorted are stored in same DataFrame
merged_pheno_dupl.sort("EID", inplace=True)
# get a duplication status
dupl_status = merged_pheno_dupl.duplicated()
# now create a new column in the dataframe named "Dup"
merged_pheno_dupl["Dup"] = dupl_status
# see if it works
print merged_pheno_dupl.head()
# reorder columns and keep only intersting ones
merged_pheno_dupl = merged_pheno_dupl[[u'EID',   u'Dup', u'Sex', u'Age']]
# examine carefully the data set (column Dupl = True)
print merged_pheno_dupl.to_string

# finish the job
###############################################################################
# now we checked and are sure we can drop and keep the first out of the 2 lines
merged_pheno = merged_pheno_dupl.drop_duplicates("EID")

merged_pheno.to_csv(outfile,index=False)
