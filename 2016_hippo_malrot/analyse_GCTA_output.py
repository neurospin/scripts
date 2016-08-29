# -*- coding: utf-8 -*-
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re
import pheno as pu
import numpy as np
import pandas as pd


if __name__ == "__main__":
    path = '/neurospin/brainomics/2016_hippo_malrot/GCTA_output/without_ICV/binary/'
    pheno = []
    h2 = []
    pval = []
    for filename in glob.glob(os.path.join(path,'*.hsq')):
        m = re.search('(.)Phe(.+?).hsq', filename)
        if m:
            pheno.append(m.group(2))
        print filename
        df = pd.read_csv(filename, delim_whitespace=True)
        h2.append(df["Variance"][3])
        pval.append(df["Variance"][8])

    print "                  h2        Pval"
    for i,phen in enumerate(pheno):
        print phen+" "*(18-len(phen))+str(h2[i-1]) +" "*(10-len(str(h2[i-1])))+ str(pval[i-1])
