# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:50:55 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
#test import
import sys, os
import numpy as np


def import_exist():
    #path in git scripts
    sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))
    import bgutils
    print "================== trying to import bgutils ====================="
    print bgutils.__file__
    print "================================================================="

if __name__=="__main__":
    import_exist()
    #
    from bgutils.build_websters import list_constraint
    print "============ First/last 4 constraints (pathways) ================"
    cl = list_constraint(con_name='go_synaptic')
    print '\n'.join(cl[:4])
    print "..."
    print '\n'.join(cl[-5:-1])
    print "================================================================="
    #
    from bgutils.build_websters import group_pw_snp
    group, group_names, snpList = group_pw_snp(nb=10)
    print "#Pathways (should be 10): ", len(group)
    print "All pathways spawn (should be (6344, )): ", snpList.shape
    print "Pathway lengthes (should be [199, 170, 923, 163, 2525, 146, 224, 57, 289, 2369]): ", [len(group[i])for i in group]
    #
    from bgutils.build_websters import get_websters_linr
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X = get_websters_linr(gene_name = 'KIF1B', snp_subset=snp_subset)
    print "============ read data for lin reg GE and SNPs   ================"
    print "dim y (should be (364,)): ", y.shape
    print "#dim X (should be (364,6344)): ", X.shape
    print "================================================================="
