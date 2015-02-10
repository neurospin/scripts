#! /usr/bin/env python
##########################################################################
# Brainomics - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import getpass
import logging
import numpy
import pickle
import os

# GENIBABEL import
from genibabel.gcas.imagen import imagen_genotype_measure_alt


# bank of pathway
p = "/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1"
bank_pw = dict(synaptic=os.path.join(p, "c7.go-synaptic.symbols.gmt"))


# Set logging level
logging.basicConfig(level=logging.INFO)


# read patwhay
def read_pw(fn):
    """ read a standard broad GMT format pathaway file
    """
    # read data from file
    rawpw = open(bank_pw['synaptic']).read().split('\n')
    rawpw = [p.split('\t') for p in rawpw]

    # organize in a dict: key is pw_name and value is a list of gene
    pw = dict()
    for r in rawpw:
        pw.setdefault(r[0], []).extend(r[2:])
    return pw


# std operation of filtering on pw
def filter_pw(pw, min_gene=1, max_gene=9999, subset=None):
    """ Method that returns the genotyped mesures files and snp list.

    Parameters
    ----------
    pw: dict : paway information
    min_gene: min gen number in a pw.
    max_gene: max gen number in a pw.
    subset: return only subset pw out of the whole dataset (def is None=all).

    Return
    ------
    pw: dict. The patway.
    """
    pw_out = dict()
    for k in pw:
        if k == '':
            continue
        if (len(pw[k]) >= min_gene) and (len(pw[k]) <= max_gene):
            pw_out[k] = pw[k]
    if subset is not None:
        wanted_keys = pw_out.keys()[:subset]
        pw_out = dict([(i, pw_out[i]) for i in wanted_keys if i in pw_out])

    return pw_out

if __name__ == "__main__":
    pw = read_pw(bank_pw['synaptic'])
    pw = filter_pw(pw, min_gene=5, max_gene=30, subset=2)

    # Ask for db login information
    # status = "qc" and postload_op = "frc_imput" considered as default
#    login = raw_input("Login on the imagen2 server:")
#    password = getpass.getpass("Password on the imagen2 server:")
    login = 'vfrouin'
    password = 'MonKennWord!'
    # dictionnary of genodata; keys are the pw names
    genodata = {}
    for k in pw.keys():
        gene_list = pw[k]
        print("======= Processing;", k)
        genodata[k] = imagen_genotype_measure_alt(login, password,
                                                    gene_names=gene_list)
    #pickle data
    outdir = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/'
    fname = os.path.join(outdir, 'synaptic' + '.pickle')
    f = open(fname, 'w')
#    pickle.dump({'genodata' : genodata, 'snpList' : snpList,
#                 'group': group, 'group_names' : group_names}, f)
    pickle.dump({'genodata' : genodata}, f)
    f.close()

