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
import pickle
import os

# System import
from genibabel.pathways import Pathways

# GENIBABEL import
from genibabel.gcas.imagen import imagen_genotype_measure_alt
from genibabel.genotypes import RawGenotype

logging.basicConfig(level=logging.INFO)


fn=('/neurospin/brainomics/bio_resources/genesets/'
                          'msigdb-v3.1/c7.go-synaptic.symbols.gmt')
origin="msigdb-v3.1",
compound_name='c7.go-synaptic'
pws = Pathways(fn=fn, origin=origin, compound_name=compound_name)

#list patwhways names
print '====================================================================='
print 'pathways names contained in this set'
for pw in pws:
    print pw['name']

Pathways.filter(pws, min_gen=5, max_gen=50, num_pw=10)

#list patwhways names: only 10 now
print '====================================================================='
print 'pathways names contained in this set limited to 10'
for pw in pws:
    print pw['name']

# create a complete set
login = raw_input("Login on the imagen2 server:")
password = getpass.getpass("Password on the imagen2 server:")
#genodata = imagen_genotype_measure_alt(login, password, status="qc",
#                                    gene_names=list(geneset))
genodata = imagen_genotype_measure_alt(login, password, status="qc", pws=pws)

fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/synaptic10.pickle'
f = open(fname,'wb')
pickle.dump(genodata, f)
f.close()


meta_pws = genodata.get_meta_pws()