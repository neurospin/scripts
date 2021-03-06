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


# GENIBABEL import
from genibabel.gcas import imagen_genotype_measure
from genibabel.gtypes import Pathways

logging.basicConfig(level=logging.INFO)


""" Usage: Each part generates a pickle file, uncomment the part(s) you need
"""

############################################################################
#################### generate new_synapticAll.pickle #######################
############################################################################
#fn = "/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1/c7.go-synaptic.symbols.gmt"
#compound_name = "c7.go-synaptic"
#origin = "msigdb-v3.1"
#pws = Pathways(fn=fn, origin=origin, compound_name=compound_name)
#
#pws.filter(min_gen=5)
#
#login = raw_input("Login on the imagen2 server:")
#password = getpass.getpass("Password on the imagen2 server:")
#
#genodata = imagen_genotype_measure(login, password, pathways=pws)
#
#fname = "/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/new_synapticAll.pickle"
#f = open(fname, 'wb')
#pickle.dump(genodata, f)
#f.close()


############################################################################
#################### generate new_synaptic10.pickle ########################
############################################################################
#fn = "/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1/c7.go-synaptic.symbols.gmt"
#compound_name = "c7.go-synaptic"
#origin = "msigdb-v3.1"
#pws = Pathways(fn=fn, origin=origin, compound_name=compound_name)
#
#pws.filter(min_gen=5)
#pws._entry = pws._entry[:10]
#
#login = raw_input("Login on the imagen2 server:")
#password = getpass.getpass("Password on the imagen2 server:")
#
#genodata = imagen_genotype_measure(login, password, pathways=pws)
#
#fname = "/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/new_synaptic10.pickle"
#f = open(fname, 'wb')
#pickle.dump(genodata, f)
#f.close()


############################################################################
#################### generate new_synaptic5.pickle #########################
############################################################################
#fn = "/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1/c7.go-synaptic.symbols.gmt"
#compound_name = "c7.go-synaptic"
#origin = "msigdb-v3.1"
#pws = Pathways(fn=fn, origin=origin, compound_name=compound_name)
#
#pws.filter(min_gen=5)
#pws._entry = pws._entry[:5]
#
#login = raw_input("Login on the imagen2 server:")
#password = getpass.getpass("Password on the imagen2 server:")
#
#genodata = imagen_genotype_measure(login, password, pathways=pws)
#
#fname = "/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/new_synaptic5.pickle"
#f = open(fname, 'wb')
#pickle.dump(genodata, f)
#f.close()


############################################################################
#################### generate new_synaptic2.pickle #########################
############################################################################
#fn = "/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1/c7.go-synaptic.symbols.gmt"
#compound_name = "c7.go-synaptic"
#origin = "msigdb-v3.1"
#pws = Pathways(fn=fn, origin=origin, compound_name=compound_name)
#
#pws.filter(min_gen=5)
#pws._entry = pws._entry[:2]
#
#login = raw_input("Login on the imagen2 server:")
#password = getpass.getpass("Password on the imagen2 server:")
#
#genodata = imagen_genotype_measure(login, password, pathways=pws)
#
#fname = "/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/new_synaptic2.pickle"
#f = open(fname, 'wb')
#pickle.dump(genodata, f)
#f.close()


############################################################################
####################### generate kegg.pickle ###############################
############################################################################

#fn = '/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1/c2.cp.kegg.v3.1.symbols.gmt'
#origin = "msigdb-v3.1"
#compound_name = 'c2.cp.kegg'
#pws = Pathways(fn=fn, origin=origin, compound_name=compound_name)
#
##list patwhways names
#print '====================================================================='
#print 'pathways names contained in this set'
#for pw in pws:
#    print pw['name']
#
#pws.filter(min_gen=5)
#
##list patwhways names:
#print '====================================================================='
#print 'pathways names contained in this set limited to 10'
#for pw in pws:
#    print pw['name']
#
## create a complete set
#login = raw_input("Login on the imagen2 server:")
#password = getpass.getpass("Password on the imagen2 server:")
##genodata = imagen_genotype_measure_alt(login, password, status="qc",
##                                    gene_names=list(geneset))
#genodata = imagen_genotype_measure(login, password, pathways=pws)
#
#fname = '/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/kegg.pickle'
#f = open(fname, 'wb')
#pickle.dump(genodata, f)
#f.close()
