#  constraint_from_meta.py
#  
#  Copyright 2013 Vincent FROUIN <vf140245@is207857>
#  
# Jun 12th 2013
import pybedtools
import json
import numpy as np


def loadrefDB():
   ref = pybedtools.BedTool('/neurospin/brainomics/2013_brainomics_genomics/data/refGene.bed')
   snps = pybedtools.BedTool("/neurospin/brainomics/2013_brainomics_genomics/data/snps_b37_ref.bed")

   return ref, snps

def tree_path_gene_snp(ontology, ref=None, snps=None, name = 'synaptic transmission'):
   go_gene_snp = dict()
   go_gene_snp[name] = dict()

   for g in ontology[name]:
      #print g
      if len(g)>0:
         subset = pybedtools.BedTool(ref.filter(lambda b: b.name.endswith('|%s'%g)>0).saveas())
         tmp = snps.intersect(subset)
         if len(tmp)>0:
            go_gene_snp[name][g] = np.unique([i.name for i in tmp]).tolist()
   return go_gene_snp

def gene_snp(name='UTS2'):
   subset = pybedtools.BedTool(ref.filter(lambda b: b.name.endswith('|%s'%name)>0).saveas())
   tmp = snps.intersect(subset)
   return np.unique([i.name for i in tmp]).tolist()




if __name__=="__main__":
   ref, snps = loadrefDB()
   
   #  load meta info from GO/GSEA, USCC refGene and snps available from mes
   go  = json.load(open("/neurospin/brainomics/2013_brainomics_genomics/data/test.json"))

   # example with a go entry
   tree = tree_path_gene_snp(go, ref=ref, snps=snps, name = 'synaptic transmission')

   # exmaple with a gene entry
   snp_list = gene_snp(ref=ref, snps=snps,name='UTS2')
