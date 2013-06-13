#  build_websters.py
#  
#  Copyright 2013 Vincent FROUIN <vf140245@is207857>
#  date : June 13th 2013
import igutils as ig
import numpy as np
import sys
def impute_data_by_med(data, verbose=0):
   """ This function cut/pasted from B DaMota (genim-stat)
   """
   eps = 1e-18
   eps2 = 1e-17
   as128 = (data == 128)
   if verbose:
      print ('impute %d data for a total of %d' %
         (as128[as128].size, data.shape[0] * data.shape[1]))
   med = np.array([int(np.median(data[:, i]))
               for i in range(0, data.shape[1])])
   if med[med > 2].size > 0:
      print 'med == 128 :', med[med > 2].size
      print 'med shape :', med.shape
      print 'shape of repetition', data.shape[0]
   else:
      med = np.array([np.median(data[:, i])
               for i in range(0, data.shape[1])])
      med[med == 0] = eps2
   med_all = np.repeat(med, data.shape[0]).reshape((-1, data.shape[0])).T
   data[as128] = med_all[as128]

   return data

def build_websters(gfn, pfn, pafn, gsubset=None, psubset=None):
   """ genotype, phenotype pahtology filename
   """
   # SNPs ###########
   genotype = ig.Geno(gfn)
   # yield a uin8 numpy.array with (0,1,2); Careful: 128 code for NA
   if gsubset==None:
      snp_data = genotype.snpGenotypeAll()
      snp_colnames = genotype.snpList()
   else:
      snp_data = genotype.snpGenotypeByName(gsubset)
      snp_colnames = gsubset
   snp_data = impute_data_by_med(snp_data)
   snp_id = ['%s%s'%(i[0], i[1])
                        for i in zip(genotype.getOrderedSubsetFamily(),
                                     genotype.getOrderedSubsetIndiv()) ]
   print "Genotype loaded..."
   sys.stdout.flush()
   
   # Phenotype : here gene expression data
   pheno = open(pfn).read().split('\n')[:-1]
   pheno_header = pheno[0].split('\t')
   pheno = np.asarray([i.split() for i in pheno[1:]])
   pheno_id = ['%s%012d'%(i[0],int(i[1])) for i in pheno[:,:2]]
   pheno_data = np.asarray(pheno[:,2:],dtype=float)
   print "Phenotype loaded..."
   sys.stdout.flush()
   
   # Patho :
   patho = open(pafn).read().split('\n')[:-1]
   patho_header = patho[0].split()
   patho = np.asarray([i.split() for i in patho[1:]])
   patho_id = ['%s%012d'%(i[0],int(i[1])) for i in patho[:,:2]]
   patho_data =  np.asarray(patho[:,2:],dtype=int)
   print "Final phenotype loaded..."
   sys.stdout.flush()
   
   # reconcile data index
   keep_id = list(set(patho_id).intersection(set(pheno_id)).intersection(set(snp_id)))
   snp_data = snp_data[np.array([snp_id.index(i) for i in keep_id]),:]
   pheno_data = pheno_data[np.array([snp_id.index(i) for i in keep_id]),:]
   patho_data = patho_data[np.array([snp_id.index(i) for i in keep_id]),:]
   print "Data realigned..."
   sys.stdout.flush()
   
   return(dict(rownames=keep_id, 
                snp_data=snp_data, snp_colnames=snp_colnames,
                pheno_data=pheno_data, pheno_colnames=pheno_header[2:],
                patho_data=patho_data, patho_colnames=patho_header[2:]))

if __name__=="__main__":
   gfn = '/neurospin/brainomics/2013_brainomics_genomics/data/genetic_control_xpt'
   pfn= '/neurospin/brainomics/2013_brainomics_genomics/data/genetic_control_xpt.phe'
   pafn = '/neurospin/brainomics/2013_brainomics_genomics/data/genetic_control_xpt.cov'
   #gsubset=None
   gsubset = ['rs2645081','rs3128309','rs2803285']
   blocks = build_websters(gfn, pfn, pafn, gsubset=gsubset)
   blocks.keys()
   
