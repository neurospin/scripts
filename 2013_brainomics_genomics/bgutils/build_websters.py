#  build_websters.py
#  
#  Copyright 2013 Vincent FROUIN <vf140245@is207857>
#  date : June 13th 2013

import sys
import numpy as np
import json
sys.path.append('/home/vf140245/gits/igutils')
import igutils as ig

basepath = '/neurospin/brainomics/2013_brainomics_genomics/'

def impute_data_by_med(data, verbose=0, nan_symbol=128):
   """ This function cut/pasted from B DaMota (genim-stat)
   """
   eps = 1e-18
   eps2 = 1e-17
   asNan = (data == nan_symbol)
   if verbose:
      print ('impute %d data for a total of %d' %
         (asNan[asNan].size, data.shape[0] * data.shape[1]))
   med = np.array([int(np.median(data[:, i]))
               for i in range(0, data.shape[1])])
   if med[med > 2].size > 0:
      print 'med == %s :'% str(nan_symbol), med[med > 2].size
      print 'med shape :', med.shape
      print 'shape of repetition', data.shape[0]
   else:
      med = np.array([np.median(data[:, i])
               for i in range(0, data.shape[1])])
      med[med == 0] = eps2
   med_all = np.repeat(med, data.shape[0]).reshape((-1, data.shape[0])).T
   data[asNan] = med_all[asNan]

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
   snp_data = impute_data_by_med(snp_data, verbose=True, nan_symbol=128)
   snp_id = ['%s%s'%(i[0], i[1])
                        for i in zip(genotype.getOrderedSubsetFamily(),
                                     genotype.getOrderedSubsetIndiv()) ]
   print "Genotype loaded..."
   sys.stdout.flush()
   
   # Phenotype : here gene expression data
   pheno = open(pfn+'.pheno').read().split('\n')[:-1]
   pheno_header = pheno[0].split('\t')
   pheno = np.asarray([i.split() for i in pheno[1:]])
   pheno_id = ['%s%012d'%(i[0],int(i[1])) for i in pheno[:,:2]]
   pheno_data = np.asarray(pheno[:,2:],dtype=float)
   pheno_data = impute_data_by_med(pheno_data, verbose=True, nan_symbol=-9.)
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
   pheno_data = pheno_data[np.array([pheno_id.index(i) for i in keep_id]),:]
   patho_data = patho_data[np.array([patho_id.index(i) for i in keep_id]),:]
   print "Data realigned..."
   sys.stdout.flush()
   
   return(dict(rownames=keep_id, 
                snp_data=snp_data, snp_colnames=snp_colnames,
                pheno_data=pheno_data, pheno_colnames=pheno_header[2:],
                patho_data=patho_data, patho_colnames=patho_header[2:]))


def get_websters_linr(gene_name='KIF1B', snp_subset=['rs12120191']):
    from os import path
    gfn = path.join(basepath,'data','geno','genetic_control_xpt')
    pfn = path.join(basepath,'data','pheno','filter_residuals')
    pafn = path.join(basepath,'data','patho','genetic_control_xpt.cov')
    blocks = build_websters(gfn, pfn, pafn, gsubset=snp_subset)
    ind = blocks['pheno_colnames'].index(gene_name)
    y =blocks['pheno_data'][:,ind].reshape((-1,))
    X =blocks['snp_data']
    
    return y, X

def get_websters_logr(snp_subset=['rs12120191']):
    from os import path
    gfn = path.join(basepath,'data','geno','genetic_control_xpt')
    pfn = path.join(basepath,'data','pheno','filter_residuals')
    pafn = path.join(basepath,'data','patho','genetic_control_xpt.cov')
    blocks = build_websters(gfn, pfn, pafn, gsubset=snp_subset)
    y = blocks['patho_data']
    y = y - 1 # code 0/1
    X =blocks['snp_data']
    
    return y, X

def pw_gene_snp(nb=10):
    from os import path
    constraint = json.load(open(path.join(basepath,'data','constraint10.json')))
    tmp = []
    for i in constraint:
        for j in constraint[i]:
            tmp+=(constraint[i][j])
    snpList = np.unique(tmp)
    
    return constraint, snpList


def group_pw_snp(nb=10):
    from os import path
    constraint = json.load(open(path.join(basepath,'data','constraint10.json')))
    tmp = []
    group_names = constraint.keys()    
    for i in constraint:
        for j in constraint[i]:
            tmp+=(constraint[i][j])
    snpList = np.unique(tmp)
    group = dict()
    for ii, i in enumerate(constraint):
        tmp = []
        for j in constraint[i]:
            tmp+=[np.where(snpList==k)[0][0] for k in constraint[i][j]]
        group[ii] = tmp
    
    return group, group_names, snpList


def build_constraint(nb=10):
    from os import path
    constraint = json.load(open(path.join(basepath,'data','constraint10.json')))
    tmp = []
    constraint_name = constraint.keys()    
    for i in constraint:
        for j in constraint[i]:
            tmp+=(constraint[i][j])
    refList = np.unique(tmp)
    constraintIndex = dict()
    for ii, i in enumerate(constraint):
        tmp = []
        for j in constraint[i]:
            tmp+=[np.where(refList==k)[0][0] for k in constraint[i][j]]
        constraintIndex[ii] = tmp
    return refList, constraint, constraintIndex, constraint_name


def list_constraint(con_name='go_synaptic'):
    from os import path
    constraint = json.load(open(path.join(basepath, 'data',con_name)+'.json'))
    return constraint.keys()

if __name__=="__main__":
   ###### get the 3 blocks geno, pheno, patho
   gfn = '/neurospin/brainomics/2013_brainomics_genomics/data/geno/genetic_control_xpt'
   pfn= '/neurospin/brainomics/2013_brainomics_genomics/data/pheno/filter_residuals'
   pafn = '/neurospin/brainomics/2013_brainomics_genomics/data/patho/genetic_control_xpt.cov'
   gsubset=None
   #gsubset = ['rs2645081','rs3128309','rs2803285']
   blocks = build_websters(gfn, pfn, pafn, gsubset=gsubset)
   blocks.keys()
   f = open(basepath+'data/blocks.pickle', 'w')
   pickle.dump(blocks, f)
   f.close()
