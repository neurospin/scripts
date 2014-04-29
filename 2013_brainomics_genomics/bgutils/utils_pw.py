# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:40:17 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import sys, os
import numpy as np
import pickle
import json

sys.path.append('/home/vf140245/gits/igutils')
msigdbpath = '/neurospin/brainomics/bio_resources/genesets/msigdb-v3.1'

def gmt_to_genejson(gmtfn='c2.cp.kegg.v3.1.symbols',outdir='', cache=True):
    '''
    '''
    #cache not used sofar
    fn = os.path.join(msigdbpath, gmtfn+'.gmt')
    tmp = open(fn).read().split('\n')[:-1]
    pw = dict()
    for i in tmp:
        detail = i.split('\t')
        pw[detail[0]] = detail [2:]
    fno = os.path.join(outdir, gmtfn+'.gjs')
    json.dump(pw, open(fno,'w'))
    
    return fno
    

def unfold_snp(pway, mask):
    sys.path.append('/home/vf140245/gits/igutils')
    import igutils as ig
    gfn = mask
    genotype = ig.Geno(gfn)
    genotypeSnp = genotype.snpList().tolist()
    sys.path.append('/home/vf140245/gits/brainomics/bioresource/examples/python')
    from bioresourcesdb import BioresourcesDB
    BioresourcesDB.login('admin', 'admin')
    
    tmp = []
    to_drop =[]
    constraint2 = dict()
    for i in pway:
        constraint2[i] = dict()
        for j in pway[i]:
            req = ("Any RS WHERE G name '%(gene)s', "
                   "G start_position B, G stop_position E, "
                   "G chromosomes C, S chromosome C, "
                   "S is Snp, S rs_id RS, S position SP HAVING SP > B, SP < E"
                     % {'gene': j})
            tmp = BioresourcesDB.rql(req)
            tmp = [item for sublist in tmp for item in sublist]
            tmp = list(set(genotypeSnp).intersection(set([str(s) for s in tmp])))
            if (len(tmp)!=0):
                constraint2[i][j] = tmp
        if (len(constraint2[i].keys())==0):
            to_drop.append(i)
        print i, '  done.'
    for i in to_drop:
        del constraint2[i]
    tmp = []
    for i in constraint2:
        for j in constraint2[i]:
            tmp+=(constraint2[i][j])
    snpList = np.unique(tmp)
    
    return constraint2, snpList


def convert_pw_group(pw, snpList):

    group_names = pw.keys()    
    group = dict()
    for ii, i in enumerate(pw):
        tmp = []
        for j in pw[i]:
            tmp+=[np.where(snpList==k)[0][0] for k in pw[i][j]]
        group[ii] = tmp
    
    return group, group_names, snpList


def build_msigdb(pw_name= 'c2.cp.kegg.v3.1.symbols', mask = '', outdir='./', cache=True):
    '''
    '''
    if cache and os.path.exists(os.path.join(outdir,pw_name+'.pickle')):
        print "Reading from cache"
        f = open(os.path.join(outdir,pw_name+'.pickle'))
        combo = pickle.load(f)
        f.close()
        return combo['group'], combo['group_names'], combo['constraint'], combo['snpList']

    gmtfn = gmt_to_genejson(pw_name, outdir=outdir, cache=cache)
    pw = json.load(open(gmtfn))#.read().split('\n')
#    pw = dict()
#    for i in tmp:
#        detail = i.split('\t')
#        pw[detail[0]] = detail [2:]
        
    #mask with a plink measurement file  
    constraint, snpList = unfold_snp(pw, mask)
    
    #transform in a group constraint
    group, group_names, snpList = convert_pw_group(constraint, snpList)
    
    # to write.        
    f = open(os.path.join(outdir,pw_name+'.pickle'), 'w')
    pickle.dump({'constraint' : constraint, 'snpList' : snpList,
                 'group': group, 'group_names' : group_names}, f)
    f.close()

    return  group, group_names, constraint, snpList


