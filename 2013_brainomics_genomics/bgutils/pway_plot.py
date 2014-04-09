# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:05:35 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))


def pway_plot(x, y):
    fig = plt.figure()
    
    #x = np.arange(100)
    #y = 3.*np.sin(x*2.*np.pi/100.)
    
    for i in range(5):
        temp = 510 + i
        ax = plt.subplot(temp)
        plt.plot(x,y)
        plt.subplots_adjust(hspace = .001)
        plt.ylim(-3,3)
        ax.set_yticklabels(())
        ax.set_xticklabels(())
        ax.title.set_visible(False)
        ax.text(.10,0.10,'chrom %2d'%i,
            horizontalalignment='center',
            transform=ax.transAxes)
    
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()



def info_gene(database, gene):
    """Return all SNPs associated to a gene.

    Parameters
    ----------
    database : ?
        Bioresource database.

    gene : unicode
        Gene name.

    Returns
    -------
    array_like
        List of rsIDs in gene.

    """
    req = ("Any G,B,E,CN WHERE G name '%(gene)s', "
           "G start_position B, "
           "G stop_position E, "
           "G chromosomes C, "
           "C name CN"
           % {'gene': gene})
#    print database.rql(req)
    return [i for i in database.rql(req)]


def pw_gene_asnp(pw, snpList, relative=False):
    """Return all SNPs associated to a gene.

    Parameters
    ----------
    realtive : ?
        positon relative to gene start.

    nb : int
        # of pathway read from the resource TOFIX
        
    Returns
    -------
    constraint : dict()
        dict with pw key each containing a dict with gene key (list of snp)
        
    array_like
        List of rsIDs in pw
                
    array_like
        List of rs positions in pw

    """
#    from os import path
#    basepath = '/neurospin/brainomics/2013_brainomics_genomics/'
#    constraint = json.load(open(path.join(basepath,'data','constraint10.json')))
#    tmp = []
#    for i in constraint:
#        for j in constraint[i]:
#            tmp+=(constraint[i][j])
#    snpList = np.unique(tmp)
    sys.path.append('/home/vf140245/gits/brainomics/bioresource/examples/python')
    from bioresourcesdb import BioresourcesDB
    BioresourcesDB.login('admin', 'admin')

    # NIMPORTE QUOI DANS LA BASE ncbi (JL a pris tous les chromosomes !!!)
    # il ne devrait pas etr necessaire de masquer par le gene
    snpPos = []
    snpChr = []
    for s in snpList:
        for pwn in pw:
            for g in pw[pwn]:
                if s in pw[pwn][g]:
                    gene =g
                    break
        req = ("Any P,GS,CN WHERE S rs_id '%(snp)s', "
                    "S position P, S chromosome C, C name CN, "
                    "S gene G, G name '%(gene)s', G start_position GS"
           % {'snp': s, 'gene':gene})
        tmp = BioresourcesDB.rql(req)
        if (len(tmp)==1):
            snpChr.append(tmp[0][2])
            if relative:
                 snpPos.append(tmp[0][0]-tmp[0][1])
            else:
                snpPos.append(tmp[0][0])
        else:
            print "One snp got no position: %s"%s
    return snpPos, snpChr

def transcode(chrom, coord_abs):
    s_start = chrom['s_start']
    s_len = chrom['s_len']
    s_offset = [0] + s_len[:-1]
    s_end = [i+j for i,j in zip(s_start,s_len)]
    x = np.zeros(coord_abs.shape, dtype=int)
    for ival, val in enumerate(coord_abs):
        js, jo = [(js,jo) for js,je,jo in zip(s_start, s_end, s_offset) if ((val>=js) & (val<=je))][0]
        x[ival] = val - js + jo
    return x


def plot_pw(beta, pway=None, snplist=None):#gene_list, beta, snpPos, snpChr):
    """Plot
    """
    if pway==None or snplist==None:
        return

    sys.path.append('/home/vf140245/gits/brainomics/bioresource/examples/python')
    from bioresourcesdb import BioresourcesDB
    BioresourcesDB.login('admin', 'admin')
    
    gene_list = get_gene_from_pw(pway)
    snpPos, snpChr = pw_gene_asnp(pway, snplist, relative=False)
    genes_annot = dict()
    for gene in gene_list:
        tmp = info_gene(BioresourcesDB, gene)
        if (len(tmp) == 1):
            genes_annot[gene] = tmp[0]
        else:
            print "One gene is des not belong exactly to one chromosome :%s"%gene
#    print genes_annot
    chr_annot = dict()
    for i in genes_annot:
        if chr_annot.has_key(genes_annot[i][3]):
            chr_annot[genes_annot[i][3]].append([i,genes_annot[i][1], genes_annot[i][2]])
        else:
            chr_annot[genes_annot[i][3]] = [[i,genes_annot[i][1], genes_annot[i][2]]]
    chr_plot_info = dict()
    for i in chr_annot:
        chr_plot_info[i] = dict()
        arg_sort = np.argsort([g[1] for g in chr_annot[i]])
        s_start = (np.asarray([g[1] for g in chr_annot[i]])[arg_sort]).tolist()
        chr_plot_info[i]['s_start'] = s_start
        s_name = (np.asarray([g[0] for g in chr_annot[i]])[arg_sort]).tolist()
        chr_plot_info[i]['s_name'] = s_name
        s_len = (np.asarray([g[2]-g[1]+1 for g in chr_annot[i]])[arg_sort]).tolist()
        tmp = (np.asarray(tmp)).tolist()
        chr_plot_info[i]['s_len'] = s_len
        chr_plot_info[i]['start'] = s_start[0]
        chr_plot_info[i]['end'] = np.sum(s_len)
    
    
    lig_plot=len(chr_plot_info)%2 + len(chr_plot_info)/2
    for i,k in enumerate(chr_plot_info):
        chrom = chr_plot_info[k]
        ax = plt.subplot(lig_plot,2,i)
        plt.subplots_adjust(hspace = .001)
        plt.ylim(-1,1)
        #plt.xlim(chrom['start'], chrom['start']+chrom['end']) 
        #x = range(chrom['start'], chrom['start']+chrom['end'])  
        plt.xlim(0, chrom['end'])
        x = range(0, chrom['end'])     
#        if k == u'chr3':
#            print 'DBG>', chrom['start'], chrom['start']+chrom['end'] 
        y = np.zeros(len(x))
        plt.plot(x,y)
        y = np.ones(len(x))*.9
        acc = 0
        y[0] = y[0] - 1
        y[-1] = y[-1] - 1
        for m in chrom['s_len']:
            acc += m-1            
            y[acc] = y[acc] - 1
        #print np.min(x), np.max(x)
        plt.plot(x,y)
        ax.set_yticklabels(())
        ax.set_xticklabels(())
        ax.title.set_visible(False)
        ax.text(.10,0.10,'%s'%str(k),
            horizontalalignment='center',
            transform=ax.transAxes)

    for i,k in enumerate(chr_plot_info): 
        chrom = chr_plot_info[k]
        ax = plt.subplot(lig_plot,2,i)
        indices = [l for l, x in enumerate(snpChr) if x == k]
        x = np.asarray(snpPos)[indices]
        x = transcode(chr_plot_info[k], x)
        y = beta[indices]
#        if k == u'chr3':
#            print indices
#            print x
#            print y
        plt.vlines(x, [0], y, lw=2)
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()

def get_gene_from_pw(pw):
    gene_list = []
    for i in pw:
        for j in pw[i]:
            gene_list.append(j)
    
    return np.unique(gene_list).tolist()
 

if __name__=="__main__":
    # read constraints : we do not use Group Constraint here
    from bgutils.build_websters import group_pw_snp2, pw_gene_snp2
    cache = True
    group, group_names, snpList = group_pw_snp2(nb=10, cache=cache)
    pw, _ = pw_gene_snp2(nb=10, cache=cache)    
  
    beta = np.random.uniform(-1, 1, 539)
    plot_pw(beta, pway=pw, snplist=snpList)
