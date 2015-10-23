# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:35:15 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import numpy as np
import os
import subprocess
import pandas as pd
import matplotlib.pylab as plt
from shutil import rmtree, move
from tempfile import mkdtemp
from glob import glob

GCTA = 'gcta64-1.24 --thread-num 4 '
PLINK = 'plink --noweb --silent '

#'/volatile/frouin/baby_imagen/reacta/dataLinks'
IMAGEN_CENTRAL = '/neurospin/brainomics/imagen_central/'
OUT = '/volatile/frouin/baby_imagen/reacta/output'


def herit_compute(relgen, qcov, cov, phen, k, n):
    """ Wrapper to prepare GCTA analysis

    Parameters:
    -----------
    relgen : string,
    genetic relationship file (gz, file)

    cov : string,
    covariate file (categorical variables)

    phen : string,
    phenotype file
    """
    tmp = mkdtemp()
    out = tmp + '/%s_Cov%s_Qcov%s_Phe%s_%s' % \
            (os.path.basename(relgen),
             os.path.basename(cov).split('.covar')[0],
             os.path.basename(qcov).split('.qcovar')[0],
             os.path.basename(phen).split('.phe')[0],
             n)
    cmd = GCTA+' --grm %s --covar %s --qcovar %s --pheno %s --reml ' % (relgen, cov, qcov, phen)
    cmd += '  --out %s --mpheno %d' % ( out, k)

    try:
        print cmd
        result = subprocess.check_output(cmd,
                                         stderr=subprocess.STDOUT,
                                         shell=True)
    except subprocess.CalledProcessError, ex:            # error code <> 0
            print "--------error------"
            print '\n'.join([ex.cmd, ex.message,str(ex.returncode), ex.output])
            exit(1)

    return tmp

if __name__ == "__main__":

    # covariates
    ############
    qcov = IMAGEN_CENTRAL + 'covar/AgeIBS.qcovar'
    cov = IMAGEN_CENTRAL + 'covar/SexScanner.covar'

    # fichiers de kinship matrix
    # see generation from genibabel.compute_kinship()
    #################################################
    relgen = IMAGEN_CENTRAL + 'kinship/pruned_m0.05_g1_h6_wsi50_wsk5_vif10.0'
    # relgen '/qc_subjects_qc_genetics_all_snps_common_indep_pruned_50_5_10.0'

    datapath =  os.path.dirname(__file__) + '/data'

    outdir = []
    # phenotype multiples
    ######################
    phen1 = datapath + '/LhippoLog.phe'
    phen1_dict = {1: u'ICV', 2: u'Mhippo', 3: u'Mthal', 4: u'Mcaud', 5: u'Mpal',
                  6: u'Mput', 7: u'Mamyg', 8: u'Maccumb'}
    for k in phen1_dict.keys():
        od = herit_compute(relgen, qcov, cov, phen1, k, phen1_dict[k])
        outdir.append(od)

    # phenotype taille
    ##################
    phen2 = datapath + '/height.phe'
    phen2_dict = {1: u'height',}    
    outdir2 = []
    for k in phen2_dict.keys():
        od = herit_compute(relgen, qcov, cov, phen2, k, phen2_dict[k]) 
        outdir.append(od)       

    # gather hsv resulting file
    tmp = mkdtemp()
    for i in outdir:
        move(glob(i+'/*hsq')[0], tmp)

    # reduce results
    title = []
    hr = []
    hsd = []
    hp = []
    phen = 'LhippoLog'
    for k in phen1_dict.keys():
        fname = tmp + '/%s_Cov%s_Qcov%s_Phe%s_%s' % \
            (os.path.basename(relgen),
             os.path.basename(cov).split('.covar')[0],
             os.path.basename(qcov).split('.qcovar')[0],
             phen,
             phen1_dict[k])
        buf = open(fname + '.hsq').read().split('\n')[:-1]
        title.append(phen1_dict[k])
        hr.extend([ float(i.split('\t')[1]) for i in buf if i.split('\t')[0].startswith('V(G)/')])
        hsd.extend([ float(i.split('\t')[2]) for i in buf if i.split('\t')[0].startswith('V(G)/')])
        hp.extend([ float(i.split('\t')[1]) for i in buf if i.split('\t')[0].startswith('Pval')])
    
    phen = 'height'
    for k in phen2_dict.keys():
        fname = tmp + '/%s_Cov%s_Qcov%s_Phe%s_%s' % \
            (os.path.basename(relgen),
             os.path.basename(cov).split('.covar')[0],
             os.path.basename(qcov).split('.qcovar')[0],
             phen,
             phen2_dict[k])
        buf = open(fname + '.hsq').read().split('\n')[:-1]
        title.append(phen2_dict[k])
        hr.extend([ float(i.split('\t')[1]) for i in buf if i.split('\t')[0].startswith('V(G)/')])
        hsd.extend([ float(i.split('\t')[2]) for i in buf if i.split('\t')[0].startswith('V(G)/')])
        hp.extend([ float(i.split('\t')[1]) for i in buf if i.split('\t')[0].startswith('Pval')])
    
    ind = np.arange(len(hr))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind+width/2., hr, width, color='r', yerr=hsd)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percent herit')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( title )
    ax.set_ylim(0,1.)

    def autolabel(rects, pval):
        # attach some text labels
        for rect,p in zip(rects, pval):
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, 'p=%0.3f'%float(p),
                    ha='center', va='bottom')
    
    autolabel(rects1, hp)
    plt.axhline(y=0.5,c="blue")
    plt.ion()
    plt.show()
    plt.savefig('.' + '/herit.png', bbox_inches='tight')
    for d in outdir+ [tmp]:
        rmtree(d)
        