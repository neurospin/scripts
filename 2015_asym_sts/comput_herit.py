# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:30:07 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import subprocess
import os
import optparse
import pheno as pu
import numpy as np
import tempfile

GCTA = 'gcta64-1.24 --thread-num 4 '
TMP = '/tmp'


def herit_compute(relgen, qcov, cov, phen, k, n):
    """ Wrapper to prepare GCTA analysis

    Parameters:
    -----------
    relgen: string,
    genetic relationship file (gz, file)
    
    qcov: string,
    covariate file (quantitative traits)

    cov: string,
    covariate file (categorical variables)

    phen: string,
    phenotype file

    k: int,
    trait number
    
    n: string,
    trait name
    """
    out = TMP + '/%s_Cov%s_Qcov%s_Phe%s_%s' % (os.path.basename(relgen),
                                              os.path.basename(cov),
                                              os.path.basename(qcov),
                                              os.path.basename(phen),
                                              n)
    local_phen = phen
    local_cov = cov
    #add some filterings
    print 'Perform filtering on subjcet from pheno'
    p = pu.readPheno(phen)
    p_copy = p.copy()
    print 'init #subjects: ', p.shape[0]
    p_copy['Pheno3'] = np.asarray(p_copy['Pheno3'], dtype=float).tolist()
#    mask = p_copy['Pheno3'].sub(p_copy['Pheno3'].mean()) <= 0.15
#    mask = p_copy['Pheno3'] - (p_copy['Pheno3'].mean()) <= 0.15
    mask = abs(p_copy['Pheno3']-p_copy['Pheno3'].mean()) < .27
    p = p.loc[mask]
    print 'filtered #subjects: ', p.shape[0]
    temp = tempfile.NamedTemporaryFile()
    pu.to_GCTA_pheno(p, temp.name)
    #pu.to_GCTA_pheno(p, '/tmp/tempphenonotdeleted.txt')  # TODEL
    local_phen = temp.name
    c = pu.readPheno(cov)
    c = c.loc[c['Pheno3'] == 'Right'][[u'FID', u'IID', u'Gender', u'Pheno2']]
    cotemp = tempfile.NamedTemporaryFile()
    pu.to_GCTA_qcovar(c, cotemp.name)
    #pu.to_GCTA_qcovar(c, '/tmp/tempcovnotdeleted.txt')  # TODEL
    local_cov = cotemp.name
    #end add some filterings

    cmd = GCTA+' --grm %s --covar %s --qcovar %s --pheno %s --reml --out %s --mpheno %d' %(relgen, local_cov, qcov, local_phen, out, k)
    print cmd
    try:
        #prints results and merges stdout and std
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        #print result
    except subprocess.CalledProcessError, ex: # error code <> 0
        print "--------error------"
        print ex.cmd
        print ex.message
        print ex.returncode
        print ex.output  # contains stdout and stderr together
        temp.close()
        cotemp.close()
    try:
        print '================ %s =================' % n
        cmd = 'cat ' + out + '.hsq'
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        print result
    except subprocess.CalledProcessError, ex: # error code <> 0
        print "--------error------"
        print ex.cmd
        print ex.message
        print ex.returncode
        print ex.output  # contains stdout and stderr together
        temp.close()
        cotemp.close()

    temp.close()
    cotemp.close()
    #p = subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    
    pheno = os.path.join('/neurospin/brainomics/2015_asym_sts/pheno',
                        'STs.phe')
    kinship = 'maf1'
    kinship_val = ['maf1', 'maf2', 'maf3', 'pairwise', 'molpsy', 'molpsymaf10',
                   'molpsyvif3']
    # options mechanics
    parser = optparse.OptionParser()
    parser.add_option('-p', '--pheno',
                      help='path to genotyping file',
                      default=pheno, type="string")
    parser.add_option('-k', '--kinship',
                      help='Kinship matrix from bank [' +
                           ' '.join(kinship_val)+']',
                      default=kinship, type="string")

    (options, args) = parser.parse_args()
    if options.kinship not in kinship_val:
        msg = 'Val: %s is not in [ ' % options.kinship
        msg = msg + ' '.join(kinship_val) + ' ]'
        print msg
        exit(-1)

    #covariates
    qcov = '/neurospin/brainomics/imagen_central/covar' + '/AgeIBS.qcovar'
    cov = os.path.join('/neurospin/brainomics/imagen_central/covar',
                       'covar_GenCitHan_GCTA.cov')

    # Fichiers de GRM
    grm = dict(maf1='pruned_m0.01_wsi50_wsk5_vif10.0',  # donne les meilleurs
               maf2='pruned_m0.02_wsi50_wsk5_vif10.0',
               maf3='pruned_m0.03_wsi50_wsk5_vif10.0',
               pairwise='qc_subjects_qc_genetics_all_snps_common_pruned_50_5_0.5',
               molpsymaf10='pruned_m0.10_g1_h6_wsi50_wsk5_vif10.0',
               molpsy='prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0',#pruned_m0.05_g1_h6_wsi50_wsk5_vif10.0',
               molpsyvif3='pruned_m0.05_g1_h6_wsi50_wsk5_vif3.0',
               )
    relgen = os.path.join('/neurospin/brainomics/imagen_central/kinship',
                          grm[options.kinship])

    # Phenotype multiples
    phen = options.pheno
    # order is hard coded : TOFIX urgently
#    sulc_pheno.columns
#    Index([u'FID', u'IID', u'STs_left_depthMax', u'STs_right_depthMax',
#           u'asym_STs_depthMax'],
#          dtype='object')
    phen_dict = {1: 'STs_left_depthMax',
                 2: 'STs_right_depthMax',
                 3: 'asym_STs_depthMax'}

    print '_______________________', relgen,  '_______________________'
    print 'This script select right handed subjects ONLY !!!!!!!!!!!'
    for k in phen_dict.keys():
#        print "===================== %s ==========================" % phen_dict[k]
#        print "===================== %s ==========================" % phen_dict[k]
        herit_compute(relgen, qcov, cov, phen, k, phen_dict[k])
#        print "===================== %s ==========================" % phen_dict[k]
#        print "===================== %s ==========================" % phen_dict[k]
