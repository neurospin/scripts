"""
@author yl247234
"""
import os, glob, re
import optparse
import subprocess
from numpy import log10

### INPUTS ###
qcovar = '/neurospin/brainomics/imagen_central/clean_covar/covar_5PCA_ICV_Bv_GCTA.qcov'
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCitHan_GCTA.cov'
pheno = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/pheno/left_STAP.phe'
columns_pheno = ['asym_mean','asym_max']
columns_pheno = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage']
GCTA = 'gcta64-1.25 --thread-num 4 '
genorate = 0.01
hwe = 1e-6
maf = 0.01
win_size = 50
win_skip = 5
vif = 10.0
grm_file = 'prunedYann_m'+str(maf)+'_g%d_h%d_wsi%d_wsk%d_vif%2.1f' % (int(100 * genorate), int(-log10(hwe)), win_size, win_skip, vif)
grm = '/neurospin/brainomics/imagen_central/kinship/'+ grm_file


### OUTPUTS ###
OUT_DIRECTORY = '/neurospin/brainomics/2016_sulcal_depth/GCTA_output/'


for i,c_pheno in enumerate(columns_pheno):
    out =  OUT_DIRECTORY+ '%s_Cov%s_Qcov%s_Phe%s' % (os.path.basename(grm),
                                                      'GenCitHan',
                                                      '5PCA_ICV_Bv',
                                                      ''+c_pheno)
    
    cmd = " ".join([GCTA,' --grm %s'% grm,
                    '--covar %s' % covar,
                    '--qcovar %s' % qcovar,
                    '--pheno %s' % pheno,
                    '--reml --out %s' % out,
                    '--mpheno %s' % str(i+1)])
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
    try:
        print '================ %s =================' % (pheno)
        cmd = 'cat ' + out + '.hsq'
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        print result
    except subprocess.CalledProcessError, ex: # error code <> 0
        print "--------error------"
        print ex.cmd
        print ex.message
        print ex.returncode
        print ex.output  # contains stdout and stderr togethe
