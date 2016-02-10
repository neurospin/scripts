"""
@author yl247234
"""
import os, glob, re
import optparse
import subprocess
from numpy import log10

### INPUTS ###
geno = '/neurospin/brainomics/imagen_central/geno/qc_sub_qc_gen_all_snps_common_autosome'
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_PLINK.cov'
pheno = '/neurospin/brainomics/2016_hippo_malrot/pheno/hippo_IHI_pruned.phe'
# to FILTER THE SNPS
genorate = 0.01
hwe = 1e-6
maf = 0.01
vif = 10
win_size = 50 
win_skip = 5


### OUTPUTS ###
WORKING_DIRECTORY = '/neurospin/brainomics/2016_hippo_malrot/PLINK_output/'
OUTPUT_FILTERED_GENO = os.path.join(WORKING_DIRECTORY,
                      'pheno_pruned_m%1.2f_g%d_h%d_wsi%d_wsk%d_vif%2.1f' % \
                      (maf, int(100 * genorate), int(-log10(hwe)), win_size, win_skip, vif))
        
OUTPUT = os.path.join(WORKING_DIRECTORY,
                      os.path.splitext(os.path.basename(pheno))[0] + '_' +
                      os.path.splitext(os.path.basename(covar))[0])



if __name__ == "__main__":
    cmd = " ".join(['plink --noweb --silent',
                    '--bfile %s' % geno,
                    '--maf %f' % maf,
                    '--geno %f' % genorate,
                    '--hwe %f' % hwe,
                    '--indep %d %d %f' % (win_size, win_skip, vif),
                    '--out %s' % OUTPUT_FILTERED_GENO])
    p = subprocess.check_call(cmd, shell=True)    
    cmd = " ".join(['plink --noweb',
                    '--maf %f' % maf,
                    '--geno %f' % genorate,
                    '--hwe %f' % hwe,
                    '--bfile %s' % geno,
                    '--make-bed',
                    '--extract %s.prune.in' % OUTPUT_FILTERED_GENO,
                    '--out %s' % OUTPUT_FILTERED_GENO])
    p = subprocess.check_call(cmd, shell=True)
    
    cmd = " ".join(['plink --noweb',
                    '--logistic',
                    '--bfile %s' % geno,
                    '--covar %s' % covar,
                    '--pheno %s' % pheno,
                    '--all-pheno',
                    '--out %s' % OUTPUT_FILTERED_GENO])
    try:
        p = subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError, ex:  # error code <> 0
        if int(ex.returncode) == 127: # This case is due to the option --all-pheno, that somehow gives this error due to PLINK
            pass
        else:
            print "--------error------"
            print 'Command ' + ex.cmd + ' returned non-zero exit status ' + str(ex.returncode)
