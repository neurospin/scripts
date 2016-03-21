"""
@author yl247234
"""
import os, glob, re
import optparse
import subprocess
from numpy import log10

### INPUTS ###
geno = '/neurospin/brainomics//2016_sulcal_depth/PLINK_output/geno/pruned_m0.01_g1_h6' 
#249058 variants remaining over the 466125 variants loaded from .bim file
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCitHan5PCA_ICV_Bv_PLINK.cov'
pheno = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/pheno/left_STAP_PLINK.phe'



### OUTPUTS ###
WORKING_DIRECTORY = '/neurospin/brainomics/2016_sulcal_depth/PLINK_output/brut_output/'
OUTPUT = os.path.join(WORKING_DIRECTORY,
                      os.path.splitext(os.path.basename(pheno))[0] + '_' +
                      os.path.splitext(os.path.basename(covar))[0])

if __name__ == "__main__":

    """Ignoring phenotypes of missing-sex samples.  If you don't want those
phenotypes to be ignored, use the --allow-no-sex flag."""
    cmd = " ".join(['plink --noweb',
                    '--logistic',
                    '--bfile %s' % geno,
                    '--covar %s' % covar,
                    '--pheno %s' % pheno,
                    '--all-pheno',
                    '--allow-no-sex',
                    '--out %s' % OUTPUT])
    try:
        p = subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError, ex:  # error code <> 0
        if int(ex.returncode) == 127: # This case is due to the option --all-pheno, that somehow gives this error due to PLINK
            pass
        else:
            print "--------error------"
            print 'Command ' + ex.cmd + ' returned non-zero exit status ' + str(ex.returncode)
