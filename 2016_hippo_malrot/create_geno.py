"""
@author yl247234
"""
import os, glob, re
import optparse
import subprocess
from numpy import log10

### INPUTS ###
geno = '/neurospin/brainomics/imagen_central/geno/qc_sub_qc_gen_all_snps_common_autosome'
# to FILTER THE SNPS
genorate = 0.01
hwe = 1e-6
maf = 0.01


### OUTPUTS ###
WORKING_DIRECTORY = '/neurospin/brainomics/2016_hippo_malrot/PLINK_output/geno/'
OUTPUT_FILTERED_GENO = os.path.join(WORKING_DIRECTORY,
                                    'pruned_m%1.2f_g%d_h%d' % \
                                    (maf, int(100 * genorate), int(-log10(hwe))))
                                    

if __name__ == "__main__":
    cmd = " ".join(['plink --noweb',
                    '--maf %f' % maf,
                    '--geno %f' % genorate,
                    '--hwe %f' % hwe,
                    '--bfile %s' % geno,
                    '--make-bed',
                    '--out %s' % OUTPUT_FILTERED_GENO])
    p = subprocess.check_call(cmd, shell=True)
