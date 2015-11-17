"""
Created  11 07 2015

@author yl247234
"""
import os, glob, re
import optparse
import subprocess
from numpy import log10

from comput_grm_gcta import select_snp_indep

if __name__ == "__main__":
    geno = ('/neurospin/brainomics/imagen_central/'
            'geno/qc_sub_qc_gen_all_snps_common_autosome')
    #covariates
    #qcov = '/neurospin/brainomics/imagen_central/covar/AgeIBS.qcovar'
    covar = '/neurospin/brainomics/imagen_central/covar/gender_centre.cov'
    ### FILTER THE SNPS ####
    genorate = 0.01
    hwe = 1e-6
    maf = 0.01
    vif = 10
    win_size = 50 
    win_skip = 5
    out = os.path.join('/neurospin/brainomics/2015_asym_sts/all_sulci_pvals_filter/',
                           'prunedYann_m%1.2f_g%d_h%d_wsi%d_wsk%d_vif%2.1f' % \
        (maf, int(100 * genorate), int(-log10(hwe)), win_size, win_skip, vif))
    cmd = " ".join(['plink --noweb --silent',
                    '--bfile %s' % geno,
                    '--maf %f' % maf,
                    '--geno %f' % genorate,
                    '--hwe %f' % hwe,
                    '--indep %d %d %f' % (win_size, win_skip, vif),
                    '--out %s' % out])
    p = subprocess.check_call(cmd, shell=True)    
    cmd = " ".join(['plink --noweb',
                    '--maf %f' % maf,
                    '--geno %f' % genorate,
                    '--hwe %f' % hwe,
                    '--bfile %s' % geno,
                    '--make-bed',
                    '--extract %s.prune.in' % out,
                    '--out %s' % out])
    p = subprocess.check_call(cmd, shell=True)

    filtered_geno = out
    # threshold relative to the number of recognize features in each subject
    tol = 0.02
    path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno'+str(tol)+'/'
    for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe')):
        pheno = filename
        parser = optparse.OptionParser()
        parser.add_option('-g', '--geno',
                          help='path to geno file',
                          default=filtered_geno, type="string")
        parser.add_option('-c', '--covar',
                          help='path to covar file',
                          default=covar, type="string")
        parser.add_option('-p', '--pheno',
                          help='path to pheno file',
                          default=pheno, type="string")
        (options, args) = parser.parse_args()
        out = os.path.join('/neurospin/brainomics/2015_asym_sts/all_sulci_pvals_filter/',
                           os.path.splitext(os.path.basename(options.pheno))[0] + '_' +
                           os.path.splitext(os.path.basename(options.covar))[0])
        cmd = " ".join(['plink --noweb',
                        '--linear',
                        '--bfile %s' % options.geno,
                        '--covar %s' % options.covar,
                        '--pheno %s' % options.pheno,
                        '--all-pheno',
                        '--out %s' % out])
        try:
            p = subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError, ex:  # error code <> 0
            if int(ex.returncode) == 127: # This case is due to the option --all-pheno, that somehow gives this error due to PLINK
                pass
            else:
                print "--------error------"
                print 'Command ' + ex.cmd + ' returned non-zero exit status ' + str(ex.returncode)
