"""
Created  11 07 2015

@author yl247234
"""
import os, glob, re
import optparse
import subprocess

if __name__ == "__main__":
    geno = ('/neurospin/brainomics/imagen_central/'
            'geno/qc_sub_qc_gen_all_snps_common_autosome')
    #covariates
    #qcov = '/neurospin/brainomics/imagen_central/covar/AgeIBS.qcovar'
    covar = '/neurospin/brainomics/imagen_central/covar/gender_centre.cov'

    # threshold relative to the number of recognize features in each subject
    tol = 0.02
    path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno'+str(tol)+'/'
    for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe')):
        pheno = filename
        parser = optparse.OptionParser()
        parser.add_option('-g', '--geno',
                          help='path to geno file',
                          default=geno, type="string")
        parser.add_option('-c', '--covar',
                          help='path to covar file',
                          default=covar, type="string")
        parser.add_option('-p', '--pheno',
                          help='path to pheno file',
                          default=pheno, type="string")
  
        (options, args) = parser.parse_args()
        out = os.path.join('/neurospin/brainomics/2015_asym_sts/all_sulci_pvals/',
                           os.path.splitext(os.path.basename(options.pheno))[0] + '_' +
                           os.path.splitext(os.path.basename(options.covar))[0])

        cmd = " ".join(['plink --noweb --linear',
                        '--bfile %s' % options.geno,
                        '--covar %s' % options.covar,
                        '--pheno %s' % options.pheno,
                        '--all-pheno',
                        '--out %s' % out])
        try:
            p = subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError, ex:  # error code <> 0
            if int(ex.returncode) == 127:
                pass
            else:
                print "--------error------"
                print 'Command ' + ex.cmd + ' returned non-zero exit status ' + str(ex.returncode)
