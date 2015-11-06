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
    # threshold relative to the number of recognize features in each subject
    tol = 0.02
    path = '/neurospin/brainomics/2015_asym_sts/all_pheno'+str(tol)+'/'
    for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe')):
        pheno = filename

    #covariates
    #qcov = '/neurospin/brainomics/imagen_central/covar/AgeIBS.qcovar'
    covar = '/neurospin/brainomics/imagen_central/covar/covar_GenCitHan_GCTA.cov'

    parser = optparse.OptionParser()
    parser.add_option('-p', '--pheno',
                      help='path to pheno file',
                      default=pheno, type="string")
    parser.add_option('-g', '--geno',
                      help='path to geno file',
                      default=geno, type="string")
    parser.add_option('-c', '--covar',
                      help='path to covar file',
                      default=covar, type="string")

    (options, args) = parser.parse_args()
    out = os.path.join(os.path.dirname(options.pheno),
                       os.path.splitext(os.path.basename(options.pheno))[0] + '_' +
                       os.path.splitext(os.path.basename(options.covar))[0])

    cmd = " ".join(['plink --noweb --linear',
           '--bfile %s' % options.geno,
           '--covar %s' % options.covar,
           '--pheno %s' % options.pheno,
           '--all-pheno',
            '--out %s' % out])
            #    ' --snps rs2483275'])
    print cmd
    try:
        p = subprocess.check_call(cmd, shell=True)
    except Exception as e:
        m = re.search('exit status (.+?)end', str(e)+"end")
        if m:
            error_number = m.group(1)
        if int(error_number) == 127:
            pass
        else:
            print (e)

    #Another way to write it, beginning:
    except subprocess.CalledProcessError, ex:  # error code <> 0
            print "--------error------"
            print '\n'.join([ex.cmd, ex.message, str(ex.returncode), ex.output])
        if int(ex.returncode) == 127:
            pass
        else:
            print (e)
