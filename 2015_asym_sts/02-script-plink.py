"""
Created  01 16 2015

@author vf140245
"""
import os
import optparse
import subprocess

if __name__ == "__main__":
    geno = ('/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/'
        '2012_imagen_shfj/genetics/'
        'qc_sub_qc_gen_all_snps_common_autosome')
    pheno = ('/neurospin/brainomics/2015_asym_sts/data/'
        'sts_asym_rightonly.phe')
    covar = ('/neurospin/brainomics/2015_asym_sts/data/'
        'sts_gender_centre.cov')
    out = ('/neurospin/brainomics/2015_asym_sts/data/'
        'sts_gender_centre.cov')

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
    print cmd
    p = subprocess.check_call(cmd, shell=True)
