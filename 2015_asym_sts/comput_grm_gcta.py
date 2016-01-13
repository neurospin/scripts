# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:32:04 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

import os
from glob import glob
import subprocess
import optparse
import shutil
from tempfile import mkdtemp
from shutil import rmtree, move
from numpy import log10



GCTA = 'gcta64-1.24 --thread-num 4 '
PLINK = 'plink --noweb --silent '



def select_snp_indep(geno, vif=10, maf=0.01, win_size=50, win_skip=5, frc=False):
    """ Linkage disequilibrium pruning. See plink doc chap 5.8

    Parameters:
    -----------
    pairwise_thresh: float,
        see plink doc chap 5.8

    win_size: int,
        see plink doc chap 5.8

    win_skip: int,
        see plink doc chap 5.8

    """
    print "GENO:  " + geno
    # 1) perform the filtering only maf vary
    genorate = 0.05  # previous 0.05
    hwe = 1e-6       # previous 1e-4
    cmd = PLINK+' --maf %f --geno %f --hwe %f ' % (maf, genorate, hwe)
    cmd = cmd+' --bfile %s --indep %d %d %f' \
        % (geno, win_size, win_skip, vif)
    temp_dir = mkdtemp()
    fout = temp_dir + '/kinship_matrix_m'+str(maf)+'_g%d_h%d_wsi%d_wsk%d_vif%2.1f' % \
        (int(100 * genorate), int(-log10(hwe)), win_size, win_skip, vif)
    cmd = cmd+' --out '+fout

    if not os.path.exists(fout+'.prune.in') or frc:
        try:
            print cmd
            #prints results and merges stdout and std
            result = subprocess.check_output(cmd,
                                             stderr=subprocess.STDOUT,
                                             shell=True)
        except subprocess.CalledProcessError, ex:  # error code <> 0
            print "--------error------"
            print '\n'.join([ex.cmd, ex.message, str(ex.returncode), ex.output])

    # 2) create new plink genotype file
    cmd = PLINK+' --maf %f --geno %f --hwe %f ' % (maf, genorate, hwe)
    fsamp = fout.split('.prune')[0]
    cmd = cmd+' --bfile %s --extract %s.prune.in --make-bed --out %s' % \
        (geno, fout, fsamp)

    if not os.path.exists(fsamp+'.bed') or frc:
        try:
            print cmd
            #prints results and merges stdout and std
            result = subprocess.check_output(cmd,
                                             stderr=subprocess.STDOUT,
                                             shell=True)
        except subprocess.CalledProcessError, ex:  # error code <> 0
            print "--------error------"
            print '\n'.join([ex.cmd, ex.message, str(ex.returncode), ex.output])

    return fsamp, temp_dir


def compute_grm(gf):
    """ Wrapper to compute gcta genetic relationship matrix
    only autosome chromosome are considered

    Parameters:
    -----------
    gf : string,
    genotyping filename original or pruned

    """
    print "GF:  " + gf
    cmd0 = GCTA+' --bfile %s --make-grm-gz' %(gf)
    #cmd0 = GCTA+' --bfile %s --make-grm' %(gf)
    log = gf.split('.prune')[0]+'_log'
    chrom = gf.split('.prune')[0]+'_chr'
    for c in range(1, 23):
        cmd = cmd0+' --chr %d --out %s%d > %s%d' % (c, chrom, c,log, c)
        try:
            print cmd
            #prints results and merges stdout and std
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError, ex:  # error code <> 0
            print "--------error------"
            print '\n'.join([ex.cmd, ex.message, str(ex.returncode), ex.output])

    tmp = ['%s_chr%d'%(gf.split('.prune')[0],i) for i in range(1, 23)]
    multi_out = gf.split('.prune')[0]+'_multi_grm.txt'
    fp = open(multi_out, 'w')
    _ = [fp.write('%s\n'%i) for i in tmp]
    fp.close()

    fout = gf.split('.prune')[0]
    cmd = GCTA+' --grm-cutoff 0.025 --mgrm-gz %s --make-grm-gz --out %s' %(multi_out, fout)
    #cmd = GCTA+' --grm-cutoff 0.025 --mgrm %s --make-grm --out %s' %(multi_out, fout)
    try:
        print cmd
        #prints results and merges stdout and std
        result = subprocess.check_output(cmd,
                                         stderr=subprocess.STDOUT,
                                         shell=True)
    except subprocess.CalledProcessError, ex:  # error code <> 0
        print "--------error------"
        print '\n'.join([ex.cmd, ex.message, str(ex.returncode), ex.output])

#    fl = glob(chrom+'*.gz')
#    fl = fl+glob(chrom+'*.id')
#    print fl
#    _ = [os.remove(i) for i in fl]
    
    return fout
    


if __name__ == "__main__":
    ### HERE IS TO CHANGE BY GETTING THE FILE DIRECTLY FROM IMAGEN2 ###
    bfile =('/volatile/yann/imagen_central/geno/'
            'qc_subjects_qc_genetics_all_snps_common')
    maf = [0.001, 0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.20]
    win_size = 50
    win_skip = 5
    vif = [2.0, 3.0, 4.0, 5.0,8.0, 10.0, 20.0]
    maf = [0.01]
    vif = [10]
    for j in range(len(vif)):
        for i in range(len(maf)):
            outdir = ('/volatile/yann/imagen_central/kinship/')

            parser = optparse.OptionParser()
            parser.add_option('-m', '--maf',
                              help='maf',
                              default=maf[i], type="float")
            parser.add_option('-v', '--vif',
                              help='vif value',
                              default=vif[j], type="float")
            parser.add_option('-b', '--bfile',
                              help='path to genotyping file',
                              default=bfile, type="string")
            parser.add_option('-o', '--outdir',
                              help='Out directory',
                              default=outdir, type="string")

            (options, args) = parser.parse_args()
            print options
            #    out = os.path.join(os.path.dirname(options.pheno),
            #                       os.path.splitext(os.path.basename(options.pheno))[0] + '_' +
            #                       os.path.splitext(os.path.basename(options.covar))[0])
    
            #    select_snp(0.5)
            pruned_geno, temp_todel = select_snp_indep(bfile, 
                                                       vif=options.vif, maf=options.maf, frc=True)
    
            fout = compute_grm(pruned_geno)
            print fout
            #for i in ['.grm.id', '.grm.bin', '.grm.N.bin']:
            for i in ['.grm.id', '.grm.gz']:
                src = fout + i
                dest = os.path.join(outdir, os.path.basename(src))
                move(src, dest)
            rmtree(temp_todel)

