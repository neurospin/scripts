"""
@author yl247234
"""
import os
import optparse
import subprocess
import pandas as pd
import tempfile


### INPUTS ###
WORKING_DIRECTORY = '/neurospin/brainomics/2016_hippo_malrot/PLINK_output/'
pheno_names = ['SCi_L', 'SCi_R','Sci_L_thresh', 'Sci_R_thresh','C0_L', 'C0_R']
for pheno_name in pheno_names:
    linear = WORKING_DIRECTORY+'pheno_pruned_m0.01_g1_h6_wsi50_wsk5_vif10.0.'+pheno_name+'.assoc.linear'

    ### OUTPUTS ###
    out = os.path.join(WORKING_DIRECTORY,
                       pheno_name+'_logistic_pruned.pval')
    
    outsel = os.path.join(WORKING_DIRECTORY,
                          pheno_name +'_logistic_pruned.sel6')


    tmp = tempfile.mktemp()
    cmd = ["head -1 %s > %s" % (linear, tmp),
           ";",
           "grep ADD %s >> %s" % (linear, tmp)]
    print " ".join(cmd)
    p = subprocess.Popen(" ".join(cmd), shell=True)
    p.wait()
    cmd = ["awk '{print $1,$2,$3,$4,$5,$6,$7,$8,$9}' %s > %s " % (tmp, out)]
    print " ".join(cmd)
    #check_call
    p = subprocess.check_call(" ".join(cmd), shell=True)
    os.remove(tmp)
    pval = pd.io.parsers.read_csv(out, sep=' ')
    pvalsub = pval.loc[pval['P'] < 5e-6]
    print pvalsub

    pvalsub.to_csv(outsel,
                    sep='\t', index=False)
