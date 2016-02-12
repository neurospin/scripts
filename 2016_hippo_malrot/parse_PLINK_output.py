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
#pheno_names = ['SCi_L', 'SCi_R','Sci_L_thresh', 'Sci_R_thresh']
case = '_binary_PLINK'
brut = WORKING_DIRECTORY+'brut_output/'
parsed = WORKING_DIRECTORY+'parse_output/'
for pheno_name in pheno_names:
    if 'C0' in pheno_name or 'thresh' in pheno_name:
        linear = brut+'hippo_IHI'+case+'_covar_GenCit5PCA_ICV_PLINK.'+pheno_name+'.assoc.logistic'
    else:
        linear = brut+'hippo_IHI'+case+'_covar_GenCit5PCA_ICV_PLINK.'+pheno_name+'.assoc.linear'

    ### OUTPUTS ###
    out = os.path.join(parsed,
                       pheno_name+'_logistic'+case+'.pval')
    
    outsel = os.path.join(parsed,
                          pheno_name +'_logistic'+case+'.sel3')


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
    pvalsub = pval.loc[pval['P'] < 5e-3]
    print pvalsub

    pvalsub.to_csv(outsel,
                    sep='\t', index=False)
