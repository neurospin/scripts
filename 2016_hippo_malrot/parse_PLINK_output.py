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
#pheno_names = ['SCi_L', 'SCi_R','Sci_L_thresh', 'Sci_R_thresh','C0_L', 'C0_R']
#pheno_names = ['SCi_L', 'SCi_R','Sci_L_thresh', 'Sci_R_thresh']
pheno_names = ['Sci_L_thresh', 'Sci_R_thresh', 'SCi_L', 'SCi_R', 'SCi_L_R']
case = ''
brut = WORKING_DIRECTORY+'brut_output_update/'
parsed = WORKING_DIRECTORY+'parse_output_update/'
for pheno_name in pheno_names:
    if 'C0' in pheno_name or 'thresh' in pheno_name:
        linear = brut+'hippo_IHI'+case+'_covar_GenCit5PCA_PLINK.'+pheno_name+'.assoc.logistic'
    else:
        linear = brut+'hippo_IHI'+case+'_covar_GenCit5PCA_PLINK.'+pheno_name+'.assoc.linear'

    ### OUTPUTS ###
    out = os.path.join(parsed,
                       pheno_name+case+'.pval')
    
    outsel = os.path.join(parsed,
                          pheno_name +case+'.sel2')


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
    pvalsub = pval.loc[pval['P'] < 5e-2]
    print pvalsub

    pvalsub.to_csv(outsel,
                    sep='\t', index=False)
