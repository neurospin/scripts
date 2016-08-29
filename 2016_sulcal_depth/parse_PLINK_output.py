"""
@author yl247234
"""
import os
import optparse
import subprocess
import pandas as pd
import tempfile


### INPUTS ###
WORKING_DIRECTORY = '/neurospin/brainomics/2016_sulcal_depth/PLINK_output/'
pheno_names = ['left']#'asym']#, 'right_depth', 'left_depth']
features1 = ['asym_max', 'asym_mean']
features2 = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage']

brut = WORKING_DIRECTORY+'brut_output_temp/'
parsed = WORKING_DIRECTORY+'parse_output_temp/'
for pheno_name in pheno_names:
    if 'asym' in pheno_name:
        features = features1
    else:
        features = features2
    for feature in features:
        if 'plis' in feature:
            linear = brut +'left_STAP_PLINK_covar_GenCitHan5PCA_ICV_Bv_PLINK.assoc.logistic'
        else:
            linear = brut +pheno_name+'_STAP_covar_GenCitHan5PCA_ICV_Bv_PLINK.'+feature+'.assoc.linear'
        

        ### OUTPUTS ###
        out = os.path.join(parsed,
                           pheno_name+'_'+feature+'_linear'+'.pval')
    
        outsel = os.path.join(parsed,
                              pheno_name+'_'+feature+'_linear'+'.sel5')


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
        pvalsub = pval.loc[pval['P'] < 5e-5]
        print pvalsub

        pvalsub.to_csv(outsel,
                       sep='\t', index=False)
