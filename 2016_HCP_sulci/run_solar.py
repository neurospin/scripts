"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re
import pandas as pd
import time
CC_ANALYSIS = True

sides = ['R', 'L']
sds = {'R' : 'right', 'L': 'left'}
working_dir = "sulci_analysis"
pheno_dir = '/neurospin/brainomics/2016_HCP/sulci_data_all_subjects/'
features = ['surface_native', 'maxdepth_native', 'meandepth_native', 'hull_junction_length_native', 'GM_thickness', 'opening']
sulci_excluded = ['F.C.M.r.AMS.ant.', 'S.Or.l.', 'S.R.sup.']

os.system("solar makeped "+ working_dir)
for side in sides:
    for i in range(5):
        print '\n'
    print "====================================Side " +sds[side]+"========================================================"
    for i in range(5):
        print '\n'
    for filename in glob.glob(os.path.join(pheno_dir,'*.csv')):
        m = re.search(pheno_dir+'(.+?)_'+sds[side]+'.csv', filename)
        if m:
            sulcus = m.group(1)
            output_dir = sulcus+'_'+sds[side]
            if sulcus not in sulci_excluded:
                for trait in features:
                    print "Heritability estimate of "+sulcus+ " "+trait
                    os.system("solar pheno_analysis "+working_dir+" "+output_dir+" "+trait+" "+filename)
                    #time.sleep(1)


