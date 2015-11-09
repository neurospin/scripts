"""
Created  11 07 2015

@author yl247234
"""
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

number_bins = 100

if __name__ == "__main__":
    # threshold relative to the number of recognize features in each subject
    tol = 0.02
    path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno'+str(tol)+'/'
    i = 0
    for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe')):
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['IID']
        m = re.search(path+'(.+?)_tol0.02.phe', filename)
        if m:
            sulcus = m.group(1)
        left_depth = np.asarray(df[sulcus+'_left_depthMax'])
        right_depth = np.asarray(df[sulcus+'_right_depthMax'])
        asym_depth = np.asarray(df['asym_'+sulcus+'_depthMax'])

        if i == 0:
            i += 1
            n, bins, patches = plt.hist(left_depth, number_bins, facecolor='blue')
            plt.xlabel('Left depthMax of '+ sulcus)
            plt.ylabel('Nb of subjects')
            plt.figure()
            n, bins, patches = plt.hist(right_depth, number_bins, facecolor='red')
            plt.xlabel('Right depthMax of '+ sulcus)
            plt.ylabel('Nb of subjects')
            plt.figure()
            n, bins, patches = plt.hist(asym_depth, number_bins, facecolor='black')
            plt.xlabel('Asym in depthMax of '+ sulcus)
            plt.ylabel('Nb of subjects')
            plt.figure()
            plt.show()
  
