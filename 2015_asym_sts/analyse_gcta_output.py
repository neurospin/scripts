# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 13:27:07 2015

@author: yl247234
Copyrignt : CEA NeuroSpin - 2014
"""

import os, glob, re
import pheno as pu
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # threshold one the number of recognize features in each subject
    tol = 0.02
    path = '/neurospin/brainomics/2015_asym_sts/all_pheno'+str(tol)+'/hsq_files/'
    
    pheno = ['right', 'left', 'asym']
    for j in range(len(pheno)):
        print '================ %s =================' % pheno[j]
        sulcus_names = []
        variance_explained = []
        p_values  = []
        std_variance_explained = []
        for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe_'+pheno[j]+'_depthMax.hsq')):
            m = re.search('prunedYann_m0.01_g1_h6_wsi50_wsk5_vif10.0_Covcovar_GenCitHan_GCTA.cov_QcovAgeIBS.qcovar_Phe(.+?)_tol0.02.phe_'+pheno[j]+'_depthMax.hsq', filename)
            if m:
                sulcus = m.group(1)
                #print "Sulcus: " + str(sulcus)
                sulcus_names.append(sulcus)
        
                df = pd.read_csv(filename, delim_whitespace=True)
                variance_explained.append(df["Variance"][3])
                p_values.append(df["Variance"][8])
                std_variance_explained.append(df["SE"][3])
                df =  pd.DataFrame({'Sulci': np.asarray(sulcus_names),
                                    'V(G)/Vp': np.asarray(variance_explained),
                                    'Pval': np.asarray(p_values)
                                    #    'SE V(G)/Vp': np.asarray(std_variance_explained)
                                })

        df.index = df['Sulci']
        df = df[['V(G)/Vp', 'Pval']]
        #print df
        
        index_p_values = np.nonzero(np.less(p_values, 0.05))[0]
        variance_explained_selected = []
        p_values_selected  = []
        std_variance_explained_selected = []
        sulcus_names_selected = []
        for j in range(len(index_p_values)):
            variance_explained_selected.append(variance_explained[index_p_values[j]])
            p_values_selected.append(p_values[index_p_values[j]])
            std_variance_explained_selected.append(std_variance_explained[index_p_values[j]])
            sulcus_names_selected.append(sulcus_names[index_p_values[j]])
        df =  pd.DataFrame({'Sulci': np.asarray(sulcus_names_selected),
                            'V(G)/Vp': np.asarray(variance_explained_selected),
                            'Pval': np.asarray(p_values_selected),
                            'SE V(G)/Vp': np.asarray(std_variance_explained_selected)
                        })

        df.index = df['Sulci']
        df = df[['V(G)/Vp', 'Pval', 'SE V(G)/Vp']]
        print df
