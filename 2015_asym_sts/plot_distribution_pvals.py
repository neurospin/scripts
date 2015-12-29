"""
Created  11 07 2015

@author yl247234
"""
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 24
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 28

if __name__ == "__main__":
    path = '/neurospin/brainomics/2015_asym_sts/all_sulci_pvals/'
    features = ["depthMax"]
    pheno = ["right", "left", "asym"]
    sulcus_names = []
    i = 0
    df = pd.DataFrame()
    for filename in glob.glob(os.path.join(path,'*_left_depthMax.assoc.pval')):
        m = re.search(path+'(.+?)_tol0.02(.*)_left_depthMax.assoc.pval', filename)
        if m:
            sulcus = m.group(1)
            print "Sulcus: " + str(sulcus)
            sulcus_names.append(sulcus)
    for j in range(len(sulcus_names)):
        df = pd.DataFrame()
        for k in range(len(pheno)):
            if k ==2:
                if sulcus_names[j] == 'SRinf' or sulcus_names[j] == 'SPasup' or  sulcus_names[j] =='SpC' or sulcus_names[j] =='SOp' or sulcus_names[j] =='SForbitaire' or sulcus_names[j]=='FIPrint1' or sulcus_names[j] == 'FCLrsc' or sulcus_names[j] == 'FCLrretroCtr':
                    break
                else:
                    filename  = path+sulcus_names[j]+'_tol0.02_gender_centre.'+pheno[k]+'_'+sulcus_names[j]+'_depthMax.assoc.pval'
            else:
                filename  = path+sulcus_names[j]+'_tol0.02_gender_centre.'+sulcus_names[j]+'_'+pheno[k]+'_depthMax.assoc.pval'
            df_temp = pd.read_csv(filename, delim_whitespace=True)
            if k == 0:
                df['BP'] = df_temp['BP']
            df['P_'+pheno[k]]= df_temp['P']
        
        df.index = df['BP']
        if i != -1:
            i += 1
            number_bins = 20
            nb_std = 3
            n, bins, patches = plt.hist(np.asarray(df['P_'+pheno[1]]), number_bins, facecolor='blue')
            plt.ylim([np.mean(n)-nb_std*np.std(n),np.mean(n)+nb_std*np.std(n)])
            plt.xlabel('Bins pval', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Nb of P values', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Left depthMax of '+ sulcus_names[j], fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom" )
            plt.figure()
            n, bins, patches = plt.hist(np.asarray(df['P_'+pheno[0]]), number_bins, facecolor='red')
            plt.ylim([np.mean(n)-nb_std*np.std(n),np.mean(n)+nb_std*np.std(n)])
            plt.xlabel('Bins pval', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Nb of P values', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title( 'Right depthMax of '+ sulcus_names[j],fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom" )
            if sulcus_names[j] == 'SRinf' or sulcus_names[j] == 'SPasup' or  sulcus_names[j] =='SpC' or sulcus_names[j] =='SOp' or sulcus_names[j] =='SForbitaire' or sulcus_names[j]=='FIPrint1' or sulcus_names[j] == 'FCLrsc' or sulcus_names[j] == 'FCLrretroCtr':
                plt.show()
                pass
            else:
                plt.figure()
                n, bins, patches = plt.hist(np.asarray(df['P_'+pheno[2]]), number_bins, facecolor='black')
                plt.ylim([np.mean(n)-nb_std*np.std(n),np.mean(n)+nb_std*np.std(n)])
                plt.xlabel('Bins pval', fontsize=text_size, fontweight = 'bold', labelpad=0)
                plt.ylabel('Nb of P values', fontsize=text_size, fontweight = 'bold', labelpad=0)
                plt.title('Asym in depthMax of ' + sulcus_names[j], fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
                plt.show()
