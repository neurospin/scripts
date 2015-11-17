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
    """# threshold relative to the number of recognize features in each subject
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
            plt.show()"""
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
    for j in range(5,len(sulcus_names)):
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
            n, bins, patches = plt.hist(np.asarray(df['P_'+pheno[1]]), number_bins, facecolor='blue')
            plt.ylim([np.mean(n)-3*np.std(n),np.mean(n)+3*np.std(n)])
            plt.xlabel('Bins pval')
            plt.title('Left depthMax of '+ sulcus_names[j])
            plt.ylabel('Nb of P values')
            plt.figure()
            n, bins, patches = plt.hist(np.asarray(df['P_'+pheno[0]]), number_bins, facecolor='red')
            plt.ylim([np.mean(n)-3*np.std(n),np.mean(n)+3*np.std(n)])
            plt.xlabel('Bins pval')
            plt.title( 'Right depthMax of '+ sulcus_names[j])
            plt.ylabel('Nb of P values')
            if sulcus_names[j] == 'SRinf' or sulcus_names[j] == 'SPasup' or  sulcus_names[j] =='SpC' or sulcus_names[j] =='SOp' or sulcus_names[j] =='SForbitaire' or sulcus_names[j]=='FIPrint1' or sulcus_names[j] == 'FCLrsc' or sulcus_names[j] == 'FCLrretroCtr':
                plt.show()
                pass
            else:
                plt.figure()
                n, bins, patches = plt.hist(np.asarray(df['P_'+pheno[2]]), number_bins, facecolor='black')
                plt.ylim([np.mean(n)-3*np.std(n),np.mean(n)+3*np.std(n)])
                plt.xlabel('Bins pval')
                plt.ylabel('Nb of subjects')
                plt.title('Asym in depthMax of ' + sulcus_names[j])
                plt.show()
