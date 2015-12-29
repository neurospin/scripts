"""
Created  11 07 2015

@author yl247234
"""
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26

number_bins = 100

if __name__ == "__main__":

    covar = '/neurospin/brainomics/imagen_central/covar/gender_centre_handedness.cov'
    df_covar = pd.read_csv(covar, delim_whitespace=True)
    df_covar.index = df_covar['IID']
    index_left = df_covar['IID'][df_covar['Handedness'] == 'Left']
    index_right = df_covar['IID'][df_covar['Handedness'] == 'Right']
    # threshold relative to the number of recognize features in each subject
    tol = 0.02
    path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno'+str(tol)+'/'
    i = 0
    sulcus_names = []
    for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe')):
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['IID']
        m = re.search(path+'(.+?)_tol0.02.phe', filename)
        if m:
            sulcus = m.group(1)
            sulcus_names.append(sulcus)
        left_depth = np.asarray(df[sulcus+'_left_depthMax'])
        left_depth_left = np.asarray(df.loc[index_left][np.isfinite(df.loc[index_left][sulcus+'_left_depthMax'])][sulcus+'_left_depthMax'])
        left_depth_right = np.asarray(df.loc[index_right][np.isfinite(df.loc[index_right][sulcus+'_left_depthMax'])][sulcus+'_left_depthMax'])
        right_depth_left = np.asarray(df.loc[index_left][np.isfinite(df.loc[index_left][sulcus+'_right_depthMax'])][sulcus+'_right_depthMax'])
        right_depth_right = np.asarray(df.loc[index_right][np.isfinite(df.loc[index_right][sulcus+'_right_depthMax'])][sulcus+'_right_depthMax'])
        right_depth = np.asarray(df[sulcus+'_right_depthMax'])
        asym_depth = np.asarray(df['asym_'+sulcus+'_depthMax'])
        asym_depth_left = np.asarray(df.loc[index_left][np.isfinite(df.loc[index_left]['asym_'+sulcus+'_depthMax'])]['asym_'+sulcus+'_depthMax'])
        asym_depth_right = np.asarray(df.loc[index_right][np.isfinite(df.loc[index_right]['asym_'+sulcus+'_depthMax'])]['asym_'+sulcus+'_depthMax'])

        if i == 0:
            a,b = min(left_depth), max(left_depth)
            n, bins, patches = plt.hist(left_depth_right, number_bins, facecolor='blue', range=(a,b), normed=True)
            print "Rights number:" + str(len(left_depth_right))
            n, bins, patches = plt.hist(left_depth_left, number_bins, facecolor='red', range=(a,b), alpha=.5, normed=True)
            print "Lefts number:" + str(len(left_depth_left))
            plt.xlabel('Depth max [mm]' ,fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Nb of subjects',fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Left part of '+ sulcus, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
            plt.figure()
            a,b = min(right_depth), max(right_depth)
            n, bins, patches = plt.hist(right_depth_right, number_bins, facecolor='blue', range=(a,b), normed=True)
            n, bins, patches = plt.hist(right_depth_left, number_bins, facecolor='red', range=(a,b), alpha=.5, normed=True)
            #n, bins, patches = plt.hist(right_depth, number_bins, facecolor='red')
            plt.xlabel('Depth max [mm]', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Nb of subjects',fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Right part of '+ sulcus, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
            plt.figure()
            a,b = min(asym_depth), max(asym_depth)
            n, bins, patches = plt.hist(asym_depth_right, number_bins, facecolor='blue', range=(a,b), normed=True)
            n, bins, patches = plt.hist(asym_depth_left, number_bins, facecolor='red', range=(a,b), alpha=.5, normed=True)
            #n, bins, patches = plt.hist(asym_depth*1e2, number_bins, facecolor='black')
            plt.xlabel('Percentage difference [%]' , fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Nb of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Asymetry in depth max of '+ sulcus, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
            plt.show()
        i += 1
