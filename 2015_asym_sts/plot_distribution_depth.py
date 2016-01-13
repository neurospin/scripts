"""
Created  11 07 2015

@author yl247234
"""
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26

number_bins = 100

if __name__ == "__main__":

    covar = '/neurospin/brainomics/imagen_central/covar/gender_centre.cov'
    df_covar = pd.read_csv(covar, delim_whitespace=True)
    df_covar.index = df_covar['IID']
    index_female = df_covar['IID'][df_covar['Gender_Female'] == 1]
    index_male = df_covar['IID'][df_covar['Gender_Female'] == 0]
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
        left_depth_female = np.asarray(df.loc[index_female][np.isfinite(df.loc[index_female][sulcus+'_left_depthMax'])][sulcus+'_left_depthMax'])
        left_depth_male = np.asarray(df.loc[index_male][np.isfinite(df.loc[index_male][sulcus+'_left_depthMax'])][sulcus+'_left_depthMax'])
        right_depth_female = np.asarray(df.loc[index_female][np.isfinite(df.loc[index_female][sulcus+'_right_depthMax'])][sulcus+'_right_depthMax'])
        right_depth_male = np.asarray(df.loc[index_male][np.isfinite(df.loc[index_male][sulcus+'_right_depthMax'])][sulcus+'_right_depthMax'])
        right_depth = np.asarray(df[sulcus+'_right_depthMax'])
        asym_depth = np.asarray(df['asym_'+sulcus+'_depthMax'])
        asym_depth_female = np.asarray(df.loc[index_female][np.isfinite(df.loc[index_female]['asym_'+sulcus+'_depthMax'])]['asym_'+sulcus+'_depthMax'])
        asym_depth_male = np.asarray(df.loc[index_male][np.isfinite(df.loc[index_male]['asym_'+sulcus+'_depthMax'])]['asym_'+sulcus+'_depthMax'])

        if i == 0:
            a,b = min(left_depth), max(left_depth)
            n, bins, patches = plt.hist(left_depth_male, number_bins, facecolor='blue', range=(a,b), normed=True, label='men')
            print "Males number:" + str(len(left_depth_male))
            n, bins, patches = plt.hist(left_depth_female, number_bins, facecolor='red', range=(a,b), alpha=.5, normed=True, label='women')
            print "Females number:" + str(len(left_depth_female))
            plt.xlabel('Depth max [mm]' ,fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Proportion of subjects',fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Left part of '+ sulcus, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
            plt.legend(loc=1,prop={'size':20})
            plt.figure()
            a,b = min(right_depth), max(right_depth)
            n, bins, patches = plt.hist(right_depth_male, number_bins, facecolor='blue', range=(a,b), normed=True, label='men')
            n, bins, patches = plt.hist(right_depth_female, number_bins, facecolor='red', range=(a,b), alpha=.5, normed=True, label='women')
            #n, bins, patches = plt.hist(right_depth, number_bins, facecolor='red')
            plt.xlabel('Depth max [mm]', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Proportion of subjects',fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Right part of '+ sulcus, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
            plt.legend(loc=1,prop={'size':20})
            plt.figure()
            a,b = min(asym_depth), max(asym_depth)
            n, bins, patches = plt.hist(asym_depth_male, number_bins, facecolor='blue', range=(a,b), normed=True, label='men')            
            n, bins, patches = plt.hist(asym_depth_female, number_bins, facecolor='red', range=(a,b), alpha=.5, normed=True, label='women')
            #n, bins, patches = plt.hist(asym_depth*1e2, number_bins, facecolor='black')
            plt.xlabel('Percentage difference [%]' , fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
            plt.title('Asymmetry in depth max of '+ sulcus, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
            plt.legend(loc=1,prop={'size':20})
            #plt.show()
        i += 1


n_m, mu_m, std_m = len(asym_depth_male), np.mean(asym_depth_male), np.std(asym_depth_male)
n_f, mu_f, std_f = len(asym_depth_female), np.mean(asym_depth_female), np.std(asym_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
dof = n_m+n_f-2
print "Degree of freedom: "+ str(dof)
print "Males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Asym t-value (saying std approximately equal): " +str(t)
print "Asym t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

n_m, mu_m, std_m = len(right_depth_male), np.mean(right_depth_male), np.std(right_depth_male)
n_f, mu_f, std_f = len(right_depth_female), np.mean(right_depth_female), np.std(right_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
dof = n_m+n_f-2
print "Degree of freedom: "+ str(dof)
print "Males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Right t-value (saying std approximately equal): " +str(t)
print "Right t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

n_m, mu_m, std_m = len(left_depth_male), np.mean(left_depth_male), np.std(left_depth_male)
n_f, mu_f, std_f = len(left_depth_female), np.mean(left_depth_female), np.std(left_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
z = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)+math.pow(std_f,2))
dof = n_m+n_f-2
print "Degree of freedom: "+ str(dof)
print "Males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Left t-value (saying std approximately equal): " +str(t)
print "Left t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

plt.show()
