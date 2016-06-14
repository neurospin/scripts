
import os, glob, re
import pandas as pd
import numpy as np
import time

path = '/neurospin/imagen/workspace/connectogram_examples/BL/'
path_mask = path +'000000106601/probtrackx2/masks.txt'
file_matrix = '/probtrackx2/fdt_network_matrix'
nb_subjects =  959
nb_sigma = 2

## Extract name pf the regions ###
df = pd.read_csv(path_mask, header=None)
clist = df[0].tolist()
areas = []
for j in range(len(clist)):
    m = re.search('/volatile/imagen/connectogram/new_results/BL/000000106601/.*/(.+?).nii.gz', clist[j])
    if m:
        areas.append(m.group(1))

## Extract connectivity matrix of each subject ##
count = 0
t = time.time()
connectivity_matrix = np.zeros((nb_subjects, len(areas)*len(areas)))
subject_list = []
for directory in glob.glob(os.path.join(path,'*')):
    if os.path.isdir(directory) and count >= 0:
        connectivity_matrix[count,:] = np.loadtxt(directory+file_matrix).flatten()
        """A = np.loadtxt(directory+file_matrix)
        B = np.sum(A, axis=0)
        C = np.zeros((len(areas), len(areas)))
        for j in range(len(areas)):
            if B[j] != 0:
                C[j,:] = A[j,:]/B[j]
        D = np.sum(A, axis=1)
        E = np.zeros((len(areas), len(areas)))
        for j in range(len(areas)):
            if D[j] != 0:
                E[:,j] = C[:,j]/D[j]
        
        connectivity_matrix[count,:] = C.flatten()
        """
        subject_list.append(os.path.basename(directory))
        count += 1
elapsed = time.time() - t
print "Elapsed time to create connectivity matrix " + str(elapsed)


df = pd.DataFrame()
df['IID'] = subject_list
df.index = df['IID']
df['FID'] = df['IID']
for j in range(len(areas)*len(areas)):
    if np.count_nonzero(connectivity_matrix[:,j]) == nb_subjects:
        df[areas[j/len(areas)]+'_'+areas[j%len(areas)]] = np.asarray(connectivity_matrix[:,j])

out_path = '/neurospin/brainomics/2016_connectogram/normalized_connecto_pheno.phe'
df.to_csv(out_path, sep= '\t',  header=True, index=False)
df_reduced = df.loc[df.index[:len(df.index)-100]]
out_path2 = '/neurospin/brainomics/2016_connectogram/normalized_connecto_pheno_reduced.phe'
df_reduced.to_csv(out_path2, sep= '\t',  header=True, index=False)

df3 = pd.read_csv('/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_MEGHA.cov', sep= '\t',  header=False )
df3.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
               u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
               u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
               u'SNPSEX', u'ICV']

df3['IID'] = ['%012d' % int(i) for i in df3['IID']]
df3.index = df3['IID']

C= set(df3['IID'])
A = set(subject_list)
B = A.intersection(C)

df4 = pd.read_csv('/neurospin/brainomics/2016_sulcal_depth/pheno/PLINK_all_pheno0.02/all_sulci_qc/concatenated_pheno.phe', delim_whitespace = True)


for column in df4.columns:
    df4 = df4.loc[np.logical_not(np.isnan(df4[column]))]
df4['IID'] = ['%012d' % int(i) for i in df4['IID']]
df4.index = df4['IID']

E= set(df4['IID'])
F = A.intersection(E)
T = B-F
T = list(T)
df5 = df3.loc[T]
for column in df5.columns:
    if column != 'IID':
        df5 = df5.loc[np.isfinite(df3[column])]
