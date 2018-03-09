#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:09:03 2018

@author: ad247405
"""


import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
from nibabel import gifti
from sklearn.cluster import KMeans


##############################################################################
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"
pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/population.csv")
site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

pop= pop[pop["site_num"]==3]
age = pop["age"].values
sex = pop["sex_num"].values



labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/thickness/\
corrected_results/3_clusters/labels_cluster.npy")
labels_cluster = labels_cluster[site==3]
pop["labels_cluster"] =labels_cluster


indir = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/genetic_nusdast"
lut_file = os.path.join(indir, "NUSDAST_id_lut.csv")
lut = pd.read_csv(lut_file)
lut["ID"] = lut["BIRNid"]
fname_map = os.path.join(indir, "gwas_file_text.map")
fname_ped = os.path.join(indir, "gwas_file_text.ped")
fname_freq = os.path.join(indir, "plink.frq")

df_map = pd.read_csv(fname_map, sep=r"\s+",names=['CHR', 'SNP', 'GenDistance', 'BP-POS'])
snps = list(df_map['SNP'])

df_ped = pd.read_csv(fname_ped, sep=" ",header = None)
names = list()
for i in range(len(snps)):
        names.append(snps[i]+"A1")
        names.append(snps[i]+"A2")

df_ped.columns = ["a","ID","b","c","d","e"] + names
df_ped = df_ped.drop(["a","b","c","d","e"],axis=1)

#Transofrm each SNP values into 0,1 or Minor allelle for each SNP
df_freq = pd.read_csv(fname_freq,sep=r"\s+",header = 0)

for i in range(len(snps)):
    minor_allele = df_freq[df_freq["SNP"] == snps[i]]["A1"].values[-1]
    major_allele = df_freq[df_freq["SNP"] == snps[i]]["A2"].values[-1]
    both_allele = df_ped[snps[i]+"A1"].values + df_ped[snps[i]+"A2"].values
    df_ped[snps[i]] = both_allele
    di = {minor_allele+minor_allele:2,\
          minor_allele+major_allele:1,\
          major_allele+minor_allele:1,\
          major_allele+major_allele:0}
    df_ped[snps[i]] = df_ped[snps[i]].map(di)


df_ped = df_ped.merge(lut,on = "ID")
df_ped["subjectid"] =df_ped["ccid"]


#df_freq["MAF"].max()
sum(df_freq["MAF"]>0.35)


df_ped = df_ped.merge(pop,on="subjectid")
df_ped = df_ped[["subjectid","dx_num","age","sex_num","labels_cluster"] +snps]
y = df_ped["dx_num"].values
labels = df_ped["labels_cluster"].values
assert sum(y==0)== 65
assert sum(y==1)== 72


assert sum(labels=='Controls')== 65
assert sum(labels=="SCZ Cluster 1")== 26
assert sum(labels=="SCZ Cluster 2")== 21
assert sum(labels=="SCZ Cluster 3")== 25

##############################################################################

# STATISTICS
##############################################################################
#anova
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_ROIs/results/thick+vol/3_clusters/nudast/genetic"

df_genetic_stats = pd.DataFrame(columns=["T","p"])
df_genetic_stats.insert(0,"SNP",snps)
for s in snps:
    print (s)
    gene = df_ped[s].values
    labels_ = labels[np.array(np.isnan(gene)==False)]
    gene = gene[np.array(np.isnan(gene)==False)]
    assert gene.shape == labels_.shape

    T, p = scipy.stats.f_oneway(gene[labels_=='SCZ Cluster 1'],\
                          gene[labels_=='SCZ Cluster 2'],\
                         gene[labels_=='SCZ Cluster 3'])
    print("SNP : %s, T = %f  and p = %f"%(s,T,p))

    df_genetic_stats.loc[df_genetic_stats.SNP==s,"T"] = round(T,3)
    df_genetic_stats.loc[df_genetic_stats.SNP==s,"p"] = round(p,4)
df_genetic_stats.to_csv(os.path.join(output,"clusters_genetic_p_values.csv"))


####################################################################

for s in snps:
    print (s)
    gene = df_ped[s].values
    labels_ = labels[np.array(np.isnan(gene)==False)]
    gene = gene[np.array(np.isnan(gene)==False)]
    assert gene.shape == labels_.shape

    T, p = scipy.stats.ttest_ind(gene[labels_=='SCZ Cluster 1'],\
                          gene[labels_=='SCZ Cluster 2'])
    print("SNP : %s, SCZ 1 vs SC 2,  T = %f  and p = %f"%(s,T,p))
    T, p = scipy.stats.ttest_ind(gene[labels_=='SCZ Cluster 1'],\
                          gene[labels_=='SCZ Cluster 3'])
    print("SNP : %s, SCZ 1 vs SC 3, T = %f  and p = %f"%(s,T,p))
    T, p = scipy.stats.ttest_ind(gene[labels_=='SCZ Cluster 3'],\
                          gene[labels_=='SCZ Cluster 2'])
    print("SNP : %s, SCZ 3 vs SC 2, T = %f  and p = %f"%(s,T,p))

    T, p = scipy.stats.ttest_ind(gene[labels_=='SCZ Cluster 1'],\
                          gene[labels_=='Controls'])
    print("SNP : %s, SCZ 1 vs Controls, T = %f  and p = %f"%(s,T,p))

    T, p = scipy.stats.ttest_ind(gene[labels_=='SCZ Cluster 2'],\
                          gene[labels_=='Controls'])
    print("SNP : %s,SCZ 2 vs Controls, T = %f  and p = %f"%(s,T,p))

    T, p = scipy.stats.ttest_ind(gene[labels_=='SCZ Cluster 3'],\
                          gene[labels_=='Controls'])
    print("SNP : %s,SCZ 3 vs Controls, T = %f  and p = %f"%(s,T,p))

    T, p = scipy.stats.ttest_ind(gene[labels_!='Controls'],\
                          gene[labels_=='Controls'])
    print("SNP : %s,All SCZ vs Controls, T = %f  and p = %f"%(s,T,p))

####################################################################

X = df_ped[['rs11210892',
       'rs10803138', 'rs6704641', 'rs2535627', 'rs7432375', 'rs10520163',
       'rs1106568', 'rs4391122', 'rs10503253', 'rs4129585', 'rs7893279',
       'rs11027857', 'rs9420', 'rs12421382', 'rs2514218', 'rs2007044',
       'rs2068012', 'rs2693698', 'rs8042374', 'rs950169', 'rs8044995',
       'rs2053079', 'rs6065094', 'rs1023500']].values

X = df_ped[['rs11210892',
       'rs10803138', 'rs6704641', 'rs2535627', 'rs7432375', 'rs10520163',
       'rs1106568', 'rs4391122', 'rs10503253', 'rs4129585', 'rs7893279',
       'rs11027857', 'rs9420', 'rs12421382', 'rs2514218', 'rs2693698', 'rs8042374', 'rs950169', 'rs8044995',
       'rs2053079', 'rs6065094', 'rs1023500']].values


LABELS_DICT = {"Controls":0, "SCZ Cluster 1":1, "SCZ Cluster 2":2, "SCZ Cluster 3":3}
lab = df_ped["labels_cluster"].map(LABELS_DICT).values

bacc,recall,auc,beta1 = svm_score(X,y)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))


#1) CONTROLS vs CLUSTER 1
X_ = X[np.logical_or(labels =='Controls',labels =='SCZ Cluster 1') ,:]
y_ = y[np.logical_or(labels_ =='Controls',labels =='SCZ Cluster 1')]
bacc,recall,auc,beta1 = svm_score(X_,y_)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))


#1) CONTROLS vs CLUSTER 2
X_ = X[np.logical_or(labels =='Controls',labels =='SCZ Cluster 2') ,:]
y_ = y[np.logical_or(labels_ =='Controls',labels =='SCZ Cluster 2')]
bacc,recall,auc,beta1 = svm_score(X_,y_)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))

#1) CONTROLS vs CLUSTER 3
X_ = X[np.logical_or(labels =='Controls',labels =='SCZ Cluster 3') ,:]
y_ = y[np.logical_or(labels_ =='Controls',labels =='SCZ Cluster 3')]
bacc,recall,auc,beta1 = svm_score(X_,y_)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))


#1) CLUSTER 1 vs CLUSTER 2
X_ = X[np.logical_or(lab ==1,lab ==2) ,:]
y_ = lab[np.logical_or(lab ==1,lab ==2)]
y_[y_==2] =0
bacc,recall,auc,beta1 = svm_score(X_,y_)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))

def svm_score(X,y):

    list_predict=list()
    list_true=list()
    list_prob_pred=list()

    clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced')
    parameters={'C':[1,1e-1,1e1]}

    def balanced_acc(t, p):
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
        ba = recall_scores.mean()
        return ba


    score=make_scorer(balanced_acc,greater_is_better=True)
    clf = grid_search.GridSearchCV(clf,parameters,cv=3,scoring=score)
    skf = StratifiedKFold(y,5)

    for train, test in skf:

        X_train=X[train,:]
        X_test=X[test,:]
        y_train=y[train]
        y_test=y[test]
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_prob_pred = clf.decision_function(X_test)
        list_predict.append(y_pred)
        list_prob_pred.append(y_prob_pred)

    t=np.concatenate(list_true)
    prob_pred=np.concatenate(list_prob_pred)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    balanced_acc= recall_scores.mean()
    auc = roc_auc_score(t,prob_pred)
    clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced')
    clf.fit(X,y)
    beta = clf.coef_
    return balanced_acc,recall_scores,auc,beta