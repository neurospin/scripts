# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 09:26:58 2016

@author: ad247405
"""


import os
import pandas as pd
import re
import matplotlib.pyplot as plt

babel_mask  = nibabel.load( '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/mask.nii.gz')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
INPUT_CSV = os.path.join(BASE_PATH,"results","multivariate_analysis/logistic_regression_tv/simple_run_no_model_selection/results.csv")

alpha = []
l1 = []
l2 = []
tv = []

data = pd.read_csv(INPUT_CSV)
for i in range(data.shape[0]):
    p = re.split(',',data.params[i])
    alpha.append(float(p[0]))
    l1.append(float(p[1]))
    l2.append(float(p[2]))
    tv.append(float(p[3]))

data["alpha"] = np.array(alpha)
data["l1"] = np.array(l1)
data["l2"] = np.array(l2)
data["tv"] = np.array(tv)




#d1=data[np.round((data.l1)/(data.l2),3) == 0.001]
#d2=data[np.round((data.l1)/(data.l2),3) == 0.01]
#d3=data[np.round((data.l1)/(data.l2),3) == 0.111]
#d4=data[np.round((data.l1)/(data.l2),3) == 1]
#d5=data[np.round((data.l1)/(data.l2),3) == 9]


data = data[data.alpha==1.0]
full_tv = data[(data.tv == 1)]


d1=data[np.round((data.l1)/(data.l2+ data.l1 ),3) == 0.001]
d2=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.01]
d3=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.100]
d4=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.500]
d5=data[np.round((data.l1)/(data.l2+ data.l1),3) == 1]

d1 = d1.append(full_tv) # add full tv for all lines
d2 = d2.append(full_tv) # add full tv for all lines
d3 = d3.append(full_tv) # add full tv for all lines
d4 = d4.append(full_tv) # add full tv for all lines
d5 = d5.append(full_tv) # add full tv for all lines


d1=d1.sort("tv")
d2=d2.sort("tv")
d3=d3.sort("tv")
d4=d4.sort("tv")
d5=d5.sort("tv")

d1_sparse = d1[d1.prop_non_zeros_mean<0.5]
d2_sparse = d2[d2.prop_non_zeros_mean<0.5]
d3_sparse = d3[d3.prop_non_zeros_mean<0.5]
d4_sparse = d4[d4.prop_non_zeros_mean<0.5]
d5_sparse = d5[d5.prop_non_zeros_mean<0.5]


plt.plot(d1.tv, d1.recall_mean,"green",label=r'$\lambda_1/(\lambda_1 + \lambda_2) = 0.001 $',linewidth=2)
plt.plot(d2.tv, d2.recall_mean,"blue",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.01 $',linewidth=2)
plt.plot(d3.tv, d3.recall_mean,"orange",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.1 $',linewidth=2)
plt.plot(d4.tv, d4.recall_mean,"red",label=r'$\lambda_1/(\lambda_1 + \lambda_2) = 0.5 $',linewidth=2)
plt.plot(d5.tv, d5.recall_mean,"black",label=r'$\lambda_1/(\lambda_1 + \lambda_2) = 1 $',linewidth=2)

plt.plot(d1_sparse.tv,d1_sparse.recall_mean,'bo',color='green')
plt.plot(d2_sparse.tv,d2_sparse.recall_mean,'bo',color='blue')
plt.plot(d3_sparse.tv,d3_sparse.recall_mean,'bo',color='orange')
plt.plot(d4_sparse.tv,d4_sparse.recall_mean,'bo',color='red')
plt.plot(d5_sparse.tv,d5_sparse.recall_mean,'bo',color='black')



#plt.plot(d1.tv, d1.recall_1,"red",label="TVl1l2- (l1=0.1,l2=0.9)",linewidth=2)
#plt.plot(d2.tv, d2.accuracy,"blue",label="TVl1l2(l1=0.9,l2=0.1",linewidth=2)
plt.ylabel("Balanced accuracy")
#plt.ylabel("Specificity")
#plt.ylabel("Sensitivity")
plt.xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_1 + \lambda_2 + \lambda_{tv})$')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.4, 0.7))
plt.title('$alpha = 1.0$')




#Select optimal parameters
