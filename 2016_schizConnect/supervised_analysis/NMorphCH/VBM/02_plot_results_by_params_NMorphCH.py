#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:42:55 2017

@author: ad247405
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:27:26 2017

@author: ad247405
"""


import os
import pandas as pd
import re
import matplotlib.pyplot as plt



INPUT_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/enettv/enettv_model_selection_5folds_NMorphCH/results_dCV.xlsx"

alpha = []
l1 = []
l2 = []
tv = []

data = pd.read_excel(INPUT_CSV)
for i in range(data.shape[0]):
    p = re.split('_',data.param_key[i])
    alpha.append(float(p[0]))
    l1.append(float(p[1]))
    l2.append(float(p[2]))
    tv.append(float(p[3]))

data["alpha"] = np.array(alpha)
data["l1"] = np.array(l1)
data["l2"] = np.array(l2)
data["tv"] = np.array(tv)


data = data[data.alpha==1]
full_tv = data[(data.tv == 1)]


d1=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0]
d2=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.1]
d3=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.2]
d4=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.3]
d5=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.5]
d6=data[np.round((data.l1)/(data.l2+ data.l1),3) == 0.9]
d7=data[np.round((data.l1)/(data.l2+ data.l1),3) == 1.0]

        


d1 = d1.append(full_tv) # add full tv for all lines
d2 = d2.append(full_tv) # add full tv for all lines
d3 = d3.append(full_tv) # add full tv for all lines
d4 = d4.append(full_tv) # add full tv for all lines
d5 = d5.append(full_tv) # add full tv for all lines
d6 = d6.append(full_tv) # add full tv for all lines
d7 = d7.append(full_tv) # add full tv for all lines



d1=d1.sort("tv")
d2=d2.sort("tv")
d3=d3.sort("tv")
d4=d4.sort("tv")
d5=d5.sort("tv")
d6=d6.sort("tv")
d7=d7.sort("tv")






plt.plot(d1.tv, d1.recall_mean,"red",label=r'$\lambda_1/(\lambda_1 + \lambda_2) = 0 $',linewidth=2)
plt.plot(d2.tv, d2.recall_mean,"green",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.1 $',linewidth=2)
plt.plot(d3.tv, d3.recall_mean,"blue",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.2 $',linewidth=2)
plt.plot(d4.tv, d4.recall_mean,"orange",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.3 $',linewidth=2)
plt.plot(d5.tv, d5.recall_mean,"yellow",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.5 $',linewidth=2)
plt.plot(d6.tv, d6.recall_mean,"cyan",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 0.9 $',linewidth=2)
plt.plot(d7.tv, d7.recall_mean,"black",label=r'$\lambda_1/(\lambda_1 + \lambda_2)= 1.0 $',linewidth=2)



plt.ylabel("Balanced accuracy")
plt.xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_1 + \lambda_2 + \lambda_{tv})$')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.4, 0.7))
plt.title('NMoprhCH dataset: 3D voxel-based GM map - $alpha =1.0$')
