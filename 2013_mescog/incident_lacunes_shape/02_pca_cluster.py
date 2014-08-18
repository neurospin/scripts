# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:13:33 2014

@author: edouard.duchesnay@cea.fr

"""
import os
import numpy as np
import pylab as plt
import pandas as pd
import sklearn.decomposition
#from sklearn import preprocessing

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"
INPUT_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
OUTPUT_clusters = os.path.join(BASE_PATH, "pc1-pc2_clusters")
OUTPUT_PCs =      os.path.join(BASE_PATH, "pc1-pc2")

data = pd.read_csv(INPUT_CSV, index_col=0)
col_invar = [col for col in data.columns if col.count('Moment_Invariant')]
X = data[col_invar]

X -= X.mean(axis=0)
#X /= X.std(axis=0)
#X = preprocessing.scale(X)
pca = sklearn.decomposition.PCA()
pca.fit(X)
# loadings:
#array([[-0.14294875, -0.17968673,  0.34669576,  0.18669825,  0.07634929,
#         0.52263794, -0.13544378,  0.17125881, -0.08050099, -0.64151083,
#         0.16133586, -0.14671097],
#       [ 0.14622265,  0.21200114, -0.76158041, -0.08229445, -0.06692794,
#         0.40739309,  0.07199359, -0.18035649,  0.06434588, -0.3293737 ,
#        -0.10317077,  0.12436119]])

pca.explained_variance_ratio_
#array([  4.71084943e-01,   2.20964783e-01,   1.31906872e-01,
#         9.03439692e-02,   4.61621312e-02,   1.69554256e-02,
#         1.21359995e-02,   7.20014382e-03,   1.71798089e-03,
#         1.34157115e-03,   1.66670657e-04,   1.95096354e-05])
#         1.50674947e-03,   1.61369200e-04,   2.01564868e-05])

np.sum(pca.explained_variance_ratio_[:2])
# 0.69204972634390094

# Project training subjects onto PC
# scikit-learn projects on min(n_samples, n_features)
PCs = pca.transform(X)

fig = plt.figure()
plt.plot(PCs[:, 0], PCs[:, 1], "ob")
for i in xrange(len(data.index)):
    plt.text(PCs[i, 0], PCs[i, 1], data.index[i])

plt.title("Invariant of Moments (pure shape descriptors)")
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.show()
fig.savefig(OUTPUT_PCs+"_annotated.svg")


# ==============================================================
# 4 patterns defined by 4 lines
# Manual clustering:
# fit 4 affines, classify points to best fit
# 4 lines extremities, [x1, y1, x2, y2]

extremities = np.vstack([
np.hstack([PCs[data.index == 18, [0, 1]], PCs[data.index == 32, [0, 1]]]),
np.hstack([PCs[data.index == 75, [0, 1]], PCs[data.index == 21, [0, 1]]]),
np.hstack([PCs[data.index == 35, [0, 1]], PCs[data.index == 94, [0, 1]]]),
np.hstack([PCs[data.index == 33, [0, 1]], PCs[data.index == 77, [0, 1]]])])


def get_affine(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b

affine = np.array([get_affine(p[0], p[1], p[2], p[3]) for p in extremities])

y_hat = (affine[:, [0]] * PCs[:, 0] + affine[:, [1]]).T
y = PCs[:, [1]]
err = (y - y_hat) ** 2
label = np.argmin(err, axis=1)

fig = plt.figure()
plt.plot(PCs[label==0, 0], PCs[label==0, 1], "ob",
        PCs[label==1, 0], PCs[label==1, 1], "or",
        PCs[label==2, 0], PCs[label==2, 1], "og",
        PCs[label==3, 0], PCs[label==3, 1], "oy")
plt.title("Invariant of Moments (pure shape descriptors)")
plt.xlabel("PC1")
plt.ylabel("PC2")
fig.savefig(OUTPUT_clusters+".svg")

for i in xrange(len(data.index)):
    plt.text(PCs[i, 0], PCs[i, 1], data.index[i])


fig.savefig(OUTPUT_clusters+"_annotated.svg")


clusters = pd.DataFrame(np.hstack([PCs[:, [0, 1]], label[:, np.newaxis]]),
                        columns=("PC1", "PC2", "label"), index=data.index)
clusters.to_csv(OUTPUT_clusters+".csv")

##############################################################################
## Color by
# inertie_max_norm
fig = plt.figure()
plt.plot(PCs[:, 0], PCs[:, 1], "o")#, color=data.inertie_max_norm)
#for i in xrange(len(data.index)):
#    plt.text(PCs[i, 0], PCs[i, 1], data.index[i])

plt.title("Invariant of Moments (pure shape descriptors)")
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.show()
fig.savefig(OUTPUT_PCs+"_color-by-inertie_max_norm.svg")

"""

"""
