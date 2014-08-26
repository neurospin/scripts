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
from matplotlib.backends.backend_pdf import PdfPages

#from sklearn import preprocessing

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"
INPUT_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
OUTPUT_MOMENT_INVARIANT = os.path.join(BASE_PATH, "results_moments_invariant")
if not os.path.exists(OUTPUT_MOMENT_INVARIANT):
        os.makedirs(OUTPUT_MOMENT_INVARIANT)
OUTPUT_MOMENT_INVARIANT_CLUST = os.path.join(OUTPUT_MOMENT_INVARIANT, "pc1-pc2_clusters")
OUTPUT_MOMENT_INVARIANT_PC = os.path.join(OUTPUT_MOMENT_INVARIANT, "pc1-pc2")

OUTPUT_TENSOR_INVARIANT = os.path.join(BASE_PATH, "results_tensor_invariant")
if not os.path.exists(OUTPUT_TENSOR_INVARIANT):
        os.makedirs(OUTPUT_TENSOR_INVARIANT)
OUTPUT_TENSOR_INVARIANT_PC = os.path.join(OUTPUT_TENSOR_INVARIANT, "pc1-pc2")

OUTPUT_ANISOTROPY_LIN_SPHE = os.path.join(BASE_PATH, "results_tensor_anisotropy_linear_spherical")
if not os.path.exists(OUTPUT_ANISOTROPY_LIN_SPHE):
        os.makedirs(OUTPUT_ANISOTROPY_LIN_SPHE)
OUTPUT_ANISOTROPY_LIN_SPHE_2D = os.path.join(OUTPUT_ANISOTROPY_LIN_SPHE, "tensor_anisotropy_linear_spherical")

moments = pd.read_csv(INPUT_CSV)

##############################################################################
## PCA + clustering on Moment invariants

col_invar = [col for col in moments.columns if col.count('moment_invariant')]
col_tensor = [col for col in moments.columns if col.count('tensor_invariant')]

## PCA
X = moments[col_invar]
X -= X.mean(axis=0)
#X /= X.std(axis=0)
#X = preprocessing.scale(X)
pca = sklearn.decomposition.PCA()
pca.fit(X)
outtxt = open(OUTPUT_MOMENT_INVARIANT_PC+".txt","wb")
outtxt.write("Loadings:\n")
outtxt.write(str(pca.components_))
outtxt.write("explained_variance_ratio_:\n")
outtxt.write(str(pca.explained_variance_ratio_))
outtxt.write("np.sum(pca.explained_variance_ratio_[:2]):\n")
outtxt.write(str(np.sum(pca.explained_variance_ratio_[:2])))
outtxt.close()

# Project training subjects onto PC
# scikit-learn projects on min(n_samples, n_features)
PCs = pca.transform(X)

pdf = PdfPages(OUTPUT_MOMENT_INVARIANT_PC+".pdf")

# Annotated
fig = plt.figure()
plt.plot(PCs[:, 0], PCs[:, 1], "ob")
for i in xrange(len(moments["lacune_id"])):
    plt.text(PCs[i, 0], PCs[i, 1], moments["lacune_id"][i])

plt.title("PC1, PC2 Annotated")
plt.xlabel("PC1")
plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

# Color by
fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_fa, s=50)
plt.title("Tensor invariant: FA"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_mode, s=50)
plt.title("Tensor invariant: Mode"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_cl, s=50)
plt.title("Tensor invariant: CL (linear anisotropy)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_cp, s=50)
plt.title("Tensor invariant: CP (planar anisotropy)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_cs, s=50)
plt.title("Tensor invariant: CS (spherical anisotropy)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.compactness, s=50)
plt.title("Compactness: Vol^(2/3) / Surface"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.perfo_angle_inertia_max, s=50)
plt.title("Angle maximum inertie axis - perforator"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

## Clustering
# 4 patterns defined by 4 lines
# Manual clustering:
# fit 4 affines, classify points to best fit
# 4 lines extremities, [x1, y1, x2, y2]

extremities = np.vstack([
np.hstack([PCs[(moments["lacune_id"] == 18).values, [0, 1]], PCs[(moments["lacune_id"] == 32).values, [0, 1]]]),
np.hstack([PCs[(moments["lacune_id"] == 75).values, [0, 1]], PCs[(moments["lacune_id"] == 21).values, [0, 1]]]),
np.hstack([PCs[(moments["lacune_id"] == 35).values, [0, 1]], PCs[(moments["lacune_id"] == 94).values, [0, 1]]]),
np.hstack([PCs[(moments["lacune_id"] == 33).values, [0, 1]], PCs[(moments["lacune_id"] == 77).values, [0, 1]]])])


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
plt.scatter(PCs[:, 0], PCs[:, 1], c=label, s=50)
plt.title("4 patterns defined by 4 lines"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()

clusters = pd.DataFrame(np.hstack([PCs[:, [0, 1]], label[:, np.newaxis]]),
                        columns=("PC1", "PC2", "label"), index=moments["lacune_id"])
clusters.to_csv(OUTPUT_MOMENT_INVARIANT_CLUST+".csv")


#
import statsmodels.api as sm
y = moments["compactness"].values
X = moments[col_tensor].values
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print results
#results.params
#print results.t_test([-1, 0])
#print results.f_test(np.identity(2))
#plt.plot(moments["fa"], moments["compactness"], "o")
moments["compactness_resid_tensor_invarient"] = results.resid
fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments["compactness_resid_tensor_invarient"], s=50)
plt.title("Compactness residual (regression on tens. invar.)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()

pdf.close()


##############################################################################
## PCA + clustering on Tensor invariants
X = moments[col_tensor]
X -= X.mean(axis=0)
#X /= X.std(axis=0)
#X = preprocessing.scale(X)
pca = sklearn.decomposition.PCA()
pca.fit(X)
outtxt = open(OUTPUT_TENSOR_INVARIANT_PC+".txt","wb")
outtxt.write("Loadings:\n")
outtxt.write(str(pca.components_))
outtxt.write("explained_variance_ratio_:\n")
outtxt.write(str(pca.explained_variance_ratio_))
outtxt.write("np.sum(pca.explained_variance_ratio_[:2]):\n")
outtxt.write(str(np.sum(pca.explained_variance_ratio_[:2])))
outtxt.close()

# Project training subjects onto PC
# scikit-learn projects on min(n_samples, n_features)
PCs = pca.transform(X)

pdf = PdfPages(OUTPUT_TENSOR_INVARIANT_PC+".pdf")

# Annotated
fig = plt.figure()
plt.plot(PCs[:, 0], PCs[:, 1], "ob")
for i in xrange(len(moments["lacune_id"])):
    plt.text(PCs[i, 0], PCs[i, 1], moments["lacune_id"][i])

plt.title("PC1, PC2 Annotated")
plt.xlabel("PC1")
plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

# Color by
fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_fa, s=50)
plt.title("Tensor invariant: FA"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_mode, s=50)
plt.title("Tensor invariant: Mode"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_cl, s=50)
plt.title("Tensor invariant: CL (linear anisotropy)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_cp, s=50)
plt.title("Tensor invariant: CP (planar anisotropy)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_cs, s=50)
plt.title("Tensor invariant: CS (spherical anisotropy)"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.compactness, s=50)
plt.title("Compactness: Vol^(2/3) / Surface "); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

fig = plt.figure()
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.perfo_angle_inertia_max, s=50)
plt.title("Angle maximum inertia axis - perforator"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig()  # saves the current figure into a pdf page

pdf.close()

##############################################################################
## 2D linear x spherical
pdf = PdfPages(OUTPUT_ANISOTROPY_LIN_SPHE_2D+".pdf")

# Annotated
fig = plt.figure()
plt.plot(moments.tensor_invariant_cl, moments.tensor_invariant_cs, "ob")
for i in xrange(len(moments["lacune_id"])):
    plt.text(moments.tensor_invariant_cl[i], moments.tensor_invariant_cs[i], moments["lacune_id"][i])

plt.title("linear vs spherical anisotropy")
plt.xlabel("Linear"); plt.ylabel("Spherical")
pdf.savefig()

fig = plt.figure()
plt.scatter(moments.tensor_invariant_cl, moments.tensor_invariant_cs, c=moments.tensor_invariant_fa, s=50)
plt.title("FA")
plt.xlabel("Linear"); plt.ylabel("Spherical")
pdf.savefig()

fig = plt.figure()
plt.scatter(moments.tensor_invariant_cl, moments.tensor_invariant_cs, c=moments.perfo_angle_inertia_max, s=50)
plt.title("Angle maximum inertia axis - perforator")
plt.xlabel("Linear"); plt.ylabel("Spherical")
pdf.savefig()

pdf.close()