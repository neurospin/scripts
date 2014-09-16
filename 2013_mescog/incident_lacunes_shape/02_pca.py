# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:13:33 2014

@author: edouard.duchesnay@cea.fr

"""
import os
import numpy as np
#import pylab as plt
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.decomposition
from matplotlib.backends.backend_pdf import PdfPages

#from sklearn import preprocessing

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"
INPUT_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
OUTPUT_MOMENT_INVARIANT = os.path.join(BASE_PATH, "results_moments_invariant")
if not os.path.exists(OUTPUT_MOMENT_INVARIANT):
        os.makedirs(OUTPUT_MOMENT_INVARIANT)
#OUTPUT_MOMENT_INVARIANT_CLUST = os.path.join(OUTPUT_MOMENT_INVARIANT, "pc1-pc2_clusters")
OUTPUT_MOMENT_INVARIANT_PC12 = os.path.join(OUTPUT_MOMENT_INVARIANT, "figures", "mnts-inv_pc12")
OUTPUT_MOMENT_INVARIANT_PCA = os.path.join(OUTPUT_MOMENT_INVARIANT, "mnts-inv_pca")

OUTPUT_TENSOR_INVARIANT = os.path.join(BASE_PATH, "results_tensor_invariant")
if not os.path.exists(OUTPUT_TENSOR_INVARIANT):
        os.makedirs(OUTPUT_TENSOR_INVARIANT)
OUTPUT_TENSOR_INVARIANT_PCA = os.path.join(OUTPUT_TENSOR_INVARIANT, "tnsr-inv_pca")
OUTPUT_TENSOR_INVARIANT_PC12 = os.path.join(OUTPUT_TENSOR_INVARIANT, "figures", "tnsr-inv_pc12")

#OUTPUT_ANISOTROPY_LIN_PLAN = os.path.join(BASE_PATH, "results_tensor_anisotropy_linear_planar")
#if not os.path.exists(OUTPUT_ANISOTROPY_LIN_PLAN):
#        os.makedirs(OUTPUT_ANISOTROPY_LIN_PLAN)
OUTPUT_ANISOTROPY_LIN_PLAN_2D = os.path.join(OUTPUT_TENSOR_INVARIANT, "figures", "tnsr-inv_lin-plan")

moments = pd.read_csv(INPUT_CSV)

# for axes dimension
def force_marging():
    xmin, xmax = plt.xlim(); xmarg = (xmax - xmin) / 10.
    ymin, ymax = plt.ylim(); ymarg = (xmax - xmin) / 10.
    axes = plt.gca()
    axes.set_xlim([xmin-xmarg, xmax+xmarg])
    axes.set_ylim([ymin-ymarg, ymax+ymarg])

##############################################################################
## PCA  on Moment invariants
# Sun et al. Automatic Inference of Sulcus Patterns Using 3D Moment Invariants
# we noticed that I6 and I10 were presenting bimodal distributions for some sulci. One mode was made
# up of positive values and the other one of negative values. There is no apparent
# correlation between the shape and the sign of I6 and I10
# These 12 invariants denoted by I1, I2, ..., I12
# => remove 'moment_invariant_5' & 'moment_invariant_9'
col_invar = [col for col in moments.columns if col.count('moment_invariant')]
col_invar.remove('moment_invariant_5')
col_invar.remove('moment_invariant_9')

col_tensor = [col for col in moments.columns if col.count('tensor_invariant')]

## PCA
X = moments[col_invar]
X -= X.mean(axis=0)
pca = sklearn.decomposition.PCA()
pca.fit(X)
PCs = pca.transform(X)

pca_df = pd.DataFrame(np.hstack([moments["lacune_id"][:, np.newaxis], PCs]), columns=["lacune_id"] + ["PC%s"%i for i in xrange(1, PCs.shape[1]+1)])
pca_df.to_csv(OUTPUT_MOMENT_INVARIANT_PCA+".csv", index=False)

comp = pd.DataFrame(dict(
    comp=["PC%s"%i for i in xrange(1, pca.components_.shape[1]+1)],
    explained_variance_ratio = pca.explained_variance_ratio_,
    explained_variance_cumulative =np.cumsum(pca.explained_variance_ratio_)
))
loadings = pd.DataFrame(pca.components_, columns=["load_%s"%m for m in col_invar])

comp = pd.concat([comp, loadings], axis=1)
comp.to_csv(OUTPUT_MOMENT_INVARIANT_PCA+"_descriptive.csv", index=False)

# SOME QC BEGIN
a = pd.read_csv(OUTPUT_MOMENT_INVARIANT_PCA+".csv")
b = pd.read_csv("/home/ed203246/data/mescog/incident_lacunes_shape/results_moments_invariant/mnts-inv_pc12_clusters.csv")
assert np.all(a[[u'lacune_id', u'PC1', u'PC2']] == b[[u'lacune_id', u'PC1', u'PC2']])
# SOME QC END

pdf = PdfPages(OUTPUT_MOMENT_INVARIANT_PC12+".pdf")
print OUTPUT_MOMENT_INVARIANT_PC12+".pdf"
# Annotated
fig = plt.figure(); plt.axis('equal')

plt.scatter(PCs[:, 0], PCs[:, 1], s=50)#, "ob", s=50)
for i in xrange(len(moments["lacune_id"])):
    plt.text(PCs[i, 0], PCs[i, 1], moments["lacune_id"][i])

plt.title("PC1, PC2 Annotated")
plt.xlabel("PC1")
plt.ylabel("PC2")
force_marging()
pdf.savefig(); plt.clf()


# Color by
fig = plt.figure(); plt.axis('equal')

plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_fa, s=50)
plt.title("Tensor's invariant: FA"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig();plt.savefig(OUTPUT_MOMENT_INVARIANT_PC12+"_fa.svg")
plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_mode, s=50)
plt.title("Tensor's invariant: Mode"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig(); plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_linear_anisotropy, s=50)
plt.title("Tensor's invariant: linear anisotropy"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig()  # saves the current figure into a pdf page
plt.savefig(OUTPUT_MOMENT_INVARIANT_PC12+"_linear_anisotropy.svg")
plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_planar_anisotropy, s=50)
plt.title("Tensor's invariant: planar anisotropy"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig()  # saves the current figure into a pdf page
plt.savefig(OUTPUT_MOMENT_INVARIANT_PC12+"_planar_anisotropy.svg")
plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_spherical_anisotropy, s=50)
plt.title("Tensor's invariant: spherical anisotropy"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig()  # saves the current figure into a pdf page
plt.savefig(OUTPUT_MOMENT_INVARIANT_PC12+"_spherical_anisotropy.svg")
plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.compactness, s=50)
plt.title("Compactness: vol^(2/3) / area"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig();plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.perfo_angle_inertia_max, s=50)
plt.title("Angle maximum inertia axis - perforator"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()

pdf.savefig()
plt.savefig(OUTPUT_MOMENT_INVARIANT_PC12+"_angle-with-perforator.svg")
plt.clf()
pdf.close()


##############################################################################
## PCA on Tensor invariants
X = moments[col_tensor]
X -= X.mean(axis=0)
pca = sklearn.decomposition.PCA()
pca.fit(X)
PCs = pca.transform(X)

pca_df = pd.DataFrame(np.hstack([moments["lacune_id"][:, np.newaxis], PCs]), columns=["lacune_id"] + ["PC%s"%i for i in xrange(1, PCs.shape[1]+1)])
pca_df.to_csv(OUTPUT_TENSOR_INVARIANT_PCA+".csv", index=False)

comp = pd.DataFrame(dict(
    comp=["PC%s"%i for i in xrange(1, pca.components_.shape[1]+1)],
    explained_variance_ratio = pca.explained_variance_ratio_,
    explained_variance_cumulative =np.cumsum(pca.explained_variance_ratio_)
))
loadings = pd.DataFrame(pca.components_, columns=["load_%s"%m for m in col_tensor])

comp = pd.concat([comp, loadings], axis=1)
comp.to_csv(OUTPUT_TENSOR_INVARIANT_PCA+"_descriptive.csv", index=False)

pdf = PdfPages(OUTPUT_TENSOR_INVARIANT_PC12+".pdf")
print OUTPUT_TENSOR_INVARIANT_PC12+".pdf"

# Annotated
fig = plt.figure()#; plt.axis('equal')
plt.plot(PCs[:, 0], PCs[:, 1], "ob")
for i in xrange(len(moments["lacune_id"])):
    plt.text(PCs[i, 0], PCs[i, 1], moments["lacune_id"][i])

plt.title("PC1, PC2 Annotated")
plt.xlabel("PC1")
plt.ylabel("PC2")
pdf.savefig(); plt.clf()

# Color by
fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_fa, s=50)
plt.title("Tensor's invariant: FA"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_mode, s=50)
plt.title("Tensor's invariant: Mode"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_linear_anisotropy, s=50)
plt.title("Tensor's invariant: linear anisotropy"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_planar_anisotropy, s=50)
plt.title("Tensor's invariant: planar anisotropy"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.tensor_invariant_spherical_anisotropy, s=50)
plt.title("Tensor's invariant: spherical anisotropy"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.compactness, s=50)
plt.title("Compactness: vol^(2/3) / area"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

fig = plt.figure()#; plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments.perfo_angle_inertia_max, s=50)
plt.title("Angle maximum inertia axis - perforator"); plt.xlabel("PC1"); plt.ylabel("PC2")
pdf.savefig(); plt.clf()

pdf.close()

##############################################################################
## 2D linear x planar
pdf = PdfPages(OUTPUT_ANISOTROPY_LIN_PLAN_2D+".pdf")

#mi, mx = np.min([moments.tensor_invariant_linear_anisotropy, moments.tensor_invariant_planar_anisotropy]), np.max([moments.tensor_invariant_linear_anisotropy, moments.tensor_invariant_planar_anisotropy])
#mi -= (mx - mi) / 20
#mx += (mx - mi) / 20

# Annotated
fig = plt.figure(); plt.axis('equal')
#plt.plot(moments.tensor_invariant_linear_anisotropy, moments.tensor_invariant_planar_anisotropy, "ob")
plt.scatter(moments.tensor_invariant_linear_anisotropy, moments.tensor_invariant_planar_anisotropy)
for i in xrange(len(moments["lacune_id"])):
    plt.text(moments.tensor_invariant_linear_anisotropy[i], moments.tensor_invariant_planar_anisotropy[i], moments["lacune_id"][i])

plt.title("linear vs planar anisotropy")
plt.xlabel("Linear"); plt.ylabel("Planar")
pdf.savefig(); plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(moments.tensor_invariant_linear_anisotropy, moments.tensor_invariant_planar_anisotropy, c=moments.tensor_invariant_fa, s=50)
plt.title("FA")
plt.xlabel("Linear"); plt.ylabel("Planar")
pdf.savefig(); plt.savefig(OUTPUT_ANISOTROPY_LIN_PLAN_2D+"_fa.svg")
plt.clf()

fig = plt.figure(); plt.axis('equal')
plt.scatter(moments.tensor_invariant_linear_anisotropy, moments.tensor_invariant_planar_anisotropy, c=moments.perfo_angle_inertia_max, s=50)
plt.title("Angle maximum inertia axis - perforator")
plt.xlabel("Linear"); plt.ylabel("Planar")
pdf.savefig(); plt.clf()

pdf.close()


"""
## Clustering
# 2 patterns defined by 2 lines

extremities = np.vstack([
np.hstack([PCs[(moments["lacune_id"] == 8).values, [0, 1]], PCs[(moments["lacune_id"] == 75).values, [0, 1]]]),
np.hstack([PCs[(moments["lacune_id"] == 21).values, [0, 1]], PCs[(moments["lacune_id"] == 64).values, [0, 1]]])])

def get_affine(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b

affine = np.array([get_affine(p[0], p[1], p[2], p[3]) for p in extremities])

y_hat = (affine[:, [0]] * PCs[:, 0] + affine[:, [1]]).T
y = PCs[:, [1]]
err = (y - y_hat) ** 2
label = np.argmin(err, axis=1)

fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=label, s=50)
plt.title("Clusters (2 groups)"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig(); plt.clf()

clusters = pd.DataFrame(np.hstack([PCs[:, [0, 1]], label[:, np.newaxis]]),
                        columns=("PC1", "PC2", "label"), index=moments["lacune_id"])
clusters.to_csv(OUTPUT_MOMENT_INVARIANT_CLUST+".csv")
clusters = pd.DataFrame(np.hstack([PCs[:, [0, 1]], label[:, np.newaxis]]),
                        columns=("PC1", "PC2", "label"), index=moments["lacune_id"])
"""

"""
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
fig = plt.figure(); plt.axis('equal')
plt.scatter(PCs[:, 0], PCs[:, 1], c=moments["compactness_resid_tensor_invarient"], s=50)
plt.title("Compactness residual (regression on tens. invar.)"); plt.xlabel("PC1"); plt.ylabel("PC2")
force_marging()
pdf.savefig(); plt.clf()

pdf.close()
"""
