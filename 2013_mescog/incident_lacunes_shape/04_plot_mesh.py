# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:48:01 2014

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import glob
import pandas as pd
import os
from brainomics import mesh_processing


BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"

INPUT_IMAGES = os.path.join(BASE_PATH, "incident_lacunes_images")
#INPUT_MOMENTS_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments_area_from_mesh.csv"
INPUT_MOMENTS_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
INPUT_PC_CLUSTERS_CSV = os.path.join(BASE_PATH, "pc1-pc2_clusters.csv")

SCALE_XY_PC12 = 300
SCALE_X_INERTY_CLUSTER = 3000
SCALE_Y_INERTY_CLUSTER = 20

moments = pd.read_csv(INPUT_MOMENTS_CSV)#, index_col=0)
moments['Vol(mm3)'].mean()
pc = pd.read_csv(INPUT_PC_CLUSTERS_CSV)#, index_col=0)

filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL_1_0_reorient.nii.gii"))
#filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/027/027_1046-M18_LacBL.nii.gz"
filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/075/075_2005-M18_LacBL_1_0_reorient.nii.gii"


#
import statsmodels.api as sm
import pylab as plt
y = moments["compactness"].values
X = moments["inertie_max_norm"].values
X = moments["fa"].values
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print results
#results.params
#print results.t_test([-1, 0])
#print results.f_test(np.identity(2))
#plt.plot(moments["fa"], moments["compactness"], "o")
moments["compactness_resid"] = results.resid


##
coord_pc12_all = list()
coord_pc12_scaled_all = list()
coord_inerty_max_vs_cluster_all = list()
coord_inerty_max_vs_cluster_scaled_all = list()

tri_all = list()
lacune_id_all = list()
labels_all = list()
fa_all = list()
inertie_max_all = list()
inertie_min_all = list()
compactness_all = list()
compactness_resid_all = list()
inertie_max_norm_all = list()
inertie_min_norm_all = list()
angle_inertie_max_perf_all = list()

n_vertex = 0
for filename in filenames:
    lacune_id = int(os.path.basename(os.path.dirname(filename)))
    m = moments[moments["lacune_id"] == lacune_id]
    comp = pc[pc["lacune_id"] == lacune_id]
    if len(m) == 0 or len(comp) == 0:
        continue
    #
    coord, tri = mesh_processing.mesh_arrays(filename)
    vol = m['Vol(mm3)'].values[0]
    scale_xyz = (moments['Vol(mm3)'].mean() / vol) ** (1. / 3.)
    # Mesh pc1 vs pc2
    coord_pc12_scaled = coord.copy()
    coord_pc12_scaled *= scale_xyz
    coord_pc12_scaled[:, 0] += comp.PC1 * SCALE_XY_PC12
    coord_pc12_scaled[:, 1] += comp.PC2 * SCALE_XY_PC12
    coord_pc12_scaled_all.append(coord_pc12_scaled)
    coord_pc12 = coord.copy()
    coord_pc12[:, 0] += comp.PC1 * SCALE_XY_PC12
    coord_pc12[:, 1] += comp.PC2 * SCALE_XY_PC12
    coord_pc12_all.append(coord_pc12)
    # Mesh inerty_max_vs_cluster
    coord_inerty_max_vs_cluster = coord.copy()
    coord_inerty_max_vs_cluster[:, 0] += m.inertie_max_norm * SCALE_X_INERTY_CLUSTER
    coord_inerty_max_vs_cluster[:, 1] += comp.label * SCALE_Y_INERTY_CLUSTER
    coord_inerty_max_vs_cluster_all.append(coord_inerty_max_vs_cluster)
    coord_inerty_max_vs_cluster_scaled = coord.copy()
    coord_inerty_max_vs_cluster_scaled *= scale_xyz
    coord_inerty_max_vs_cluster_scaled[:, 0] += m.inertie_max_norm * SCALE_X_INERTY_CLUSTER
    coord_inerty_max_vs_cluster_scaled[:, 1] += comp.label * SCALE_Y_INERTY_CLUSTER
    coord_inerty_max_vs_cluster_scaled_all.append(coord_inerty_max_vs_cluster_scaled)

    tri_all.append(tri + n_vertex)
    ids = np.zeros(coord.shape[0], dtype=int)
    ids[::] = lacune_id
    lacune_id_all.append(ids)
    label = np.zeros(coord.shape[0], dtype=int)
    label[::] = comp.label
    labels_all.append(label)
    #
    fa = np.zeros(coord.shape[0], dtype=float)
    fa[::] = m["fa"]
    fa_all.append(fa)
    #
    compact = np.zeros(coord.shape[0], dtype=float)
    compact[::] = m["compactness"]
    compactness_all.append(compact)
    #
    compact_resid = np.zeros(coord.shape[0], dtype=float)
    compact_resid[::] = m["compactness_resid"]
    compactness_resid_all.append(compact_resid)
    #
    inertie_max_norm = np.zeros(coord.shape[0], dtype=float)
    inertie_max_norm[::] = m["inertie_max_norm"]
    inertie_max_norm_all.append(inertie_max_norm)
    #
    inertie_min_norm = np.zeros(coord.shape[0], dtype=float)
    inertie_min_norm[::] = m["inertie_min_norm"]
    inertie_min_norm_all.append(inertie_min_norm)
    #
    angle_inertie_max_perf = np.zeros(coord.shape[0], dtype=float)
    angle_inertie_max_perf[::] = m["angle_inertie_max_perf"]
    angle_inertie_max_perf_all.append(angle_inertie_max_perf)

    n_vertex += coord.shape[0]


coord_pc12_scaled_all = np.vstack(coord_pc12_scaled_all)
coord_pc12_all = np.vstack(coord_pc12_all)
coord_inerty_max_vs_cluster_scaled_all = np.vstack(coord_inerty_max_vs_cluster_scaled_all)
coord_inerty_max_vs_cluster_all = np.vstack(coord_inerty_max_vs_cluster_all)

tri_all = np.vstack(tri_all)
labels_all = np.hstack(labels_all)
lacune_id_all = np.hstack(lacune_id_all)
fa_all = np.hstack(fa_all)
compactness_all = np.hstack(compactness_all)
compactness_resid_all = np.hstack(compactness_resid_all)
inertie_max_norm_all = np.hstack(inertie_max_norm_all)
inertie_min_norm_all = np.hstack(inertie_min_norm_all)
angle_inertie_max_perf_all = np.hstack(angle_inertie_max_perf_all)

mesh_processing.mesh_from_arrays(coord_pc12_scaled_all, tri_all, path="/tmp/mesh_pc12_scaled.gii")
mesh_processing.mesh_from_arrays(coord_pc12_all, tri_all, path="/tmp/mesh_pc12.gii")
mesh_processing.mesh_from_arrays(coord_inerty_max_vs_cluster_all, tri_all, path="/tmp/mesh_inerty-max_vs_cluster.gii")
mesh_processing.mesh_from_arrays(coord_inerty_max_vs_cluster_scaled_all, tri_all, path="/tmp/mesh_inerty-max_vs_cluster_scaled.gii")
mesh_processing.save_texture(path="/tmp/tex_lacunes_id.gii", data=lacune_id_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_clust-labels.gii", data=labels_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_fa.gii", data=fa_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_compactness.gii", data=compactness_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_compactness_resid.gii", data=compactness_resid_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_inerty_max_norm.gii", data=inertie_max_norm_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_inerty_min_norm.gii", data=inertie_min_norm_all, intent='NIFTI_INTENT_NONE')
mesh_processing.save_texture(path="/tmp/tex_angle_inerty_max_perf.gii", data=angle_inertie_max_perf_all, intent='NIFTI_INTENT_NONE')

"""
anatomist /tmp/mesh_*.gii
"""

