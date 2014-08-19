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
INPUT_PC_CLUSTERS_CSV = os.path.join(BASE_PATH, "results_moments_invariant", "pc1-pc2_clusters.csv")
OUTPUT = "/tmp"

SCALE_XY_PC12 = 300
SCALE_anisotropy_linear = 500
SCALE_anisotropy_spherical = 500

moments = pd.read_csv(INPUT_MOMENTS_CSV)#, index_col=0)
pc = pd.read_csv(INPUT_PC_CLUSTERS_CSV)#, index_col=0)

filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL_1_0_reorient.nii.gii"))
#filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/027/027_1046-M18_LacBL.nii.gz"
filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/075/075_2005-M18_LacBL_1_0_reorient.nii.gii"

##
coord_pc_moment_invariant_all = list()
coord_pc_moment_invariant_scaled_all = list()
coord_anisotropy_linear_spherical_all = list()
coord_anisotropy_linear_spherical_scaled_all = list()


tri_all = list()
lacune_id_all = list()
labels_all = list()
#fa_all = list()
#inertie_max_all = list()
#inertie_min_all = list()
#compactness_all = list()
#compactness_resid_all = list()
#inertie_max_norm_all = list()
#inertie_min_norm_all = list()
#angle_inertie_max_perf_all = list()

textures = {t:list() for t in
["lacune_id", "tensor_invariant_fa", "tensor_invariant_cl", "tensor_invariant_cs", "perfo_angle_inertia_max"]}

n_vertex = 0
for filename in filenames:
    lacune_id = int(os.path.basename(os.path.dirname(filename)))
    m = moments[moments["lacune_id"] == lacune_id]
    comp = pc[pc["lacune_id"] == lacune_id]
    if len(m) == 0 or len(comp) == 0:
        continue
    #
    coord, tri = mesh_processing.mesh_arrays(filename)
    vol = m['vol_mm3'].values[0]
    scale_xyz = (moments['vol_mm3'].mean() / vol) ** (1. / 3.)
    # Mesh coord pc1 vs pc2
    coord_pc_moment_invariant_scaled = coord.copy()
    coord_pc_moment_invariant_scaled *= scale_xyz
    coord_pc_moment_invariant_scaled[:, 0] += comp.PC1 * SCALE_XY_PC12
    coord_pc_moment_invariant_scaled[:, 1] += comp.PC2 * SCALE_XY_PC12
    coord_pc_moment_invariant_scaled_all.append(coord_pc_moment_invariant_scaled)
    coord_pc_moment_invariant = coord.copy()
    coord_pc_moment_invariant[:, 0] += comp.PC1 * SCALE_XY_PC12
    coord_pc_moment_invariant[:, 1] += comp.PC2 * SCALE_XY_PC12
    coord_pc_moment_invariant_all.append(coord_pc_moment_invariant)
    # Mesh coord inerty_max_vs_cluster
    coord_anisotropy_linear_spherical = coord.copy()
    coord_anisotropy_linear_spherical[:, 0] += m.tensor_invariant_cl * SCALE_anisotropy_linear
    coord_anisotropy_linear_spherical[:, 1] += m.tensor_invariant_cs * SCALE_anisotropy_spherical
    coord_anisotropy_linear_spherical_all.append(coord_anisotropy_linear_spherical)
    coord_anisotropy_linear_spherical_scaled = coord.copy()
    coord_anisotropy_linear_spherical_scaled *= scale_xyz
    coord_anisotropy_linear_spherical_scaled[:, 0] += m.tensor_invariant_cl * SCALE_anisotropy_linear
    coord_anisotropy_linear_spherical_scaled[:, 1] += m.tensor_invariant_cs * SCALE_anisotropy_spherical
    coord_anisotropy_linear_spherical_scaled_all.append(coord_anisotropy_linear_spherical_scaled)
    # Tri
    tri_all.append(tri + n_vertex)
    #ids = np.zeros(coord.shape[0], dtype=int)
    #ids[::] = lacune_id
    #lacune_id_all.append(ids)
    #label = np.zeros(coord.shape[0], dtype=int)
    #label[::] = comp.label
    #labels_all.append(label)
    #
    for k in textures:
        arr = np.zeros(coord.shape[0], dtype=float)
        arr[::] = m[k]
        textures[k].append(arr)
    n_vertex += coord.shape[0]


#    fa = np.zeros(coord.shape[0], dtype=float)
#    fa[::] = m["fa"]
#    fa_all.append(fa)
#    #
#    compact = np.zeros(coord.shape[0], dtype=float)
#    compact[::] = m["compactness"]
#    compactness_all.append(compact)
#    #
#    compact_resid = np.zeros(coord.shape[0], dtype=float)
#    compact_resid[::] = m["compactness_resid"]
#    compactness_resid_all.append(compact_resid)
#    #
#    inertie_max_norm = np.zeros(coord.shape[0], dtype=float)
#    inertie_max_norm[::] = m["inertie_max_norm"]
#    inertie_max_norm_all.append(inertie_max_norm)
#    #
#    inertie_min_norm = np.zeros(coord.shape[0], dtype=float)
#    inertie_min_norm[::] = m["inertie_min_norm"]
#    inertie_min_norm_all.append(inertie_min_norm)
#    #
#    angle_inertie_max_perf = np.zeros(coord.shape[0], dtype=float)
#    angle_inertie_max_perf[::] = m["angle_inertie_max_perf"]
#    angle_inertie_max_perf_all.append(angle_inertie_max_perf)



coord_pc_moment_invariant_scaled_all = np.vstack(coord_pc_moment_invariant_scaled_all)
coord_pc_moment_invariant_all = np.vstack(coord_pc_moment_invariant_all)
coord_anisotropy_linear_spherical_scaled_all = np.vstack(coord_anisotropy_linear_spherical_scaled_all)
coord_anisotropy_linear_spherical_all = np.vstack(coord_anisotropy_linear_spherical_all)
tri_all = np.vstack(tri_all)
mesh_processing.mesh_from_arrays(coord_pc_moment_invariant_scaled_all, tri_all, path=os.path.join(OUTPUT, "pca_moment_invariant_lacunes-scaled.gii"))
mesh_processing.mesh_from_arrays(coord_pc_moment_invariant_all, tri_all, path=os.path.join(OUTPUT, "pca_moment_invariant.gii"))
mesh_processing.mesh_from_arrays(coord_anisotropy_linear_spherical_scaled_all, tri_all,
 path=os.path.join(OUTPUT, "anisotropy_linear_spherical_lacunes-scaled.gii"))
mesh_processing.mesh_from_arrays(coord_anisotropy_linear_spherical_all, tri_all, path=os.path.join(OUTPUT, "anisotropy_linear_spherical_.gii"))


for k in textures:
    textures[k] = np.hstack(textures[k])
    print textures[k]
    mesh_processing.save_texture(path=os.path.join(OUTPUT,"tex_%s.gii" % k), data=textures[k], intent='NIFTI_INTENT_NONE')

#labels_all = np.hstack(labels_all)
#lacune_id_all = np.hstack(lacune_id_all)
#fa_all = np.hstack(fa_all)
#compactness_all = np.hstack(compactness_all)
#compactness_resid_all = np.hstack(compactness_resid_all)
#inertie_max_norm_all = np.hstack(inertie_max_norm_all)
#inertie_min_norm_all = np.hstack(inertie_min_norm_all)
#angle_inertie_max_perf_all = np.hstack(angle_inertie_max_perf_all)

#mesh_processing.save_texture(path="/tmp/tex_lacunes_id.gii", data=lacune_id_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_clust-labels.gii", data=labels_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_fa.gii", data=fa_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_compactness.gii", data=compactness_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_compactness_resid.gii", data=compactness_resid_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_inerty_max_norm.gii", data=inertie_max_norm_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_inerty_min_norm.gii", data=inertie_min_norm_all, intent='NIFTI_INTENT_NONE')
#mesh_processing.save_texture(path="/tmp/tex_angle_inerty_max_perf.gii", data=angle_inertie_max_perf_all, intent='NIFTI_INTENT_NONE')

"""
anatomist /tmp/mesh_*.gii
"""

