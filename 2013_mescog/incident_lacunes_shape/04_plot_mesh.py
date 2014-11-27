# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:48:01 2014

@author: edouard.duchesnay@cea.fr


Generate lacunes in 3 spaces
- brain
- mnts_inv_pc12
- tnsr_inv_lin_plan

Generate 3 types of lacunes

- native
- lacunes__max_inertia_to_yaxis
- lacunes__max_inertia_to_yaxis_scaled
- lacunes__perfo_to_yaxis
- lacunes__perfo_to_yaxis_scaled


"""

import numpy as np
import glob
import pandas as pd
import os
from brainomics import mesh_processing
from soma import aims

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"

INPUT_IMAGES = os.path.join(BASE_PATH, "incident_lacunes_images")
#INPUT_MOMENTS_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments_area_from_mesh.csv"
INPUT_MOMENTS_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
INPUT_PC_CLUSTERS_CSV = os.path.join(BASE_PATH, "results_moments_invariant", "mnts-inv_pca.csv")
OUTPUT = os.path.join(BASE_PATH, "results_lacunes-mesh")
OUTPUT_MESH_IN_BRAIN = os.path.join(OUTPUT, "brain")

SCALE_XY_PC12 = 300
SCALE_anisotropy_linear = 500
SCALE_anisotropy_planar = 500

moments = pd.read_csv(INPUT_MOMENTS_CSV)#, index_col=0)
pc = pd.read_csv(INPUT_PC_CLUSTERS_CSV)#, index_col=0)

#ima_filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL_1_0_reorient.nii.gii"))
ima_filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL.nii*"))

##
mnts_inv_pc12__lacunes__max_inertia_to_yaxis_all = list()
mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled_all = list()
tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_all = list()
tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled_all = list()

mnts_inv_pc12__lacunes__perfo_to_yaxis_all = list()
mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled_all = list()
tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_all = list()
tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled_all = list()

brain_mesh_lacunes_native_all = None
brain_mesh_perfo_native_all = None
#brain_mesh_perfos = None
#brain_mesh_lacunes = None

tri_lacunes_all = list()
#tri_perforators_all=list()
lacune_id_all = list()
labels_all = list()

textures = {t:dict(lacunes=list(), perforators=list()) for t in
["lacune_id", "tensor_invariant_fa",
"tensor_invariant_linear_anisotropy",
"tensor_invariant_spherical_anisotropy",
"tensor_invariant_planar_anisotropy",
"perfo_angle_inertia_max"]}

n_vertex_lacunes = 0
#n_vertex_perforators = 0

for ima_filename in ima_filenames: # ima_filename = ima_filenames[0]
    lacune_id = int(os.path.basename(os.path.dirname(ima_filename)))
    print lacune_id
    mesh_lacunes_filename = ima_filename.replace(".nii.gz", "_native_1_0.gii")
    ima_perforator_filename = os.path.join(os.path.dirname(ima_filename), "%s-Perf.nii.gz" % lacune_id)
    #mesh_lacunes_filename = ima_filename.replace(".nii.gz", "_1_0.nii.gii")
    mesh_lacunes__max_inertia_to_yaxis_filename = ima_filename.replace(".nii.gz", "_centered_max_inertia_to_yaxis.gii")
    mesh_lacunes__perfo_to_yaxis_filename = ima_filename.replace(".nii.gz", "_centered_perfo_to_yaxis.nii.gii")
    mesh_perfo_native_filename = ima_perforator_filename.replace(".nii.gz", "_native.gii")
    mesh_perfo_centered_filename = ima_perforator_filename.replace(".nii.gz", "_centered.gii")
    #mesh_lacunes_filename = ima_filename.replace(".nii.gz", "_1_0.nii.gii")
    #mesh_lacunes__max_inertia_to_yaxis_filename = ima_filename.replace(".nii.gz", "_1_0_lacunes__max_inertia_to_yaxis.gii")
    #mesh_lacunes__perfo_to_yaxis_filename = ima_filename.replace(".nii.gz", "_1_0_lacunes__perfo_to_yaxis.nii.gii")
    #mesh_perfo_native_filename = ima_filename.replace(".nii.gz", "_perfo_native.gii")
    #mesh_perfo_centered_filename = ima_filename.replace(".nii.gz", "_perfo_centered.gii")
    #lacune_id = int(os.path.basename(os.path.dirname(ima_filename)))
    m = moments[moments["lacune_id"] == lacune_id]
    comp = pc[pc["lacune_id"] == lacune_id]
    if len(m) == 0 or len(comp) == 0:
        continue
    #
    mesh_lacunes_native = aims.read(mesh_lacunes_filename)
    mesh_lacunes__max_inertia_to_yaxis = aims.read(mesh_lacunes__max_inertia_to_yaxis_filename)
    mesh_lacunes__perfo_to_yaxis = aims.read(mesh_lacunes__perfo_to_yaxis_filename)
    mesh_perfo_native = aims.read(mesh_perfo_native_filename)
    mesh_perfo_centered = aims.read(mesh_perfo_centered_filename)
    #
    xyz_lacunes__max_inertia_to_yaxis = np.array(mesh_lacunes__max_inertia_to_yaxis.vertex()).copy()
    tri_lacunes__max_inertia_to_yaxis = np.array(mesh_lacunes__max_inertia_to_yaxis.polygon()).copy()
    xyz_lacunes__perfo_to_yaxis = np.array(mesh_lacunes__perfo_to_yaxis.vertex()).copy()
    tri_lacunes__perfo_to_yaxis = np.array(mesh_lacunes__perfo_to_yaxis.polygon()).copy()
    #xyz_lacune, tri = mesh_processing.mesh_arrays(filename)
    vol = m['vol_mm3'].values[0]
    scale_xyz = (moments['vol_mm3'].mean() / vol) ** (1. / 3.)
    ### =======================================================================
    ### mnts_inv_pc12__lacunes__max_inertia_to_yaxis
    ### =======================================================================
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled = xyz_lacunes__max_inertia_to_yaxis.copy()
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled *= scale_xyz
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled[:, 0] += comp.PC1 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled[:, 1] += comp.PC2 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled_all.append(mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled)
    #
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis = xyz_lacunes__max_inertia_to_yaxis.copy()
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis[:, 0] += comp.PC1 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis[:, 1] += comp.PC2 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__max_inertia_to_yaxis_all.append(mnts_inv_pc12__lacunes__max_inertia_to_yaxis)
    ### =======================================================================
    ### mnts_inv_pc12__lacunes__perfo_to_yaxis
    ### =======================================================================
    mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled = xyz_lacunes__perfo_to_yaxis.copy()
    mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled *= scale_xyz
    mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled[:, 0] += comp.PC1 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled[:, 1] += comp.PC2 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled_all.append(mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled)
    #
    mnts_inv_pc12__lacunes__perfo_to_yaxis = xyz_lacunes__perfo_to_yaxis.copy()
    mnts_inv_pc12__lacunes__perfo_to_yaxis[:, 0] += comp.PC1 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__perfo_to_yaxis[:, 1] += comp.PC2 * SCALE_XY_PC12
    mnts_inv_pc12__lacunes__perfo_to_yaxis_all.append(mnts_inv_pc12__lacunes__perfo_to_yaxis)
    ### =======================================================================
    ###  tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis
    ### =======================================================================
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis = xyz_lacunes__max_inertia_to_yaxis.copy()
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis[:, 0] += m.tensor_invariant_linear_anisotropy * SCALE_anisotropy_linear
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis[:, 1] += m.tensor_invariant_planar_anisotropy * SCALE_anisotropy_planar
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_all.append(tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis)
    #
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled = xyz_lacunes__max_inertia_to_yaxis.copy()
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled *= scale_xyz
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled[:, 0] += m.tensor_invariant_linear_anisotropy * SCALE_anisotropy_linear
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled[:, 1] += m.tensor_invariant_planar_anisotropy * SCALE_anisotropy_planar
    tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled_all.append(tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled)
    ### =======================================================================
    ###  tnsr_inv_lin_plan__lacunes__perfo_to_yaxis
    ### =======================================================================
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis = xyz_lacunes__perfo_to_yaxis.copy()
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis[:, 0] += m.tensor_invariant_linear_anisotropy * SCALE_anisotropy_linear
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis[:, 1] += m.tensor_invariant_planar_anisotropy * SCALE_anisotropy_planar
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_all.append(tnsr_inv_lin_plan__lacunes__perfo_to_yaxis)
    #
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled = xyz_lacunes__perfo_to_yaxis.copy()
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled *= scale_xyz
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled[:, 0] += m.tensor_invariant_linear_anisotropy * SCALE_anisotropy_linear
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled[:, 1] += m.tensor_invariant_planar_anisotropy * SCALE_anisotropy_planar
    tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled_all.append(tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled)

   # Tri
    tri_lacunes_all.append(tri_lacunes__max_inertia_to_yaxis + n_vertex_lacunes)
    ### =======================================================================
    ### brain
    ### =======================================================================
    if brain_mesh_lacunes_native_all is None:
        brain_mesh_lacunes_native_all = mesh_lacunes_native
    else:
        aims.SurfaceManip.meshMerge(brain_mesh_lacunes_native_all, mesh_lacunes_native)

    if brain_mesh_perfo_native_all is None:
        brain_mesh_perfo_native_all = mesh_perfo_native
    else:
        aims.SurfaceManip.meshMerge(brain_mesh_perfo_native_all, mesh_perfo_native)

    ### =======================================================================
    ### Mesh perforator by a cylindre, centered on the lacune gravity center
    ### =======================================================================
    ext1 = m[['perfo_ext1_x', 'perfo_ext1_y', 'perfo_ext1_z']].values.ravel()
    ext2 = m[['perfo_ext2_x', 'perfo_ext2_y', 'perfo_ext2_z']].values.ravel()
    # brain
#    mesh_perfo = aims.SurfaceGenerator.cylinder(ext1, ext2, 1, 1, 100, 1, 1)
#    if brain_mesh_perfos is None:
#        brain_mesh_perfos = mesh_perfo
#    else:
#        aims.SurfaceManip.meshMerge(brain_mesh_perfos, mesh_perfo)

    #ids = np.zeros(xyz_lacune.shape[0], dtype=int)
    #ids[::] = lacune_id
    #lacune_id_all.append(ids)
    #label = np.zeros(xyz_lacune.shape[0], dtype=int)
    #label[::] = comp.label
    #labels_all.append(label)
    #
    for k in textures:
        arr = np.zeros(xyz_lacunes__max_inertia_to_yaxis.shape[0], dtype=float)
        arr[::] = m[k]
        textures[k]["lacunes"].append(arr)
        arr = np.zeros(len(mesh_lacunes__perfo_to_yaxis.vertex()), dtype=float)
        arr[::] = m[k]
        textures[k]["perforators"].append(arr)
    n_vertex_lacunes += xyz_lacunes__max_inertia_to_yaxis.shape[0]
"""
    if brain_mesh_lacunes is None:
        brain_mesh_lacunes = mesh_lacune
    else:
        aims.SurfaceManip.meshMerge(brain_mesh_lacunes, mesh_lacune)

    if brain_mesh_perfo is None:
        brain_mesh_perfo = mesh_perfo
    else:
        aims.SurfaceManip.meshMerge(brain_mesh_perfo, mesh_perfo)


    mesh_perfo_centered = aims.SurfaceGenerator.cylinder(ext1, ext2, 1, 1, 100, 1, 1)
"""



tri_lacunes_all = np.vstack(tri_lacunes_all)

### =======================================================================
### mnts_inv_pc12__lacunes__max_inertia_to_yaxis
### =======================================================================
mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled_all = \
    np.vstack(mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled_all)
mesh_processing.mesh_from_arrays(mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "mnts_inv_pc12__lacunes__max_inertia_to_yaxis_scaled.gii"))
mnts_inv_pc12__lacunes__max_inertia_to_yaxis_all = \
    np.vstack(mnts_inv_pc12__lacunes__max_inertia_to_yaxis_all)
mesh_processing.mesh_from_arrays(mnts_inv_pc12__lacunes__max_inertia_to_yaxis_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "mnts_inv_pc12__lacunes__max_inertia_to_yaxis.gii"))

### =======================================================================
### mnts_inv_pc12__lacunes__perfo_to_yaxis
### =======================================================================
mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled_all = \
    np.vstack(mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled_all)
mesh_processing.mesh_from_arrays(mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "mnts_inv_pc12__lacunes__perfo_to_yaxis_scaled.gii"))

mnts_inv_pc12__lacunes__perfo_to_yaxis_all = \
    np.vstack(mnts_inv_pc12__lacunes__perfo_to_yaxis_all)
mesh_processing.mesh_from_arrays(mnts_inv_pc12__lacunes__perfo_to_yaxis_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "mnts_inv_pc12__lacunes__perfo_to_yaxis.gii"))

### =======================================================================
###  tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis
### =======================================================================

tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled_all = \
    np.vstack(tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled_all)
mesh_processing.mesh_from_arrays(tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scaled_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_scale.gii"))
tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_all = \
    np.vstack(tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_all)
mesh_processing.mesh_from_arrays(tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "tnsr_inv_lin_plan__lacunes__max_inertia_to_yaxis.gii"))

### =======================================================================
###  tnsr_inv_lin_plan__lacunes__perfo_to_yaxis
### =======================================================================
tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled_all = \
    np.vstack(tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled_all)
mesh_processing.mesh_from_arrays(tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scaled_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_scale.gii"))
tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_all = \
    np.vstack(tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_all)
mesh_processing.mesh_from_arrays(tnsr_inv_lin_plan__lacunes__perfo_to_yaxis_all, tri_lacunes_all,
                                 path=os.path.join(OUTPUT, "tnsr_inv_lin_plan__lacunes__perfo_to_yaxis.gii"))

### =======================================================================
### brain__lacunes_native & brain__perfo_native
### =======================================================================

aims.write(brain_mesh_lacunes_native_all,
           os.path.join(OUTPUT, "brain__lacunes__native.gii"))
aims.write(brain_mesh_perfo_native_all,
           os.path.join(OUTPUT, "brain__perforators__native.gii"))


for k in textures:
    textures[k]["lacunes"] = np.hstack(textures[k]["lacunes"])
    #print textures[k]
    mesh_processing.save_texture(filename=os.path.join(OUTPUT,"tex__lacunes__%s.gii" % k),
                                 data=textures[k]["lacunes"])#, intent='NIFTI_INTENT_NONE')
    textures[k]["perforators"] = np.hstack(textures[k]["perforators"])
    #print textures[k]
    mesh_processing.save_texture(filename=os.path.join(OUTPUT,"tex__perforators__%s.gii" % k),
                                 data=textures[k]["perforators"])#, intent='NIFTI_INTENT_NONE')

"""
anatomist /tmp/mesh_*.gii
freeview -f mnts-inv_pc12_scaled.gii:overlay=tex_tensor_invariant_fa.gii
"""

