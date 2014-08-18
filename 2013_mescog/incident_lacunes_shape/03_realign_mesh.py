# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:54:24 2014

@author: edouard.duchesnay@cea.fr

Realign lacunes mesh :
Translation: center (0, 0, 0) => center_of_mass
Rotation: align main inertie axis such
 v1 => y
 v2 => x
 v3 => z
"""
import numpy as np
import glob
import pandas as pd
import os
#import re

INPUT_IMAGES = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images"
INPUT_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments.csv"
#OUTPUT_AREA_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments_area_from_mesh.csv"

data = pd.read_csv(INPUT_CSV)

filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL.nii*"))
#filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/027/027_1046-M18_LacBL.nii.gz"
filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/075/075_2005-M18_LacBL.nii.gz"
area_vol_all = list()
for filename in filenames:
    lacune_id = int(os.path.basename(os.path.dirname(filename)))
    print lacune_id
    m = data[data["lacune_id"] == lacune_id]
    if len(m) == 0:
        continue
    inerties = m[["Orientation_Inerty_0", "Orientation_Inerty_1", "Orientation_Inerty_2"]].values.ravel()
    mass_center = m[["Center_of_Mass_0", "Center_of_Mass_1", "Center_of_Mass_2"]].values.ravel()
    v1 = m[["Orientation_V1_0", "Orientation_V1_1", "Orientation_V1_2"]].values.T
    v2 = m[["Orientation_V2_0", "Orientation_V2_1", "Orientation_V2_2"]].values.T
    v3 = m[["Orientation_V3_0", "Orientation_V3_1", "Orientation_V3_2"]].values.T
    # 1) QC check center_of_mass == the one computed by AimsMoments
    import scipy.ndimage as ndimage
    from soma import aims
    vol_aims = aims.Reader().read(filename)
    # t,z,y,x
    arr = vol_aims.arraydata().squeeze()
    center = ndimage.measurements.center_of_mass(arr)
    voxel_size = vol_aims.header()['voxel_size'].arraydata()
    assert np.allclose((center * voxel_size[::-1])[::-1], mass_center)
    # QC END
    # 3) Compute rotation such
    #v1 => y
    #v2 => x
    #v3 => z
    V = np.hstack([v2, v1, v3])
    # Find rotation matrix
    A = np.zeros((9, 9))
    b = np.zeros((9, 1))
    i = 0
    while i < 9:
        for j in range(0, 9, 3):
            #print i, j, int(i / 3)
            A[i, j:(j + 3)] = V[:, int(i / 3)]
            if (i % 3) == int(i / 3):
                b[i] = 1
            i += 1
    import scipy.linalg
    x, res, rank, s = scipy.linalg.lstsq(A, b)
    rot = x.reshape((3, 3))
    assert np.allclose(np.dot(rot, v1).ravel(), [0, 1, 0])
    assert np.allclose(np.dot(rot, v2).ravel(), [1, 0, 0])
    assert np.allclose(np.dot(rot, v3).ravel(), [0, 0, 1])
    # Apply transfo to mesh
    filename_mesh = filename.replace(".nii.gz", "_1_0.nii.gii")
    filename_mesh_reorient = filename.replace(".nii.gz", "_1_0_reorient.nii.gii")
    mesh_aims = aims.Reader().read(filename_mesh)
    coord = np.array(mesh_aims.vertex())
    triangles = np.array(mesh_aims.polygon())
    coord_new = np.zeros(coord.shape)
    for i in xrange(coord.shape[0]):
        coord_new[i, :] = np.dot(rot, (coord[i, :] - mass_center))
    from brainomics import mesh_processing
    mesh_processing.mesh_from_arrays(coord_new, triangles, path=filename_mesh_reorient)

#area_vol = pd.DataFrame(area_vol_all, columns=["lacune_id", "area_mesh", "vol_mesh"])
#area_vol["lacune_id"] = area_vol.lacune_id
#data["lacune_id"] = data["lacune_id"]
#moments_with_area = data.merge(area_vol)
#moments_with_area.to_csv(OUTPUT_AREA_CSV, index=False)
#INPUT_MOMENTS_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments.csv"
#moments = pd.read_csv(INPUT_MOMENTS_CSV, index_col=0)

    #ndimage.find_objects(arr)
#    label, num_features = ndimage.label(arr)
#    voxel_size = np.array(  [ 0.898425996303558, 1.20071411132812, 0.898452758789062 ])
#    voxel_size * arr.shape
#    center = ndimage.measurements.center_of_mass(arr)
#    center = np.array([arr.shape[0] - center[0], arr.shape[1] - center[1], arr.shape[2] - center[2]])
#    center * voxel_size
#    np.array([center[0], arr.shape[1] - 1 - center[1], arr.shape[2] - 1 - center[2]])* voxel_size
#    mass_center

    #nzeros = np.where(arr == 1)
    #[np.sum(nzeros[i] * voxel_size[i])/len(nzeros[i]) for i in xrange(len(voxel_size))]
"""
cd /home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images
anatomist 018/*.gii 075/*.gii 035/*.gii 033/*.gii 021/*.gii
    'volume_dimension' : [ 256, 160, 256 ],
'volume_dimension' : [ 256, 160, 256 ],

"""
