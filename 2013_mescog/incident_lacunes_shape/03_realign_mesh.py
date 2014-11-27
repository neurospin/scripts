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
#import scipy.ndimage as ndimage
from soma import aims
import sys
sys.path.append("/home/ed203246/git/scripts/2013_mescog/incident_lacunes_shape")

from math_transformation import angle_between_vectors, unit_vector
from math_transformation import rotation_matrix, vector_product
from brainomics import mesh_processing

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"

INPUT_IMAGES = os.path.join(BASE_PATH, "incident_lacunes_images")
INPUT_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
# OUPUT 1

data = pd.read_csv(INPUT_CSV)

ima_filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL.nii*"))
#filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/027/027_1046-M18_LacBL.nii.gz"
#filename = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images/075/075_2005-M18_LacBL.nii.gz"
brain_mesh_lacunes = None
brain_mesh_perfo = None

for ima_filename in ima_filenames: #filename = filenames[0]
    lacune_id = int(os.path.basename(os.path.dirname(ima_filename)))
    print lacune_id
    mesh_lacunes_filename = ima_filename.replace(".nii.gz", "_native_1_0.gii")
    ima_perforator_filename = os.path.join(os.path.dirname(ima_filename), "%s-Perf.nii.gz" % lacune_id)
    #mesh_lacunes_filename = ima_filename.replace(".nii.gz", "_1_0.nii.gii")
    mesh_lacunes_max_inertia_to_yaxis_filename = ima_filename.replace(".nii.gz", "_centered_max_inertia_to_yaxis.gii")
    mesh_lacunes_perfo_to_yaxis_filename = ima_filename.replace(".nii.gz", "_centered_perfo_to_yaxis.nii.gii")
    mesh_perfo_native_filename = ima_perforator_filename.replace(".nii.gz", "_native.gii")
    mesh_perfo_centered_filename = ima_perforator_filename.replace(".nii.gz", "_centered.gii")
#    mesh_filename = ima_filename.replace(".nii.gz", "_1_0.nii.gii")
#    mesh_lacunes_max_inertia_to_yaxis_filename = ima_filename.replace(".nii.gz", "_1_0_lacunes_max_inertia_to_yaxis.gii")
#    mesh_lacunes_perfo_to_yaxis_filename = ima_filename.replace(".nii.gz", "_1_0_lacunes_perfo_to_yaxis.nii.gii")
#    mesh_filename_perfo = ima_filename.replace(".nii.gz", "_perfo.gii")
#    mesh_filename_perfo = ima_filename.replace(".nii.gz", "_perfo.gii")

    m = data[data["lacune_id"] == lacune_id]
    if len(m) == 0:
        continue
    inerties = m[["orientation_v1_inertia", "orientation_v2_inertia", "orientation_v3_inertia"]].values.ravel()
    mass_center = m[["center_of_mass_x", "center_of_mass_y", "center_of_mass_z"]].values.ravel()
    v1 = m[["orientation_v1_x", "orientation_v1_y", "orientation_v1_z"]].values.T
    v2 = m[["orientation_v2_x", "orientation_v2_y", "orientation_v2_z"]].values.T
    v3 = m[["orientation_v3_x", "orientation_v3_y", "orientation_v3_z"]].values.T
    perfo = unit_vector(np.array(m[["perfo_orientation_x",
                                    "perfo_orientation_y",
                                    "perfo_orientation_z"]]).ravel())
    ###
    ### 1) QC check center_of_mass == the one computed by AimsMoments
    ###
    ima = aims.Reader().read(ima_filename)
    mesh_lacune = aims.Reader().read(mesh_lacunes_filename)
    mesh_xyz = np.array(mesh_lacune.vertex())
    mesh_tri = np.array(mesh_lacune.polygon())
    # t,z,y,x
    arr = ima.arraydata().squeeze()
    voxel_size = ima.header()['voxel_size'].arraydata()

    # recompute mass_center from image to check that zyx*voxel_size == what is csv
    # file
    zyx = np.where(arr != 0)
    xyz_mm = np.vstack([zyx[2]*voxel_size[0],
             zyx[1]*voxel_size[1],
             zyx[0]*voxel_size[2]]).T
    xyz_mass_center_mm = xyz_mm.mean(axis=0)
    # Check we retrive mass_center
    assert np.allclose(xyz_mass_center_mm, mass_center)

    ### =======================================================================
    ### Rotate mesh such:
    ### - perforator is aligned to y-axis
    ### - max inertia in (x-z plan) is aligned to x-axis
    ### =======================================================================
    # center on mass center
    xyz_mm -= xyz_mass_center_mm
    assert np.allclose(xyz_mm.mean(axis=0), [0, 0, 0])
    # Find rotation that aligns perforator to y-axis
    y_axis = unit_vector(np.array([0., 1., 0.]))
    Rperf_align = rotation_matrix(angle_between_vectors(perfo, y_axis), vector_product(perfo, y_axis))
    # check perfo is aligned to y-axis
    assert np.allclose(unit_vector(y_axis),
                       unit_vector(np.dot(perfo, Rperf_align[:3,:3].T)))
    # Rotate lacune with perfo to y-axis rotation
    xyz_perfo = np.dot(xyz_mm, Rperf_align[:3,:3].T)
    # Find rotation that aligns ax inertia in (x-z plan) to x-axis
    x_axis = unit_vector(np.array([1., 0, 0.]))
    xyz_inertia = np.sum((xyz_perfo - xyz_perfo.mean(axis=0))**2, axis=0)
    xyz_inertia[1] = 0.
    xz_inertia = unit_vector(xyz_inertia)

    Rmxinertiaxz_align = rotation_matrix(angle_between_vectors(xz_inertia, x_axis),
                                    vector_product(xz_inertia, x_axis))
    assert np.allclose( # check max inertia in x-z plan is aligned to x-axis
        unit_vector(x_axis),
        unit_vector(np.dot(xz_inertia, Rmxinertiaxz_align[:3,:3].T)))

    # Compose the two rotations
    Rcomp = np.dot(Rmxinertiaxz_align, Rperf_align)
    assert np.allclose(  # check composition works
        np.dot(np.dot(xyz_mm, Rperf_align[:3,:3].T), Rmxinertiaxz_align[:3,:3].T),
        np.dot(xyz_mm, Rcomp[:3,:3].T))
    # Apply transfo to mesh
    mesh_xyz_perfo_to_yaxis = np.dot((mesh_xyz - mass_center),  Rcomp[:3,:3].T)
    mesh_processing.mesh_from_arrays(mesh_xyz_perfo_to_yaxis,
                                     mesh_tri, path=mesh_lacunes_perfo_to_yaxis_filename)

    ### =======================================================================
    ### Rotate mesh such:
    ### Max inertia is aligned to y-axis
    ### =======================================================================
    # Compute rotation such
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
    mesh_xyz_new = np.zeros(mesh_xyz.shape)
    for i in xrange(mesh_xyz.shape[0]):
        mesh_xyz_new[i, :] = np.dot(rot, (mesh_xyz[i, :] - mass_center))
    mesh_processing.mesh_from_arrays(mesh_xyz_new, mesh_tri,
                                     path=mesh_lacunes_max_inertia_to_yaxis_filename)

    ### =======================================================================
    ### Mesh perforator by a cylindre, centered on the lacune gravity center
    ### =======================================================================
    ext1 = m[['perfo_ext1_x', 'perfo_ext1_y', 'perfo_ext1_z']].values.ravel()
    ext2 = m[['perfo_ext2_x', 'perfo_ext2_y', 'perfo_ext2_z']].values.ravel()
    mesh_perfo = aims.SurfaceGenerator.cylinder(ext1, ext2, .5, .5, 100, 1, 1)
    aims.write(mesh_perfo, mesh_perfo_native_filename)
    ext1 -= mass_center
    ext2 -= mass_center
    #print ext1, ext2, "=>",mesh_filename_perfo
    mesh_perfo_centered = aims.SurfaceGenerator.cylinder(ext1, ext2, 1, 1, 100, 1, 1)
    aims.write(mesh_perfo_centered, mesh_perfo_centered_filename)


"""
    # QC (again) check center_of_mass == the one computed by AimsMoments
#    import scipy.ndimage as ndimage
#    #from soma import aims
#    #vol_aims = aims.Reader().read(filename)
#    # t,z,y,x
#    arr = ima.arraydata().squeeze()
#    center = ndimage.measurements.center_of_mass(arr)
#    voxel_size = ima.header()['voxel_size'].arraydata()
#    assert np.allclose((center * voxel_size[::-1])[::-1], mass_center)
    # QC END
"""