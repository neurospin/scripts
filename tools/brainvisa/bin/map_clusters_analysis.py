#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyse a map of values (possibly signed) and generate informations to vizualize
it. The map can be thresholded, then left out-values are clusterized into
components of connected voxels. For each cluster a mesh if generated and a information
about the cluster is stored in a csv file.

Generated information:
(i) a mesh file for all small cluster (meshed by a sphere);
(ii) a mesh file for all big cluster;
(iii) a csv file that contains size, min, max, mean, coordinates of clusters;
(iv) an image of cluster label.
"""

import os, sys, optparse
import numpy as np
import scipy, scipy.ndimage
from soma import aims
import tempfile
import pandas as pd


def xyz_to_mm(trm, vox_size):
    trmcp = aims.Motion(trm)
    trmcp.scale(aims.Point3d(vox_size), aims.Point3d(1, 1, 1))
    return trmcp


# Transformation operation
def vol_get_trm_xyz_to_mm(vol, refNb=0):
    """
    Return the transformation XYZ=>mm of a volume
    """
    trm_vol_mm2ref_mm = aims.Motion(vol.header()['transformations'][refNb])
    vol_voxel_size = vol.header()['voxel_size'][:3]
    return(xyz_to_mm(trm_vol_mm2ref_mm, vol_voxel_size))


def cluster_info(arr, clust_labeled, clust_sizes, labels, centers,
                 trm_xyz_to_mm):
    """Compute Information about cluster: label, size, mean, max, min,
    min_coord_mm, max_coord_mm, center_coord_mm"""
    clusters_info = list()
    arr_abs = np.abs(arr)
    print "Scan clusters:"
    for i in xrange(len(labels)):
        print i,
        mask = clust_labeled == labels[i]
        center_xyz = centers[i]
        center_mm = np.round(np.asarray(trm_xyz_to_mm.transform(center_xyz)),
                             3)
        if clust_sizes[i] > 1:
            max_val = np.max(np.abs(arr[mask]))
            max_xyz = np.asarray([c[0] for c in
                np.where((arr_abs == max_val) & mask)])
            max_mm = np.round(np.asarray(trm_xyz_to_mm.transform(max_xyz)), 3)
            min_val = np.min(arr[mask])
            mean_val = np.mean(arr[mask])
        else:
            max_val = min_val = mean_val = arr[mask][0]
            max_mm = center_mm
        info = [labels[i], clust_sizes[i]] + \
        max_mm.tolist() + center_mm.tolist() + [mean_val, max_val, min_val]
        clusters_info.append(info)
        header = ["label", "size",
                  "x_max_mm",  "y_max_mm",  "z_max_mm",
                   "x_center_mm",  "y_center_mm", "z_center_mm",
                  "mean", "max", "min"]
    return header, clusters_info


def mesh_small_clusters(arr, clust_labeled, clust_sizes, labels,
    output_mesh_small_cluster_filename, centers, vol, thresh_size):
    """Mesh small cluster"""
    vox_size = np.asarray(vol.header()['voxel_size'])
    vox_vol = vox_size.prod()
    small_cluster_meshs = None
    for i in xrange(len(labels)):
        if clust_sizes[i] > thresh_size:
            continue
        sphere_radius = (3. / 4. * clust_sizes[i] * vox_vol / np.pi)\
            ** (1. / 3.)
        center_mm = np.asarray(centers[i]) * vox_size
        mesh = aims.SurfaceGenerator.icosahedron(center_mm, sphere_radius)
        if small_cluster_meshs is None:
            small_cluster_meshs = mesh
        else:
            aims.SurfaceManip.meshMerge(small_cluster_meshs, mesh)
    writer = aims.Writer()
    # Save mesh of small clusters
    small_cluster_meshs.header()['referentials'] = vol.header()['referentials']
    small_cluster_meshs.header()['transformations'] = vol.header()['transformations']
    writer.write(small_cluster_meshs, output_mesh_small_cluster_filename)


##############################################################################
# Mesh big clusters using AimsClusterArg
def mesh_big_clusters(arr, clust_labeled, clust_sizes, labels,
                      output_mesh_large_cluster_filename, tempdir, vol,
                      thresh_size):
    big_clust_vol = aims.Volume(vol)
    big_clust_arr = np.asarray(big_clust_vol).squeeze()
    big_clust_arr[:] = 0
    for i in xrange(len(labels)):
        label = labels[i]
        if clust_sizes[i] > thresh_size:
            big_clust_arr[clust_labeled == label] = 1
    big_clust_vol_filename = os.path.join(tempdir, "big_clusters.nii")
    writer = aims.Writer()
    writer.write(big_clust_vol, big_clust_vol_filename)
    big_clust_graph_filename = os.path.join(tempdir, "big_clusters.arg")
    cmd = 'AimsClusterArg --input %s --output %s' % \
    (big_clust_vol_filename, big_clust_graph_filename)
    os.popen(cmd)
    graph = aims.read(big_clust_graph_filename)
    big_clust_meshs = None
    for v in graph.vertices():
        if big_clust_meshs is None:
            big_clust_meshs = v['aims_Tmtktri']
        else:
            aims.SurfaceManip.meshMerge(big_clust_meshs, v['aims_Tmtktri'])
    if big_clust_meshs is not None:
        big_clust_meshs.header()['referentials'] = vol.header()['referentials']
        big_clust_meshs.header()['transformations'] = vol.header()['transformations']
        writer.write(big_clust_meshs, output_mesh_large_cluster_filename)
    else:
        print "No big cluster generated"
if __name__ == "__main__":
    # Set default values to parameters
    thresh_size = 10
    thresh_neg_low = -np.inf
    thresh_neg_high = 0
    thresh_pos_low = 0
    thresh_pos_high = np.inf
    # parse command line options
    parser = optparse.OptionParser(description=__doc__)

    parser.add_option('--input',
        help='Input map volume', type=str)
    parser.add_option('--thresh_size',
        help='Threshold, in voxels nb, between small and big clusters'
        ' (default %i).' % thresh_size, default=thresh_size, type=float)
    parser.add_option('--thresh_neg_low',
        help='Negative lower bound threshold (default %f)' % thresh_neg_low, default=thresh_neg_low, type=float)
    parser.add_option('--thresh_neg_high',
        help='Negative upper bound threshold (default %f)' % thresh_neg_high, default=thresh_neg_high, type=float)
    parser.add_option('--thresh_pos_low',
        help='Positive lower bound threshold (default %f)' % thresh_pos_low, default=thresh_pos_low, type=float)
    parser.add_option('--thresh_pos_high',
        help='Positive upper bound threshold (default %f)' % thresh_pos_high, default=thresh_pos_high, type=float)

    options, args = parser.parse_args(sys.argv)
    #print __doc__
    if options.input is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)

#    map_filename = "/neurospin/brainomics/neuroimaging_ressources/examples_images/weights_map_mixte.nii"
#    map_filename = "/neurospin/brainomics/2013_imagen_subdepression/results/svm_feature_selection/betas.nii"
    map_filename = options.input
    thresh_size = options.thresh_size
    thresh_neg_low = options.thresh_neg_low
    thresh_neg_high = options.thresh_neg_high
    thresh_pos_low = options.thresh_pos_low
    thresh_pos_high = options.thresh_pos_high
    ##########################################################################

    map_filename_noext, _ = os.path.splitext(map_filename)
    output_csv_cluster_info_filename  = map_filename_noext + "_info_clusters.csv"
    output_labels_cluster_filename  = map_filename_noext + "_labels_clusters.nii"
    output_mesh_small_cluster_filename  = map_filename_noext + "_small_clusters.mesh"
    output_mesh_large_cluster_filename  = map_filename_noext + "_large_clusters.mesh"
    tempdir = tempfile.mkdtemp()

    ##########################################################################
    # Read volume
    vol = aims.read(map_filename)
    trm_xyz_to_mm = vol_get_trm_xyz_to_mm(vol)
    arr = np.asarray(vol).squeeze()

    ##########################################################################
    # Find clusters (connected component abov a given threshold)
    clust_bool = np.zeros(arr.shape, dtype=bool)
    #((arr > thresh_neg_low) & (arr < thresh_neg_high) | (arr > thresh_pos_low) & (arr < thresh_pos_high)).sum()
    clust_bool[(arr > thresh_neg_low) & (arr < thresh_neg_high) |
               (arr > thresh_pos_low) & (arr < thresh_pos_high)] = True
    clust_labeled, n_clusts = scipy.ndimage.label(clust_bool)
    clust_sizes = scipy.ndimage.measurements.histogram(clust_labeled, 1,
                                                       n_clusts, n_clusts)
    labels = np.unique(clust_labeled)[1:]
    centers = scipy.ndimage.center_of_mass(clust_bool, clust_labeled, labels)
    labels_vol = aims.Volume(vol)
    labels_vol_arr = np.asarray(labels_vol).squeeze()
    labels_vol_arr[:] = clust_labeled[:]
    writer = aims.Writer()
    writer.write(labels_vol, output_labels_cluster_filename)
    ##########################################################################
    # Get clusters information
    header_info, info = cluster_info(arr, clust_labeled, clust_sizes, labels,
                                     centers, trm_xyz_to_mm)
    df = pd.DataFrame(info, columns=header_info)
    df.to_csv(output_csv_cluster_info_filename, sep="\t", index=False)

    ##########################################################################
    # Mesh small clusters by spheres
    mesh_small_clusters(arr, clust_labeled, clust_sizes, labels,
        output_mesh_small_cluster_filename, centers, vol, thresh_size)

    ##########################################################################
    # Mesh big clusters
    mesh_big_clusters(arr, clust_labeled, clust_sizes, labels,
        output_mesh_large_cluster_filename, tempdir, vol, thresh_size)