#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyse a map of values (possibly signed) and generate informations to vizualize
it. The map can be thresholded, then left out-values are clusterized into
components of connected voxels. For each cluster a mesh if generated and a information
about the cluster is stored in a csv file.

Generated information:
(i) a mesh file for all small cluster (meshed by a sphere);
(ii) a mesh file for all large cluster;
(iii) a csv file that contains size, min, max, mean, coordinates of clusters;
(iv) an image of cluster label.
"""

import os, sys, argparse
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
def ima_get_trm_xyz_to_mm(ima, refNb=0):
    """
    Return the transformation XYZ=>mm of a volume
    """
    trm_ima_mm2ref_mm = aims.Motion(ima.header()['transformations'][refNb])
    ima_voxel_size = ima.header()['voxel_size'][:3]
    return(xyz_to_mm(trm_ima_mm2ref_mm, ima_voxel_size))


def clusters_info(arr, clust_labeled, clust_sizes, labels, centers,
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
    output_clusters_small_mesh_filename, centers, ima, thresh_size):
    """Mesh small cluster"""
    vox_size = np.asarray(ima.header()['voxel_size'])[:3]
    vox_ima = vox_size.prod()
    small_clusters_meshs = None
    for i in xrange(len(labels)):
        if clust_sizes[i] > thresh_size:
            continue
        sphere_radius = (3. / 4. * clust_sizes[i] * vox_ima / np.pi)\
            ** (1. / 3.)
        center_mm = np.asarray(centers[i]) * vox_size
        mesh = aims.SurfaceGenerator.icosahedron(center_mm, sphere_radius)
        if small_clusters_meshs is None:
            small_clusters_meshs = mesh
        else:
            aims.SurfaceManip.meshMerge(small_clusters_meshs, mesh)
    if small_clusters_meshs is not None:
        writer = aims.Writer()
        # Save mesh of small clusters
        small_clusters_meshs.header()['referentials'] = ima.header()['referentials']
        small_clusters_meshs.header()['transformations'] = ima.header()['transformations']
        writer.write(small_clusters_meshs, output_clusters_small_mesh_filename)
        return small_clusters_meshs
    else:
        print "No small cluster generated"
        return None

##############################################################################
# Mesh large clusters using AimsClusterArg
def mesh_large_clusters(arr, clust_labeled, clust_sizes, labels,
                      output_clusters_large_mesh_filename, tempdir, ima,
                      thresh_size):
    large_clust_ima = aims.Volume(ima)
    large_clust_arr = np.asarray(large_clust_ima).squeeze()
    large_clust_arr[:] = 0
    for i in xrange(len(labels)):
        label = labels[i]
        if clust_sizes[i] > thresh_size:
            large_clust_arr[clust_labeled == label] = 1
    large_clust_ima_filename = os.path.join(tempdir, "large_clusters.nii")
    writer = aims.Writer()
    writer.write(large_clust_ima, large_clust_ima_filename)
    large_clust_graph_filename = os.path.join(tempdir, "large_clusters.arg")
    cmd = 'AimsClusterArg --input %s --output %s' % \
    (large_clust_ima_filename, large_clust_graph_filename)
    os.popen(cmd)
    graph = aims.read(large_clust_graph_filename)
    large_clust_meshs = None
    for v in graph.vertices():
        if large_clust_meshs is None:
            large_clust_meshs = v['aims_Tmtktri']
        else:
            aims.SurfaceManip.meshMerge(large_clust_meshs, v['aims_Tmtktri'])
    if large_clust_meshs is not None:
        large_clust_meshs.header()['referentials'] = ima.header()['referentials']
        large_clust_meshs.header()['transformations'] = ima.header()['transformations']
        writer.write(large_clust_meshs, output_clusters_large_mesh_filename)
        return large_clust_meshs
    else:
        print "No large cluster generated"
        return None

if __name__ == "__main__":
    # Set default values to parameters
    thresh_size = 10
    thresh_neg_low = -np.inf
    thresh_neg_high = 0
    thresh_pos_low = 0
    thresh_pos_high = np.inf
    referential = 'Talairach-MNI template-SPM'
    fsl_warp_cmd = "fsl5.0-applywarp -i %s -r %s -o %s"
    MNI152_T1_1mm_brain_filename = "/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz"

    # parse command line options
    #parser = optparse.OptionParser(description=__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
        help='Input map volume', type=str)
    parser.add_argument('--thresh_size',
        help='Threshold, in voxels nb, between small and large clusters'
        ' (default %i).' % thresh_size, default=thresh_size, type=float)
    parser.add_argument('--thresh_neg_low',
        help='Negative lower bound threshold (default %f)' % thresh_neg_low, default=thresh_neg_low, type=float)
    parser.add_argument('--thresh_neg_high',
        help='Negative upper bound threshold (default %f)' % thresh_neg_high, default=thresh_neg_high, type=float)
    parser.add_argument('--thresh_pos_low',
        help='Positive lower bound threshold (default %f)' % thresh_pos_low, default=thresh_pos_low, type=float)
    parser.add_argument('--thresh_pos_high',
        help='Positive upper bound threshold (default %f)' % thresh_pos_high, default=thresh_pos_high, type=float)

    options = parser.parse_args()
    #print __doc__
    if options.input is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)

    map_filename = options.input
    thresh_size = options.thresh_size
    thresh_neg_low = options.thresh_neg_low
    thresh_neg_high = options.thresh_neg_high
    thresh_pos_low = options.thresh_pos_low
    thresh_pos_high = options.thresh_pos_high
    ##########################################################################
    #map_filename = "/tmp/beta_0.001_0.5_0.5_0.0_-1.0.nii_thresholded:0.003706/beta_0.001_0.5_0.5_0.0_-1.0.nii.gz"

    output, ext = os.path.splitext(map_filename)
    if ext == ".gz":
        output, _ = os.path.splitext(output)
    if not os.path.exists(output):
        os.mkdir(output)
    map_filename_symlink =  os.path.join(output, os.path.basename(map_filename))
    if not os.path.exists(map_filename_symlink):
        os.symlink(map_filename, map_filename_symlink)
    #print map_filename_symlink
    #sys.exit(0)
    output_csv_clusters_info_filename  = os.path.join(output, "clust_info.csv")
    output_clusters_labels_filename  = os.path.join(output, "clust_labels.nii.gz")
    output_clusters_values_filename  = os.path.join(output, "clust_values.nii.gz")
    output_clusters_small_mesh_filename  = os.path.join(output, "clust_small.gii")
    output_clusters_large_mesh_filename  = os.path.join(output, "clust_large.gii")
    output_clusters_mesh_filename  = os.path.join(output, "clust.gii")
    output_MNI152_T1_1mm_brain_filename  = os.path.join(output, os.path.basename(MNI152_T1_1mm_brain_filename))

    tempdir = tempfile.mkdtemp()

    writer = aims.Writer()
    ##########################################################################
    # Read volume
    ima = aims.read(map_filename)

    # force referential to MNI
    has_to_force_to_mni = False
    for i in xrange(len(ima.header()['referentials'])):
        if ima.header()['referentials'][i] != referential:
             ima.header()['referentials'][i] = referential
             has_to_force_to_mni = True
    if has_to_force_to_mni:
        writer.write(ima, map_filename)
        ima = aims.read(map_filename)
    trm_xyz_to_mm = ima_get_trm_xyz_to_mm(ima)
    arr = np.asarray(ima).squeeze()

    
    #MNI152_T1_1mm_brain.header()['referentials']
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
    # _clusters.nii.gz
    clusters_values_ima = aims.Volume(ima)
    clusters_values_ima_arr = np.asarray(clusters_values_ima).squeeze()
    clusters_values_ima_arr[clust_bool] = arr[clust_bool]
    writer = aims.Writer()
    writer.write(clusters_values_ima, output_clusters_values_filename)
    # _clusters_labels.nii.gz
    clusters_labels_ima = aims.Volume(ima)
    clusters_labels_ima_arr = np.asarray(clusters_labels_ima).squeeze()
    clusters_labels_ima_arr[:] = clust_labeled[:]
    writer.write(clusters_labels_ima, output_clusters_labels_filename)

    ##########################################################################
    # Get clusters information
    header_info, info = clusters_info(arr, clust_labeled, clust_sizes, labels,
                                     centers, trm_xyz_to_mm)
    df = pd.DataFrame(info, columns=header_info)
    df.to_csv(output_csv_clusters_info_filename, index=False)

    ##########################################################################
    # Mesh small clusters by spheres
    clusters_small_mesh = mesh_small_clusters(arr, clust_labeled, clust_sizes, labels,
        output_clusters_small_mesh_filename, centers, ima, thresh_size)

    ##########################################################################
    # Mesh large clusters
    clusters_large_mesh = mesh_large_clusters(arr, clust_labeled, clust_sizes, labels,
        output_clusters_large_mesh_filename, tempdir, ima, thresh_size)

    if clusters_small_mesh and clusters_large_mesh:
        aims.SurfaceManip.meshMerge(clusters_small_mesh, clusters_large_mesh)
        #print "TOTO", clusters_mesh, clusters_small_mesh, clusters_large_mesh
        clusters_mesh = clusters_small_mesh
    elif clusters_small_mesh:
        clusters_mesh = clusters_small_mesh
    elif clusters_large_mesh:
        clusters_mesh = clusters_large_mesh

    writer.write(clusters_mesh, output_clusters_mesh_filename)
    # warp  MNI152_T1_1mm into map referential
    os.system(fsl_warp_cmd % (MNI152_T1_1mm_brain_filename, map_filename, 
                              output_MNI152_T1_1mm_brain_filename))
    #print MNI152_T1_1mm_brain_filename, map_filename, output_MNI152_T1_1mm_brain_filename
    # Force same referential
    MNI152_T1_1mm_brain =  aims.read(output_MNI152_T1_1mm_brain_filename)
    MNI152_T1_1mm_brain.header()['referentials'] = ima.header()['referentials']
    MNI152_T1_1mm_brain.header()['transformations'] = ima.header()['transformations']
    writer.write(MNI152_T1_1mm_brain, output_MNI152_T1_1mm_brain_filename)

    print "Output directory:", output


"""
{'connected_components': '/tmp/beta_count_nonnull_5cv_0.001_0.3335_0.3335_0.333_-1.0.nii_thresholded:1.000000/connected_components.nii.gz',
  'mesh_file': '/tmp/beta_count_nonnull_5cv_0.001_0.3335_0.3335_0.333_-1.0.nii_thresholded:1.000000/clusters.mesh',
  'cluster_mask_file': '/tmp/beta_count_nonnull_5cv_0.001_0.3335_0.3335_0.333_-1.0.nii_thresholded:1.000000/clusters_mask.nii.gz',
  'cluster_file': '/tmp/beta_count_nonnull_5cv_0.001_0.3335_0.3335_0.333_-1.0.nii_thresholded:1.000000/clusters.nii.gz'}
/home/ed203246/.local/share/nsap/MNI152_T1_1mm_Bothhemi.gii

mesh_file = output_clusters_mesh_filename
white_mesh_file = "/neurospin/brainomics/neuroimaging_ressources/mesh/MNI152_T1_1mm_Bothhemi.gii"

# run render
do_mesh_cluster_rendering(mesh_file = outputs["mesh_file"],
                             texture_file = outputs["cluster_file"],
                             white_mesh_file = get_sample_data("mni_1mm").mesh,
                             anat_file = target)
"""