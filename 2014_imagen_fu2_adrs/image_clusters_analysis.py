# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:28:17 2014

@author: cp243490
"""

import os, sys, argparse
import numpy as np
import scipy, scipy.ndimage
from soma import aims
import tempfile
import pandas as pd
from brainomics import array_utils
from numpy import linalg as LA


def xyz_to_mm(trm, vox_size):
    trmcp = aims.Motion(trm)
    trmcp.scale(aims.Point3d(vox_size), aims.Point3d(1, 1, 1))
    return trmcp


def harvardOxford_atlases_infos():
    import xml.etree.ElementTree as ET
    xml_cort_filename = '/usr/share/data/harvard-oxford-atlases/HarvardOxford-Cortical.xml'
    xml_sub_filename = '/usr/share/data/harvard-oxford-atlases/HarvardOxford-Subcortical.xml'
    tree_cort, tree_sub = ET.parse(xml_cort_filename), ET.parse(xml_sub_filename)
    root_cort, root_sub = tree_cort.getroot(), tree_sub.getroot()
    dict_label_cort, dict_label_sub = {}, {}

    for label in root_cort.iter('label'):
        index_lab = label.attrib['index']
        dict_label_cort[index_lab] = label.text

    for label in root_sub.iter('label'):
        index_lab = label.attrib['index']
        dict_label_sub[index_lab] = label.text

    return dict_label_cort, dict_label_sub

# Transformation operation
def ima_get_trm_xyz_to_mm(ima, refNb=0):
    """
    Return the transformation XYZ=>mm of a volume
    """
    trm_ima_mm2ref_mm = aims.Motion(ima.header()['transformations'][refNb])
    ima_voxel_size = ima.header()['voxel_size'][:3]
    return(xyz_to_mm(trm_ima_mm2ref_mm, ima_voxel_size))


def clusters_info(arr, clust_labeled, clust_sizes, labels, centers,
                 trm_xyz_to_mm, atlas_sub, atlas_cort):
    """Compute Information about cluster: label, size, mean, max, min,
    min_coord_mm, max_coord_mm, center_coord_mm, regions_involved"""
    clusters_info = list()
    #arr_abs = np.abs(arr)

    # get harvard_oxford atalses (sub and cort) infos
    ima_atlas_cort = aims.read(atlas_cort_filename)
    arr_atlas_cort = np.asarray(ima_atlas_cort).squeeze()
    ima_atlas_sub = aims.read(atlas_sub_filename)
    arr_atlas_sub = np.asarray(ima_atlas_sub).squeeze()
    dict_label_cort, dict_label_sub = harvardOxford_atlases_infos()
    print "Scan clusters:"
    for i in xrange(len(labels)):
        print i
        mask = clust_labeled == labels[i]
        center_xyz = centers[i]
        center_mm = np.round(np.asarray(trm_xyz_to_mm.transform(center_xyz)),
                             3)

#        atlas_sub_clust = np.copy(arr_atlas_sub)
#        atlas_sub_clust[np.logical_not(mask)]=0
#        atlas_cort_clust = np.copy(arr_atlas_cort)
#        atlas_cort_clust[np.logical_not(mask)]=0
        atlas_sub_clust = arr_atlas_sub[mask]
        atlas_cort_clust = arr_atlas_cort[mask]
        labels_sub_ROI = np.unique(atlas_sub_clust)
        labels_cort_ROI = np.unique(atlas_cort_clust)
        regions_cort_atlas = {}
        regions_sub_atlas = {}
        regions_cort_atlas_weight = {}
        regions_sub_atlas_weight = {}
        # subcortical atlas
        for lab_sub in labels_sub_ROI:
            lab_cluster = arr[np.logical_and(mask, arr_atlas_sub == lab_sub)]
            if lab_sub == 0:
                ROI_name = "outside"
            elif lab_sub > 0:
                ROI_name = dict_label_sub[str(int(lab_sub - 1))]
            regions_sub_atlas[ROI_name] = int(np.round(
                                      np.sum(atlas_sub_clust == lab_sub) \
                                        / float(clust_sizes[i]) * 100))
            regions_sub_atlas_weight[ROI_name] = int(np.round(LA.norm(lab_cluster) \
                                                / float(LA.norm(arr[mask])) * 100))
        # cortical atlas
        for lab_cort in labels_cort_ROI:
            lab_cluster = arr[np.logical_and(mask, arr_atlas_cort == lab_cort)]
            if lab_cort == 0:
                ROI_name = "outside"
            elif lab_cort > 0:
                ROI_name = dict_label_cort[str(int(lab_cort - 1))]
            regions_cort_atlas[ROI_name] = int(np.round(
                                      np.sum(atlas_cort_clust == lab_cort) / \
                                        float(clust_sizes[i]) * 100))                                
            regions_cort_atlas_weight[ROI_name] = int(np.round(LA.norm(lab_cluster) \
                                                / float(LA.norm(arr[mask])) * 100))

        if clust_sizes[i] > 1:
            max_val, max_ind = np.max((arr[mask])), np.argmax(arr[mask])
            # max_xyz = np.asarray([c[0] for c in
            #   np.where((arr_abs == max_val) & mask)])
            max_xyz = np.asarray([c[0] for c in
                np.where((arr == max_val) & mask)])
            max_mm = np.round(np.asarray(trm_xyz_to_mm.transform(max_xyz)), 3)
            min_val, min_ind = np.min(arr[mask]), np.argmin(arr[mask])
            # min_xyz = np.asarray([c[0] for c in
            #    np.where((arr_abs == min_val) & mask)])
            min_xyz = np.asarray([c[0] for c in
                np.where((arr == min_val) & mask)])
            min_mm = np.round(np.asarray(trm_xyz_to_mm.transform(min_xyz)), 3)
            mean_val = np.mean(arr[mask])

        else:
            max_val = min_val = mean_val = arr[mask][0]
            max_ind = min_ind = 0
            max_mm = center_mm
            min_mm = center_mm

        prop_norm2_weight = LA.norm(arr[mask]) / float(LA.norm(arr))

        # negative peak (cortical and subcortical atlases)
        if min_val >= 0:
            region_cort_peak_neg, region_sub_peak_neg = None, None
        else:
            lab_cort = atlas_cort_clust.ravel()[min_ind]
            if lab_cort == 0:
                region_cort_peak_neg = "outside"
            else:
                region_cort_peak_neg = dict_label_cort[str(int(lab_cort - 1))]
            lab_sub = atlas_sub_clust.ravel()[min_ind]
            if lab_sub == 0:
                region_sub_peak_neg = "outside"
            else:
                region_sub_peak_neg = dict_label_sub[str(int(lab_sub - 1))]
        # positive peak (cortical and subcortical atlases)
        if max_val <= 0:
            region_cort_peak_pos, region_sub_peak_pos = None, None
        else:
            lab_cort = atlas_cort_clust.ravel()[max_ind]
            if lab_cort == 0:
                region_cort_peak_pos = "outside"
            else:
                region_cort_peak_pos = dict_label_cort[str(int(lab_cort - 1))]
            lab_sub = atlas_sub_clust.ravel()[max_ind]
            if lab_sub == 0:
                region_sub_peak_pos = "outside"
            else:
                region_sub_peak_pos = dict_label_sub[str(int(lab_sub - 1))]

        info = [labels[i], clust_sizes[i], prop_norm2_weight] + \
        max_mm.tolist() + min_mm.tolist() + center_mm.tolist() + \
        [mean_val, max_val, min_val,
         regions_cort_atlas, regions_cort_atlas_weight, region_cort_peak_pos,
         region_cort_peak_neg, regions_sub_atlas, regions_sub_atlas_weight,
         region_sub_peak_pos, region_sub_peak_neg]
        clusters_info.append(info)
        header = ["label", "size", "prop_norm2_weight",
                  "x_max_mm",  "y_max_mm",  "z_max_mm",
                  "x_min_mm",  "y_min_mm",  "z_min_mm",
                  "x_center_mm",  "y_center_mm", "z_center_mm",
                  "mean", "max", "min", "Regions_cort_prop (%)",
                  "Regions_cort_weight_prop (%)", "Region_cort_peak_pos",
                  "Region_cort_peak_neg", "Regions_sub_prop (%)",
                  "Regions_sub_weight_prop (%)",
                  "Region_sub_peak_pos", "Region_sub_peak_neg"
                  ]
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
    print "\n============="
    print cmd
    print "============="
    os.popen(cmd)
    graph = aims.read(large_clust_graph_filename)
    print "-----"
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
    thresh_norm_ratio = 1.
    referential = 'Talairach-MNI template-SPM'
    fsl_warp_cmd = "fsl5.0-applywarp -i %s -r %s -o %s"
    MNI152_T1_1mm_brain_filename = "/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz"
    
    atlas_cort_filename = "/neurospin/brainomics/2014_imagen_fu2_adrs/ADRS_dataset/atlas/HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz"
    atlas_sub_filename = "/neurospin/brainomics/2014_imagen_fu2_adrs/ADRS_dataset/atlas/HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz"

    # parse command line options
    #parser = optparse.OptionParser(description=__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input map volume', type=str)
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
    parser.add_argument('-t', '--thresh_norm_ratio',
        help='Threshold image such ||v[|v| >= t]||2 / ||v||2 == thresh_norm_ratio (default %f)'% thresh_norm_ratio,
        default=thresh_norm_ratio, type=float)

    options = parser.parse_args()
#    options.input = "/neurospin/brainomics/2014_deptms/results_enettv/MRI_Maskdep/analysis_results/fold0_0.05_0.7_0.0_0.3_-1.0/beta.nii.gz"
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
    thresh_norm_ratio = options.thresh_norm_ratio

#    map_filename = "/neurospin/brainomics/2014_deptms/results_enettv/MRI_Maskdep/analysis_results/fold0_0.05_0.7_0.0_0.3_-1.0/beta.nii.gz"
#    map_filename = "/neurospin/brainomics/2014_deptms/results_univariate/MRI/t_stat_rep_min_norep_MRI_brain.nii.gz"
#    thresh_neg_high = -3
#    thresh_pos_low = 3
    ##########################################################################
    
    output, ext = os.path.splitext(map_filename)
    if ext == ".gz":
        output, _ = os.path.splitext(output)
    if not os.path.exists(output):
        os.mkdir(output)
#    map_filename_symlink =  os.path.join(output, os.path.basename(map_filename))
#    if not os.path.exists(map_filename_symlink):
#        print map_filename, map_filename_symlink
#        os.symlink(map_filename, map_filename_symlink)
    #print map_filename_symlink
    #sys.exit(0)
    output_csv_clusters_info_filename = os.path.join(output, "clust_info.csv")
    output_clusters_labels_filename = os.path.join(output, "clust_labels.nii.gz")
    output_clusters_values_filename = os.path.join(output, "clust_values.nii.gz")
    output_clusters_small_mesh_filename = os.path.join(output, "clust_small.gii")
    output_clusters_large_mesh_filename = os.path.join(output, "clust_large.gii")
    output_clusters_mesh_filename = os.path.join(output, "clust.gii")
    output_MNI152_T1_1mm_brain_filename = os.path.join(output, os.path.basename(MNI152_T1_1mm_brain_filename))

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
    if len(arr.shape) > 3:
        print "input image is more than 3D split them first using"
        print 'fsl5.0-fslsplit %s ./%s -t' % (map_filename, output)
        sys.exit(0)
    #MNI152_T1_1mm_brain.header()['referentials']
    ##########################################################################
    # Find clusters (connected component abov a given threshold)
    print thresh_neg_low, thresh_neg_high, thresh_pos_low, thresh_pos_high
    if thresh_norm_ratio < 1:
        arr = array_utils.arr_threshold_from_norm2_ratio(arr, .99)[0]
    clust_bool = np.zeros(arr.shape, dtype=bool)
    #((arr > thresh_neg_low) & (arr < thresh_neg_high) | (arr > thresh_pos_low) & (arr < thresh_pos_high)).sum()
    clust_bool[((arr > thresh_neg_low) & (arr < thresh_neg_high)) |
               ((arr > thresh_pos_low) & (arr < thresh_pos_high))] = True
    arr[np.logical_not(clust_bool)] = False
    clust_labeled, n_clusts = scipy.ndimage.label(clust_bool)
    clust_sizes = scipy.ndimage.measurements.histogram(clust_labeled, 1,
                                                       n_clusts, n_clusts)
    labels = np.unique(clust_labeled)[1:]
    centers = scipy.ndimage.center_of_mass(clust_bool, clust_labeled, labels)
    # _clusters.nii.gz
    clusters_values_ima = aims.Volume(ima)
    clusters_values_ima_arr = np.asarray(clusters_values_ima).squeeze()
    #clusters_values_ima_arr[clust_bool] = arr[clust_bool]
    clusters_values_ima_arr[::] = arr[::]
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
                                     centers, trm_xyz_to_mm, atlas_cort_filename, atlas_sub_filename)
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