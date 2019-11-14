#!/usr/bin/env python
# -*- coding: utf-8 -*-

epilog = """
Analyse a map of values (possibly signed) and generate informations to vizualize
it. The map can be thresholded, then left out-values are clusterized into
components of connected voxels. For each cluster information
about the cluster is stored in a csv file.

Example
-------

./image_clusters_analysis_nilearn.py /tmp/weight_map.nii.gz -o /tmp/deptms_map --vmax 0.001 --thresh_norm_ratio 0.99 --thresh_size 10

"""

import os, sys, argparse, os.path
import numpy as np
import scipy, scipy.ndimage
import pandas as pd
from brainomics import array_utils
from numpy import linalg as LA
import nilearn
import nilearn.datasets
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting
import xml.etree.ElementTree as ET


def ijk_to_mni(affine, ijk):
    M = affine[:3, :3]
    abc = affine[:3, 3]
    return M.dot(ijk.T).T + abc


def clusters_info(map_arr, map_clustlabels_arr, clust_sizes, labels, centers,
                 affine_ijk2mm, atlas1_arr, atlas1_labels, atlas2_arr, atlas2_labels, atlas):
    """Compute Information about cluster: label, size, mean, max, min,
    min_coord_mni, max_coord_mni, center_coord_mni, regions_involved"""
    centers = np.asarray(centers)
    clusters_info = list()

    for i in range(len(labels)):
        # i = 0
        print(labels[i])
        mask = map_clustlabels_arr == labels[i]
        center_ijk = centers[i]
        center_mni = np.round(np.asarray(ijk_to_mni(affine_ijk2mm, center_ijk)),
                             3)
        atlas2_clust = atlas2_arr[mask]
        atlas1_clust = atlas1_arr[mask]

        # Cluster overlapp with atlases ROIs
        def dict_sort_by_value_to_str(d):
            s = sorted(d.items(), key=lambda x:x[1], reverse=True)
            return " ".join(["%s:%.0f" % (x[0], x[1]) for x in s])

        regions_atlas1 = {}
        regions_atlas2 = {}
        regions_atlas1_weight = {}
        regions_atlas2_weight = {}
        # atlas2 (subcortical atlas)
        for atlas_lab in np.unique(atlas2_clust):
            #atlas_lab = 16
            lab_cluster = map_arr[np.logical_and(mask, atlas2_arr == atlas_lab)]
            ROI_name = atlas2_labels[atlas_lab]
            regions_atlas2[ROI_name] = int(np.round(
                                      np.sum(atlas2_clust == atlas_lab) \
                                        / float(clust_sizes[i]) * 100))
            regions_atlas2_weight[ROI_name] = int(np.round(LA.norm(lab_cluster) \
                                                / float(LA.norm(map_arr[mask])) * 100))
        # atlas21 (cortical atlas)
        for atlas_lab in np.unique(atlas1_clust):
            lab_cluster = map_arr[np.logical_and(mask, atlas1_arr == atlas_lab)]
            ROI_name = atlas1_labels[atlas_lab]
            regions_atlas1[ROI_name] = int(np.round(
                                      np.sum(atlas1_clust == atlas_lab) / \
                                        float(clust_sizes[i]) * 100))
            regions_atlas1_weight[ROI_name] = int(np.round(LA.norm(lab_cluster) \
                                                / float(LA.norm(map_arr[mask])) * 100))

        regions_atlas1 = dict_sort_by_value_to_str(regions_atlas1)
        regions_atlas2 = dict_sort_by_value_to_str(regions_atlas2)
        regions_atlas1_weight = dict_sort_by_value_to_str(regions_atlas1_weight)
        regions_atlas2_weight = dict_sort_by_value_to_str(regions_atlas2_weight)

        # Min, Max
        max_val, max_idx = np.max(map_arr[mask]), np.argmax(map_arr[mask])
        max_ijk = np.array(np.where((map_arr == max_val) & mask)).T
        max_mni = np.round(np.asarray(ijk_to_mni(affine_ijk2mm, max_ijk)), 3)
        min_val, min_idx = np.min(map_arr[mask]), np.argmin(map_arr[mask])
        min_ijk = np.array(np.where((map_arr == min_val) & mask)).T
        min_mni = np.round(np.asarray(ijk_to_mni(affine_ijk2mm, min_ijk)), 3)
        mean_val = np.mean(map_arr[mask])

        # Proportion of weight: ||cluster||^2 / ||map||^2
        prop_norm2_weight = np.sum(map_arr[mask] ** 2) / np.sum(map_arr ** 2)

        # negative peak (cortical and subcortical atlases)
        if min_val >= 0:
            region_cort_peak_neg, region_sub_peak_neg = None, None
        else:
            atlas_lab = atlas1_clust[min_idx]
            region_cort_peak_neg = atlas1_labels[atlas_lab]
            atlas_lab = atlas2_clust[min_idx]
            region_sub_peak_neg = atlas2_labels[atlas_lab]
        # positive peak (cortical and subcortical atlases)
        if max_val <= 0:
            region_cort_peak_pos, region_sub_peak_pos = None, None
        else:
            atlas_lab = atlas1_clust[max_idx]
            region_cort_peak_pos = atlas1_labels[atlas_lab]
            atlas_lab = atlas2_clust[max_idx]
            region_sub_peak_pos = atlas2_labels[atlas_lab]

        info = [labels[i], clust_sizes[i], prop_norm2_weight] + \
        max_mni[0, :].tolist() + min_mni[0, :].tolist() + center_mni.tolist() + \
        [mean_val, max_val, min_val,
         regions_atlas1, regions_atlas1_weight, region_cort_peak_pos,
         region_cort_peak_neg, regions_atlas2, regions_atlas2_weight,
         region_sub_peak_pos, region_sub_peak_neg]
        clusters_info.append(info)

    header_info = ["label", "size", "prop_norm2_weight",
              "x_max_mni",  "y_max_mni",  "z_max_mni",
              "x_min_mni",  "y_min_mni",  "z_min_mni",
              "x_center_mni",  "y_center_mni", "z_center_mni",
              "mean", "max", "min",
              "ROIs_{sn}-{a1}_prop (%)".format(**atlas), "ROIs_{sn}-{a1}_weight_prop (%)".format(**atlas),
              "ROI_{sn}-{a1}_peak_pos".format(**atlas), "ROI_{sn}-{a1}_peak_neg".format(**atlas),
              "ROIs_{sn}-{a2}_prop (%)".format(**atlas), "ROIs_{sn}-{a2}_weight_prop (%)".format(**atlas),
              "ROI_{sn}-{a2}_peak_pos".format(**atlas), "ROI_{sn}-{a2}_peak_neg".format(**atlas)
              ]

    df = pd.DataFrame(clusters_info, columns=header_info)
    df = df.sort_values(by="prop_norm2_weight", ascending=False)
    return df

if __name__ == "__main__":
    # Set default values to parameters
    thresh_size = 1
    thresh_neg_low = -np.inf
    thresh_neg_high = 0
    thresh_pos_low = 0
    thresh_pos_high = np.inf
    thresh_norm_ratio = 1.
    vmax = 0.001
    # MNI152_T1_1mm_brain_filename = "/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz"

    atlas =  dict(name="harvard_oxford", sn="HO" , a1="cort", a2="sub") # atlas name, short-name, atlas 1 and 2 names
    #atlas_cort_filename = '/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz'
    #atlas_sub_filename = '/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz'

    # parse command line options
    #parser = optparse.OptionParser(description=__doc__)
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('input', help='Input map volume', type=str)
    parser.add_argument('-o', '--output',
        help='Output directory', type=str)
    parser.add_argument('--thresh_size',
        help='Threshold, in voxels nb, between small and large clusters'
        '(default %i)' % thresh_size, default=thresh_size, type=float)
    parser.add_argument('-t', '--thresh_norm_ratio',
        help='Threshold image such ||v[|v| >= t]||2 / ||v||2 == thresh_norm_ratio (default %f)'% thresh_norm_ratio,
        default=thresh_norm_ratio, type=float)
    parser.add_argument('--vmax',
        help='Upper bound for plotting, passed to matplotlib.pyplot.imshow (default %f)'% vmax,
        default=vmax, type=float)
    parser.add_argument('--thresh_neg_low',
        help='Negative lower bound threshold (default %f)' % thresh_neg_low, default=thresh_neg_low, type=float)
    parser.add_argument('--thresh_neg_high',
        help='Negative upper bound threshold (default %f)' % thresh_neg_high, default=thresh_neg_high, type=float)
    parser.add_argument('--thresh_pos_low',
        help='Positive lower bound threshold (default %f)' % thresh_pos_low, default=thresh_pos_low, type=float)
    parser.add_argument('--thresh_pos_high',
        help='Positive upper bound threshold (default %f)' % thresh_pos_high, default=thresh_pos_high, type=float)
    parser.add_argument('--atlas',
        help='Atlas in harvard_oxford (CM cortical and subcortical) JHU for (dti WM and track) (default %s) ' % atlas["name"],
        default=atlas, type=str)

    # To debug: manually set command line
    # argv_ = ['/neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000350_1.000_0.001/components-brain-maps_PC0000.nii.gz']
    # options = parser.parse_args(argv_)

    options = parser.parse_args()

    #print __doc__
    if options.input is None:
        #print("Error: Input is missing.")
        parser.print_help()
        raise SystemExit("Error: Input is missing.")

    map_filename = options.input

    ##########################################################################
    # Read volume
    import nibabel as nib
    map_img = nib.load(map_filename)

    #trm_ijk_to_mni = ima_get_trm_ijk_to_mni(map_img)
    map_arr = map_img.get_data()
    if len(map_arr.shape) > 3:
        print("input image is more than 3D split them first using")
        print('fsl5.0-fslsplit %s ./%s -t' % (map_filename, "prefix"))
        sys.exit(0)

    map_filename = options.input
    thresh_size = options.thresh_size
    thresh_norm_ratio = options.thresh_norm_ratio
    vmax = options.vmax
    thresh_neg_low = options.thresh_neg_low
    thresh_neg_high = options.thresh_neg_high
    thresh_pos_low = options.thresh_pos_low
    thresh_pos_high = options.thresh_pos_high

    # Fetch atlases
    if options.atlas == "harvard_oxford":
        atlas1 = nilearn.datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm", data_dir=None, symmetric_split=False, resume=True, verbose=1)
        atlas1_labels = atlas1.labels
        atlas1_filename = atlas1.maps
        atlas2 = nilearn.datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm", data_dir=None, symmetric_split=False, resume=True, verbose=1)
        atlas2_labels = atlas2.labels
        atlas2_filename = atlas2.maps
        atlas = dict(name="harvard_oxford", sn="HO", a1="cort", a2="sub")  # atlas name, short-name, atlas 1 and 2 names
        # FIX bug nilearn.datasets.fetch_atlas_harvard_oxford: Errors in HarvardOxford.tgz / sub-maxprob-thr0-1mm
        # atlas2.maps = os.path.join('/usr/share/data/harvard-oxford-atlases/HarvardOxford', os.path.basename(atlas2.maps))
    elif options.atlas == 'JHU':
        atlas1_filename = "/usr/share/data/jhu-dti-whitematter-atlas/JHU/JHU-ICBM-labels-1mm.nii.gz"
        atlas2_filename = "/usr/share/data/jhu-dti-whitematter-atlas/JHU/JHU-ICBM-tracts-maxprob-thr0-1mm.nii.gz"
        # Read xml, get label and name and make sure order is ok
        tree = ET.parse("/usr/share/data/jhu-dti-whitematter-atlas/JHU-labels.xml")
        atlas1_labels = [str(name) for name in pd.DataFrame([[int(item.attrib['index']), item.text] for item in tree.getroot()[1]], columns=["lab", "name"]).sort_values("lab")["name"]]
        tree = ET.parse("/usr/share/data/jhu-dti-whitematter-atlas/JHU-tracts.xml")
        atlas2_labels = [str(name) for name in pd.DataFrame([[int(item.attrib['index']), item.text] for item in tree.getroot()[1]], columns=["lab", "name"]).sort_values("lab")["name"]]
        # In JHU-tracts.xml index="0" => Anterior thalamic radiation L but in image atlsas it is 1 add item 0 as being 'Unclassified'
        atlas2_labels += ['Unclassified']
        atlas = dict(name="JHU", sn="JHU", a1="wm-icbm-dti-81", a2="wm-tracts")  # atlas name, short-name, atlas 1 and 2 names

    atlas1_img = nilearn.image.resample_to_img(source_img=atlas1_filename, target_img=map_filename, interpolation='nearest', copy=True, order='F')
    atlas1_arr  = atlas1_img.get_data().astype(int)
    assert len(np.unique(atlas1_arr)) == len(atlas1_labels), "Atlas %s : array labels must match labels table" %  options.atlas

    atlas2_img = nilearn.image.resample_to_img(source_img=atlas2_filename, target_img=map_filename, interpolation='nearest', copy=True, order='F')
    atlas2_arr = atlas2_img.get_data().astype(int)
    assert len(np.unique(atlas2_arr)) == len(atlas2_labels), "Atlas %s : array labels must match labels table" %  options.atlas

    assert np.all((map_img.affine == atlas2_img.affine) & (map_img.affine == atlas1_img.affine))
    ##########################################################################
    map_basename, ext = os.path.splitext(map_filename)
    if ext == ".gz":
        map_basename, _ = os.path.splitext(map_basename)
    map_basename = os.path.basename(map_basename)

    output = os.getcwd()
    if options.output:
        output = options.output
        try:
            os.makedirs(output)
        except FileExistsError:
            pass
    output = os.path.join(output, map_basename)

    output_figure_filename  = output + "_clust_info.pdf"
    output_csv_clusters_info_filename  = output + "_clust_info.csv"
    output_clusters_labels_filename  = output + "_clust_labels.nii.gz"
    output_clusters_values_filename  = output + "_clust_values.nii.gz"

    print("Outputs:", output, output_figure_filename)
    ##########################################################################
    # Find clusters (connected component above a given threshold)
    print(thresh_neg_low, thresh_neg_high, thresh_pos_low, thresh_pos_high)
    if thresh_norm_ratio < 1:
        map_arr, thres = array_utils.arr_threshold_from_norm2_ratio(map_arr, thresh_norm_ratio)
        print("Threshold image as %f" % thres)
    clust_bool = np.zeros(map_arr.shape, dtype=bool)
    clust_bool[((map_arr > thresh_neg_low) & (map_arr < thresh_neg_high)) |
               ((map_arr > thresh_pos_low) & (map_arr < thresh_pos_high))] = True
    map_arr[np.logical_not(clust_bool)] = False
    map_clustlabels_arr, n_clusts = scipy.ndimage.label(clust_bool)
    clust_sizes = scipy.ndimage.measurements.histogram(map_clustlabels_arr, 1,
                                                       n_clusts, n_clusts)
    labels = np.unique(map_clustlabels_arr)[1:]
    centers = scipy.ndimage.center_of_mass(clust_bool, map_clustlabels_arr, labels)

    # _clusters.nii.gz
    img = nib.Nifti1Image(map_arr, map_img.get_affine())
    img.to_filename(output_clusters_values_filename)

    # _clusters_labels.nii.gz
    img = nib.Nifti1Image(map_clustlabels_arr, map_img.get_affine())
    img.to_filename(output_clusters_labels_filename)

    ##########################################################################
    print("Get clusters information")
    affine_ijk2mm = map_img.affine
    df =  clusters_info(map_arr, map_clustlabels_arr, clust_sizes, labels, centers,
                  affine_ijk2mm,
                  atlas1_arr, atlas1_labels,
                  atlas2_arr, atlas2_labels,
                  atlas)


    df.to_csv(output_csv_clusters_info_filename, index=False)

    ##########################################################################
    print("Plot clusters")
    def print_info(ax, row):
        j, nrow = 0, 12
        ax.text(0, 1-j/nrow, '%s: %s, %s: %s, |clust|/|map|: %i%%' % ('label', row['label'], 'size', row['size'], row['prop_norm2_weight'] * 100), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "max[%i,%i,%i]: %.2E" % (row['x_max_mni'], row['y_max_mni'], row['z_max_mni'], row['max']), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "min[%i,%i,%i]: %.2E" % (row['x_min_mni'], row['y_min_mni'], row['z_min_mni'], row['min']), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "center[%i,%i,%i], mean: %.2E" % (row['x_center_mni'], row['y_center_mni'], row['z_center_mni'], row['mean']), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROI_{sn}-{a1}_peak_pos".format(**atlas), str(row["ROI_{sn}-{a1}_peak_pos".format(**atlas)])), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROI_{sn}-{a1}_peak_neg".format(**atlas), str(row["ROI_{sn}-{a1}_peak_neg".format(**atlas)])), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROIs_{sn}-{a1}_prop (%)".format(**atlas), str(row["ROIs_{sn}-{a1}_prop (%)".format(**atlas)])), fontsize=6); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROIs_{sn}-{a1}_weight_prop (%)".format(**atlas), str(row["ROIs_{sn}-{a1}_weight_prop (%)".format(**atlas)])), fontsize=6); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROI_{sn}-{a2}_peak_pos".format(**atlas), str(row["ROI_{sn}-{a2}_peak_pos".format(**atlas)])), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROI_{sn}-{a2}_peak_neg".format(**atlas), str(row["ROI_{sn}-{a2}_peak_neg".format(**atlas)])), fontsize=8); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROIs_{sn}-{a2}_prop (%)".format(**atlas), str(row["ROIs_{sn}-{a2}_prop (%)".format(**atlas)])), fontsize=6); j += 1
        ax.text(0, 1-j/nrow, "%s: %s" % ("ROIs_{sn}-{a2}_weight_prop (%)".format(**atlas), str(row["ROIs_{sn}-{a2}_weight_prop (%)".format(**atlas)])), fontsize=6); j += 1
        ax2.axis('off')

    pdf = PdfPages(output_figure_filename)

    # Glass brain
    fig = plt.figure(figsize=(11.69, 8.27))
    plotting.plot_glass_brain(output_clusters_values_filename,
                                      colorbar=True, plot_abs=False,
                                      cmap=plt.cm.bwr,
                                      threshold = max(thresh_pos_low, abs(thresh_neg_high)),
                                      vmax=abs(vmax), vmin =-abs(vmax))
    pdf.savefig()
    plt.close(fig)

    for i, row in df.iterrows():
        #row = df.ix[3, :]
        if row['size'] < thresh_size:
            continue
        print(row['size'])
        # A cluster have at least a negative pek or a positive one, but it
        # can have both
        # positive peak
        if row["ROI_{sn}-{a1}_peak_pos".format(**atlas)]:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(211)
            title = "%i %s/%s" % (row["label"], row["ROI_{sn}-{a1}_peak_pos".format(**atlas)], row["ROI_{sn}-{a2}_peak_pos".format(**atlas)])
            cut_coords = row[['x_max_mni', 'y_max_mni', 'z_max_mni']]
            m = plotting.plot_stat_map(output_clusters_values_filename, display_mode='ortho', vmax=vmax,
                                   cmap=plt.cm.bwr,
                                   threshold=thresh_pos_low,
                                   cut_coords=cut_coords, figure=fig,axes=ax,#(0, 0, 100, 100),
                                   title=title)

            ax2 = fig.add_subplot(212)
            print_info(ax2, row)
            pdf.savefig()
            plt.close(fig)

        # Negative peak
        if row["ROI_{sn}-{a1}_peak_neg".format(**atlas)]:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(211)
            title = "%i %s/%s" % (row["label"], row["ROI_{sn}-{a1}_peak_neg".format(**atlas)], row["ROI_{sn}-{a2}_peak_neg".format(**atlas)])
            cut_coords = row[['x_min_mni', 'y_min_mni', 'z_min_mni']]
            m = plotting.plot_stat_map(output_clusters_values_filename, display_mode='ortho', vmax=vmax,
                                   cmap=plt.cm.bwr, threshold=abs(thresh_neg_high),
                                   cut_coords=cut_coords, figure=fig,axes=ax,#(0, 0, 100, 100),
                                   title=title)

            ax2 = fig.add_subplot(212)
            print_info(ax2, row)
            pdf.savefig()
            plt.close(fig)

    pdf.close()

"""
Debug:
F5
sys.argv = "image_clusters_analysis_nilearn.py /tmp/weight_map.nii.gz --thresh_norm_ratio 0.99".split()
"""
