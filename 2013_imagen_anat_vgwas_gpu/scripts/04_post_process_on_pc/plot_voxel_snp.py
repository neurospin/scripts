# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:01:25 2014

@author: jl237561
"""
import nibabel as ni
import numpy as np
import joblib
import os
import tables
import matplotlib.pyplot as plt
from numpy import array, ones, linalg
import random
import copy


def add_shakes(x, percent=0.2):
    cp_x = copy.copy(x)
    cp_x = cp_x.astype(float)
    for i in xrange(len(cp_x)):
        cp_x[i] = cp_x[i] + (random.random() - 0.5) * percent
    return cp_x


def get_linear_weights(x, y):
    '''
    Example
    -------
    x = arange(0,9)
    y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24]
    weights = get_linear_weights(x, y)
    predict_y = weights[0] * x + weights[1]
    '''
    A = array([x, ones(len(x))])
    # linearly generated sequence
    w = linalg.lstsq(A.T, y)[0]  # obtaining the parameters
    return w


def get_sub_array(seek_x, x_pts, y_pts):
    data = []
    for i in xrange(len(x_pts)):
        if x_pts[i] == seek_x:
            data.append(y_pts[i])
    data = np.asarray(data)
    return data


def t_test(x_pts, y_pts, x_group1, x_group2):
    '''
    Example
    -------
    x_pts = np.asarray([1,1,1,2,2,2,3,3,3])
    y_pts = np.asarray([1,2,5,1,2,5,1,2,5.0])
    print t_test(x_pts, y_pts, [1,2], [3])
    '''
    from scipy import stats
    data1 = None
    for each_x in x_group1:
        res_data = get_sub_array(each_x, x_pts, y_pts)
        if data1 is None:
            data1 = res_data
        else:
            data1 = np.concatenate([data1, res_data])
    data2 = None
    for each_x in x_group2:
        res_data = get_sub_array(each_x, x_pts, y_pts)
        if data2 is None:
            data2 = res_data
        else:
            data2 = np.concatenate([data2, res_data])
    return stats.ttest_ind(data1, data2)



# ax.broken_barh([ (110, 30), (150, 10) ] , (10, 9), facecolors='blue')
def boxplot_mean_std(ax, x_pts, y_pts):
    unique_x_pts = np.unique(x_pts)
    group_unique_x = {}
    group_mean_x = {}
    group_std_x = {}
    # Initilize as list and put data into each group
    for i in xrange(len(unique_x_pts)):
        unique_x_pt = unique_x_pts[i]
        group_unique_x[unique_x_pt] = []
        for j in xrange(len(x_pts)):
            if x_pts[j] == unique_x_pts[i]:
                group_unique_x[unique_x_pt].append(y_pts[j])
    for i in xrange(len(unique_x_pts)):
        unique_x_pt = unique_x_pts[i]
        group_mean_x[unique_x_pt] = np.mean(group_unique_x[unique_x_pt])
        group_std_x[unique_x_pt] = np.std(group_unique_x[unique_x_pt])
    plot_x = []
    plot_y = []
    plot_y_err = []
    for unique_x in group_unique_x:
        plot_x.append(unique_x)
        plot_y.append(group_mean_x[unique_x])
        plot_y_err.append(group_std_x[unique_x])
    plot_x = np.asarray(plot_x)
    plot_y = np.asarray(plot_y)
    plot_y_err = np.asarray(plot_y_err)
    # print "plot_x=", plot_x
    # print "plot_y=", plot_y
    # print "plot_y_err=", plot_y_err
    ax.errorbar(plot_x, plot_y, yerr=plot_y_err, marker="s", linestyle=" ")



#fig = plt.figure("for test")
#ax = fig.add_subplot(111)
#boxplot_mean_std(ax, [1,1,1,2,2,2,3,3,3], [1,1,2,5,1,2,3,5,6])
#plt.show()
#
#
#x = np.arange(0.1, 4, 0.5)
#y = np.exp(-x)
#yerr = 0.1 + 0.2 * np.sqrt(x)
#xerr = 0.1 + yerr
#fig = plt.figure("for test")
#plt.show()
#ax = fig.add_subplot(211)
#ax.errorbar(x, y, xerr=0.2, yerr=0.4, marker="s", linestyle=" ")
#ax = fig.add_subplot(212)
#ax.errorbar(x, y, xerr=0.2, yerr=0.4, marker="s", linestyle=" ")
#plt.show()


BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
PATH_RES_RESULT = os.path.join(BASE_DIR,
                               'data',
                               'tmp_reduce',
                               'result.joblib')
DATA_DIR = os.path.join(BASE_DIR, '2013_imagen_bmi', 'data')
MASK_FILE = os.path.join(DATA_DIR, 'mask', 'mask.nii')
mask_file = MASK_FILE
OUT_IMAGES = os.path.join(BASE_DIR,
                          'interesting_snp_brain_img')

OUT_DIR = os.path.join(BASE_DIR, 'data')
OUT_HDF5_FILE_FULRES_INTER = os.path.join(OUT_DIR,
                                            'cache_full_res_inter.hdf5')
OUT_SNP_NPZ = os.path.join(OUT_DIR, 'snp')
OUT_SNP_LIST_NPY = os.path.join(OUT_DIR, 'snp_list')

# Load image data
h5file = tables.openFile(OUT_HDF5_FILE_FULRES_INTER, mode="r")
images = h5file.getNode(h5file.root, 'images')

# Load geno data
geno_data = np.load(OUT_SNP_NPZ + ".npz")
geno_data = geno_data[geno_data.files[0]]

# Load snp list
snp_list_name = np.load(OUT_SNP_LIST_NPY + ".npz")
snp_list_name = snp_list_name[snp_list_name.files[0]]

ar = joblib.load(PATH_RES_RESULT)
h1 = ar['h1']
h0 = ar['h0']
snp_of_interest = [122664, 379105]
n_voxels = 336188

for snp in snp_of_interest:
    sp_h1_snp = h1[h1['x_id'] == snp]
    # compute average grep matter value
    voxels_snp = images[:, sp_h1_snp['y_id']]
    subjects_snp = geno_data[:, snp]
    print "======================================"
    print "The snp name is ", snp_list_name[snp]
    for snp_val in np.unique(subjects_snp):
        print "The number of '", snp_val, "'=", np.sum(subjects_snp == snp_val)
    # Plot new figure
    plot_x = subjects_snp
    shake_plot_x = add_shakes(plot_x)
    plot_y_avg = np.average(voxels_snp, axis=1)
    plot_y_max = np.max(voxels_snp, axis=1)
    # Linear regression using least square
    avg_weights = get_linear_weights(plot_x, plot_y_avg)
    max_weights = get_linear_weights(plot_x, plot_y_max)
    regression_line_x = np.sort(np.unique(plot_x))
    print "avg_weights =", avg_weights
    print "max_weights =", max_weights
    # Plot first subplot
    fig = plt.figure("%s (a point denotes a subject)" % snp_list_name[snp])
    ax = fig.add_subplot(211)
    _ = ax.set_title('%s (a point denotes a subject)' % snp_list_name[snp])
    _ = ax.set_xlabel('snp value')
    _ = ax.set_ylabel('average voxel value')
    _ = ax.scatter(shake_plot_x, plot_y_avg, s=30, facecolors='none', edgecolors='r', alpha=0.3)
    boxplot_mean_std(ax, plot_x, plot_y_avg)
    print "Using average values"
    print "T-test between snp(0) and snp(1,2) (t-statistic, p-value)"
    print t_test(plot_x, plot_y_avg, [0], [1, 2])
    # ax.plot(regression_line_x, regression_line_x * avg_weights[0] + avg_weights[1])
    # Plot second subplot
    ax = fig.add_subplot(212)
    _ = ax.set_title('%s (a point denotes a subject)' % snp_list_name[snp])
    _ = ax.set_xlabel('snp value')
    _ = ax.set_ylabel('max voxel value')
    _ = ax.scatter(shake_plot_x, plot_y_max, s=30, facecolors='none', edgecolors='r', alpha=0.3)
    boxplot_mean_std(ax, plot_x, plot_y_max)
    print "Using max values"
    print "T-test between snp(0) and snp(1,2) (t-statistic, p-value):"
    print t_test(plot_x, plot_y_max, [0], [1, 2])
    # ax.plot(regression_line_x, regression_line_x * max_weights[0] + max_weights[1])

plt.show()
