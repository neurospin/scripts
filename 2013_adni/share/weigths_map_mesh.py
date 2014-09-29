#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:31:53 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import argparse, os
import nibabel, numpy as np

from nsap.use_cases.utils.brainvisa_map_cluster_analysis import *
from nsap.plugins.nipype import set_environment


#def get_threshold_from_norm2_ratio(v, ratio=.99):
#    """Threshold to apply an input_vector such
#    norm2(output_vector) / norm2(input_vector) == ratio
#    return the thresholded vector and the threshold"""
#    #shape = v.shape
#    import numpy as np
#    v = v.copy().ravel()
#    v2 = (v ** 2)
#    v2.sort()
#    v2 = v2[::-1]
#    v_n2 = np.sqrt(np.sum(v2))
#    #(v_n2 * ratio) ** 2
#    cumsum2 = np.cumsum(v2)  #np.sqrt(np.cumsum(v2))
#    select = cumsum2 <= (v_n2 * ratio) ** 2
#    thres = np.sqrt(v2[select][-1])
#    return thres

#############################################################################
## Read argument
#############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-k', help='Input image')
options = parser.parse_args()


if not options.input:
    print 'Input image required'
    sys.exit(1)


INPUT_IMAGE = options.input
beta_image = nibabel.load(INPUT_IMAGE)
beta3d = beta_image.get_data()
mask = beta3d != 0

print "Beta stat min, max, mean", beta3d[mask].min(), beta3d[mask].max(), beta3d[mask].mean()


#############################################################################
## Threshold at 99%
#############################################################################
beta = beta3d[mask]
from brainomics import array_utils
beta_thres, thres = array_utils.arr_threshold_from_norm2_ratio(beta, ratio = .99)
#THRESH = get_threshold_from_norm2_ratio(beta, ratio = .99)
#beta_thres = np.zeros(beta.shape)
#beta_thres[(np.sqrt(beta**2))>THRESH] = beta[(np.sqrt(beta**2))>THRESH]
print "THRESH:", thres, ", Ratio:", np.sqrt(np.sum(beta_thres ** 2)) / np.sqrt(np.sum(beta ** 2))


arr = np.zeros(mask.shape)
arr[mask] = beta_thres
im_out = nibabel.Nifti1Image(arr, affine=beta_image
.get_affine())#, header=mask_image.get_header().copy())
OUTPUT_DIR = os.path.splitext(INPUT_IMAGE)[0]+"_thresholded:%f" %thres
if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
im_out.to_filename(os.path.join(OUTPUT_DIR, os.path.basename(INPUT_IMAGE)))

#############################################################################
## Mesh
#############################################################################
set_environment(set_matlab=False, set_brainvisa=True)
target = get_sample_data("mni_1mm").brain

outputs = {}
outputs.update(do_brainvisa_mesh_cluster(OUTPUT_DIR,
                                          INPUT_IMAGE,
                                          #thresh_size=0,
#                                              outputs["register_map_image"],
                                              thresh_neg_bound=(-np.inf,-THRESH),
                                              thresh_pos_bound=(THRESH, np.inf)) )

print outputs
# run render
do_mesh_cluster_rendering(mesh_file = outputs["mesh_file"],
                             texture_file = outputs["cluster_file"],
                             white_mesh_file = get_sample_data("mni_1mm").mesh,
                             anat_file = target)

"""
bv_env weigths_map_mesh.py --input /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/split_vizu/1-1.0-0.0-0.0_beta.nii
"""
