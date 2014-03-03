# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:31:53 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import nibabel, numpy as np

# import modules
from nsap.use_cases.utils.brainvisa_map_cluster_analysis import *
from nsap.plugins.nipype import set_environment
#

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif"
INPUT_MASK = os.path.join(BASE_PATH,"SPM", "template_FinalQC_CTL_AD", "mask.nii")
OUTPUT_DIR = "/tmp/mesh"
# set image of wheights and taget volume
INPUT_IMAGE = os.path.join(BASE_PATH, "tv/split/0.1-0.0-0.05-0.95/beta.nii")


mask_im = nibabel.load(INPUT_MASK)
mask = mask_im.get_data() != 0

beta3d = nibabel.load(INPUT_IMAGE).get_data()
print "Beta stat min, max, mean", beta3d[mask].min(), beta3d[mask].max(), beta3d[mask].mean()

THRESH = 5e-4
#############################################################################

# init local environ for nipype
set_environment(set_matlab=False, set_brainvisa=True)
target = get_sample_data("mni_1mm").brain

outputs = {}
outputs.update( do_brainvisa_mesh_cluster(OUTPUT_DIR,
                                          INPUT_IMAGE,
#                                              outputs["register_map_image"],
                                              thresh_neg_bound=(-np.inf,-THRESH),
                                              thresh_pos_bound=(THRESH, np.inf)) )
                                              
# run render
do_mesh_cluster_rendering(mesh_file = outputs["mesh_file"],
                             texture_file = outputs["cluster_file"],
                             white_mesh_file = get_sample_data("mni_1mm").mesh,
                             anat_file = target)

#bv_env python 05_visu_weigths_maps.py
### Si déjà dans le MNI