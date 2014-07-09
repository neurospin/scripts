# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:31:53 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import optparse, os.path
import nibabel, numpy as np

# import modules
from nsap.use_cases.utils.brainvisa_map_cluster_analysis import *
from nsap.plugins.nipype import set_environment
#
SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif_AD-CTL")
sys.path.append(SRC_PATH)
import utils_proj_classif


BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif_AD-CTL"
INPUT_MASK = os.path.join(BASE_PATH,"SPM", "template_FinalQC_CTL_AD", "mask.nii")
OUTPUT_DIR = "/tmp/mesh"
# set image of wheights and taget volume
mask_im = nibabel.load(INPUT_MASK)
mask = mask_im.get_data() != 0


#INPUT_IMAGE = os.path.join(BASE_PATH, "tv/split_vizu/1-0.1-0.1-0.8_beta.nii")

#############################################################################
## READ INPUT IMAGE
#############################################################################
parser = optparse.OptionParser(description=__doc__)
parser.add_option('--input',
    help='Inpput image', type=str)

options, args = parser.parse_args(sys.argv)

INPUT_IMAGE = options.input
beta3d = nibabel.load(INPUT_IMAGE).get_data()
print "Beta stat min, max, mean", beta3d[mask].min(), beta3d[mask].max(), beta3d[mask].mean()


#############################################################################
## Threshold at 99%
#############################################################################
beta = beta3d[mask]
THRESH = utils_proj_classif.get_threshold_from_norm2_ratio(beta, ratio = .99)
beta_thres = np.zeros(beta.shape)
beta_thres[(np.sqrt(beta**2))>THRESH] = beta[(np.sqrt(beta**2))>THRESH]
print "THRESH:", THRESH, ", Ratio:", np.sqrt(np.sum(beta_thres ** 2)) / np.sqrt(np.sum(beta ** 2))


arr = np.zeros(mask.shape)
arr[mask] = beta_thres
im_out = nibabel.Nifti1Image(arr, affine=mask_im.get_affine())#, header=mask_im.get_header().copy())
OUTPUT_DIR = os.path.splitext(INPUT_IMAGE)[0]+"_thresholded:%f" %THRESH
if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
im_out.to_filename(os.path.join(OUTPUT_DIR, os.path.basename(INPUT_IMAGE)))

#############################################################################
## MEsh
#############################################################################
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

"""
bv_env python 05_visu_weigths_maps.py --input=/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/split_vizu/1-1.0-0.0-0.0_beta.nii
bv_env python 05_visu_weigths_maps.py --input=/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/split_vizu/1-0.1-0.0-0.9_beta.nii
bv_env python 05_visu_weigths_maps.py --input=/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/split_vizu/1-0.9-0.1-0.0_beta.nii

bv_env python 05_visu_weigths_maps.py --input=/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/split_vizu/1-0.0-0.1-0.9_beta.nii
bv_env python 05_visu_weigths_maps.py --input=/neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv/split_vizu/1-0.1-0.1-0.8_beta.nii

"""
