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
SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "share")
sys.path.append(SRC_PATH)
import utils_proj_classif
BASE_PATH = "/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL"
INPUT_MASK = os.path.join(BASE_PATH, "mask.nii.gz")
keys = [
"0.001_0.9_0.0_0.1_-1.0",
"0.001_0.45_0.45_0.1_-1.0",
"0.001_0.5_0.5_0.0_-1.0",
"0.001_1.0_0.0_0.0_-1.0"]
penalty_start = 2

mask_image = nibabel.load(INPUT_MASK)
mask = mask_image.get_data() != 0
outfilenames = list()
for key in keys:
    #key = keys[0]
    beta = np.load(os.path.join(BASE_PATH, "results/0", key, "beta.npz"))['arr_0']
    arr = np.zeros(mask.shape)
    arr[mask] = beta[penalty_start:].ravel()
    out_im = nibabel.Nifti1Image(arr,affine=mask_image.get_affine())
    outfilename = os.path.join(BASE_PATH, "results/0", key, "beta_%s.nii.gz" % key)
    out_im.to_filename(outfilename)
    outfilenames.append(outfilename)

outfilenames
 ['/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/results/0/0.001_0.9_0.0_0.1_-1.0/beta_0.001_0.9_0.0_0.1_-1.0.nii.gz',
 '/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/results/0/0.001_0.45_0.45_0.1_-1.0/beta_0.001_0.45_0.45_0.1_-1.0.nii.gz',
 '/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/results/0/0.001_0.5_0.5_0.0_-1.0/beta_0.001_0.5_0.5_0.0_-1.0.nii.gz',
 '/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/results/0/0.001_1.0_0.0_0.0_-1.0/beta_0.001_1.0_0.0_0.0_-1.0.nii.gz']
 
#############################################################################
## beta => image
#############################################################################

OUTPUT_DIR = "/tmp/mesh"
# set image of wheights and taget volume


INPUT_IMAGE = os.path.join(BASE_PATH, "tv/split_vizu/1-0.1-0.1-0.8_beta.nii")

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
