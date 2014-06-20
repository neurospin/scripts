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


BASE_PATH =  '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
DATA_DIR = os.path.join(BASE_PATH, '2013_imagen_bmi', 'data')
MASK_FILE = os.path.join(BASE_PATH, '2013_imagen_bmi', 'data', 'mask', 'mask.nii')
#INPUT_IMAGE = os.path.join(BASE_PATH, "interesting_snp_brain_img", "snp_379105_perm_pcorr.nii.gz")
OUTPUT_DIR = "/tmp/mesh"
# set image of wheights and taget volume
mask_im = nibabel.load(MASK_FILE)
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



#############################################################################
## MEsh
#############################################################################
set_environment(set_matlab=False, set_brainvisa=True)
target = get_sample_data("mni_1mm").brain

outputs = {}
outputs.update( do_brainvisa_mesh_cluster(OUTPUT_DIR, INPUT_IMAGE))
#                                              thresh_neg_bound=(-np.inf,-0),
#                                              thresh_pos_bound=(0, np.inf)) )
                                              
# run render
do_mesh_cluster_rendering(mesh_file = outputs["mesh_file"],
                             texture_file = outputs["cluster_file"],
                             white_mesh_file = get_sample_data("mni_1mm").mesh,
                             anat_file = target)

"""
bv_env python weigths_maps.py --input=/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/interesting_snp_brain_img/snp_379105_perm_pcorr.nii.gz
bv_env python weigths_maps.py --input=/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/interesting_snp_brain_img/snp_122664_perm_pcorr.nii.gz

"""
