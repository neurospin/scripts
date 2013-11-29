# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:11:47 2013

@author: md238665

The mask voxel size is [1.5, 1.5, 1.5] so we need to resample TD at this resolution.
We use nearest interpolation to be sure to not introduce foreign values.
We also set the stomrage to memory orientation in order to have voxel-wise aligned images in memory.
We also resample the MNI for reference.

Aims python's binding doesn't contain a resampler for U8 images so we convert to S16 and convert back.

This script replaces resample_TD_lobe (bash version) which didn't set the storage to memory properly.

"""
import os
from soma import aims, aimsalgo # aimsalgo is need for resampling

BASE_DIR='/neurospin/brainomics/2013_imagen_bmi/'

# Input file names
ATLAS_FILE='/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/TD_lobe.nii'
MASK_FILE=os.path.join(BASE_DIR, 'data/mask.nii')
MNI_FILE='/volatile/brainvisa/source/brainvisa-share/trunk/anatomical_templates/MNI152_T1_2mm.nii.gz'

# Output file names
OUT_DIR=os.path.join(BASE_DIR, 'data/mask_without_cerebellum')
RESAMPLED_ATLAS_FILE=os.path.join(OUT_DIR, 'TD_lobe_1.5mm.nii')
RESAMPLED_MNI_FILE=os.path.join(OUT_DIR, 'MNI152_T1_1.5mm.nii')

if ~os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Converters
U8_2_S16_converter = aims.Converter(aims.Volume_U8, aims.Volume_S16)
S16_2_U8_converter = aims.Converter(aims.Volume_S16, aims.Volume_U8)

# Nearest-neighbor resampler
resp = aims.ResamplerFactory_S16().getResampler(0)
resp.setDefaultValue(-1) # set background to -1
I = aims.AffineTransformation3d()

# Open mask and get parameters from it
mask = aims.read(MASK_FILE)
mask_header = mask.header()
VOLUME_SIZE=mask_header['volume_dimension']
VOXEL_SIZE=mask_header['voxel_size']
STOM = mask_header['storage_to_memory']

# Open atlas file & convert it
atlas = aims.read(ATLAS_FILE)
atlas_S16 = U8_2_S16_converter(atlas)

# Resampling
resp.setRef(atlas_S16) # volume to resample
resampled_atlas_S16 = resp.doit(I, VOLUME_SIZE[0], VOLUME_SIZE[1], VOLUME_SIZE[2], VOXEL_SIZE)

# Write resampled image
resampled_atlas_U8 = S16_2_U8_converter(resampled_atlas_S16)
resampled_atlas_U8.header()['storage_to_memory'] = STOM
aims.write(resampled_atlas_U8, RESAMPLED_ATLAS_FILE)

# Open MNI file (already in S16)
MNI = aims.read(MNI_FILE)
# Resampling
resp.setRef(MNI) # volume to resample
resampled_MNI = resp.doit(I, VOLUME_SIZE[0], VOLUME_SIZE[1], VOLUME_SIZE[2], VOXEL_SIZE)

# Write resampled image
resampled_MNI.header()['storage_to_memory'] = STOM
aims.write(resampled_MNI, RESAMPLED_MNI_FILE)
