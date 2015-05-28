#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:50:15 2015

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause


http://brainvisa.info/doc/pyaims-4.4/sphinx/pyaims_tutorial.html#types-conversion
"""
import argparse
import numpy as np
from soma import aims
from soma import aimsalgo

"""
Do the same job than fsl5.0-applywarp

cd /tmp/
mkdir test
cd test
cp /usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz .
cp /neurospin/brainomics/2013_adni/MCIc-CTL_csi/mask.nii .
fsl5.0-applywarp -i HarvardOxford-cort-maxprob-thr0-1mm.nii.gz -r mask.nii -o atlas_in_ref_fsl.nii.gz --interp=nn

src_filename = "HarvardOxford-cort-maxprob-thr0-1mm.nii.gz"
ref_filename = "mask.nii"
out_filename = "atlas_in_ref_bv_fsl.nii.gz"
interp = "nn"

"""
def get_transformation(src, ref):
    src3mni = aims.AffineTransformation3d(src.header()['transformations'][0])
    ref2mni = aims.AffineTransformation3d(ref.header()['transformations'][0])
    src2ref = ref2mni.inverse() * src3mni
    return src2ref


def resample(src, ref):
    interp_dict = {"nn": 0, "lin": 1, "quad": 2}
    conv = aims.Converter(intype=src, outtype=aims.Volume('FLOAT'))
    src = conv(src)
    conv = aims.Converter(intype=ref, outtype=aims.Volume('FLOAT'))
    ref = conv(ref)

    # resampler
    resp = aims.ResamplerFactory_FLOAT().getResampler(interp_dict[interp])
    resp.setRef(src)  # volume to resample
    resp.setDefaultValue(0)  # set background to 0

    # resample
    voxel_size = np.array(ref.header()['voxel_size'])    
    src2ref_trm = get_transformation(src, ref)    
    output_ima = resp.doit(src2ref_trm, ref.getSizeX(), ref.getSizeY(), ref.getSizeZ(), voxel_size)    
    output_ima.header()['referentials'] = ref.header()['referentials']
    output_ima.header()['transformations'] = ref.header()['transformations']
    return output_ima

if __name__ == "__main__":
    interp = "nn"
    # Set default values to parameters
    # parse command line options
    #parser = optparse.OptionParser(description=__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input image to resample', type=str)
    parser.add_argument('--ref', help='Reference volume', type=str)
    parser.add_argument('--interp', help='Interpolation method nn, lin, quad', type=str)
    parser.add_argument('--output', help='Ouptut', type=str)

    options = parser.parse_args()
    src_filename = options.input
    ref_filename = options.ref
    src = aims.read(src_filename)
    ref = aims.read(ref_filename)
    output_ima = resample(src, ref)
    assert output_ima.maximum() == src.arraydata().max()

    writer = aims.Writer()
    writer.write(output_ima, options.output)