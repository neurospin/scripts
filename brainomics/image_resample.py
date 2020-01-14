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
def down_sample(src_img, factor=2):
    """

    Parameters
    ----------
    src_img: nii image
    factor: subsmapling factor (default 2)

    Returns
    -------
    nii subsample image

    Examples
    --------
    >>> import nilearn.datasets
    >>> import brainomics.image_resample
    >>> mni159_img = nilearn.datasets.load_mni152_template()

    >>> resamp_img = brainomics.image_resample.down_sample(src_img=mni159_img, factor=2)

    >>> print("SRC:", mni159_img.header.get_zooms(), mni159_img.get_data().shape)
    >>> print("DST:", resamp_img.header.get_zooms(), resamp_img.get_data().shape)

    >>> mni159_img.to_filename("/tmp/mni152_%imm.nii.gz" % mni159_img.header.get_zooms()[0])
    >>> resamp_img.to_filename("/tmp/mni152_%imm.nii.gz" % resamp_img.header.get_zooms()[0])
    """
    from nilearn.image import resample_img
    target_affine = np.copy(src_img.affine)[:3, :][:, :3]
    target_affine[:3, :3] *= factor
    return resample_img(src_img, target_affine=target_affine)


def aims_get_transformation(src, ref):
    from soma import aims
    src3mni = aims.AffineTransformation3d(src.header()['transformations'][0])
    ref2mni = aims.AffineTransformation3d(ref.header()['transformations'][0])
    src2ref = ref2mni.inverse() * src3mni
    return src2ref


def aims_resample(src, ref):
    from soma import aims
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

def aims__main__():
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