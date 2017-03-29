# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:16:46 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
#!/usr/bin/env python

import argparse
import sys
import os.path
import traceback
import subprocess
import nibabel as ni
import numpy as np
import shutil
import tempfile
import json
import yaml
from glob import glob
import re


# Script documentation
doc = """
Bias field correction
~~~~~~~~~~~~~~~~~~~~~

This code uses http://dx.doi.org/10.1016/j.nicl.2014.08.008
Shinohara RT, Sweeney EM, Goldsmith J, et al. Statistical normalization 
techniques for magnetic resonance imaging. 
NeuroImage : Clinical. 2014;6:9-19. doi:10.1016/j.nicl.2014.08.008.

It uses R and R packages (fslr and WhiteStripe) so far to pave the way to the
obtention of the radiomics features.

Steps:


Command:
python $HOME/gits/scripts/2017_rr/metastasis/RadiomicFeaturesExtraction.py \
    -i /neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_WS.nii.gz \
    -l /neurospin/radiomics/studies/metastasis/base/187962757123/model10/187962757123_model10_mask_lesion-1.nii.gz \
    -d /neurospin/radiomics/studies/metastasis/base/187962757123/model50
python $HOME/gits/scripts/2017_rr/metastasis/RadiomicFeaturesExtraction.py \
    -i /neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_WS.nii.gz \
    -l /neurospin/radiomics/studies/metastasis/base/187962757123/model10/187962757123_model10_mask_edema_1.nii.gz \
    -d /neurospin/radiomics/studies/metastasis/base/187962757123/model50
"""


def get_tissue_meta(roi):
    """ Return a dict read from a jsonfile up in the tree that harbor the
        given roi filename

    Parameters
    ----------
    roi: nifti image (mandatory)
        A nifti image in which contains the sampling ROI.

    Returns
    -------
    retval: dict
        Read from the 'tissuetype.json' found on the filesystem.

    """
    fn = roi.get_filename()
    while fn is not '/':
        fn = os.path.dirname(fn)
        expected_name = os.path.join(fn, 'tissuetype.json')
        if os.path.exists(expected_name):
            with open(expected_name) as fp:
                retval = json.load(fp)
                return retval

    raise Exception('Cannot find a tissuetype.json for metadata completion!')


def label_to_shortname(label, roi):
    """ Return a string by looking up label from the tissuetype.json 
        inferred by the get_tissue_roi macro
    """
    dtissue = get_tissue_meta(roi)
    for t in dtissue:
        if dtissue[t]['Index'] == label:
            return t
            
    raise Exception('Cannot find a tissue {0} from {1}'.format(label, roi.get_filename()))

def edit_header_name(jsonout, paramfile):
    params = yaml.load(open(paramfile).read())
    if 'inputImage' in params:
        if 'Wavelet' in params['inputImage']:
            if 'wavelet' in params['inputImage']['Wavelet']:
                replace_val = params['inputImage']['Wavelet']['wavelet']
                with open(jsonout) as fp:
                    json_string = fp.read()
                json_string = json_string.replace('wavelet', replace_val)
                with open(jsonout, 'w') as fp:
                    fp.write(json_string)


def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
#    if glob(os.path.join(dirarg, '*')) != []:
#        raise argparse.ArgumentError(
#            "The dir '{0}' is not empty!".format(dirarg))
    return dirarg

def is_image(image):
    template = '/neurospin/radiomics/studies/metastasis/base'
    image = os.path.abspath(image)
    if not image.find(template) > -1:
        raise argparse.ArgumentError(
            "The path '{0}' is not correct (should be in {1}) !".format(image, template))
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', metavar='FILE',
                    type=is_image,
                    help='Image to correct for bias field')
parser.add_argument('--lesion', '-l', metavar='FILE',
                    help='Lesion(4D-Image) file')
parser.add_argument('--outdir', '-d', metavar='PATH',
                    type=is_dir,
                    help='Output directory to create the file in.')
parser.add_argument('--param', '-p', metavar='FILE', nargs=1, type=str, default=None,
                    help='Parameter file containing the settings to be used in extraction')


"""

"""
def main():
    args = parser.parse_args()
#    args = parser.parse_args([
#    '-i', '/neurospin/radiomics/studies/metastasis/base/187962757123/model03/187962757123_enh-gado_T1w_bfc_WS.nii.gz',
#    '-l', '/neurospin/radiomics/studies/metastasis/base/187962757123/model10/187962757123_model10_mask_lesion-1.nii.gz',
#    '-d', '/neurospin/radiomics/studies/metastasis/base/187962757123/model50'])


    # Initialize Logging if any


    # Initialize extractor
    try:
        if args.param is not None:
            raise Exception('Reading from param file not implemented yet.')
        # 
        image = args.image
        lesion = args.lesion
        outdir = args.outdir 
        template = '/neurospin/radiomics/studies/metastasis/base'
        subject = os.path.split(os.path.abspath(image))[1].split("_")[0]
        paramfile = os.path.join(os.getenv('HOME'), 'gits',
                                 'scripts', '2017_rr', 'metastasis',
                                 'basicWavelet.yaml')

        # get an tmp dir
        tmpdir = tempfile.mkdtemp()
        log_level = 'DEBUG'
        log_file = os.path.join(tmpdir, 'pyradiomicsbasicWavelet.log')
        prefix_outfile = os.path.join(
                            tmpdir, 
                            '{0}_enh-gado_T1w_bfc_WS_rad-'.format(subject))

        lesion_name=re.search(r"lesion-[0-9][0-9]?", lesion) 
        print("### Radiomics for "+lesion_name.group(0)+" ###")
        lesion_nb=re.search(r"[0-9]+", lesion_name.group(0))
        # run tissue type by tissue type
        vois = ni.load(lesion)
        cumul = np.zeros(vois.get_shape()[:-1], dtype='int16')
        imgs = ni.four_to_three(vois)
        for t, img3d in enumerate(imgs):
            cumul += img3d.get_data()
            labels = np.unique(img3d.get_data())
            if labels.shape[0] == 2:
                sname = label_to_shortname(np.sort(labels)[1], vois)
                fmask = prefix_outfile + 'ttype-{0}{1}.nii.gz'.format(sname, lesion_nb.group(0))
                jsonout = prefix_outfile + 'ttype-{0}{1}.json'.format(sname, lesion_nb.group(0))
                bin_img3d = np.asarray((img3d.get_data() > 0) * 1, dtype='uint16')
                ni.save(ni.Nifti1Image(bin_img3d, affine=vois.get_affine()), fmask)
                #
                cmd = ['pyradiomics',
                       '--logging-level', log_level,
                       '--log-file', log_file,
                       '--param', paramfile,
                       '--out', jsonout, '--format', 'json',
                       image, fmask]
                print " ".join(cmd)
                results = subprocess.check_call(cmd)
                # try to edit generated json file
                edit_header_name(jsonout, paramfile)
            else:
                raise Exception('Cannot find a tissue from tissuetype.json.')
        # run for the whole lesion
        cumul = np.asarray((cumul > 0) * 1, dtype='uint16')
        fmask = prefix_outfile + 'ttype-{0}.nii.gz'.format(lesion_name.group(0))
        jsonout = prefix_outfile + 'ttype-{0}.json'.format(lesion_name.group(0))
        ni.save(ni.Nifti1Image(cumul, affine=vois.get_affine()), fmask)        
        #
        cmd = ['pyradiomics',
               '--logging-level', log_level,
               '--log-file', log_file,
               '--param', paramfile,
               '--out', jsonout, '--format', 'json',
               image, fmask]
        print " ".join(cmd)
        results = subprocess.check_call(cmd)
        # try to edit generated json file
        edit_header_name(jsonout, paramfile)
        print(jsonout+" done")
        
        #move
        flist = glob(os.path.join(tmpdir,'*ttype*json')) + \
                glob(os.path.join(tmpdir,'*log'))
        for f in flist:
            shutil.move(f, outdir)
        #House keeping
        shutil.rmtree(tmpdir)


    except Exception:
        print 'Radiomic feature extraction FAILED:\n%s', traceback.format_exc()


if __name__ == "__main__":
    main()

