# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:29:35 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import argparse
import os.path
from glob import glob
import json
import numpy as np
import nibabel as ni


HABITAT = ['edema', 'enhancement', 'necrosis']
HABITAT_CODE = {'edema': 1, 'enhancement': 2, 'necrosis': 3}


def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg


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

    raise Exception('Cannot find a tissue {0} from {1}'.format(
                                                    label, roi.get_filename()))


doc = """
python $HOME/gits/scripts/2017_rr/metastasis/lesion_masher.py \
   --subjectdir /neurospin/radiomics/studies/metastasis/base/ \
   --out /tmp/mash
"""


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', metavar='PATH', required=True, type=is_dir,
                    help='outputdir.')
parser.add_argument('-s', '--subjectdir', metavar='PATH', type=is_dir,
                    required=True,
                    help='Base directory to read files from.')

#
args = parser.parse_args()
subject = os.path.basename(args.subjectdir)
subjectdir = args.subjectdir
fout = os.path.join(args.out, '{}_lesion.nii.gz'.format(subject))


lesions = glob(os.path.join(subjectdir, 'model10',
               '{}_model*_mask_lesion-*.nii.gz'.format(subject)))

cumul = np.zeros(ni.load(lesions[0]).get_shape()[:-1], dtype='int16')
for lesion in lesions:
    print lesion
    vois = ni.load(lesion)
    imgs = ni.four_to_three(vois)
    imgs_dict = {}
    for t, img3d in enumerate(imgs):
        # get the labels cont'd in the image should be 0 and [1 or 2 or 3]
        labels = np.unique(img3d.get_data())
        # exactly two labels mandatory
        if labels.shape[0] != 2:
            raise Exception('Cannot find a tissue from tissuetype.json.')
        label = max(labels)
        imgs_dict[label_to_shortname(label, vois)] = t
        print label,  " found in ", t
    for h in HABITAT:  # in this order
        if imgs_dict.has_key(h):
            cumul[imgs[imgs_dict[h]].get_data() != 0] = HABITAT_CODE[h]
ni.save(ni.Nifti1Image(cumul, affine=vois.affine), fout)
