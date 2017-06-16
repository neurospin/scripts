#!/usr/bin/env python
import argparse
import sys
import os.path
import subprocess
import nibabel as ni
import traceback
import shutil
import json
import tempfile
from glob import glob
from nilearn import plotting
import numpy as np
from clindmri.registration.fsl import flirt

MNI_BRAIN = '/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
HABITAT = ['edema', 'enhancement', 'both']

def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg


doc = """
python $HOME/gits/scripts/2017_rr/metastasis/m90_niche.py \
   --base /neurospin/radiomics/studies/metastasis/base/ \
   --out /tmp/niche \
   --model model10 \
   --habitat edema
"""
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out', metavar='PATH', required=True, type=is_dir,
                    help='outputdir.')
parser.add_argument('-b', '--base', metavar='PATH', type=is_dir, required=True,
                    help='Base directory to read files from.')
parser.add_argument('-m', '--model', metavar='string', required=True,
                    help='model entry (eg. model10).')
parser.add_argument('--habitat', '-a', choices=HABITAT,
                    default='both', help='Habitat to study')



def regular(args):
    # Initialize extractor
    if glob(os.path.join(args.outdir, '*')) != []:
        raise Exception("The dir '{0}' is not empty!".format(args.outdir))

    try:

        # get an tmp dir change dir
        tmpdir = tempfile.mkdtemp()
        prevdir = os.getcwd()
        os.chdir(tmpdir)

        # move selected files
        flist = np.unique(glob('*'))
        for f in flist:
            shutil.move(f, OutDirPath)

    except Exception:
        print 'm90_niche analysis FAILED:\n%s', traceback.format_exc()

    # final housekeeping
    os.chdir(prevdir)
    shutil.rmtree(tmpdir)


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



#
# if __name__ == "__main__": #################################################
# 

#args = parser.parse_args()
args = parser.parse_args([
            '-b', '/neurospin/radiomics/studies/metastasis/base',
            '-o', '/tmp/niche',
            '-m', 'model10',
            '-a', 'edema'
             ])
# basic anchor dir
dirent = glob(os.path.join(args.base, "*"))
dirent =[i for i in dirent if os.path.basename(i).isdigit()]
subjects = [os.path.basename(i) for i in dirent]
outdir = args.out
model = args.model
tag = args.habitat

# forge path to lesions and xform (are in model03)
res_dict = {}
path_dict = {}
for s in subjects:
    path_dict[s] = {}
    res_dict[s] = {}
    # get the putative lesion files
    path_dict[s]['xfm'] = glob(os.path.join(args.base, s, 'model03',
                                            'nat2mni'))
    path_dict[s]['lesions'] = glob(os.path.join(args.base, s, model,
                                    '{}_model*_mask_lesion-*.nii.gz'.format(s)
                                             ))
# get the AAL in 2mm MNI
aal_name = '/volatile/newfrouin/17_Study_Localisation/aal/atlas/AAL2.nii'
aal = ni.load(aal_name)
aal_label = np.unique(aal.get_data())[1:].tolist()
aal_data = aal.get_data()
aal_dict = {}
for l in aal_label:
    aal_dict[l] = (aal_data == l)
for s in path_dict:
    for l in aal_label:
        res_dict[s][l] = 0

try:

    # get an tmp dir change dir
    tmpdir = tempfile.mkdtemp()
    prevdir = os.getcwd()
    os.chdir(tmpdir)

    # get in cumul array the data corresponding to the lesions(s)
    for s in path_dict:
        print s
        cumul = np.zeros(ni.load(path_dict[s]['lesions'][0]).get_shape()[:-1],
                         dtype='int16')
        for lesion in path_dict[s]['lesions']:
            print lesion
            vois = ni.load(lesion)
            imgs = ni.four_to_three(vois)
            for t, img3d in enumerate(imgs):
                # get the labels cont'd in the image should be 0 and [1 or 2 or 3]
                labels = np.unique(img3d.get_data())
                # exactly two labels mandatory
                if labels.shape[0] != 2:
                    raise Exception('Cannot find a tissue from tissuetype.json.')
                label = max(labels)
                sname = label_to_shortname(label, vois)
                if sname not in HABITAT:
                    continue
                if tag == 'both':
                    cumul += img3d.get_data()
                elif tag == sname:  # should be edema or enhancement
                    cumul = img3d.get_data()

            # resample lesion images in MNI 1mm to form a dict indexed by subject
            tmp_lesion_name = os.path.basename(lesion)
            res_lesion_name = 'res_{}'.format(tmp_lesion_name)
            cumul[cumul != 0] = 1
            ni.save(ni.Nifti1Image(cumul, affine=vois.affine), tmp_lesion_name)
            flirt(in_file=tmp_lesion_name,
                  ref_file=aal_name,
                  applyxfm=True, 
                  init=path_dict[s]['xfm'][0],
                  out=res_lesion_name,
                  interp='nearestneighbour')
            resamp_lesion = ni.load(res_lesion_name).get_data()
            # crawl the AAL labels to count lesions occ to form a dict indexed by l
            for l in aal_label:  # acummulate for the cases with multiples lesions
                res_dict[s][l] += np.sum(resamp_lesion[aal_dict[l]])
                
    # save as a json for manipulation in R
    with open("stat_localisation.json", 'w') as fpout:
        json.dump(res_dict, fpout, indent=4)
        fpout.write('\n')

    # move selected files
    flist = ["stat_localisation.json"]
    for f in flist:
        shutil.move(f, outdir)

    # finale housekeeping
    os.chdir(prevdir)
    shutil.rmtree(tmpdir)

except Exception:
    print 'm90_niche analysis FAILED:\n%s', traceback.format_exc()
