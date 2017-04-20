#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
from nilearn import plotting
import nibabel
import argparse
import os
from clindmri.registration.fsl import flirt
from glob import glob

from hopla.converter import hopla

# Parameters to keep trace
#__hopla__ = ["tool", "version"]

doc = """
python hopla_m50_radiomic_feat_extraction.py \
-b /neurospin/radiomics/studies/metastasis/base \
-d /neurospin/radiomics/studies/metastasis/base \
-c 2 \
-p
"""


# Parsing
def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg

parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
    help="increase the verbosity level: 0 silent, [1, 2] verbose.")
parser.add_argument(
    "-b", "--basedir", dest="basedir", required=True, metavar="PATH",
    help="the subject main directory to parse files",
    type=is_dir)
parser.add_argument(
    "-d", "--destdir", dest="destdir", required=True, metavar="PATH",
    help="An existing destination dir")
parser.add_argument(
    "-p", "--process", dest="process", action='store_true',
    help="if activated execute.")
parser.add_argument(
    "-c", "--cpus", dest="cpus", type=int, default=2,
    help="# number of concurrent tasks run: default is 2.")


args = parser.parse_args()
#args = parser.parse_args([
#            '-b', '/neurospin/radiomics/studies/metastasis/base',
#            '-d', '/tmp/WSres/'
#             ])


#"""
#python $HOME/gits/scripts/2017_rr/metastasis/getGCLM.py
#--param $HOME/gits/scripts/2017_rr/metastasis/minimal.yaml
#--out /tmp/GLCM4
#--format json
#--habitat edema
#/neurospin/radiomics/studies/metastasis/base/187962757123/model04/187962757123_enh-gado_T1w_bfc_WS.nii.gz
#/neurospin/radiomics/studies/metastasis/base/187962757123/model10/187962757123_model10_mask_lesion-1.nii.gz
#"""

dirent = glob(os.path.join(args.basedir, "*"))
dirent =[i for i in dirent if os.path.basename(i).isdigit()]
subjects = [os.path.basename(i) for i in dirent]
rootdir = args.destdir

PARAM_FILE = '/neurospin/radiomics/studies/metastasis/base/all_feat_1.yaml'

p_dict = {}
for s in subjects:
    p_dict[s]={}
    p_dict[s]['image'] = os.path.join(args.basedir, s, 'model4',
                                    '{}_enh-gado_T1w_bfc_WS.nii.gz'.format(s))
    # get the putative lesion files
    p_dict[s]['lesions'] = glob(os.path.join(args.basedir, s, 'model10',
                                    '{}_model10_mask_lesion-*.nii.gz'.format(s)))
imafiles = []
maskfiles = []
destdirs = []
for s in p_dict:
#    print s, p_dict[s]['lesions']
    for m in p_dict[s]['lesions']:
        destdirs.append(os.path.join(args.destdir, '{}'.format(s), 'model52'))
        imafiles.append(p_dict[s]['image'])
        maskfiles.append(m)

#print "imafiles (", len(imafiles), ") [", imafiles[0], ",...,", imafiles[-1]
#print "destdirs (", len(destdirs), ") [", destdirs[0], ",...,", destdirs[-1]
#print "maskfiles (", len(maskfiles), ") [", maskfiles[0], ",...,", maskfiles[-1]
#print imafiles, destdirs, maskfiles

if args.process:
    logfile = "{}/log/RadFeaExtlog.txt".format(args.destdir)
    if not os.path.isdir(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))
    #
    cmd = os.path.join(os.getenv('HOME'), 'gits',
                       'scripts', '2017_rr', 'metastasis',
                       'm50_radiomic_feat_extraction.py')

    status, exitcodes = hopla(cmd,
                              '--param', PARAM_FILE,
                              '--format', 'json',
                              '--habitat', 'edema',
                              '--out', destdirs,
                              '--image', imafiles,
                              '--segment', maskfiles,
                              hopla_iterative_kwargs=["i", "s", "-o"],
                              hopla_cpus=args.cpu,
                              hopla_logfile=logfile,
                              hopla_verbose=args.verbose)
