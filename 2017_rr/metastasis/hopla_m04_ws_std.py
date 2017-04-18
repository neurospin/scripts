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
python hopla_m04_ws_std.py \
-b /neurospin/radiomics/studies/metastasis/base \
-d /tmp/WSres \
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


args = parser.parse_args()
#args = parser.parse_args([
#            '-b', '/neurospin/radiomics/studies/metastasis/base',
#            '-d', '/tmp/WSres/',
#            '-p'])

dirent = glob(os.path.join(args.basedir, "*"))
dirent =[i for i in dirent if os.path.basename(i).isdigit()]
subjects = [os.path.basename(i) for i in dirent]
rootdir = args.destdir

infiles = [os.path.join(args.basedir, '{}'.format(i),
                        'model02',
                        '{}_enh-gado_T1w_bfc.nii.gz'.format(i))
           for i in subjects]
pvefiles = [os.path.join(args.basedir, '{}'.format(i),
                         'model03',
                         '{}_enh-gado_T1w_bfc_betmask_pve_2.nii.gz'.format(i))
            for i in subjects]
maskfiles = [os.path.join(os.path.dirname(i.replace('model02', 'model03')),
                          'native_hatbox.nii.gz')
             for i in infiles]
destfiles = [os.path.join(rootdir, '{}'.format(i), 'model04')
             for i in subjects]
for d in destfiles:
    if not os.path.exists(d):
        os.makedirs(d)

print "infiles (", len(infiles), ") [", infiles[0], ",...,", infiles[-1]
print "destfiles (", len(destfiles), ") [", destfiles[0], ",...,", destfiles[-1]
print "maskfiles (", len(maskfiles), ") [", maskfiles[0], ",...,", maskfiles[-1]
print "pvefiles (", len(pvefiles), ") [", pvefiles[0], ",...,", pvefiles[-1]


if args.process:
    logfile = "{}/WSlog/WSlog.txt".format(args.destdir)
    if not os.path.isdir(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))
    #
    cmd = os.path.join(os.getenv('HOME'), 'gits',
                       'scripts', '2017_rr', 'metastasis',
                       'm04_ws_std.py')

    status, exitcodes = hopla(cmd,
                              i=infiles,
                              d=destfiles,
                              m=maskfiles,
                              p=pvefiles,
                              hopla_iterative_kwargs=["i", "d", "m", "p"],
                              hopla_cpus=3,
                              hopla_logfile=logfile,
                              hopla_verbose=args.verbose)
