#! /usr/bin/env python
from nilearn import plotting
import nibabel as ni
import numpy as np
import argparse
import os
from clindmri.registration.fsl import flirt
from glob import glob
import pandas as pd

from hopla.converter import hopla

# Parameters to keep trace
#__hopla__ = ["tool", "version"]

doc = """
python hopla_m02a_mni.py \
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
#            '-d', '/tmp/m02/',
#            '-p',
#            ])

dirent = glob(os.path.join(args.basedir, "*"))
dirent =[i for i in dirent if os.path.basename(i).isdigit()]
subjects = [os.path.basename(i) for i in dirent]
destdir = args.destdir

imagefiles = [os.path.join(args.basedir, '{}'.format(i),
                           'model02',
                           '{}_enh-gado_T1w_bfc.nii.gz'.format(i))
              for i in subjects]
destfiles = [os.path.join(destdir, '{}'.format(i), 'model03')
                for i in subjects]
for d in destfiles:
    if not os.path.exists(d):
        os.makedirs(d)
#
print imagefiles, destfiles
#
#if args.process:
#    logfile = "{}/log/MniSegLog/log.txt".format(args.destdir)
#    if not os.path.isdir(os.path.dirname(logfile)):
#        os.makedirs(os.path.dirname(logfile))
#
#    # regular processing
#    cmd = os.path.join(os.getenv('HOME'), 'gits',
#                     'scripts', '2017_rr', 'metastasis',
#                     'm03_mni.py')
#    status, exitcodes = hopla(cmd,
#                              i=imagefiles,
#                              d=destfiles,
#                              hopla_iterative_kwargs=["i", "d"],
#                              hopla_cpus=3,
#                              hopla_logfile=logfile,
#                              hopla_verbose=args.verbose)


# check and potential rescue mode
dirtocheck = [os.path.dirname(i.replace('model02', 'model03'))
              for i in imagefiles]
hatlist = [os.path.join(i, 'hatbox.nii.gz') for i in dirtocheck]
m2nlist = [os.path.join(i, 'mni2nat') for i in dirtocheck]
# criteria to trigger a rescue process (inion, nasion respective position)
imagekeeps = []
destkeeps = []
for i, (hat, m2n) in enumerate(zip(hatlist, m2nlist)):
    hat_scan = ni.load(hat).get_affine()
    m2n_trans = np.asarray(pd.read_csv(m2n, sep='  ',header=None))
    inion = np.asarray([100, 20, 80, 1])
    nasion = np.asarray([100, 190, 80, 1])
    # performe the xform in real (mm) world
    inion = m2n_trans.dot(hat_scan.dot(inion))
    nasion = m2n_trans.dot(hat_scan.dot(nasion))
    if not((inion[1] < 0) and (nasion[1] > 0)):
        imagekeeps.append(imagefiles[i])
        destkeeps.append(destfiles[i])

print imagekeeps, destkeeps
#
if args.process and len(imagekeeps) > 0:
    # rescue processing
    logfile = "{}/log/MniSegLog/rescuelog.txt".format(args.destdir)
    if not os.path.isdir(os.path.dirname(logfile)):
        os.makedirs(os.path.dirname(logfile))

    cmd = os.path.join(os.getenv('HOME'), 'gits',
                       'scripts', '2017_rr', 'metastasis',
                       'm03_mni.py')
    status, exitcodes = hopla(cmd,
                              i=imagekeeps,
                              d=destkeeps,
                              r=True,
                              hopla_iterative_kwargs=["i", "d"],
                              hopla_cpus=3,
                              hopla_logfile=logfile,
                              hopla_verbose=args.verbose)
