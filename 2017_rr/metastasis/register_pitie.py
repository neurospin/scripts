#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from nilearn import plotting
import nibabel
import argparse
import os
from clindmri.registration.fsl import flirt
from glob import glob

try:
    import bredala
    bredala.USE_PROFILER = False
    bredala.register("clindmri.registration.fsl.flirt")    
    bredala.register("nilearn.plotting")
except:
    pass

doc = """
Perform unit registration of T2 on T1 image using fsl/flirt command.
Use mutual info and trigger sinc resampling

Create results according to path and prepare QC plots (from nilearn)

Command:
========

python register_pitie.py -r AxT1enhanced.nii.gz  -i  AxT2.nii.gz -o  rAxT2.nii.gz

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
    "-i", "--in", dest="infile", metavar="FILE",
    help="Image to register")
parser.add_argument(
    "-r", "--ref", dest="refile", metavar="FILE",
    help="Target/Reference image file")
parser.add_argument(
    "-o", "--out", dest="outfile", metavar="FILE",
    help="Image file that will contain the resampled result")
args = parser.parse_args()


# transform pathnames
outfile = os.path.abspath(args.outfile)
anat = args.refile
infile = args.infile
if not os.path.isdir(os.path.dirname(outfile)):
    os.makedirs(os.path.dirname(outfile))
omatfile = os.path.splitext(outfile)[0]
if os.path.splitext(omatfile)[0] != '':
    omatfile = '{}.txt'.format(os.path.splitext(outfile)[0])
else:
    omatfile = '{}.txt'.format(omatfile)
outfileAxi = '{}/qc_axi.pdf'.format(os.path.dirname(outfile))
outfileSag = '{}/qc_sag.pdf'.format(os.path.dirname(outfile))


# Register : call to the wrapping function
flirt(in_file=infile,
      ref_file=anat,
      omat=omatfile,
      out=outfile, 
      cost='mutualinfo',interp='sinc')

# QC :  pdf sheet
bg = nibabel.load(outfile)
anat = nibabel.load(anat)
# image axial
display = plotting.plot_anat(bg, title="T1 Gado contours", 
                             display_mode = 'z',
                             cut_coords = 10)
display.add_edges(anat)
display.savefig(outfileAxi)
display.close()
# mage coronal
display = plotting.plot_anat(bg, title="T1 Gado contours", 
                             display_mode = 'x',
                             cut_coords = 10)
display.add_edges(anat)
display.savefig(outfileSag)
display.close()