#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import argparse
import os
from zipfile import ZipFile
from numpy import unique
from shutil import rmtree
from glob import glob
import re
import sys


# Bredala import
try:
    import bredala
    bredala.USE_PROFILER = False
    #bredala.register("pydcmio.dcmreader.reader",
    #                 names=["get_values"])
except:
    pass



from hopla.converter import hopla

# Script documentation
doc = """
Dicom parser for Pitie NeuroRadio Import
========================================

This code fix and list the file to be processed to import NeuroRadio onco 
images.

Hopla nesting.

Steps:



Command:

python ~/gits/script/2017_rr/lpsnc/hl import_lpsnc.py \
    -i master/lpscnc/test32 \
    -o /volatile/frouin/radiomique_radiogenomique/vv \
    -t /volatile/frouin/radiomique_radiogenomique/base3/transcoding.json \
    -p 

"""

def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg
    
    
def define_parser():
    """
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
        help="increase the verbosity level: 0 silent, [1, 2] verbose.")
    parser.add_argument(
        "-p", "--process", dest="process", action='store_true',
        help="if activated execute.")
    parser.add_argument(
        "-t", "--transcoding", dest="transcoding", default=None, 
        help="the transcoding table")
    parser.add_argument(
        "-o", "--outdirectory", dest="outdir", default=None, metavar="PATH",
        help="the outdir that contains decoded nifti files")
    parser.add_argument(
        "-i", "--indir", dest="indir", required=True, metavar="PATH",
        help="the subject main directory with splitted DICOM files",
        type=is_dir)
    return parser


if __name__ == "__main__":

    # parsing
    parser = define_parser()
    args = parser.parse_args()
    logfile = os.path.join(args.outdir, 'logfile.txt')
    indirs = glob(os.path.join(args.indir, '*', 'Subj*'))[:1]
    
    if len(indirs) > 0:
        print("indirs {0} [{1} ... {2}])".format(
            len(indirs), indirs[0], indirs[-1]))
    else:
        print("no data")
        sys.exit(0)

    if args.process:
        #
        status, exitcodes = hopla(
            os.path.join(os.getenv('HOME'), "gits", "scripts", "2017_rr",
                         "lpsnc", "import_lpsnc.py"),
            i=indirs,
            o=args.outdir,
            t=args.transcoding,
            s='c',
            hopla_iterative_kwargs=["i"],
            hopla_cpus=1,
            hopla_logfile=logfile,
            hopla_verbose=1)
