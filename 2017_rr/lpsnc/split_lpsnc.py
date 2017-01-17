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
from numpy import unique
from shutil import rmtree
from glob import glob

# Bredala import
try:
    import bredala
    bredala.USE_PROFILER = False
    #bredala.register("pydcmio.dcmreader.reader",
    #                 names=["get_values"])
except:
    pass

# Dcmio import
from pydcmio import __version__ as version
from pydcmio.dcmconverter.spliter import split_series

from hopla.converter import hopla

# Parameters to keep trace
__hopla__ = ["tool", "version"]


# Script documentation
doc = """
Dicom converter for LPSNC pitie Import
======================================

This code split one dicom dir according to the contained series

Steps:

Command:

python  ~/gits/scripts/2017_rr/lpscnc/split_lpsnc.py \
    -i master/lpscnc/test32/1071388/Subj4 \
    -o working2

"""


def is_dir(dirarg):
    """ Type for argparse - checks that output dir exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The dir '{0}' does not exist!".format(dirarg))
    return dirarg


def define_parser():
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2],
        default=0,
        help="increase the verbosity level: 0 silent, [1, 2] verbose.")
    parser.add_argument(
        "-p", "--process", dest="process", action='store_true',
        help="if activated execute.")
    parser.add_argument(
        "-l", "--logfile", dest="logfile", metavar="FILE",
        help="logfile filename")
    parser.add_argument(
        "-i", "--indirectory", dest="dcmdir", required=True, metavar="PATH",
        help="the subject main directory to parse for DICOM files",
        type=is_dir)
    parser.add_argument(
        "-o", "--outdirectory", dest="outdir", default=None, metavar="PATH",
        help="the outdir that will contain dicom splitted files",
        type=is_dir)
    return parser


if __name__ == "__main__":

    # parsing
    parser = define_parser()
    args = parser.parse_args()
    log_msg = []

    indir = glob(os.path.join(args.dcmdir, '*', 'Subj*'))
    outdir = []
    for i in indir:
        h, s = os.path.split(i)
        _, n = os.path.split(h)
        outdir.append(os.path.join(args.outdir, n, s))

    if not args.process:
        os.makedirs(outdir[0])
        split_series(indir[0], outdir[0])
    else:
        if args.process:
            logfile = os.path.join(os.path.realpath(args.outdir), "log.txt")
            #
            for i in outdir:
                os.makedirs(i)
            #
            status, exitcodes = hopla(
                os.path.join(os.getenv('HOME'),
                             "gits", "pydcmio", "pydcmio",
                             "scripts", "pydcmio_splitseries"),
                i=indir,
                o=outdir,
                v=0,
                hopla_iterative_kwargs=["i", "o"],
                hopla_cpus=2,
                hopla_logfile=logfile,
                hopla_verbose=1)
