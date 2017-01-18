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
from numpy import unique, zeros, argmax
from shutil import rmtree
from glob import glob
import re
import subprocess
import tempfile

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
from pydcmio.dcmreader.reader import walk
from pydcmio.dcmreader.reader import get_values
from pydcmio.dcmreader.reader import STANDARD_EXTRACTOR
from pydcmio.dcmconverter.spliter import split_series
from pydcmio.dcmconverter.converter import dcm2niix

from hopla.converter import hopla

# Parameters to keep trace
__hopla__ = ["tool", "version", "sids"]

SEQ_TES_MAP={'AxT1enhanced':{'min':3, 'max':5},  #durations for TEs
             'AxT2FLAIR':{'min':110, 'max':150}, #durations for TEs
             'AxT2':{'min':80, 'max':109},       #durations for TEs
             }
SEQ_NUMF_MAP={'AxT1enhanced':{'min':100, 'max':300},  #numfiles
             'AxT2FLAIR':{'min':25, 'max':150},      #numfiles
             'AxT2':{'min':20, 'max':109},           #numfiles
             }
SEQ_NAME_MAP ={
    'AX FSPGR G':               'AxT1enhanced',
    'AX FSPGR 1.2 MM 1 PLAN':   'AxT1enhanced',
    'AX FSPGR 3D 1.2MM':        'AxT1enhanced',
    'AX FSPGR 3D HD 1.2MM':     'AxT1enhanced',
    'AX CUBE T1':               'AxT1enhanced',
    'T1-3D-GADO.*':               'AxT1enhanced',
    '3D-IR-FSPGR-G':            'AxT1enhanced',
    '3D GADO':                  'AxT1enhanced',
    '3D T1 BRAVO GADO':         'AxT1enhanced',
    '3D-T1-BRAVO-G.*':         'AxT1enhanced',
    '3D-T1-GADO':               'AxT1enhanced',
    '3D-T1-G':                  'AxT1enhanced',
    'AX-T1-SE-G.*':            'AxT1enhanced',
    'AX T2* GRE' :              'AxT2FLAIR',
    'AXFLAIR GADO' :            'AxT2FLAIR',
    'AXFLAIR' :                 'AxT2FLAIR',
    'AX FLAIR' :                'AxT2FLAIR',
    'AX T2 FLAIR 5/0' :         'AxT2FLAIR',
    '.*FLAIR' :                 'AxT2FLAIR',}
SEQ_NAME = SEQ_NAME_MAP.keys()
REQ_SEQ_NAME = unique(SEQ_NAME_MAP.values())

# Script documentation
doc = """
Dicom converter for LPSNC pitie Import
======================================

This code list the file to be processed to import NeuroRadio onco 
images.

Based on split_series choose one serie for AxT1enhanced, AxT2, and AxT2FLAIR

Steps:

Command:

python  ~/gits/scripts/2017_rr/lpscnc/import_lpsnc.py \
    -i master/lpscnc/test32/1071388/Subj4 \
    -o working2 \
    -t /volatile/frouin/radiomique_radiogenomique/base3/transcoding.json \
    --steps d


"""


def searchfor(target, snames, snums, stes):
    """ Look for specific expected series

    search in snames list the first item that matches a ref_seq_name with 
    correct type. ref_seq_name is read from SEQ_NAME_MAP 
    
    Return:
    =======
        an int. The index in the list or -1 if none found
    """

    votes = zeros(len(snames)).tolist()
    #
    if not target in unique(SEQ_NAME_MAP.values()):
        print('target should be {}'.format(unique(SEQ_NAME_MAP.values())))
        return votes
    for r, sname in enumerate(snames):
        for ref_seq_name in SEQ_NAME_MAP.keys():
            if SEQ_NAME_MAP[ref_seq_name] == target:
                if re.compile(ref_seq_name, re.I).match(sname.upper()) is not None:
                    votes[r] += 1
        if votes[r] >= 1:
            if ((SEQ_TES_MAP[target]['min'] < stes[r]) &
                (SEQ_TES_MAP[target]['max'] > stes[r])):
                votes[r] += 10
            if ((SEQ_NUMF_MAP[target]['min'] < snums[r]) &
                (SEQ_NUMF_MAP[target]['max'] > snums[r])):
                votes[r] += 100
    return votes


def detect_series(wd):
    """
    """
    # init output struct
    ret = dict()
    for i in REQ_SEQ_NAME:
        ret[i] = None

    # double loop
    series = glob(os.path.join(wd, '*'))
    # print
    for nd in series:
#        print(nd)
#        print(glob(nd))
        print(os.path.basename(nd).split('_'),
              'numfiles: ',
              len(glob(os.path.join(nd, '*'))))
    #
    snames = [os.path.basename(i).split('_')[0] for i in series]
    snums = [len(glob(os.path.join(i, '*'))) for i in series]
    stes = [float(os.path.basename(i).split('_')[1]) for i in series]
    # look for all the required series
    for reqserie in REQ_SEQ_NAME:
        votes = searchfor(reqserie, snames, snums, stes)
        for k, v in enumerate(votes):
            if v > 0.:
                print('[{0}] {1}: {2}'.format(reqserie, k, v))
#        print(">>>>>", votes, argmax(votes), series[argmax(votes)])
        if any([i > 0 for i in votes]):
            ret[reqserie] = series[argmax(votes)]
#            print('<<<<< ', ret)

    return ret


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
        "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
        help="increase the verbosity level: 0 silent, [1, 2] verbose.")
    parser.add_argument(
        "-i", "--indir", dest="indir", required=True, metavar="PATH",
        help="entry containing splitted series",
        type=is_dir)
    parser.add_argument(
        "-o", "--outdirectory", dest="outdir", default=None, metavar="PATH",
        help="the outdir that contains decoded nifti files")
    parser.add_argument(
        "-t", "--transcoding", dest="transcoding", default=None, 
        help="transcoding table")
    parser.add_argument(
        "-s", "--step", dest="steps",  action='store', default='d',
        choices=['d', 'c'],
        help='Specify jobs todo d is detect and'
             'c is detect&convert.')
    return parser

# parsing
tool = "import_lpsnc"
parser = define_parser()
args = parser.parse_args()
#
#Pass 1
if 'd' in args.steps:
    print('=======', args.indir)
    selseries = detect_series(args.indir)
    print('=======')
    print(selseries)
print('\n')

#Pass 2
if 'c' in args.steps:
    selseries = detect_series(args.indir)
    sids = []
    returncode = 0
    if args.outdir is not None:
        for reqserie in REQ_SEQ_NAME:
            if selseries[reqserie] is not None:
                sid = selseries[reqserie].split('/')[-3]
                sids.append(sid)
                indir =  os.path.abspath(selseries[reqserie])
                cmd = ['python',
                       '/home/vf140245/gits/pydcmio/pydcmio/scripts/pydcmio_dicom2nifti',
                       '-r', args.transcoding,
                       '-t', '-x',
                       '-d', indir,
                       '-s', sid,
                       '-p', reqserie,
                       '-o', args.outdir]
                print(' '.join(cmd))
                returncode += subprocess.call(cmd)

    if returncode > 0:
        raise ValueError("One of the conversion failed.")
